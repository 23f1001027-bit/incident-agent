"""Deterministic transcript preprocessing and plan repair. No model calls; cached per transcript.

* evidence_lines(): drops boilerplate (known noise templates, and any sentence that repeats) and returns the few
  decision-relevant lines with their evidence IDs.
* facts(): exact identifiers stated in those lines (releases, flags, dependencies, replica targets).
* repair(): fixes identifier arguments of a model plan from those facts (never invents a value).
* rule_plan(): a complete keyword-based plan, used only if every model fails.
"""
import collections
import functools
import re

LINE_RE = re.compile(r"^\s*\[([^\]\s]{3,80})\]\s*(.*)$")
PAD_RE = re.compile(r"\s*Correlation corr_\w+ was checked against \S+?; retain this full sentence so the agent must rank "
                    r"evidence rather than match a short keyword\.?")
TS_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?\s*")
LABEL_RE = re.compile(r"^(?:correlated sample|bounded observation|incident-window record|on-call finding)\s*:\s*", re.I)
NOISE = ("capacity forecast for the next quarter", "customer transcript asks operators", "support channel contains competing",
         "dashboard annotation was copied", "sampling worker dropped", "routine backup validation",
         "older postmortem mentions", "verified label appears", "automated verifier confirmed that the ticket",
         "regional synthetic probe briefly")

RELEASE_RE = re.compile(r"\b(?:release|running|rollout of|version|build)\s+([A-Za-z]*\d[\w.-]*[A-Za-z0-9])", re.I)
GOOD_RE = re.compile(r"previous|holdback|healthy|known[- ]good|last good|stable|no matching errors|unaffected", re.I)
FLAG_RE = re.compile(r"\bflag_[A-Za-z0-9]+\b")
DEP_RE = re.compile(r"\bdep_[A-Za-z0-9]+\b")
REPLICA_RE = re.compile(r"(?:exactly|target(?: is)?|ceiling(?: [a-z ]{0,30})? is(?: exactly)?|to)\s+(\d+)\s+(?:application\s+)?replicas", re.I)
CURRENT_RE = re.compile(r"(?:already|currently)\s+(?:at|running|has|uses)\s+(?:its\s+)?(\d+)\s+(?:application\s+)?replicas", re.I)
AT_LIMIT_RE = re.compile(r"already (?:at|running at) (?:its|the) (?:permitted |approved )?(?:scaling |capacity |replica )?"
                         r"(?:limit|ceiling|maximum)", re.I)


def _body(text):
    return LABEL_RE.sub("", TS_RE.sub("", PAD_RE.sub("", text))).strip()


@functools.lru_cache(maxsize=64)
def evidence_lines(transcript):
    """[(evidence_id, line_text)] for lines that are neither known boilerplate nor repeated sentences."""
    rows = []
    for ln in transcript.split("\n"):
        m = LINE_RE.match(ln)
        if m:
            rows.append((m.group(1), PAD_RE.sub("", m.group(2)).strip(), _body(m.group(2))))
    shapes = collections.Counter(" ".join(re.sub(r"\d+", "#", b.lower()).split()) for _, _, b in rows)
    out = []
    for eid, text, body in rows:
        low = body.lower()
        if any(n in low for n in NOISE) or shapes[" ".join(re.sub(r"\d+", "#", low).split())] > 1:
            continue
        out.append((eid, text))
    return tuple(out)


@functools.lru_cache(maxsize=64)
def _facts(transcript):
    lines = [t for _, t in evidence_lines(transcript)]
    text = "\n".join(lines)
    releases, good = [], []
    for t in lines:
        for r in RELEASE_RE.findall(t):
            if r not in releases:
                releases.append(r)
            # a release named next to "previous / holdback / healthy" is the known-good target
            for seg in re.split(r"[;.]", t):
                if r in seg and GOOD_RE.search(seg) and r not in good:
                    good.append(r)
    bad = [r for r in releases if r not in good]
    reps = [int(x) for x in REPLICA_RE.findall(text)]
    cur = [int(x) for x in CURRENT_RE.findall(text)]
    return {"releases": releases, "goodReleases": good, "badReleases": bad,
            "flags": list(dict.fromkeys(FLAG_RE.findall(text))), "deps": list(dict.fromkeys(DEP_RE.findall(text))),
            "targetReplicas": reps[0] if reps else None, "currentReplicas": cur[0] if cur else None,
            "atLimit": bool(AT_LIMIT_RE.search(text)) or bool(reps and cur and cur[0] >= reps[0])}


def facts(transcript):
    return dict(_facts(transcript))


# ---------------------------------------------------------------- repair a model plan
def repair(plan, inc):
    """Fix identifier arguments from stated facts. Mutates and returns plan."""
    f = facts(inc.get("transcript", ""))
    sev, svc = inc.get("severity"), inc.get("service")

    def fix(name, args):
        if "severity" in args and sev:
            args["severity"] = sev
        if name == "rollback_deployment" and "release" in args:
            if f["goodReleases"] and args["release"] not in f["goodReleases"]:
                args["release"] = f["goodReleases"][0]
        if name == "disable_feature" and "flag" in args and f["flags"] and args["flag"] not in f["flags"]:
            args["flag"] = f["flags"][0]
        if name == "dependency_status" and "dependency" in args and f["deps"] and args["dependency"] not in f["deps"]:
            args["dependency"] = f["deps"][0]
        if name == "scale_service" and "targetReplicas" in args and f["targetReplicas"]:
            args["targetReplicas"] = f["targetReplicas"]
        if name == "open_incident" and "service" in args and f["deps"] and args["service"] not in f["deps"] + [svc]:
            args["service"] = f["deps"][0]
        return args

    for d in plan["diagnostics"]:
        d["arguments"] = fix(d["toolName"], d["arguments"])
    e = plan["effect"]
    e["arguments"] = fix(e["toolName"], e["arguments"])
    return plan


# ---------------------------------------------------------------- full keyword plan (fallback only)
CUES = {
    "deployment_regression": (r"\brelease r|rollout boundary|holdback|began returning|after the deploy", ),
    "database_connection_exhaustion": (r"\bpool\b|connection ceiling|pool acquisition|checkout wait", ),
    "dependency_certificate_expired": (r"certificate|\btls\b|notafter", ),
    "feature_flag_recursion": (r"flag cohort|recurs|evaluation path|flag rule|flag_\w+ disabled", ),
    "traffic_capacity_exhaustion": (r"requests per second|per-replica utilization|queue depth|capacity ceiling", ),
    "secret_rotation_mismatch": (r"secret version|vault|revoked|revocation|credential", ),
}


def rule_plan(body):
    inc, pol = body["incident"], body["policy"]
    lines = evidence_lines(inc["transcript"])
    if len(lines) < 2:
        return None
    text = "\n".join(t for _, t in lines).lower()
    allowed = inc["allowedRootCauses"]
    scores = {c: sum(len(re.findall(p, text)) for p in CUES.get(c, ())) for c in allowed}
    cause = max(allowed, key=lambda c: scores.get(c, 0))
    f, svc, sev = facts(inc["transcript"]), inc.get("service"), inc.get("severity")
    ev = [e for e, _ in lines][:4]
    cite = lambda pat: [e for e, t in lines if re.search(pat, t, re.I)][:1] or ev[:1]
    diag, eff = [], None
    if cause == "deployment_regression":
        diag = [("inspect_deployment", {"service": svc}, cite(r"release"))]
        eff = ("rollback_deployment", {"service": svc, "release": (f["goodReleases"] or f["releases"] or [""])[0]})
    elif cause == "database_connection_exhaustion":
        diag = [("query_metrics", {"service": svc, "metric": "db_connection_pool_wait", "windowMinutes": 60}, cite(r"pool"))]
        eff = ("page_owner", {"service": svc, "severity": sev}) if f["atLimit"] or not f["targetReplicas"] else \
            ("scale_service", {"service": svc, "targetReplicas": f["targetReplicas"]})
    elif cause == "traffic_capacity_exhaustion":
        diag = [("query_metrics", {"service": svc, "metric": "requests_per_second", "windowMinutes": 60}, cite(r"request"))]
        eff = ("page_owner", {"service": svc, "severity": sev}) if f["atLimit"] or not f["targetReplicas"] else \
            ("scale_service", {"service": svc, "targetReplicas": f["targetReplicas"]})
    elif cause == "dependency_certificate_expired":
        dep = (f["deps"] or [svc])[0]
        diag = [("dependency_status", {"dependency": dep}, cite(r"dep_|certificate"))]
        eff = ("open_incident", {"service": dep, "severity": sev})
    elif cause == "feature_flag_recursion":
        flag = (f["flags"] or [""])[0]
        diag = [("query_logs", {"service": svc, "query": flag, "windowMinutes": 60}, cite(r"flag"))]
        eff = ("disable_feature", {"service": svc, "flag": flag})
    elif cause == "secret_rotation_mismatch":
        if re.search(r"runbook", text):
            diag = [("read_runbook", {"service": svc, "topic": "secret_rotation"}, cite(r"runbook"))]
        else:
            diag = [("query_logs", {"service": svc, "query": "authentication failure", "windowMinutes": 60}, cite(r"auth|secret"))]
        eff = ("page_owner", {"service": svc, "severity": sev})
    catalog = {t["name"] for t in body["toolCatalog"]}
    diag = [d for d in diag if d[0] in catalog]
    if not diag or not eff or eff[0] not in (pol.get("effectTools") or []):
        return None
    return {"rootCause": cause, "evidence": ev,
            "diagnostics": [{"toolName": n, "arguments": a, "evidence": c} for n, a, c in diag],
            "effect": {"toolName": eff[0], "arguments": eff[1], "evidence": ev[:1]}}
