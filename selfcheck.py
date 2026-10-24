"""GET /selfcheck: a local mock of the grader. Uses no Check.

Runs the real incidents through this deployment (including the real model call), plays the grader's role
(receipts, a 503 retry, a timeout, an approval, replays, conflicts) and scores the same categories the grader
reports. It verifies everything that is observable from the spec. It cannot know the grader's hidden answer key:
the "decision" rows compare against our own reading of each case and are labelled as such.
"""
import copy
import hashlib
import json
import re
import threading
import time
import uuid

import rules
import selfcheck_cases as cases

TP_RE = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-01$")
TOP = ["runId", "status", "diagnosis", "dispatches", "approvals"]
DISPATCH = ["actionId", "callId", "phase", "toolName", "arguments", "evidence", "attempt", "traceparent"]
INCOMING = ("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01", "vendor=selfcheck")
SCENARIO = {"deploy": "approval", "flag": "approval", "db": "happy", "cert": "happy", "traffic": "retry503",
            "secret": "timeout", "audit": "audit", "audit2": "audit"}


class Run:
    def __init__(self, handle, name):
        self.handle, self.name = handle, name
        self.run_id = f"run_selfcheck_{name}_{uuid.uuid4().hex[:10]}"
        self.body = cases.body(name, self.run_id, seed=11)
        self.canaries = list(self.body["sensitive"].values())
        self.problems = {k: [] for k in ("transport", "proposal", "decision", "semantics", "topology",
                                         "correlation", "lifecycle", "durability", "redaction")}
        self.dispatches, self.outcomes, self.approval_receipts, self.latency = [], [], [], []
        self.final = self.first = None

    # -------------------------------------------------------------- plumbing
    def bad(self, cat, msg):
        self.problems[cat].append(msg)

    def call(self, method, path, body=None, headers=None):
        raw = json.dumps(body).encode() if body is not None else b""
        t = time.time()
        status, h, payload = self.handle(method, path, headers or {}, raw)
        secs = time.time() - t
        text = payload.decode("utf-8", "replace")
        for c in self.canaries:
            if c in text:
                self.bad("redaction", f"{method} {path} response contains a sensitive value")
        if h.get("content-type", "").split(";")[0] != "application/json":
            self.bad("transport", f"{method} {path} content-type {h.get('content-type')}")
        if len(payload) > 768 * 1024:
            self.bad("transport", f"{method} {path} response larger than 768 KiB")
        try:
            data = json.loads(text)
        except ValueError:
            self.bad("transport", f"{method} {path} returned non-JSON")
            data = None
        return status, data, secs

    def receipt(self, rid, outcomes=None, approvals=None):
        body = {"receiptId": rid}
        if outcomes:
            body["outcomes"] = outcomes
        if approvals:
            body["approvals"] = approvals
        status, data, _ = self.call("POST", f"/v2/incidents/{self.run_id}/receipts", body)
        if status != 200:
            self.bad("lifecycle", f"receipt {rid} rejected with HTTP {status}: {data}")
            return body, None
        self.outcomes += [{**o, "receiptId": rid} for o in outcomes or []]
        self.approval_receipts += [{**a, "receiptId": rid} for a in approvals or []]
        self.collect(data)
        return body, data

    def collect(self, data):
        for d in data.get("dispatches") or []:
            self.dispatches.append(copy.deepcopy(d))

    @staticmethod
    def outcome(d, status=200, **kw):
        return {"actionId": d["actionId"], "callId": d["callId"], "attempt": d["attempt"], "status": status,
                "resultClass": "diagnosis_confirmed" if status == 200 else "error", "nonce": str(uuid.uuid4()), **kw}

    # -------------------------------------------------------------- the run
    def execute(self):
        headers = {"traceparent": INCOMING[0], "tracestate": INCOMING[1]} if self.name == "deploy" else {}
        self.headers = headers
        status, first, secs = self.call("POST", "/v2/incidents", self.body, headers)
        self.latency.append(round(secs, 1))
        if secs > 12:
            self.bad("transport", f"first POST took {secs:.1f}s (grader posts incidents one after another; keep < ~10s)")
        if status != 200 or not first:
            self.bad("transport", f"first POST returned HTTP {status}")
            return
        self.first = first
        self.collect(first)
        self.check_proposal(first)
        scen = SCENARIO[self.name]
        pending = [d for d in first.get("dispatches", [])]
        effect_seen = approval_seen = False
        timed_out = False
        rid = 0
        guard = 0
        while pending and guard < 8:
            guard += 1
            rid += 1
            outs = []
            for i, d in enumerate(pending):
                if scen == "retry503" and i == 0 and d["attempt"] == 1 and d["phase"] == "diagnostic":
                    outs.append(self.outcome(d, 503))
                elif scen == "timeout" and d["phase"] == "diagnostic" and (
                        d["toolName"] == "read_runbook" or (i == 0 and not any(x["toolName"] == "read_runbook" for x in pending))):
                    outs.append(self.outcome(d, 0, errorType="timeout"))
                    timed_out = True
                else:
                    outs.append(self.outcome(d))
            _, data = self.receipt(f"rcpt_{rid}_{uuid.uuid4().hex[:6]}", outs)
            if data is None:
                return
            if any(o["status"] == 503 for o in outs):
                if data.get("approvals"):
                    self.bad("lifecycle", "a retry response contained an approval request")
                retry = [d for d in data.get("dispatches", []) if d["attempt"] == 2]
                if not retry:
                    self.bad("lifecycle", "503 did not produce exactly one retry")
            pending = list(data.get("dispatches") or [])
            if any(d["phase"] != "diagnostic" for d in pending):
                effect_seen = True
            if data.get("approvals"):
                approval_seen = True
                ap = data["approvals"][0]
                if pending:
                    self.bad("lifecycle", "approval request returned together with dispatches")
                self.check_approval_request(ap)
                nonce = str(uuid.uuid4())
                _, data2 = self.receipt(f"apr_{uuid.uuid4().hex[:8]}",
                                        approvals=[{"approvalId": ap["approvalId"], "decision": "approved", "nonce": nonce}])
                if data2 is None:
                    return
                eff = [d for d in data2.get("dispatches", []) if d["phase"] != "diagnostic"]
                if len(eff) != 1 or eff[0].get("approvalId") != ap["approvalId"] or eff[0].get("approvalNonce") != nonce \
                        or eff[0]["actionId"] != ap["actionId"]:
                    self.bad("lifecycle", "approved effect dispatch missing approvalId/approvalNonce or wrong actionId")
                pending = list(data2.get("dispatches") or [])
                effect_seen = True
            if data.get("status") in ("completed", "failed"):
                break
        status, final, _ = self.call("GET", f"/v2/incidents/{self.run_id}")
        if status != 200 or not final:
            self.bad("durability", "GET of the run failed")
            return
        self.final = final
        if final.get("status") not in ("completed", "failed"):
            self.bad("lifecycle", f"run never reached a final state (status {final.get('status')})")
            return
        if timed_out:
            if effect_seen:
                self.bad("lifecycle", "an effect was dispatched after its prerequisite diagnostic timed out")
            if final["status"] != "failed" or not final.get("suppressed"):
                self.bad("lifecycle", "timeout run should end 'failed' with the effect listed in 'suppressed'")
        else:
            if final["status"] != "completed":
                self.bad("lifecycle", f"run ended '{final['status']}' instead of 'completed'")
        needs_approval = (final.get("chosenEffect") or "") in self.body["policy"]["approvalRequiredFor"]
        if needs_approval and not approval_seen:
            self.bad("lifecycle", "destructive effect completed without an approval round")
        self.check_semantics(final)
        self.check_final_logs(final)
        self.check_trace(final)
        if scen == "audit":
            self.lifecycle_probes(final)

    # -------------------------------------------------------------- category checks
    def check_proposal(self, r):
        if list(r)[:5] != TOP:
            self.bad("proposal", f"first keys {list(r)[:5]} != {TOP}")
        for k in ("actionLog", "receiptLog"):
            if not isinstance(r.get(k), list):
                self.bad("correlation", f"waiting response lacks mandatory array {k}")
        if r.get("status") != "waiting" or r.get("runId") != self.run_id:
            self.bad("proposal", "status/runId wrong")
        inc = self.body["incident"]
        dg = r.get("diagnosis", {})
        if dg.get("rootCause") not in inc["allowedRootCauses"]:
            self.bad("proposal", "rootCause not allowed")
        ids = set(re.findall(r"^\s*\[([^\]\s]+)\]", inc["transcript"], re.M))
        ev = dg.get("evidence") or []
        if not 2 <= len(ev) <= 4 or len(set(ev)) != len(ev) or not set(ev) <= ids:
            self.bad("proposal", f"diagnosis evidence invalid {ev}")
        cat = {t["name"]: t for t in self.body["toolCatalog"]}
        effects = set(self.body["policy"]["effectTools"])
        ds = r.get("dispatches") or []
        if not 1 <= len(ds) <= self.body["policy"]["maximumDiagnostics"]:
            self.bad("proposal", f"{len(ds)} diagnostic dispatches")
        if r.get("approvals") != []:
            self.bad("proposal", "approvals not empty in first response")
        for d in ds:
            want = DISPATCH + (["tracestate"] if "tracestate" in d else [])
            if list(d) != want:
                self.bad("proposal", f"dispatch keys {list(d)}")
            if d.get("phase") != "diagnostic" or d.get("toolName") not in cat or d["toolName"] in effects:
                self.bad("proposal", f"dispatch {d.get('toolName')} is not a diagnostic catalog tool")
            for k in cat.get(d.get("toolName"), {}).get("inputSchema", {}).get("required", []):
                if d.get("arguments", {}).get(k) in (None, "", [], {}):
                    self.bad("proposal", f"{d.get('toolName')} missing argument {k}")
            de = d.get("evidence") or []
            if not de or len(set(de)) != len(de) or not set(de) <= set(ev):
                self.bad("proposal", "dispatch evidence not a subset of diagnosis evidence")
            if d.get("attempt") != 1:
                self.bad("proposal", "attempt != 1")
            for k in ("actionId", "callId"):
                if not isinstance(d.get(k), str) or len(d[k]) < 8:
                    self.bad("proposal", f"{k} shorter than 8")
            m = TP_RE.match(d.get("traceparent", ""))
            if not m:
                self.bad("topology", "bad dispatch traceparent")
            elif self.headers and m.group(1) != INCOMING[0].split("-")[1]:
                self.bad("topology", "dispatch does not continue the incoming trace")
            if self.headers and d.get("tracestate") != INCOMING[1]:
                self.bad("topology", "incoming tracestate not preserved")
            if not self.headers and "tracestate" in d:
                self.bad("topology", "tracestate present without an incoming one")
        _, decisive, exp = cases.CASES[self.name]
        if dg.get("rootCause") != exp["rootCause"]:
            self.bad("decision", f"rootCause {dg.get('rootCause')} (we expect {exp['rootCause']})")
        if set(ev) - {e for e, _ in decisive}:
            self.bad("decision", "cites boilerplate lines as evidence")

    def check_approval_request(self, ap):
        if set(ap) != {"approvalId", "actionId", "toolName", "argumentsDigest"}:
            self.bad("lifecycle", f"approval keys {sorted(ap)}")
        eff = self.final_effect_args(ap)
        if eff is not None:
            dg = hashlib.sha256(json.dumps(eff, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if ap.get("argumentsDigest") != dg:
                self.bad("lifecycle", "argumentsDigest does not match the effect arguments")

    def final_effect_args(self, ap):
        self._ap = ap
        return None          # verified after the effect is dispatched (check_semantics)

    def check_semantics(self, final):
        eff = [d for d in final.get("actionLog", []) if d.get("phase") != "diagnostic"]
        _, _, exp = cases.CASES[self.name]
        tool, args = exp["effect"]
        if SCENARIO[self.name] == "timeout":
            if eff:
                self.bad("semantics", "effect executed although a prerequisite timed out")
            return
        if len({d["actionId"] for d in eff}) != 1:
            self.bad("semantics", f"expected exactly one logical effect, got {len({d['actionId'] for d in eff})}")
            return
        e = eff[0]
        if e["toolName"] != tool or any(e["arguments"].get(k) != v for k, v in args.items()):
            self.bad("decision", f"effect {e['toolName']} {e['arguments']} (we expect {tool} {args})")
        if final.get("chosenEffect") != e["toolName"]:
            self.bad("semantics", "chosenEffect does not name the executed effect")
        ap = getattr(self, "_ap", None)
        if ap:
            dg = hashlib.sha256(json.dumps(e["arguments"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if ap["argumentsDigest"] != dg or ap["toolName"] != e["toolName"]:
                self.bad("lifecycle", "approval digest/tool does not match the executed effect")

    def check_final_logs(self, final):
        if final.get("dispatches") not in (None, []) or final.get("approvals") not in (None, []):
            self.bad("lifecycle", "final response still has pending work")
        if final.get("actionLog") != self.dispatches:
            self.bad("correlation", "actionLog is not every dispatch exactly as issued")
        rl = final.get("receiptLog")
        if not isinstance(rl, list):
            self.bad("correlation", "receiptLog missing")
            return
        want = [{k: o[k] for k in ("receiptId", "actionId", "callId", "attempt", "status", "resultClass", "nonce")}
                for o in self.outcomes]
        got = [{k: r.get(k) for k in ("receiptId", "actionId", "callId", "attempt", "status", "resultClass", "nonce")}
               for r in rl if "callId" in r]
        if got != want:
            self.bad("correlation", "receiptLog tool outcomes do not match the receipts sent")
        apr = [r for r in rl if "approvalId" in r]
        if [(r["receiptId"], r["approvalId"], r["decision"], r["nonce"]) for r in apr] != \
                [(a["receiptId"], a["approvalId"], a["decision"], a["nonce"]) for a in self.approval_receipts]:
            self.bad("correlation", "receiptLog approval entries do not match")

    def check_trace(self, final):
        try:
            spans = final["otlp"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
        except (KeyError, IndexError, TypeError):
            self.bad("topology", "otlp missing")
            return
        A = lambda s: {a["key"]: list(a["value"].values())[0] for a in s.get("attributes", [])}
        by = {}
        for s in spans:
            if s["spanId"] in by:
                self.bad("topology", "duplicate span id")
            by[s["spanId"]] = s
            for k in ("traceId", "spanId"):
                if not re.fullmatch(r"[0-9a-f]+", s.get(k, "")) or int(s[k], 16) == 0:
                    self.bad("topology", f"bad {k}")
            a = A(s)
            if a.get("ga5.run.id") != self.run_id or a.get("ga5.public.marker") != self.body["publicMarker"]:
                self.bad("topology", f"span {s['name']} lacks run id / public marker")
            if "gen_ai.tool.call.arguments" in a or "gen_ai.tool.call.result" in a:
                self.bad("redaction", "tool arguments/results exported")
        names = [s["name"] for s in spans]
        server = [s for s in spans if s["name"] == "POST /v2/incidents" and s["kind"] == 2]
        agent = [s for s in spans if s["name"] == "invoke_agent incident-response" and s["kind"] == 1]
        chat = [s for s in spans if s["name"] == "chat incident-plan" and s["kind"] == 3]
        if len(server) != 1 or len(agent) != 1 or len(chat) != 1:
            self.bad("topology", f"need exactly one SERVER/agent/chat span, got {len(server)}/{len(agent)}/{len(chat)}")
            return
        tid = server[0]["traceId"]
        if any(s["traceId"] != tid for s in spans):
            self.bad("topology", "spans use different trace ids")
        if self.headers:
            if tid != INCOMING[0].split("-")[1] or server[0].get("parentSpanId") != INCOMING[0].split("-")[2]:
                self.bad("topology", "SERVER span does not continue the incoming trace")
        if agent[0].get("parentSpanId") != server[0]["spanId"] or chat[0].get("parentSpanId") != agent[0]["spanId"]:
            self.bad("topology", "agent/chat parent links wrong")
        ca = A(chat[0])
        if ca.get("gen_ai.operation.name") != "chat" or not ca.get("gen_ai.request.model"):
            self.bad("topology", "chat span lacks gen_ai attributes")
        execs = {A(s).get("ga5.action.id"): s for s in spans if s["name"].startswith("execute_tool ")}
        actions = list(dict.fromkeys(d["actionId"] for d in final["actionLog"]))
        if set(execs) != set(actions) or len(execs) != len([s for s in spans if s["name"].startswith("execute_tool ")]):
            self.bad("topology", "execute_tool spans != logical executed actions")
        for aid, s in execs.items():
            a = A(s)
            if s.get("parentSpanId") != agent[0]["spanId"] or s["kind"] != 1 or a.get("gen_ai.operation.name") != "execute_tool" \
                    or not a.get("gen_ai.tool.name") or not a.get("gen_ai.tool.call.id"):
                self.bad("topology", f"execute_tool span for {aid} malformed")
        clients = {s["spanId"]: s for s in spans if s["name"].startswith("POST tool/")}
        if len(clients) != len(final["actionLog"]):
            self.bad("topology", "CLIENT tool spans != physical attempts")
        recs = {(r["callId"], r["attempt"]): r for r in final["receiptLog"] if "callId" in r}
        for d in final["actionLog"]:
            sid = d["traceparent"].split("-")[2]
            s = clients.get(sid)
            if not s:
                self.bad("correlation", f"no CLIENT span with the dispatch span id of {d['toolName']}")
                continue
            a = A(s)
            ex = execs.get(d["actionId"])
            if s["kind"] != 3 or not ex or s.get("parentSpanId") != ex["spanId"]:
                self.bad("topology", "CLIENT tool span parent/kind wrong")
            r = recs.get((d["callId"], d["attempt"]))
            if not r:
                continue
            if a.get("ga5.action.id") != d["actionId"] or a.get("ga5.attempt") != d["attempt"] \
                    or a.get("ga5.receipt.id") != r["receiptId"] or a.get("ga5.receipt.nonce") != r["nonce"] \
                    or a.get("http.request.method") != "POST" or a.get("http.request.resend_count") != d["attempt"] - 1:
                self.bad("correlation", f"CLIENT span attributes do not match dispatch/receipt for {d['toolName']}#{d['attempt']}")
            code = s.get("status", {}).get("code")
            if r["status"] == 503 and (code != 2 or a.get("error.type") != "503"):
                self.bad("topology", "503 attempt not marked as error 503")
            if r["status"] == 0 and (code != 2 or a.get("error.type") != "timeout"):
                self.bad("topology", "timeout attempt not marked as error timeout")
            if 200 <= r["status"] < 300 and (code == 2 or "error.type" in a):
                self.bad("topology", "successful attempt marked as error")
        diags = [aid for aid in actions if any(d["actionId"] == aid and d["phase"] == "diagnostic" for d in final["actionLog"])]
        joins = [s for s in spans if s["name"] == "incident.join"]
        if len(diags) >= 2:
            if len(joins) != 1 or joins[0].get("parentSpanId") != agent[0]["spanId"] or \
                    {l["spanId"] for l in joins[0].get("links", [])} != {execs[a]["spanId"] for a in diags if a in execs}:
                self.bad("topology", "incident.join missing or not linked to every diagnostic execute_tool span")
        elif joins:
            self.bad("topology", "incident.join present without fan-out")
        gates = [s for s in spans if s["name"] == "approval_gate"]
        if self.approval_receipts:
            ar = self.approval_receipts[0]
            vals = set(A(gates[0]).values()) if len(gates) == 1 else set()
            if len(gates) != 1 or ar["approvalId"] not in vals or ar["nonce"] not in vals:
                self.bad("topology", "approval_gate missing approval id / receipt nonce")
        elif gates:
            self.bad("topology", "approval_gate without approval")
        blob = json.dumps(final["otlp"])
        if any(c in blob for c in self.canaries) or self.body["incident"]["transcript"][:60] in blob:
            self.bad("redaction", "trace exports sensitive values or transcript text")

    def lifecycle_probes(self, final):
        s, again, _ = self.call("POST", "/v2/incidents", self.body, self.headers)
        if s != 200 or again != self.first:
            self.bad("durability", "repeating the original request did not return the identical first response")
        changed = copy.deepcopy(self.body)
        changed["incident"]["title"] += " (changed)"
        if self.call("POST", "/v2/incidents", changed)[0] != 409:
            self.bad("lifecycle", "same runId with changed content did not return 409")
        bad = copy.deepcopy(self.body)
        bad["profile"], bad["runId"] = "nope/v0", self.run_id + "_badprofile"
        if self.call("POST", "/v2/incidents", bad)[0] not in (400, 422) or \
                self.call("GET", f"/v2/incidents/{bad['runId']}")[0] != 404:
            self.bad("lifecycle", "unsupported profile not rejected cleanly (must create nothing)")
        if self.outcomes:
            o = self.outcomes[0]
            rb = {"receiptId": o["receiptId"], "outcomes": [{k: v for k, v in x.items() if k != "receiptId"}
                                                          for x in self.outcomes if x["receiptId"] == o["receiptId"]]}
            s1, r1, _ = self.call("POST", f"/v2/incidents/{self.run_id}/receipts", rb)
            s2, r2, _ = self.call("POST", f"/v2/incidents/{self.run_id}/receipts", rb)
            if s1 != 200 or r1 != r2:
                self.bad("durability", "identical receipt replay did not return the same JSON")
            rc = copy.deepcopy(rb)
            rc["outcomes"][0]["nonce"] = str(uuid.uuid4())
            if self.call("POST", f"/v2/incidents/{self.run_id}/receipts", rc)[0] != 409:
                self.bad("lifecycle", "same receiptId with changed content did not return 409")
        late = {"receiptId": "late_" + uuid.uuid4().hex[:8], "outcomes": [self.outcome(final["actionLog"][0])]}
        if self.call("POST", f"/v2/incidents/{self.run_id}/receipts", late)[0] not in (400, 409, 422):
            self.bad("lifecycle", "invalid state change after the final state was accepted")
        s, after, _ = self.call("GET", f"/v2/incidents/{self.run_id}")
        if after != final:
            self.bad("durability", "stored final state changed after the replay/conflict probes")

    def report(self):
        p = self.first or {}
        plan = {"rootCause": (p.get("diagnosis") or {}).get("rootCause"),
                "evidence": (p.get("diagnosis") or {}).get("evidence"),
                "diagnostics": [f"{d['toolName']} {json.dumps(d['arguments'], separators=(',', ':'))}" for d in p.get("dispatches") or []],
                "effect": next((f"{d['toolName']} {json.dumps(d['arguments'], separators=(',', ':'))}"
                                for d in (self.final or {}).get("actionLog", []) if d["phase"] != "diagnostic"), None),
                "finalStatus": (self.final or {}).get("status"), "suppressed": (self.final or {}).get("suppressed")}
        return {"incident": self.name, "scenario": SCENARIO[self.name], "firstPostSeconds": self.latency,
                "plan": plan, "problems": {k: v for k, v in self.problems.items() if v}}


def run(handle, sequential=False):
    t0 = time.time()
    runs = [Run(handle, n) for n in cases.CASES]
    if sequential:
        for r in runs:
            r.execute()
    else:
        th = [threading.Thread(target=r.execute) for r in runs]
        [t.start() for t in th]
        [t.join() for t in th]
    reports = [r.report() for r in runs]
    cats = ["transport", "proposal", "semantics", "topology", "correlation", "lifecycle", "durability", "redaction"]
    summary = {c: f"{sum(1 for r in runs if not r.problems[c])}/{len(runs)}" for c in cats}
    engineering_ok = all(summary[c].split("/")[0] == summary[c].split("/")[1] for c in cats)
    decisions_ok = sum(1 for r in runs if not r.problems["decision"])
    import planner as pm
    return {
        "verdict": ("ENGINEERING OK: every observable check passed" if engineering_ok else
                    "PROBLEMS FOUND: see 'problems' per incident"),
        "engineeringChecks": summary,
        "decisionsMatchingOurReading": f"{decisions_ok}/{len(runs)} (our reading of each case, NOT the grader's hidden key)",
        "modelOrder": pm.models(), "diagProbe": __import__("incident").DIAG_PROBE,
        "mode": "sequential" if sequential else "parallel", "totalSeconds": round(time.time() - t0, 1),
        "incidents": reports,
    }
