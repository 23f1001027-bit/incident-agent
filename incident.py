"""Incident agent: state machine, receipts, approvals and OTLP trace.

Framework independent: handle(method, path, headers, body, planner) -> (status, headers, body_bytes).
The planner is called exactly once per new runId; receipts, GET, replays and the trace never call it.
"""
import copy
import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time

PROFILE = "ga5-incident-agent/v2"
DB_PATH = os.environ.get("DB_PATH", "/tmp/incident_agent.db")
MAX_BODY = 700 * 1024
TERMINAL = {"completed", "failed"}
TP_RE = re.compile(r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")
EV_RE = re.compile(r"^\s*\[([^\]\s]{3,80})\]", re.M)


# ---------------------------------------------------------------- helpers
def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hexid(n):
    while True:
        v = secrets.token_hex(n // 2)
        if int(v, 16) != 0:
            return v


def parse_traceparent(h):
    m = TP_RE.match((h or "").strip())
    if not m or m.group(1) == "ff" or int(m.group(2), 16) == 0 or int(m.group(3), 16) == 0:
        return None
    return m.group(2), m.group(3)


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)


def scrub(obj, secrets_list):
    """Replace any sensitive value found inside strings of obj."""
    if not secrets_list:
        return obj
    if isinstance(obj, str):
        for s in secrets_list:
            if s in obj:
                obj = obj.replace(s, "[REDACTED]")
        return obj
    if isinstance(obj, dict):
        return {k: scrub(v, secrets_list) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v, secrets_list) for v in obj]
    return obj


def sensitive_values(body):
    vals = {s for s in strings(body.get("sensitive", {})) if len(s) >= 6}
    vals |= {s for s in (body.get("policy", {}) or {}).get("doNotExport", []) if isinstance(s, str) and len(s) >= 6}
    # field names of the sensitive object are not secrets; only values (and explicit doNotExport entries)
    names = set((body.get("sensitive") or {}).keys())
    return sorted(vals - names, key=len, reverse=True)


# ---------------------------------------------------------------- storage
class Store:
    def __init__(self, path):
        self.lock = threading.RLock()
        self.runs = {}
        self.db = None
        try:
            self.db = sqlite3.connect(path, check_same_thread=False)
            self.db.execute("create table if not exists runs(run_id text primary key, body text)")
            self.db.commit()
            for rid, body in self.db.execute("select run_id, body from runs"):
                self.runs[rid] = json.loads(body)
        except Exception as exc:
            print("storage warning, in-memory only:", repr(exc), flush=True)
            self.db = None

    def save(self, state):
        with self.lock:
            self.runs[state["runId"]] = state
            if self.db is not None:
                try:
                    self.db.execute("insert or replace into runs values(?,?)", (state["runId"], json.dumps(state)))
                    self.db.commit()
                except Exception as exc:
                    print("db write failed:", repr(exc), flush=True)


STORE = Store(DB_PATH)
_locks, _guard = {}, threading.Lock()


def run_lock(run_id):
    with _guard:
        return _locks.setdefault(run_id, threading.Lock())


def reply(status, obj):
    body = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return status, {"content-type": "application/json"}, body


def error(status, code, message="Request rejected."):
    return reply(status, {"error": {"code": code, "message": message}})


# ---------------------------------------------------------------- plan validation
def type_ok(t, v):
    if t is None:
        return True
    if isinstance(t, list):
        return any(type_ok(x, v) for x in t)
    return ((t == "string" and isinstance(v, str)) or
            (t == "integer" and isinstance(v, int) and not isinstance(v, bool)) or
            (t == "number" and isinstance(v, (int, float)) and not isinstance(v, bool)) or
            (t == "boolean" and isinstance(v, bool)) or (t == "array" and isinstance(v, list)) or
            (t == "object" and isinstance(v, dict)) or (t == "null" and v is None))


def check_schema(args, schema):
    if not isinstance(args, dict):
        return None
    schema = schema or {}
    props = schema.get("properties") or {}
    out = {}
    for k, v in args.items():
        if props and k not in props and schema.get("additionalProperties") is False:
            continue
        spec = props.get(k) or {}
        if not type_ok(spec.get("type"), v):
            # tolerate numbers sent as strings, and the reverse
            if spec.get("type") in ("integer", "number") and isinstance(v, str):
                try:
                    v = int(v) if spec["type"] == "integer" else float(v)
                except ValueError:
                    return None
            elif spec.get("type") == "string" and isinstance(v, (int, float)) and not isinstance(v, bool):
                v = str(v)
            else:
                return None
        if "enum" in spec and v not in spec["enum"]:
            return None
        out[k] = v
    for k in schema.get("required") or []:
        if k not in out:
            return None
    return out


def ev_ids(transcript):
    return list(dict.fromkeys(EV_RE.findall(transcript or "")))


def validate_plan(raw, body):
    """Returns (plan, None) or (None, reason)."""
    try:
        inc, pol = body["incident"], body["policy"]
        known = set(ev_ids(inc["transcript"]))
        if raw.get("rootCause") not in inc["allowedRootCauses"]:
            return None, f"rootCause {raw.get('rootCause')!r} is not one of allowedRootCauses {inc['allowedRootCauses']}"
        ev = []
        for e in raw.get("evidence") or []:
            e = str(e).strip().strip("[]")
            if e in known and e not in ev:
                ev.append(e)
        ev = ev[:4]
        if len(ev) < 2:
            return None, "evidence must contain 2 to 4 IDs that appear at the start of transcript lines"
        catalog = {t["name"]: t for t in body["toolCatalog"]}
        effect_tools = set(pol.get("effectTools") or [])
        limit = max(1, min(3, int(pol.get("maximumDiagnostics", 3))))
        diags, seen = [], set()
        for d in raw.get("diagnostics") or []:
            name = d.get("toolName")
            if name not in catalog or name in effect_tools:
                continue
            args = check_schema(d.get("arguments", {}), catalog[name].get("inputSchema"))
            if args is None:
                continue
            key = (name, canon(args))
            if key in seen:
                continue
            seen.add(key)
            cite = [e for e in dict.fromkeys(str(x).strip("[] ") for x in d.get("evidence") or []) if e in ev] or ev[:1]
            diags.append({"toolName": name, "arguments": args, "evidence": cite})
        diags = diags[:limit]
        if not diags:
            return None, "no valid diagnostic call: use catalog tools that are not effect tools, with arguments matching inputSchema"
        eff = raw.get("effect") or {}
        if eff.get("toolName") not in effect_tools or eff.get("toolName") not in catalog:
            return None, f"effect.toolName must be one of {sorted(effect_tools)}"
        eargs = check_schema(eff.get("arguments", {}), catalog[eff["toolName"]].get("inputSchema"))
        if eargs is None:
            return None, "effect.arguments do not match the tool inputSchema"
        ecite = [e for e in dict.fromkeys(str(x).strip("[] ") for x in eff.get("evidence") or []) if e in ev] or ev[:1]
        return {"rootCause": raw["rootCause"], "evidence": ev, "diagnostics": diags,
                "effect": {"toolName": eff["toolName"], "arguments": eargs, "evidence": ecite}}, None
    except Exception as exc:
        return None, f"malformed plan ({type(exc).__name__})"


def fallback_plan(body):
    """Deterministic, conservative plan used only if the model output is unusable."""
    inc, pol = body["incident"], body["policy"]
    ev = ev_ids(inc["transcript"])[:2]
    catalog = {t["name"]: t for t in body["toolCatalog"]}
    effect_tools = list(pol.get("effectTools") or [])
    diag_name = next((n for n in catalog if n not in effect_tools), None)

    def fill(schema):
        out = {}
        for k in (schema or {}).get("required") or []:
            out[k] = inc.get("service", "") if k == "service" else ""
        return out
    eff = "page_owner" if "page_owner" in effect_tools else (effect_tools[0] if effect_tools else None)
    if len(ev) < 2 or not diag_name or not eff:
        return None
    return {"rootCause": inc["allowedRootCauses"][0], "evidence": ev,
            "diagnostics": [{"toolName": diag_name, "arguments": fill(catalog[diag_name].get("inputSchema")), "evidence": ev[:1]}],
            "effect": {"toolName": eff, "arguments": fill(catalog[eff].get("inputSchema")), "evidence": ev[:1]}}


_sample_logged = {"n": 0}


def log_sample(body):
    """First two incidents: print the structure (not the sensitive object) so prompts can be tuned."""
    if _sample_logged["n"] >= 2:
        return
    _sample_logged["n"] += 1
    inc, pol = body["incident"], body["policy"]
    lines = inc["transcript"].split("\n")
    print("SAMPLE INCIDENT:", json.dumps({
        "title": inc.get("title"), "service": inc.get("service"), "severity": inc.get("severity"),
        "allowedRootCauses": inc["allowedRootCauses"], "transcriptLines": len(lines),
        "firstLines": lines[:6],
        "tools": [{"name": t["name"], "schema": t.get("inputSchema")} for t in body["toolCatalog"]],
        "policy": {k: pol.get(k) for k in ("maximumDiagnostics", "effectTools", "approvalRequiredFor")},
    }, ensure_ascii=False)[:5000], flush=True)


# ---------------------------------------------------------------- dispatch / views
def make_dispatch(state, call, attempt):
    span = hexid(16)
    d = {
        "actionId": call["actionId"], "callId": call["callId"], "phase": call["phase"],
        "toolName": call["toolName"], "arguments": copy.deepcopy(call["arguments"]),
        "evidence": list(call["evidence"]), "attempt": attempt,
        "traceparent": f"00-{state['trace']['traceId']}-{span}-01",
    }
    if state["trace"].get("tracestate"):
        d["tracestate"] = state["trace"]["tracestate"]
    if call.get("approvalId"):
        d["approvalId"] = call["approvalId"]
        d["approvalNonce"] = call["approvalNonce"]
    call["attempt"], call["state"] = attempt, "pending"
    state["actionLog"].append(d)
    state["times"][f"d:{call['callId']}:{attempt}"] = time.time_ns()
    return d


def latest_dispatch(state, call):
    for d in reversed(state["actionLog"]):
        if d["callId"] == call["callId"]:
            return d


def view(state, new_dispatches=None, new_approvals=None, include_pending=False):
    out = {"runId": state["runId"], "status": state["status"], "diagnosis": state["diagnosis"]}
    waiting = state["status"] == "waiting"
    if waiting:
        if include_pending:
            out["dispatches"] = [latest_dispatch(state, c) for c in state["calls"].values() if c["state"] == "pending"]
            ap = state.get("approval")
            out["approvals"] = [ap["request"]] if ap and not ap.get("decided") else []
        else:
            out["dispatches"] = new_dispatches or []
            out["approvals"] = new_approvals or []
    else:
        out["dispatches"], out["approvals"] = [], []
    out["actionLog"] = copy.deepcopy(state["actionLog"])      # snapshots: stored responses must never change
    out["receiptLog"] = copy.deepcopy(state["receiptLog"])
    out["diagnosis"] = copy.deepcopy(state["diagnosis"])
    out["dispatches"] = copy.deepcopy(out["dispatches"])
    if not waiting:
        out["chosenEffect"] = state.get("chosenEffect")
        out["suppressed"] = list(state.get("suppressed", []))
        out["otlp"] = copy.deepcopy(state["otlp"])
    return out


# ---------------------------------------------------------------- create run
def create_run(body, headers, planner):
    t0 = time.time_ns()
    run_id = body["runId"]
    inc = body["incident"]
    incoming = parse_traceparent(headers.get("traceparent"))
    trace = {"traceId": incoming[0] if incoming else hexid(32), "serverSpanId": hexid(16),
             "agentSpanId": hexid(16), "chatSpanId": hexid(16), "joinSpanId": hexid(16), "gateSpanId": hexid(16)}
    if incoming:
        trace["parentSpanId"] = incoming[1]
        ts = (headers.get("tracestate") or "").strip()
        if ts:
            trace["tracestate"] = ts

    log_sample(body)
    plan_start = time.time_ns()
    raw, model = None, os.environ.get("LLM_MODEL", "gpt-4.1")
    reason, t_first = "planner failed", time.time()
    try:
        raw, model = planner(body)
    except Exception as exc:
        print("planner error:", repr(exc), flush=True)
    plan = None
    if raw:
        plan, reason = validate_plan(raw, body)
        if plan is None and time.time() - t_first < 7:          # one corrective retry, same logical chat step
            try:
                raw2, model = planner(body, reason)
                plan, reason = validate_plan(raw2, body)
                raw = raw2
            except Exception as exc:
                print("planner retry error:", repr(exc), flush=True)
    plan_end = time.time_ns()
    used_fallback = plan is None
    if used_fallback:
        print(f"RUN {run_id[:14]} FALLBACK plan used. reason: {reason}", flush=True)
        if raw:
            print("RUN raw model plan:", json.dumps(raw, ensure_ascii=False)[:1500], flush=True)
        plan = fallback_plan(body)
    if plan is None:
        return error(422, "PLAN_UNAVAILABLE")
    secrets_list = sensitive_values(body)
    plan = scrub(plan, secrets_list)                 # scrub BEFORE anything is logged or stored
    print(f"RUN {run_id[:14]} plan={'FALLBACK' if used_fallback else 'model'} rootCause={plan['rootCause']} "
          f"evidence={plan['evidence']} diagnostics={[(d['toolName'], d['arguments']) for d in plan['diagnostics']]} "
          f"effect=({plan['effect']['toolName']}, {plan['effect']['arguments']})", flush=True)

    state = {
        "runId": run_id, "status": "waiting", "reqHash": sha(canon(body)),
        "agentName": body.get("agentName") or "incident-response", "marker": body["publicMarker"],
        "trace": trace, "model": model, "secrets": [],   # secrets are never persisted
        "diagnosis": {"rootCause": plan["rootCause"], "evidence": plan["evidence"]},
        "plan": plan, "calls": {}, "actionLog": [], "receiptLog": [], "receipts": {},
        "times": {"t0": t0, "planStart": plan_start, "planEnd": plan_end},
        "approval": None, "suppressed": [], "chosenEffect": None, "phase": "diagnostics",
        "firstResponse": None,
    }
    approval_needed = plan["effect"]["toolName"] in set(body["policy"].get("approvalRequiredFor") or [])
    state["approvalNeeded"] = approval_needed

    new = []
    for d in plan["diagnostics"]:
        call = {"actionId": "act_" + hexid(16), "callId": "call_" + hexid(16), "phase": "diagnostic",
                "toolName": d["toolName"], "arguments": d["arguments"], "evidence": d["evidence"],
                "execSpanId": hexid(16), "retried": False, "state": "pending", "attempt": 0}
        state["calls"][call["callId"]] = call
        new.append(make_dispatch(state, call, 1))
    eff = plan["effect"]
    state["effectCall"] = {"actionId": "act_" + hexid(16), "callId": "call_" + hexid(16), "phase": "effect",
                           "toolName": eff["toolName"], "arguments": eff["arguments"], "evidence": eff["evidence"],
                           "execSpanId": hexid(16), "retried": False, "state": "planned", "attempt": 0}
    state["dispatchSecrets"] = None
    resp = view(state, new_dispatches=new, new_approvals=[])
    state["firstResponse"] = resp
    STORE.save(state)
    return reply(200, resp)


# ---------------------------------------------------------------- receipts
def finalize(state, status):
    state["status"] = status
    now = time.time_ns()
    state["times"]["final"] = now
    state["otlp"] = build_otlp(state)
    return None


def process_receipt(state, body, now):
    """Mutates `state`. Returns (new_dispatches, new_approvals) or raises ValueError for invalid input."""
    receipt_id = body["receiptId"]
    new_d, new_a = [], []
    outcomes = body.get("outcomes")
    approvals = body.get("approvals")
    if not outcomes and not approvals:
        raise ValueError("empty receipt")

    # ---- validate everything before mutating
    seen = set()
    if outcomes is not None:
        if not isinstance(outcomes, list) or not outcomes:
            raise ValueError("bad outcomes")
        for o in outcomes:
            if not isinstance(o, dict):
                raise ValueError("bad outcome")
            call = state["calls"].get(o.get("callId")) or (
                state["effectCall"] if o.get("callId") == state["effectCall"]["callId"] else None)
            if call is None or call["actionId"] != o.get("actionId") or call["state"] != "pending" \
                    or o.get("attempt") != call["attempt"] or call["callId"] in seen:
                raise ValueError("outcome does not match a pending call")
            if not isinstance(o.get("status"), int) or isinstance(o.get("status"), bool) \
                    or not isinstance(o.get("nonce"), str) or not o["nonce"]:
                raise ValueError("bad outcome fields")
            seen.add(call["callId"])
    if approvals is not None:
        ap = state.get("approval")
        if not isinstance(approvals, list) or not approvals or ap is None or ap.get("decided"):
            raise ValueError("no pending approval")
        for a in approvals:
            if not isinstance(a, dict) or a.get("approvalId") != ap["request"]["approvalId"] \
                    or not isinstance(a.get("nonce"), str) or not a["nonce"] \
                    or a.get("decision") not in ("approved", "denied", "rejected"):
                raise ValueError("approval does not match")

    # ---- apply outcomes
    for o in outcomes or []:
        call = state["calls"].get(o["callId"]) or state["effectCall"]
        entry = {"receiptId": receipt_id, "actionId": o["actionId"], "callId": o["callId"], "attempt": o["attempt"],
                 "status": o["status"], "resultClass": o.get("resultClass"), "nonce": o["nonce"]}
        if o.get("errorType"):
            entry["errorType"] = o["errorType"]
        state["receiptLog"].append(entry)
        state["times"][f"r:{o['callId']}:{o['attempt']}"] = now
        if 200 <= o["status"] < 300:
            call["state"] = "succeeded"
        elif o["status"] == 503 and not call["retried"]:
            call["retried"] = True
            new_d.append(make_dispatch(state, call, call["attempt"] + 1))
        else:
            call["state"] = "failed"

    # ---- apply approvals
    for a in approvals or []:
        ap = state["approval"]
        state["receiptLog"].append({"receiptId": receipt_id, "approvalId": a["approvalId"],
                                    "decision": a["decision"], "nonce": a["nonce"]})
        ap["decided"], ap["decision"], ap["nonce"], ap["receiptId"] = True, a["decision"], a["nonce"], receipt_id
        state["times"]["approvalReceipt"] = now
        if a["decision"] == "approved":
            ec = state["effectCall"]
            ec["approvalId"], ec["approvalNonce"] = a["approvalId"], a["nonce"]
            new_d.append(make_dispatch(state, ec, 1))
            state["phase"] = "effect"
        else:
            state["suppressed"].append(state["effectCall"]["toolName"])
            finalize(state, "failed")
            return [], []

    if new_d:
        return new_d, []
    diag_pending = [c for c in state["calls"].values() if c["state"] == "pending"]
    if state["phase"] == "diagnostics":
        if diag_pending:
            return [], []
        if any(c["state"] == "failed" for c in state["calls"].values()):
            state["suppressed"].append(state["effectCall"]["toolName"])
            finalize(state, "failed")
            return [], []
        ec = state["effectCall"]
        if state["approvalNeeded"]:
            req = {"approvalId": "apr_" + hexid(16), "actionId": ec["actionId"], "toolName": ec["toolName"],
                   "argumentsDigest": sha(canon(ec["arguments"]))}
            state["approval"] = {"request": req, "decided": False}
            state["times"]["approvalRequest"] = now
            state["phase"] = "approval"
            return [], [req]
        state["phase"] = "effect"
        return [make_dispatch(state, ec, 1)], []
    if state["phase"] == "effect":
        ec = state["effectCall"]
        if ec["state"] == "succeeded":
            state["chosenEffect"] = ec["toolName"]
            finalize(state, "completed")
        elif ec["state"] == "failed":
            finalize(state, "failed")
        return [], []
    return [], []


def receipts_route(run_id, body_bytes):
    try:
        body = json.loads(body_bytes.decode("utf-8"))
        rid = body["receiptId"]
        if not isinstance(body, dict) or not isinstance(rid, str) or not rid:
            raise ValueError
    except Exception:
        return error(400, "INVALID_REQUEST")
    with run_lock(run_id):
        state = STORE.runs.get(run_id)
        if state is None:
            return error(404, "NOT_FOUND")
        h = sha(canon(body))
        prior = state["receipts"].get(rid)
        if prior:
            if prior["hash"] != h:
                return error(409, "CONFLICT")
            return reply(200, prior["response"])
        if state["status"] in TERMINAL:
            return error(422, "INVALID_STATE")
        snapshot = copy.deepcopy(state)
        try:
            new_d, new_a = process_receipt(state, body, time.time_ns())
        except ValueError:
            STORE.runs[run_id] = snapshot          # nothing changed
            return error(422, "INVALID_STATE")
        resp = view(state, new_dispatches=new_d, new_approvals=new_a)
        state["receipts"][rid] = {"hash": h, "response": resp}
        STORE.save(state)
        return reply(200, resp)


def incidents_route(body_bytes, headers, planner):
    try:
        body = json.loads(body_bytes.decode("utf-8"))
        inc = body["incident"]
        ok = (isinstance(body, dict) and isinstance(body["runId"], str) and body["runId"]
              and isinstance(body.get("publicMarker"), str) and isinstance(inc["transcript"], str)
              and isinstance(inc["allowedRootCauses"], list) and isinstance(body["toolCatalog"], list)
              and isinstance(body["policy"], dict))
    except Exception:
        ok = False
    if not ok:
        return error(400, "INVALID_REQUEST")
    if body.get("profile") != PROFILE:
        return error(400, "UNSUPPORTED_PROFILE")
    with run_lock(body["runId"]):
        state = STORE.runs.get(body["runId"])
        if state is not None:
            if state["reqHash"] != sha(canon(body)):
                return error(409, "CONFLICT")
            return reply(200, state["firstResponse"])
        return create_run(body, headers, planner)


def get_route(run_id):
    with run_lock(run_id):
        state = STORE.runs.get(run_id)
        if state is None:
            return error(404, "NOT_FOUND")
        return reply(200, view(state, include_pending=True))


# ---------------------------------------------------------------- OTLP
def attr(k, v):
    if isinstance(v, bool):
        return {"key": k, "value": {"boolValue": v}}
    if isinstance(v, int):
        return {"key": k, "value": {"intValue": v}}
    return {"key": k, "value": {"stringValue": str(v)}}


def build_otlp(state):
    T, tid = state["trace"], state["trace"]["traceId"]
    times, base = state["times"], {"ga5.run.id": state["runId"], "ga5.public.marker": state["marker"]}
    t_end = times.get("final", time.time_ns())
    spans = []

    def span(sid, parent, name, kind, start, end, attrs, status=None, links=None):
        s = {"traceId": tid, "spanId": sid, "name": name, "kind": kind,
             "startTimeUnixNano": str(start), "endTimeUnixNano": str(max(end, start)),
             "attributes": [attr(k, v) for k, v in {**base, **attrs}.items()],
             "status": {"code": status if status is not None else 1}}
        if parent:
            s["parentSpanId"] = parent
        if links:
            s["links"] = [{"traceId": tid, "spanId": x} for x in links]
        if T.get("tracestate"):
            s["traceState"] = T["tracestate"]
        spans.append(s)

    span(T["serverSpanId"], T.get("parentSpanId"), "POST /v2/incidents", 2, times["t0"], t_end + 3000,
         {"http.request.method": "POST", "url.path": "/v2/incidents"})
    span(T["agentSpanId"], T["serverSpanId"], f"invoke_agent {state['agentName']}", 1, times["t0"] + 1000, t_end + 2000,
         {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": state["agentName"]})
    span(T["chatSpanId"], T["agentSpanId"], "chat incident-plan", 3, times["planStart"], times["planEnd"],
         {"gen_ai.operation.name": "chat", "gen_ai.request.model": state["model"] or "unknown"})

    calls = list(state["calls"].values())
    if state["actionLog"] and any(d["callId"] == state["effectCall"]["callId"] for d in state["actionLog"]):
        calls.append(state["effectCall"])
    exec_ids = {}
    for c in calls:
        ds = [d for d in state["actionLog"] if d["callId"] == c["callId"]]
        rs = [times.get(f"r:{c['callId']}:{d['attempt']}") for d in ds]
        start = times[f"d:{c['callId']}:{ds[0]['attempt']}"]
        end = max([r for r in rs if r] or [start])
        exec_ids[c["callId"]] = c["execSpanId"]
        span(c["execSpanId"], T["agentSpanId"], f"execute_tool {c['toolName']}", 1, start, end,
             {"ga5.action.id": c["actionId"], "gen_ai.tool.name": c["toolName"],
              "gen_ai.tool.call.id": c["callId"], "gen_ai.operation.name": "execute_tool"})
        for d in ds:
            sid = d["traceparent"].split("-")[2]
            rec = next((r for r in state["receiptLog"] if r.get("callId") == c["callId"] and r.get("attempt") == d["attempt"]), None)
            ds_start = times[f"d:{c['callId']}:{d['attempt']}"]
            ds_end = times.get(f"r:{c['callId']}:{d['attempt']}", ds_start)
            a = {"ga5.action.id": c["actionId"], "ga5.attempt": d["attempt"], "http.request.method": "POST",
                 "http.request.resend_count": d["attempt"] - 1}
            status = 1
            if rec:
                a["ga5.receipt.id"], a["ga5.receipt.nonce"] = rec["receiptId"], rec["nonce"]
                if rec["status"] > 0:
                    a["http.response.status_code"] = rec["status"]
                if rec["status"] == 0 or rec.get("errorType") == "timeout":
                    a["error.type"], status = "timeout", 2
                elif not 200 <= rec["status"] < 300:
                    a["error.type"], status = str(rec["status"]), 2
            span(sid, c["execSpanId"], f"POST tool/{c['toolName']}", 3, ds_start, ds_end, a, status)

    diag_calls = list(state["calls"].values())
    if len(diag_calls) >= 2:
        starts = [times[f"d:{c['callId']}:1"] for c in diag_calls]
        ends = [times.get(f"r:{c['callId']}:{c['attempt']}", t_end) for c in diag_calls]
        span(T["joinSpanId"], T["agentSpanId"], "incident.join", 1, min(starts), max(ends),
             {"ga5.join.count": len(diag_calls)}, links=[exec_ids[c["callId"]] for c in diag_calls])

    ap = state.get("approval")
    if ap and ap.get("decided"):
        span(T["gateSpanId"], T["agentSpanId"], "approval_gate", 1,
             times.get("approvalRequest", t_end), times.get("approvalReceipt", t_end),
             {"ga5.approval.id": ap["request"]["approvalId"], "ga5.approval.nonce": ap["nonce"],
              "ga5.approval.receipt.nonce": ap["nonce"], "ga5.receipt.nonce": ap["nonce"],
              "ga5.receipt.id": ap["receiptId"], "ga5.action.id": ap["request"]["actionId"],
              "ga5.approval.decision": ap["decision"]})

    return {"resourceSpans": [{"resource": {"attributes": [attr("service.name", "incident-agent")]},
                               "scopeSpans": [{"scope": {"name": "incident-agent"}, "spans": spans}]}]}


# ---------------------------------------------------------------- router
def handle(method, path, headers, body, planner):
    headers = {k.lower(): v for k, v in headers.items()}
    path = re.sub(r"/{2,}", "/", path).rstrip("/") or "/"
    if method == "GET" and path in ("/healthz", "/"):
        return reply(200, {"status": "ok"})
    if len(body) > MAX_BODY * 4:
        return error(413, "TOO_LARGE")
    if method == "POST" and path == "/v2/incidents":
        return incidents_route(body, headers, planner)
    m = re.fullmatch(r"/v2/incidents/([^/]+)/receipts", path)
    if method == "POST" and m:
        return receipts_route(m.group(1), body)
    m = re.fullmatch(r"/v2/incidents/([^/]+)", path)
    if method == "GET" and m:
        return get_route(m.group(1))
    return error(404, "NOT_FOUND")
