"""Hostile edge-case sweep: every module must degrade, never crash.

Every public entry point is fed inputs it was not designed for - empty
directories, corrupt images, malformed JSON, null fields, missing files - and
must return a structured result rather than raising. Modules that cannot
assess their target must say so explicitly (NOT_ASSESSED) rather than
reporting the absence of findings as a clean result.

The 'regressions' block pins bugs that were found and fixed, so they cannot
silently return.

Run directly::

    python tests/test_edge_cases.py
"""
import sys,tempfile,shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
fails=[]
def check(name,fn):
    try:
        r=fn()
        print(f"  ok   {name}")
        return r
    except Exception as e:
        fails.append((name,f"{type(e).__name__}: {e}"))
        print(f"  FAIL {name}: {type(e).__name__}: {e}")

tmp=Path(tempfile.mkdtemp())
(tmp/'empty').mkdir()
(tmp/'garbage').mkdir()
(tmp/'garbage'/'notanimage.jpg').write_text("this is not a jpeg")
(tmp/'garbage'/'empty.jpg').write_bytes(b"")
(tmp/'badjson.json').write_text("{not valid json")
(tmp/'emptylist.json').write_text("[]")
(tmp/'wrongshape.json').write_text('{"foo":"bar"}')

print("== data scanner ==")
from src.scanners import scan_dataset
check("empty dir",       lambda: scan_dataset(str(tmp/'empty')))
check("garbage images",  lambda: scan_dataset(str(tmp/'garbage')))
check("nonexistent",     lambda: scan_dataset(str(tmp/'nope')))
check("file not dir",    lambda: scan_dataset(str(tmp/'badjson.json')))
check("single image",    lambda: scan_dataset('demo/data/clean/contributor_alpha', max_images=1))

print("== crypto ==")
from src.scanners import verify_records
check("bad json",     lambda: verify_records(str(tmp/'badjson.json')))
check("empty list",   lambda: verify_records(str(tmp/'emptylist.json')))
check("wrong shape",  lambda: verify_records(str(tmp/'wrongshape.json')))
check("missing file", lambda: verify_records(str(tmp/'nope.json')))

print("== model auditor ==")
from src.scanners import audit_model
check("missing model", lambda: audit_model(str(tmp/'nope.pt')))
check("garbage model", lambda: audit_model(str(tmp/'badjson.json')))

print("== drift ==")
from src.scanners import detect_drift
check("empty vs empty",   lambda: detect_drift(str(tmp/'empty'),str(tmp/'empty')))
check("real vs empty",    lambda: detect_drift('demo/data/clean',str(tmp/'empty'),max_images=20))
check("identical corpus", lambda: detect_drift('demo/data/clean','demo/data/clean',max_images=20))

print("== intelligence ==")
from src.intelligence import build_threat_story, compute_immunity_score, generate_briefing
check("story {}",      lambda: build_threat_story({}))
check("immunity {}",   lambda: compute_immunity_score({}))
ts=build_threat_story({}); im=compute_immunity_score({},ts)
check("briefing {}",   lambda: generate_briefing({},ts,im,use_llm=False))
check("story junk",    lambda: build_threat_story({'data_scanner':{'findings':None}}))
check("immunity junk", lambda: compute_immunity_score({'data_scanner':{'findings':None}}))

print("== reporting ==")
from src.reporting import generate_report, summarize_findings
check("report {}",     lambda: generate_report({},persist=False))
check("summarize []",  lambda: summarize_findings([]))
check("summarize junk",lambda: summarize_findings([{'nope':1}]))

print("== agents ==")
from src.agents import run_office, hold_meeting
check("office empty",  lambda: run_office(str(tmp/'empty')))
check("office missing",lambda: run_office(str(tmp/'nope')))
o=run_office(str(tmp/'empty'))
check("meeting empty", lambda: hold_meeting(o))
check("meeting junk",  lambda: hold_meeting({'reports':[]}))


print("== regressions (bugs found and fixed) ==")

def corrupt_not_clean():
    """A corpus of unreadable files must never be reported as CLEAN."""
    d = tmp / "corrupt"
    d.mkdir(exist_ok=True)
    (d / "a.jpg").write_text("not a jpeg")
    (d / "b.jpg").write_bytes(b"")
    r = scan_dataset(str(d))
    assert r["summary"]["verdict"] == "NOT_ASSESSED", r["summary"]["verdict"]
    assert r.get("assessment_failed") is True
    return r

check("corrupt corpus -> NOT_ASSESSED", corrupt_not_clean)

def null_findings_key():
    """A null `findings` value must not crash coverage computation."""
    r = build_threat_story({"data_scanner": {"findings": None}})
    assert r["coverage"]["stages_covered"] == 1
    return r

check("null findings key", null_findings_key)

def payload_tamper_detected():
    """Editing a recorded output must break the chain even after re-hashing."""
    from src.crypto.chain import compute_record_hash, verify_chain, _as_dict
    from src.loaders.record_loader import load_records
    recs = [dict(_as_dict(x)) for x in load_records("demo/records/clean_chain.json")]
    assert verify_chain(recs).valid, "clean chain must verify"
    t = [dict(x) for x in recs]
    t[5]["output"] = {"label": "TAMPERED"}
    t[5]["record_hash"] = compute_record_hash(t[5])
    out = verify_chain(t)
    assert not out.valid, "payload tamper went undetected"
    return out

check("payload tamper + rehash detected", payload_tamper_detected)

def risk_severity_dominated():
    """Volume of trivial findings must not outrank a single CRITICAL."""
    def mk(sev, c, n):
        return [{"severity": sev, "confidence": c, "attack_class": "X",
                 "disposition": "ACCEPT"} for _ in range(n)]
    crit = summarize_findings(mk("CRITICAL", 0.95, 1))["risk_score"]
    noise = summarize_findings(mk("LOW", 0.25, 100))["risk_score"]
    assert noise < crit, f"{noise} >= {crit}"
    return {"critical": crit, "noise": noise}

check("risk score severity-dominated", risk_severity_dominated)

shutil.rmtree(tmp, ignore_errors=True)
print()
print("TOTAL FAILURES:",len(fails))
for n,e in fails: print("  !",n,e)
sys.exit(1 if fails else 0)
