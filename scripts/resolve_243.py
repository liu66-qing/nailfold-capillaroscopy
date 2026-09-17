"""Use flow_score additivity to infer the true microthrombus class for recovered_archive3/243."""
import json
import pandas as pd

RULES = json.load(open("/root/nailfold/artifacts/labels/score_rules_v3.json"))
FR = RULES["field_rules"]
FLOW = RULES["groups"]["flow_score"]

m = pd.read_csv(
    "/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv",
    dtype={"exam_case_id": str},
)


def score_of(field, val):
    rule = FR.get(field, {})
    if rule.get("type") == "categorical_lookup":
        return rule["mapping"].get(str(val), None)
    return ("NUMERIC", rule.get("type"))


for case in ["recovered_archive3/243", "recovered_archive1/51"]:
    row = m[m.exam_case_id == case].iloc[0]
    print("=" * 66)
    print(case, "flow_score =", row["flow_score"], " total =", row["total_score"])
    known = 0.0
    for f in FLOW:
        s = score_of(f, row[f])
        print("   %-18s val=%-12r -> %r" % (f, row[f], s))
        if isinstance(s, float):
            known += s
    print("   sum of resolvable flow fields (incl. microthrombus as-is):", round(known, 4))
    others = 0.0
    for f in FLOW:
        if f == "microthrombus":
            continue
        s = score_of(f, row[f])
        if isinstance(s, float):
            others += s
    print("   sum EXCLUDING microthrombus:", round(others, 4))
    fs = row["flow_score"]
    if pd.notna(fs):
        print("   => residual attributable to microthrombus:", round(float(fs) - others, 4))
        for k, v in FR["microthrombus"]["mapping"].items():
            print("        matches %-6s (%.2f)? %s" % (k, v, abs(float(fs) - others - v) < 1e-6))

# also print the numeric-field rule shapes so we know which flow fields are non-categorical
print("=" * 66)
for f in FLOW:
    print(f, "->", FR.get(f, {}).get("type"))
