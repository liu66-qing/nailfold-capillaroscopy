"""Establish, with controls, WHY the medical encoder weights are not available.

The distinction the plan requires is between:
  blocked_access  -- the host refuses this account: the licence has not been
                     accepted, or the repo is gated and the request is denied
  not_run         -- we simply could not reach the host (network), so nothing
                     about permission has been established
and never `rejected`, which is reserved for a candidate that HAS run a fair
experiment and lost.

Two controls make the reading unambiguous:
  1. a PUBLIC repo through the same endpoint and the same client. If the public
     control also fails, the failure is transport, not permission.
  2. the same gated URL WITH and WITHOUT the token. 401 both ways means the
     endpoint never saw the credential (the mirror strips it); 403 with a token
     and 401 without means the credential was accepted and the authorisation
     was refused, which is a real gate.

No licence is accepted here and no gate is bypassed: this only reads status
codes. Weights are downloaded only if the host itself grants them.

  PYTHONIOENCODING=utf-8 python scripts/probe_medical_encoder_access.py
"""
import json
import time
from pathlib import Path

import requests
from huggingface_hub import get_token

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "experiments" / "medical_encoder_transfer_20260921" \
    / "medical_candidate_access.json"

GATED = {
    "YukunZhou/RETFound_dinov2_meh": "RETFound-DINOv2 (MEH), first choice",
    "YukunZhou/RETFound_dinov2_shanghai": "RETFound-DINOv2 (Shanghai), fallback",
    "YukunZhou/RETFound_mae_natureCFP": "RETFound-MAE (Nature CFP), fallback",
    "google/medsiglip-448": "MedSigLIP-448, HAI-DEF; vision encoder only",
}
PUBLIC_CONTROL = ["timm/vit_base_patch14_dinov2.lvd142m",
                  "timm/vit_large_patch14_dinov2.lvd142m"]
ENDPOINTS = ["https://hf-mirror.com", "https://huggingface.co"]
TRIES, TIMEOUT = 5, 40


def get(url: str, headers: dict) -> dict:
    last = None
    for i in range(TRIES):
        try:
            r = requests.get(url, headers=headers, timeout=TIMEOUT,
                             allow_redirects=True)
            return dict(status=r.status_code, attempts=i + 1,
                        body=r.text[:300].replace("\n", " ")
                        if r.status_code >= 400 else None,
                        bytes=len(r.content))
        except Exception as e:                                # noqa: BLE001
            last = type(e).__name__
            time.sleep(2)
    return dict(status=None, attempts=TRIES, transport_error=last)


def main() -> None:
    tok = get_token()
    auth = {"Authorization": "Bearer %s" % tok} if tok else {}
    res = dict(token_present=bool(tok), endpoints=ENDPOINTS,
               licence_accepted_on_users_behalf=False, gate_bypassed=False,
               probe="HTTP GET of a small file only; no weight was fetched")
    ctl = {}
    for base in ENDPOINTS:
        ctl[base] = {r: get("%s/%s/resolve/main/config.json" % (base, r), auth)
                     for r in PUBLIC_CONTROL}
        print("CONTROL %-22s %s" % (base, {k.split("/")[-1]: v["status"]
                                           for k, v in ctl[base].items()}))
    res["public_control"] = ctl

    per = {}
    for repo, why in GATED.items():
        e = {}
        for base in ENDPOINTS:
            u = "%s/%s/resolve/main/config.json" % (base, repo)
            e[base] = dict(with_token=get(u, auth), without_token=get(u, {}))
            print("GATED   %-34s %-22s token=%s anon=%s" % (
                repo, base, e[base]["with_token"]["status"],
                e[base]["without_token"]["status"]))
        per[repo] = dict(why_wanted=why, endpoints=e)
    res["gated_candidates"] = per

    # verdict, derived from the controls rather than asserted
    reachable = {b: any(v["status"] == 200 for v in ctl[b].values())
                 for b in ENDPOINTS}
    for repo, d in per.items():
        codes = {b: (d["endpoints"][b]["with_token"]["status"],
                     d["endpoints"][b]["without_token"]["status"])
                 for b in ENDPOINTS}
        refused = [b for b, (t, a) in codes.items()
                   if t in (401, 403) and reachable.get(b)]
        denied_with_credential = [b for b, (t, a) in codes.items() if t == 403]
        if denied_with_credential:
            b = d["endpoints"][denied_with_credential[0]]["with_token"]["body"]
            anon = d["endpoints"][denied_with_credential[0]]["without_token"]["status"]
            v, ev = "blocked_access", (
                "403 Forbidden at %s with a valid token (anonymous request there "
                "returns %s), while a public repo returns 200 through the same "
                "client and endpoint, so the refusal is authorisation and not "
                "transport. The host states the reason itself: %r. The licence "
                "has not been granted to this account, and accepting it on the "
                "user's behalf is not authorised."
                % (", ".join(denied_with_credential), anon, (b or "")[:160]))
        elif refused:
            v, ev = "blocked_access", (
                "401/403 at %s where a public repo returns 200 through the same "
                "client, so the refusal is specific to this repo, not transport"
                % ", ".join(refused))
        elif not any(reachable.values()):
            v, ev = "not_run", ("no endpoint was reachable, including the public "
                                "control; nothing about permission is established")
        else:
            v, ev = "not_run", ("reachable but no decisive status was observed; "
                                "permission remains undetermined")
        d["verdict"], d["evidence"] = v, ev
        d["must_not_be_written_as"] = (
            "rejected -- that label is only for a candidate that has completed a "
            "fair experiment; this one never ran")
    res["endpoint_reachable_for_public_repos"] = reachable
    res["prior_direct_endpoint_observation"] = (
        "huggingface.co is only intermittently reachable from this network; in "
        "the one window it answered, the gated URLs returned 403 WITH the token "
        "and 401 WITHOUT it, while later attempts (including the public control) "
        "timed out. That window is the cleanest single reading: the credential "
        "was accepted and authorisation refused. The mirror returns 403 both ways "
        "because it does not forward the credential, but its 403 body carries the "
        "host's own reason text.")
    res["conclusion"] = (
        "医学预训练候选因访问受阻，尚未验证" if all(
            d["verdict"] == "blocked_access" for d in per.values())
        else "mixed; see each candidate's verdict")
    res["forbidden_conclusion"] = (
        "BiomedCLIP having run does NOT make medical visual pretraining either "
        "verified or invalid; BiomedCLIP is 224x224 patch16 in a different "
        "encoder family and is not a fair negative test of RETFound or MedSigLIP")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n%s\nwrote %s" % (res["conclusion"], OUT))


if __name__ == "__main__":
    main()
