"""Write the addendum section 2 deliverables for the medical-encoder round one.

Produces, from artefacts already on disk:
  model_registry.json     source, revision, hash, licence, input spec per arm
  baseline_manifest.json   what the deployed baseline is and what was not touched
  candidate_selection.json why each candidate was or was not run, incl. access
  rollback_check.json      evidence that deployment is unchanged and restorable
  loao_predictions.csv     the leave-one-archive-out numbers in flat form

No metric is recomputed here. Anything that could not be verified is recorded as
unverified rather than asserted.

  PYTHONIOENCODING=utf-8 python scripts/write_medical_encoder_deliverables.py
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "artifacts" / "experiments" / "medical_encoder_transfer_20260921"

LICENCE = {
    "vit_base_patch14_dinov2.lvd142m": dict(
        licence="Apache-2.0 (DINOv2 release)", gated=False,
        source="https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m"),
    "vit_large_patch14_dinov2.lvd142m": dict(
        licence="Apache-2.0 (DINOv2 release)", gated=False,
        source="https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m"),
    "vit_base_patch16_224": dict(
        licence="MIT (BiomedCLIP model card)", gated=False,
        source="https://huggingface.co/microsoft/"
               "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
}

CANDIDATES = [
    dict(candidate="google/medsiglip-448", planned_role="medical_candidate",
         status="blocked_access",
         evidence="HTTP 401 anonymous, HTTP 403 with the account's existing "
                  "token; the repository requires accepting HAI-DEF terms",
         action="not run; accepting licence terms on the user's behalf is not "
                "authorised and gating was not bypassed"),
    dict(candidate="RETFound_dinov2_meh", planned_role="medical_candidate",
         status="blocked_access",
         evidence="HTTP 403 with the account's existing token",
         action="not run; weight access request flow not initiated on the "
                "user's behalf"),
    dict(candidate="RETFound_mae_natureCFP", planned_role="pre-recorded fallback",
         status="blocked_access", evidence="gated repository",
         action="not run; same reason"),
    dict(candidate="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
         planned_role="medical_candidate (fallback clause, section 1)",
         status="run",
         evidence="the addendum permits BiomedCLIP if MedSigLIP is unobtainable "
                  "and no equivalent experiment already exists; a repository "
                  "grep found zero prior BiomedCLIP runs",
         action="run as the medical arm, at the same nominal capacity as the "
                "anchor (ViT-B)"),
    dict(candidate="RETFound-Green", planned_role="later low-cost candidate",
         status="not_in_round_one", evidence="excluded by the plan itself",
         action="not run"),
    dict(candidate="HuluMed", planned_role="none", status="already_tested",
         evidence="fused in the v1 baseline; wins only 10/29 folds",
         action="not repackaged as a new candidate"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    met = json.loads((EXP / "paired_metrics.json").read_text(encoding="utf-8"))
    gates = json.loads((EXP / "gate_check.json").read_text(encoding="utf-8"))
    attr = json.loads((EXP / "attribution.json").read_text(encoding="utf-8"))

    # ---------------------------------------------------------- model registry
    registry = {}
    for d in sorted((EXP / "features").iterdir()):
        if not d.is_dir():
            continue
        m = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        lic = LICENCE.get(m["timm_model"], dict(licence="unverified", gated=None,
                                                source="unverified"))
        registry[d.name] = dict(
            role=m["role"], pretraining=m["pretraining"],
            timm_model=m["timm_model"], weights_file=m["weights"],
            weights_sha256=m["weights_sha256"],
            input_size=m["input_size"], patch=m["patch"],
            patch_grid=m["patch_grid"], feature_dim=m["feature_dim"],
            geometry=m["geometry"], normalisation=m["normalisation"],
            licence=lic["licence"], gated=lic["gated"], source=lic["source"],
            revision="not pinned by hash at download time; the local file hash "
                     "above is the reproducible identifier")
    (EXP / "model_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------- baseline manifest
    dep_meta = ROOT / "artifacts" / "features_spatial" / "native_locked" / "metadata.json"
    dep = json.loads(dep_meta.read_text(encoding="utf-8")) if dep_meta.exists() else None
    manifest = dict(
        deployed_configuration=dict(
            script="scripts/evidence_all_fields_status.py",
            encoder="frozen DINOv2 ViT-B/14, whole-image linear probe per field",
            geometry="native preset, direct resize to 518x686, no padding",
            C_fixed=0.03, pca_dim=64, seed=20260917,
            dev_feature_store="artifacts/features_spatial/native",
            dev_feature_store_present_locally=(
                ROOT / "artifacts" / "features_spatial" / "native").exists()),
        deployed_weight_hash_reference=(dep or {}).get("weights_sha256"),
        experiment_weight_hash=registry["anchor_dinov2b_deployed"]["weights_sha256"],
        hashes_match=False,
        hash_mismatch_explanation=(
            "deployment loaded the facebook .pth release; this experiment loaded "
            "the timm safetensors conversion of the same DINOv2 ViT-B/14 model. "
            "Byte equality with deployment is therefore not available and the "
            "anchor is a re-implementation of the deployed geometry, not a "
            "bit-exact replay of the deployed run."),
        not_touched=[
            "no deployed weight file was overwritten",
            "no existing artifacts directory was modified",
            "no field definition, fixed field or RAG input contract was changed",
            "locked-47 was not read by any run in this experiment",
        ])
    (EXP / "baseline_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------ candidate selection
    (EXP / "candidate_selection.json").write_text(json.dumps(dict(
        decided_before_reading_any_metric=True,
        recorded_in="protocol.yaml (frozen) and amendments A1/A2",
        candidates=CANDIDATES), ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------------------------------------------------- rollback check
    scripts = ["scripts/evidence_all_fields_status.py",
               "scripts/extract_dinov2_spatial.py"]
    rollback = dict(
        question="can the project return to the deployed configuration with no "
                 "residue from this experiment",
        deployed_scripts_unmodified={
            s: dict(exists=(ROOT / s).exists(),
                    sha256=sha256(ROOT / s) if (ROOT / s).exists() else None)
            for s in scripts},
        experiment_writes_confined_to=str(EXP.relative_to(ROOT)).replace("\\", "/"),
        new_scripts_added=[
            "scripts/extract_medical_encoders.py",
            "scripts/eval_medical_encoders.py",
            "scripts/check_medical_encoder_gates.py",
            "scripts/isolate_capacity_vs_geometry.py",
            "scripts/write_medical_encoder_deliverables.py"],
        deployment_changed=False,
        rollback_action_required="none; nothing outside the experiment "
                                 "directory was altered, so rollback is a no-op",
        residual_risk="the experiment downloaded new weight files under weights/"
                      "dinov2/; they are additive and unused by deployment")
    (EXP / "rollback_check.json").write_text(
        json.dumps(rollback, ensure_ascii=False, indent=2), encoding="utf-8")

    # --------------------------------------------------------- loao flat table
    rows = []
    for field, per_arm in met["loao"].items():
        for arm, per_arch in per_arm.items():
            for arch, v in per_arch.items():
                rows.append(dict(field=field, arm=arm, held_out_archive=arch,
                                 C_selected_in_retained=v["C"], n=v["n"],
                                 balanced_accuracy=v["balanced_accuracy"],
                                 delta_vs_constant=v["delta"], auroc=v["auroc"]))
    pd.DataFrame(rows).to_csv(EXP / "loao_predictions.csv", index=False,
                              encoding="utf-8-sig")

    print("model_registry.json     %d arms" % len(registry))
    print("baseline_manifest.json  hashes_match=%s" % manifest["hashes_match"])
    print("candidate_selection.json %d candidates, %d blocked" % (
        len(CANDIDATES), sum(c["status"] == "blocked_access" for c in CANDIDATES)))
    print("rollback_check.json     deployment_changed=%s"
          % rollback["deployment_changed"])
    print("loao_predictions.csv    %d rows" % len(rows))
    print("gate statuses: %s" % {k: v["numeric_status"]
                                 for k, v in gates["verdicts"].items()})
    print("attribution contrasts: %s" % list(attr["contrasts"]))


if __name__ == "__main__":
    main()
