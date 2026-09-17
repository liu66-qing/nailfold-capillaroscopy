# Omission scan checklist

Status legend: CONFIRMED = documented; VERIFY = requires a read-only check or experiment; BLOCKED = cannot be claimed with current evidence.

| Area | Status | What is still missing |
|---|---|---|
| locked isolation | CONFIRMED | Recheck after every new feature build |
| duplicate/augmentation grouping | CONFIRMED | Verify remote copy preserves group ids |
| frame-to-case leakage | VERIFY | Available non-blank files: zero valid cross-fold hash leaks; 244 video hashes unavailable and 1,222 records absent locally |
| precomputed-feature leakage | VERIFY | Build-time role and timestamp provenance |
| fold-local preprocessing | VERIFY | Audit all scripts, not only main classifier |
| class support/vocabulary | VERIFY | Per-fold rare-class table and remap log |
| missing vs negative semantics | VERIFY | Field-level missingness contract |
| probability calibration | VERIFY | Reliability/Brier/ECE and abstention policy |
| acquisition/device drift | VERIFY | Session/device metadata or explicit limitation |
| augmentation distribution drift | VERIFY | Compare source and derived quality statistics |
| ROI localization | BLOCKED | Requires landmarks or validated proposal |
| instance/graph ground truth | BLOCKED | Requires instance/centerline annotation |
| absolute micron calibration | BLOCKED | Requires pixel scale/device calibration |
| clinical semantic agreement | BLOCKED | Requires qualified raters and agreement study |
| multiple-comparison control | VERIFY | Pre-register primary route and gate |
| reproducibility | VERIFY | Pin environment, seeds, hashes, command lines; video provenance remains open |
| deployment shift | VERIFY | Frame-count and unreadable-rate stress test |
