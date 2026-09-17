# v4 pooling/classifier experiments

All runs use the fixed development-only 5-fold case-level protocol and `locked_cases_seen=0`.

## Runs

- `pooling_correct_mean`: YOLO v3 frame features aggregated by mean + std, appended to frozen v1 features. BA: clarity 0.715341, blood_color 0.636599, exudation 0.684415, SVP 0.646299, papilla 0.549622.
- `pooling_correct_median`: detector median + std. BA: clarity 0.715986, blood_color 0.641975, exudation 0.683687, SVP 0.674576, papilla 0.549622.
- `pooling_correct_topk`: detector top 30% by detector score + std. BA: clarity 0.693643, blood_color 0.636599, exudation 0.688842, SVP 0.674576, papilla 0.541001.

None passes the field gate against v1: no field simultaneously has BA at least v1, at least 3/5 fold wins, and no >2pp minority-recall loss. CatBoost was attempted as a genuine classifier experiment but the remote environment lacks the `catboost` package; no result is claimed.

The prior v3 detector route remains the current best: clarity 0.715986, blood_color 0.710474, exudation 0.710855, papilla 0.576955; SVP remains v1 at 0.698672.
