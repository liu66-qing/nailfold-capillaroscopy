Subject: Re: Request for Access to the CAPIDATA Nailfold Capillaroscopy Dataset

Dear Dr. Gracia Tello,

Thank you again for your time. I am writing to follow up on my message of 8 September, and also to correct two of the numbers I reported in it.

**A correction to my earlier figures.** The per-field results I sent (clarity 0.837, blood_color 0.739, exudation 0.799, SVP 0.862, papilla 0.627; mean 0.7729) were obtained by selecting, for each field, the best-performing model on the test data. That is a cherry-picking procedure, and I should not have reported it as a single baseline. Under a proper nested protocol (5 seeds x 5 folds, with epoch and aggregation both selected on an inner validation fold), the same configurations give:

| field | as reported | nested protocol |
|---|---|---|
| SVP | 0.862 | 0.757 |
| exudation | 0.799 | 0.728 |
| blood_color | 0.739 | 0.710 |
| clarity | 0.837 | 0.820 |
| papilla | 0.627 | 0.617 |

The optimistic bias was about +0.02 overall, and it was concentrated in exactly the field where I had claimed the strongest result. This also explains most of the gap to the locked test set that puzzled me when I wrote to you: I was comparing a selected-on-test number against an honest one.

**A second correction, about papilla.** I described its 77.3% positive rate as a source of label noise. The more basic problem is that balanced accuracy hid the imbalance from me. In accuracy terms the model reaches 0.715 against a majority-class baseline of 0.773 — that is, it performs worse than always predicting the majority class. I have removed papilla from our deliverable set rather than continue to report it.

**Where the work now stands.** I have shifted from whole-case grading to per-field delivery, and I now report accuracy against a fold-internal majority baseline with bootstrap confidence intervals rather than balanced accuracy alone. Four classification fields survive this test (clarity, exudation, SVP, blood_color; improvements of +0.19 to +0.34 over baseline, CIs excluding zero). For the quantitative fields I abandoned absolute regression: our diameter annotations turn out to have only 24 distinct values across 165 cases, with half the cases falling on four integer values, so the labels do not support micron-level prediction. Recasting them as relative within-cohort comparisons (above/below the cohort median) yields usable results for loop length and afferent diameter, while apex diameter does not survive.

Your own papers, together with Dinsdale et al. (2017) and the SCLEROCAP study, were what convinced me that this coarser granularity is the honest one rather than a compromise — if inter-observer ICC for capillary density is 0.64 among ten experts, and kappa for the standard classification schemes is around 0.47 to 0.49 before consensus training, then claiming finer resolution from a model trained on single-reader labels is not defensible.

**What I would still value from you, whether or not data access is possible.** Two questions in particular:

1. In CAPIDATA, are the quantitative measurements (apical width, loop length) recorded per capillary at identified positions, or as a summary reading per field of view? In our data they appear to be the latter, and I suspect this is why our models correlate with vessel count and area rather than with width.

2. Did your multi-centre design include any repeated or independent double reading? Without it I cannot estimate the human reproducibility ceiling on our own data, and so cannot state what level of agreement would actually be sufficient.

If the CAPIDATA dataset cannot be shared, I would be grateful for any guidance on either question, and I fully understand the constraints around clinical data.

With thanks and best regards,

Liu Junqing
Undergraduate, Taiyuan University of Technology
Research Assistant, Zhejiang University
liujunqing200602@gmail.com
