"""甲襞项目标准指标口径 v2 —— 三层尺子

所有新实验必须用这里的函数报告，禁止各自实现。
设计依据（均为 dev n=183 实测）：

1. MAE 的 95% CI = [2.442, 3.050]，**包含常数基线 2.940**。
   MAE 从主指标降级为诊断量：它测不出能力，且会掩盖字段级的大幅提升。
2. 已排除基线的三个指标：C-index [0.578,0.666]、重症 BA [0.538,0.672]、
   QWK [0.051,0.257]。**主报告指标用 C-index。**
3. 最小可检测差异（= CI 半宽，n=183）：
   中度+ BA 0.043 | C-index 0.044 | 重症 BA 0.067 | 精确等级 0.074 | QWK 0.103
   **小于这个幅度的改进不要声称，也不要为它花 GPU。**
4. 本队列异常率 86.3%（正常 9 + 大致正常 16 = 25/183）。
   灵敏度和 PPV 天然虚高，**在此队列上做不出筛查声明**。
   一律用 balanced accuracy 和特异度。
5. 无任何重复标注 → QWK 的人类可复现上限**测不出**。
   QWK 只能对 0.000 基线读，禁止说"接近人类水平"。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                            f1_score, recall_score, confusion_matrix)

LEVELS = ["正常", "大致正常", "轻度异常", "中度异常", "重度异常"]
L2I = {l: i for i, l in enumerate(LEVELS)}
THRESHOLDS = [1.0, 2.0, 4.0, 8.0]          # 含等号，已验证 183/183 复现

# n=183 实测的最小可检测差异，低于此值的改进不可声称
MDD = {'mod_BA': 0.043, 'cindex': 0.044, 'sev_BA': 0.067,
       'exact': 0.074, 'qwk': 0.103, 'mae': 0.304}


def cut_to_level(score):
    """把分数切成等级。含等号（<=），已验证。
    只用于把【预测分数】转等级；真实等级一律直接用 overall_assessment 列。"""
    for t, l in zip(THRESHOLDS, LEVELS):
        if score <= t:
            return l
    return LEVELS[4]


def c_index(true_s, pred_s):
    """成对一致率。与阈值无关，是当前最可靠的能力指标。"""
    t, p = np.asarray(true_s, float), np.asarray(pred_s, float)
    conc = tie = disc = 0
    for i in range(len(t)):
        dt, dp = t[i+1:] - t[i], p[i+1:] - p[i]
        v = dt != 0
        conc += np.sum((dt[v] * dp[v]) > 0)
        tie += np.sum(dp[v] == 0)      # 预测打平：算 0.5，否则常数预测会得 0 而非 0.5
        disc += np.sum(v)
    return (conc + 0.5 * tie) / disc if disc else np.nan


def bootstrap_ci(fn, n, n_boot=1000, seed=17, alpha=0.05):
    """对 case 重采样求 95% CI。返回 (lo, hi, width)。
    fn(idx) -> 标量。至少 2 个类别才计入。"""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = fn(idx)
        except Exception:
            continue
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < 50:
        return (np.nan, np.nan, np.nan)
    lo, hi = np.percentile(vals, [100*alpha/2, 100*(1-alpha/2)])
    return (float(lo), float(hi), float(hi - lo))


# ---------------- 第一层：字段能力 ----------------

def field_metrics(y_true, y_pred, classes=None, score_map=None,
                  true_score=None, pred_score=None, n_boot=1000):
    """单字段分类能力。y_true/y_pred 是类别标签（可为字符串）。

    score_map: {类别: 分数}，用于算字段分数 MAE —— 那才是composes 进 total_score 的量。
    与多数类基线和随机基线对比，BA 带 bootstrap CI。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if classes is None:
        classes = sorted(set(y_true.tolist()))
    n = len(y_true)
    maj = pd.Series(y_true).value_counts()
    maj_label, maj_rate = maj.index[0], maj.iloc[0] / n

    out = {
        'n': int(n),
        'classes': list(map(str, classes)),
        'class_counts': {str(k): int(v) for k, v in maj.items()},
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'accuracy': float((y_true == y_pred).mean()),
        'majority_rate': float(maj_rate),
        'majority_label': str(maj_label),
        'random_ba': 1.0 / len(classes),
        'per_class_recall': {
            str(c): float(recall_score(y_true, y_pred, labels=[c],
                                       average='macro', zero_division=0))
            for c in classes},
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        'confusion_labels': list(map(str, classes)),
    }
    out['lift_over_random'] = out['balanced_acc'] - out['random_ba']
    out['lift_over_majority_acc'] = out['accuracy'] - out['majority_rate']

    lo, hi, w = bootstrap_ci(
        lambda i: (balanced_accuracy_score(y_true[i], y_pred[i])
                   if len(set(y_true[i].tolist())) > 1 else None),
        n, n_boot)
    out['balanced_acc_ci'] = [lo, hi]
    out['balanced_acc_ci_width'] = w
    out['ba_excludes_random'] = bool(lo > out['random_ba']) if np.isfinite(lo) else None

    if score_map is not None:
        ts = np.array([score_map.get(str(c), np.nan) for c in y_true], float)
        ps = np.array([score_map.get(str(c), np.nan) for c in y_pred], float)
        m = np.isfinite(ts) & np.isfinite(ps)
        if m.sum():
            out['field_score_mae'] = float(np.abs(ps[m] - ts[m]).mean())
            mode_sc = score_map.get(str(maj_label), np.nan)
            if np.isfinite(mode_sc):
                out['field_score_mae_mode_baseline'] = float(
                    np.abs(mode_sc - ts[m]).mean())
    return out


# ---------------- 第二层：风险分层能力 ----------------

def stratification_metrics(true_level, pred_level, true_score, pred_score,
                           n_boot=1000):
    """五级风险分层。全部带常数基线对照和 95% CI。

    true_level 必须来自报告的 overall_assessment，不要从 total_score 反推。
    """
    a = np.array([L2I[l] if isinstance(l, str) else int(l) for l in true_level])
    b = np.array([L2I[l] if isinstance(l, str) else int(l) for l in pred_level])
    ts = np.asarray(true_score, float)
    ps = np.asarray(pred_score, float)
    n = len(a)

    # 常数基线：全部预测训练集中位数所在的等级
    const_s = float(np.median(ts))
    const_lvl = L2I[cut_to_level(const_s)]
    cb = np.full(n, const_lvl)

    def pack(a_, b_, ts_, ps_):
        d = {
            'mod_BA': float(balanced_accuracy_score((a_ >= 3).astype(int),
                                                    (b_ >= 3).astype(int))),
            'sev_BA': float(balanced_accuracy_score((a_ >= 4).astype(int),
                                                    (b_ >= 4).astype(int))),
            'exact': float((a_ == b_).mean()),
            'within_1': float((np.abs(a_ - b_) <= 1).mean()),
            'macro_mae': float(np.mean([np.abs(a_[a_ == k] - b_[a_ == k]).mean()
                                        for k in np.unique(a_)])),
            'mae': float(np.abs(ps_ - ts_).mean()),
        }
        try:
            d['qwk'] = float(cohen_kappa_score(a_, b_, weights='quadratic'))
            d['lwk'] = float(cohen_kappa_score(a_, b_, weights='linear'))
        except Exception:
            d['qwk'] = d['lwk'] = np.nan
        d['cindex'] = float(c_index(ts_, ps_))
        # 重症漏检：真重度被判为 正常/大致正常
        sev = a_ == 4
        d['severe_miss_rate'] = float((b_[sev] <= 1).mean()) if sev.sum() else np.nan
        d['severe_soft_miss'] = float((b_[sev] <= 2).mean()) if sev.sum() else np.nan
        d['severe_recall'] = float((b_[sev] == 4).mean()) if sev.sum() else np.nan
        return d

    out = {'n': int(n), 'model': pack(a, b, ts, ps),
           'constant_baseline': pack(a, cb, ts, np.full(n, const_s)),
           'true_level_counts': {LEVELS[k]: int((a == k).sum())
                                 for k in range(5)},
           'abnormal_prevalence': float((a >= 2).mean())}

    # 各级召回
    out['per_level_recall'] = {
        LEVELS[k]: float((b[a == k] == k).mean()) if (a == k).sum() else np.nan
        for k in range(5)}

    # CI + 是否排除基线 + 是否超过最小可检测差异
    ci = {}
    for key in ['mod_BA', 'sev_BA', 'exact', 'qwk', 'cindex', 'mae']:
        def f(i, key=key):
            if len(set(a[i].tolist())) < 2:
                return None
            return pack(a[i], b[i], ts[i], ps[i])[key]
        lo, hi, w = bootstrap_ci(f, n, n_boot)
        base = out['constant_baseline'][key]
        excl = (hi < base) if key == 'mae' else (lo > base)
        ci[key] = {'point': out['model'][key], 'ci': [lo, hi], 'width': w,
                   'baseline': base,
                   'excludes_baseline': bool(excl) if np.isfinite(lo) else None,
                   'min_detectable_diff': MDD.get(key)}
    out['ci'] = ci
    return out


def print_report(strat, title=''):
    """人看的报告。基线和 CI 强制并列，不给单独看点估计的机会。"""
    print('=' * 78)
    if title:
        print(title)
    print(f"n={strat['n']}  异常率={strat['abnormal_prevalence']:.3f}"
          f"  等级分布={strat['true_level_counts']}")
    print('-' * 78)
    print(f"{'指标':12s} {'模型':>7s} {'基线':>7s} {'95% CI':>17s}"
          f" {'宽度':>6s} {'MDD':>6s}  判定")
    for k, v in strat['ci'].items():
        lo, hi = v['ci']
        verdict = ('超过基线' if v['excludes_baseline'] else '与基线重叠')
        print(f"{k:12s} {v['point']:7.3f} {v['baseline']:7.3f}"
              f" [{lo:6.3f},{hi:6.3f}] {v['width']:6.3f}"
              f" {v['min_detectable_diff'] or float('nan'):6.3f}  {verdict}")
    print('-' * 78)
    print('各级召回: ' + '  '.join(
        f"{k}={v:.3f}" for k, v in strat['per_level_recall'].items()))
    m = strat['model']
    print(f"重症漏检(判为正常/大致正常)={m['severe_miss_rate']:.3f}"
          f"  软漏检(<=轻度)={m['severe_soft_miss']:.3f}"
          f"  重症召回={m['severe_recall']:.3f}")
    print('注: 异常率 %.1f%% → 灵敏度/PPV 虚高，勿作筛查声明。'
          % (100 * strat['abnormal_prevalence']))
    print('注: 无重复标注 → QWK 人类上限未知，勿称"接近人类水平"。')
    print('=' * 78)
