"""EXP-I: microthrombus 三分类可学性 go/no-go

为什么做这个：microthrombus 占 total_score 方差 42.64%，是唯一一个
"做对它就能大幅推动指标"的字段。天花板分析说做对 2 个字段 -> 中度+ BA 0.800
（当前 0.511，提升 +0.29，远超最小可检测差异 0.043）。
但它的原始记录单位是"个/min" —— 医生是在动态观察下计数的。
所以必须先证明静态帧里有没有判据，再决定要不要投入。

三臂（自身对照，同特征同折同 seed，只改聚合方式）：
  A static_mean   帧特征均值池化 -> 线性分类
  B static_max    帧特征最大池化 -> 线性分类   （血栓稀疏，任一帧出现就该判正）
  C majority      恒定预测多数类（无）

判定标准（不看点估计，只看 CI）：
  BA 的 bootstrap 95% CI 下界 > 1/3（随机）才算有信号。
  两个 static 臂都没排除随机 -> 静态图学不到，走视频或放弃该字段。

标签：无 138 / 1--2 56 / >2 34（全库 233）。脏值 [未见]->无, 个/min->丢弃。
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, '/tmp')
from metrics_v2 import bootstrap_ci
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

FEAT_DIR = '/root/autodl-tmp/nailfold/exp_i_feats'
MANIFEST = '/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
OUT = '/root/nailfold/artifacts/experiments/exp_i_microthrombus'
CLASSES = ['无', '1--2', '>2']
SEEDS = [17, 42, 123]
NFOLD = 5


def load_labels():
    m = pd.read_csv(MANIFEST)
    kc = [c for c in m.columns if c in ('case', 'exam_case_id')][0]
    if 'split' in m.columns:
        m = m[m['split'] != 'test']          # locked_test 绝对不碰
    s = m['microthrombus'].astype(str).str.strip()
    s = s.replace({'[未见]': '无'})            # 语义无损
    keep = s.isin(CLASSES)
    out = pd.DataFrame({'case': m.loc[keep, kc].values,
                        'y': s[keep].map({c: i for i, c in enumerate(CLASSES)}).values})
    return out.reset_index(drop=True)


def load_feats(cases):
    """返回 {case: {'cls': (F,768), 'patch': (F,1369,768)}}。

    特征保留了完整 patch 网格（37x37），这对稀疏小目标是关键：
    血栓是单 patch 尺度的目标，帧级池化会把它稀释掉。
    """
    X = {}
    for c in cases:
        p = os.path.join(FEAT_DIR, c.replace('/', '__') + '.npz')
        if os.path.exists(p):
            d = np.load(p)
            X[c] = {'cls': d['cls'].astype(np.float32),
                    'patch': d['patch'].astype(np.float32)}
    return X


def pool(a, how):
    """a: {'cls': (F,768), 'patch': (F,P,768)} -> (D,)

    mean       帧级均值池化 CLS —— 常规基线
    max        帧级最大池化 CLS —— 稀疏发现在任一帧出现即可
    patchmax   全帧全 patch 上取最大 —— 单 patch 尺度目标的对症聚合
    patchmix   CLS均值 ⊕ patch全局max，兼顾整体与局部
    """
    cls, pat = a['cls'], a['patch']
    if how == 'mean':
        return cls.mean(0)
    if how == 'max':
        return cls.max(0)
    if how == 'patchmax':
        return pat.reshape(-1, pat.shape[-1]).max(0)
    if how == 'patchmix':
        return np.concatenate([cls.mean(0),
                               pat.reshape(-1, pat.shape[-1]).max(0)])
    raise ValueError(how)


def run_arm(df, X, how, n_epoch=60, lr=1e-3, wd=1e-2):
    """5-fold x 3-seed，返回 OOF 预测（seed 平均 logit 后取 argmax）。"""
    cases = df['case'].tolist()
    y = df['y'].values
    Z = np.stack([pool(X[c], how) for c in cases])
    # 逐折标准化，避免跨折泄漏
    n, D = Z.shape
    K = len(CLASSES)
    logit_sum = np.zeros((n, K), np.float64)

    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    folds = np.array_split(order, NFOLD)

    for seed in SEEDS:
        torch.manual_seed(seed)
        for fi in range(NFOLD):
            te = folds[fi]
            tr = np.concatenate([folds[j] for j in range(NFOLD) if j != fi])
            mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-6
            Xtr = torch.tensor((Z[tr] - mu) / sd)
            Xte = torch.tensor((Z[te] - mu) / sd)
            ytr = torch.tensor(y[tr], dtype=torch.long)

            # 类权重只从训练折算（GPT 第 5 条，避免泄漏）
            cnt = np.bincount(y[tr], minlength=K).astype(np.float64)
            w = torch.tensor((cnt.sum() / (K * np.maximum(cnt, 1))),
                             dtype=torch.float32)

            net = nn.Linear(D, K)
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
            lossf = nn.CrossEntropyLoss(weight=w)
            net.train()
            for _ in range(n_epoch):
                opt.zero_grad()
                lossf(net(Xtr), ytr).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                logit_sum[te] += net(Xte).numpy()

    pred = logit_sum.argmax(1)
    return y, pred


def evaluate(y, pred, name):
    n = len(y)
    rand_ba = 1.0 / len(CLASSES)
    cnt = np.bincount(y, minlength=len(CLASSES))
    r = {
        'arm': name, 'n': int(n),
        'class_counts': {CLASSES[i]: int(cnt[i]) for i in range(len(CLASSES))},
        'balanced_acc': float(balanced_accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'accuracy': float((y == pred).mean()),
        'random_ba': rand_ba,
        'majority_rate': float(cnt.max() / n),
        'per_class_recall': {CLASSES[i]: (float((pred[y == i] == i).mean())
                                         if (y == i).sum() else None)
                             for i in range(len(CLASSES))},
        'confusion_matrix': confusion_matrix(
            y, pred, labels=list(range(len(CLASSES)))).tolist(),
    }
    lo, hi, w = bootstrap_ci(
        lambda i: (balanced_accuracy_score(y[i], pred[i])
                   if len(set(y[i].tolist())) > 1 else None), n, 1000)
    r['ba_ci'] = [lo, hi]
    r['ba_ci_width'] = w
    r['ba_excludes_random'] = bool(lo > rand_ba) if np.isfinite(lo) else None
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    df = load_labels()
    X = load_feats(df['case'].tolist())
    df = df[df['case'].isin(X)].reset_index(drop=True)
    nf = [X[c]['cls'].shape[0] for c in df['case']]
    p0 = X[df['case'][0]]
    print(f'cases={len(df)}  frames/case: min={min(nf)} med={int(np.median(nf))} '
          f'max={max(nf)}  cls_D={p0["cls"].shape[1]} '
          f'patch_grid={p0["patch"].shape[1]}', flush=True)
    print('label dist:', df['y'].value_counts().sort_index().to_dict(), flush=True)

    results = []
    for name, how in [('A_static_mean', 'mean'), ('B_static_max', 'max'),
                      ('D_patchmax', 'patchmax'), ('E_patchmix', 'patchmix')]:
        print(f'--- {name} ---', flush=True)
        y, pred = run_arm(df, X, how, n_epoch=a.epochs)
        r = evaluate(y, pred, name)
        results.append(r)
        # 每臂算完立刻落盘 —— 这台共享机器会静默杀进程，别把已算的结果丢掉
        json.dump(results, open(os.path.join(OUT, 'results_partial.json'), 'w'),
                  ensure_ascii=False, indent=2, default=float)
        np.save(os.path.join(OUT, f'{name}_oof_pred.npy'), pred)
        print(f"  BA={r['balanced_acc']:.4f} CI=[{r['ba_ci'][0]:.4f},"
              f"{r['ba_ci'][1]:.4f}] rand={r['random_ba']:.4f} "
              f"排除随机={r['ba_excludes_random']}  macroF1={r['macro_f1']:.4f}",
              flush=True)
        print(f"  各类召回: {r['per_class_recall']}", flush=True)

    # C: 恒定多数类
    y = df['y'].values
    maj = int(np.bincount(y).argmax())
    r = evaluate(y, np.full(len(y), maj), 'C_majority')
    results.append(r)
    print(f"--- C_majority ---\n  BA={r['balanced_acc']:.4f} "
          f"acc={r['accuracy']:.4f}", flush=True)

    json.dump(results, open(os.path.join(OUT, 'results.json'), 'w'),
              ensure_ascii=False, indent=2, default=float)

    print('\n' + '=' * 74)
    print('GO/NO-GO 判定')
    print('=' * 74)
    ok = [r for r in results if r['arm'] != 'C_majority'
          and r['ba_excludes_random']]
    for r in results:
        print(f"{r['arm']:16s} BA={r['balanced_acc']:.4f} "
              f"CI=[{r['ba_ci'][0]:.4f},{r['ba_ci'][1]:.4f}] "
              f"宽度={r['ba_ci_width']:.4f}")
    if ok:
        best = max(ok, key=lambda r: r['balanced_acc'])
        print(f"\nGO: {best['arm']} 的 BA CI 下界 {best['ba_ci'][0]:.4f} "
              f"> 随机 {best['random_ba']:.4f}")
        print("静态图含判据，可投入该字段的专家模型。")
    else:
        print("\nNO-GO: 四个静态臂(帧mean/帧max/patchmax/patchmix)的 BA CI "
              "都未排除随机 1/3。")
        print("已排除'聚合方式不对'这一解释 —— patch 级最大池化是对稀疏小目标"
              "最对症的聚合，仍无信号。")
        print("静态帧不含 microthrombus 判据 —— 与'个/min'的动态记录方式一致。")
        print("含义: 占 42.64% 方差的字段不可达 -> '做对2字段到 BA 0.800' 路线不成立。")
    print('=' * 74)


if __name__ == '__main__':
    main()
