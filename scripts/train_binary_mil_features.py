"""Five-field binary bag-level attention MIL on development cases only."""
from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from torch import nn

FIELDS = ("clarity", "blood_color", "exudation", "subpapillary_venous_plexus", "papilla")
EXPLORATORY_FIELDS = {"papilla"}
VOCABULARIES = {
    "clarity": ["clear", "poor"],
    "blood_color": ["dark", "light"],
    "exudation": ["absent", "present"],
    "subpapillary_venous_plexus": ["absent", "present"],
    "papilla": ["flat", "wavy"],
}


def binary_label(field: str, value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if field == "clarity":
        return "poor" if text in {"不清", "模糊"} else "clear" if text == "清晰" else None
    if field == "blood_color":
        return "dark" if text in {"暗红", "暗紫"} else "light" if text in {"浅红", "淡红"} else None
    if field == "exudation":
        return "absent" if text == "无" else "present" if text in {"+", "++", "+++"} else None
    if field == "subpapillary_venous_plexus":
        return "absent" if text == "不见" else "present" if text in {"可见1排", "可见2排", ">2排,扩张"} else None
    if field == "papilla":
        return "flat" if text == "平坦" else "wavy" if text in {"浅波纹状", "波纹状"} else None
    raise ValueError(field)


@dataclass
class Case:
    case_id: str
    split: str
    features: torch.Tensor
    targets: dict[str, int]


class BinaryAttentionMIL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, attention_heads: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, attention_heads))
        pooled_dim = hidden_dim * (attention_heads + 2)
        self.case_encoder = nn.Sequential(nn.LayerNorm(pooled_dim), nn.Linear(pooled_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({field: nn.Linear(hidden_dim, 2) for field in FIELDS})

    def forward(self, bag: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        encoded = self.encoder(bag)
        weights = self.attention(encoded).transpose(0, 1).softmax(dim=1)
        pooled = torch.cat(((weights @ encoded).flatten(), encoded.mean(0), encoded.max(0).values), dim=0)
        representation = self.case_encoder(pooled)
        return {field: head(representation) for field, head in self.heads.items()}, weights


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def build_cases(matrix: np.ndarray, index: pd.DataFrame, labels: pd.DataFrame, roles: pd.DataFrame, test_fold: int) -> tuple[list[Case], dict[str, torch.Tensor]]:
    roles = roles[roles.evaluation_role.eq("development")].copy()
    roles.development_fold = roles.development_fold.astype(int)
    validation_fold = (test_fold + 1) % 5
    roles["split"] = np.where(roles.development_fold.eq(test_fold), "test", np.where(roles.development_fold.eq(validation_fold), "val", "train"))
    frame = roles[["exam_case_id", "split"]].merge(labels[["exam_case_id", *FIELDS]], on="exam_case_id", validate="one_to_one")
    label_rows = frame.set_index("exam_case_id")
    cases = []
    for case_id, positions in index.groupby("exam_case_id", sort=True).indices.items():
        case_id = str(case_id)
        if case_id not in label_rows.index:
            continue
        row = label_rows.loc[case_id]
        targets = {}
        for field in FIELDS:
            value = binary_label(field, row[field])
            targets[field] = VOCABULARIES[field].index(value) if value is not None else -1
        cases.append(Case(case_id, str(row.split), torch.from_numpy(np.asarray(matrix[np.asarray(positions)], dtype=np.float32)), targets))
    train = [case for case in cases if case.split == "train"]
    class_weights = {}
    for field in FIELDS:
        counts = np.bincount([c.targets[field] for c in train if c.targets[field] >= 0], minlength=2).astype(float)
        weights = np.sqrt(max(counts.sum(), 1.0) / np.maximum(counts, 1.0)); class_weights[field] = torch.tensor(weights / weights.mean(), dtype=torch.float32)
    return cases, class_weights


def loss_for_case(model: BinaryAttentionMIL, case: Case, weights: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    outputs, _ = model(case.features.to(device)); losses=[]
    for field, target in case.targets.items():
        if target >= 0:
            losses.append(nn.functional.cross_entropy(outputs[field].unsqueeze(0), torch.tensor([target], device=device), weight=weights[field].to(device), label_smoothing=0.04))
    return torch.stack(losses).mean()


@torch.inference_mode()
def evaluate(model: BinaryAttentionMIL, cases: list[Case], split: str, device: torch.device) -> dict[str, object]:
    model.eval(); selected=[c for c in cases if c.split == split]; truth={f:[] for f in FIELDS}; pred={f:[] for f in FIELDS}; ids={f:[] for f in FIELDS}; attention=[]
    for case in selected:
        outputs, weights = model(case.features.to(device))
        attention.append({"exam_case_id":case.case_id,"frame_count":len(case.features),"attention":weights.cpu().tolist()})
        for field,target in case.targets.items():
            if target >= 0: truth[field].append(target); pred[field].append(int(outputs[field].argmax())); ids[field].append(case.case_id)
    result={}
    for field in FIELDS:
        labels=[0,1]; y=np.asarray(truth[field]); p=np.asarray(pred[field]); recalls=recall_score(y,p,labels=labels,average=None,zero_division=0)
        result[field]={"n":len(y),"exploratory":field in EXPLORATORY_FIELDS,"balanced_accuracy":float(balanced_accuracy_score(y,p)),"macro_f1":float(f1_score(y,p,labels=labels,average="macro",zero_division=0)),"class_recall":{VOCABULARIES[field][i]:float(recalls[i]) for i in labels},"confusion_labels":VOCABULARIES[field],"confusion_matrix":confusion_matrix(y,p,labels=labels).tolist(),"records":[{"exam_case_id":c,"truth":VOCABULARIES[field][int(a)],"prediction":VOCABULARIES[field][int(b)]} for c,a,b in zip(ids[field],y,p)]}
    result["attention_records"] = attention
    return result


def objective(metrics: dict[str, object]) -> float:
    # Papilla is trained but cannot select checkpoints for delivery fields.
    return float(np.mean([metrics[f]["balanced_accuracy"] for f in FIELDS if f not in EXPLORATORY_FIELDS]))


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--features",type=Path,required=True); ap.add_argument("--feature-index",type=Path,required=True); ap.add_argument("--labels",type=Path,required=True); ap.add_argument("--folds-file",type=Path,required=True); ap.add_argument("--roles",type=Path,required=True); ap.add_argument("--test-fold",type=int,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--seeds",type=int,nargs="+",default=[17,29,43,71,101]); ap.add_argument("--epochs",type=int,default=250); ap.add_argument("--patience",type=int,default=35); ap.add_argument("--learning-rate",type=float,default=3e-4); ap.add_argument("--weight-decay",type=float,default=3e-3); ap.add_argument("--hidden-dim",type=int,default=256); ap.add_argument("--attention-heads",type=int,default=4); ap.add_argument("--dropout",type=float,default=.35); ap.add_argument("--device",choices=["cpu"],default="cpu"); args=ap.parse_args()
    folds=pd.read_csv(args.folds_file,usecols=["exam_case_id","fold"]); roles=pd.read_csv(args.roles,usecols=["exam_case_id","evaluation_role","development_fold"])
    for frame in (folds,roles): frame["exam_case_id"]=frame.exam_case_id.astype(str)
    if set(roles.evaluation_role.unique()) != {"development","locked_test"}: raise ValueError("roles must contain exactly development and locked_test")
    dev=set(roles.loc[roles.evaluation_role.eq("development"),"exam_case_id"]); locked=set(roles.loc[roles.evaluation_role.eq("locked_test"),"exam_case_id"])
    if len(dev)!=186 or len(locked)!=47 or dev & locked: raise ValueError(f"unexpected role boundary: development={len(dev)}, locked={len(locked)}")
    fold_check=roles.loc[roles.evaluation_role.eq("development"),["exam_case_id","development_fold"]].merge(folds,on="exam_case_id",validate="one_to_one")
    if len(fold_check)!=len(dev) or not np.array_equal(fold_check.development_fold.astype(int),fold_check["fold"].astype(int)): raise ValueError("development folds disagree between roles and folds file")
    index=pd.read_csv(args.feature_index); index["exam_case_id"]=index.exam_case_id.astype(str); full_matrix=np.load(args.features,mmap_mode="r")
    if len(index)!=len(full_matrix): raise ValueError(f"feature/index row mismatch: {len(full_matrix)} vs {len(index)}")
    if set(index.exam_case_id) & locked: raise ValueError("locked feature row reached MIL input")
    if set(index.exam_case_id)!=dev: raise ValueError(f"development feature case mismatch: features={index.exam_case_id.nunique()}, expected={len(dev)}")
    keep=index.exam_case_id.isin(dev).to_numpy(); matrix=full_matrix[keep]; index=index.loc[keep].reset_index(drop=True)
    labels=pd.read_csv(args.labels); labels["exam_case_id"]=labels.exam_case_id.astype(str); labels=labels[labels.exam_case_id.isin(dev)].copy()
    if labels.exam_case_id.duplicated().any() or set(labels.exam_case_id)!=dev: raise ValueError("development labels are missing or duplicated")
    cases,class_weights=build_cases(matrix,index,labels,roles,args.test_fold); device=torch.device("cpu"); train=[c for c in cases if c.split=="train"]; args.output_dir.mkdir(parents=True,exist_ok=True)
    report={"schema_version":"binary-attention-mil-development/1.0","method":"bag-level attention MIL","evaluation_role":"development_cv","test_fold":args.test_fold,"validation_fold":(args.test_fold+1)%5,"locked_cases_seen":0,"locked_cases_in_source_manifest":len(locked),"gpu_used":False,"device":str(device),"fields":list(FIELDS),"exploratory_fields":sorted(EXPLORATORY_FIELDS),"cases":{s:sum(c.split==s for c in cases) for s in ("train","val","test")},"seeds":{}}
    for seed in args.seeds:
        set_seed(seed); model=BinaryAttentionMIL(matrix.shape[1],args.hidden_dim,args.attention_heads,args.dropout).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay); best=-float("inf"); best_epoch=0; best_state=None; stale=0
        for epoch in range(1,args.epochs+1):
            model.train(); random.shuffle(train); optimizer.zero_grad(set_to_none=True)
            for i,case in enumerate(train,1):
                (loss_for_case(model,case,class_weights,device)/8).backward()
                if i%8==0 or i==len(train): nn.utils.clip_grad_norm_(model.parameters(),2.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            val=evaluate(model,cases,"val",device); score=objective(val)
            if score>best+1e-4: best=score; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict()); stale=0
            else: stale+=1
            if stale>=args.patience: break
        model.load_state_dict(best_state); seed_result={"best_epoch":best_epoch,"validation_objective_delivery_fields":best,"val":evaluate(model,cases,"val",device),"test":evaluate(model,cases,"test",device)}; report["seeds"][str(seed)]=seed_result; torch.save({"state_dict":best_state,"input_dim":matrix.shape[1],"vocabularies":VOCABULARIES,"fields":FIELDS,"exploratory_fields":sorted(EXPLORATORY_FIELDS),"seed":seed},args.output_dir/f"binary_mil_seed_{seed}.pt"); print(json.dumps({"fold":args.test_fold,"seed":seed,"best_epoch":best_epoch,"test_ba":{f:round(seed_result['test'][f]['balanced_accuracy'],4) for f in FIELDS}},ensure_ascii=False),flush=True)
    (args.output_dir/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__": main()
