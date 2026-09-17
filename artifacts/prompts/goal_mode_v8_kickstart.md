你刚才陷入了死循环：反复读取目标文件、报告"未完成"，但从不执行训练。这不可接受。

# 当前进度

已完成：B1、B2（blood_color LoRA 二分类，rank4 和 rank8）
未完成：B3、B4、路线 A 全部、路线 C 全部

# 你现在必须做的事（不是"检查"，是"执行"）

## 立即执行：路线 A — exudation 过拟合修复

exudation 当前 dev BA 0.772 / locked BA 0.618，落差 15pp，过拟合严重。

**现在运行以下命令**（不要再读目标文件，不要再报告状态，直接跑）：

```bash
cd /root/甲劈微循环  # 或实际项目路径

# 如果没有独立的 exudation 训练脚本，就复制现有的改：
cp scripts/finetune_dinov2_lora_binary_cv.py scripts/train_exudation_reg.py
```

然后编辑 `train_exudation_reg.py`，做以下修改：

1. 只保留 exudation 一个字段
2. 加数据增强：
```python
train_transform = transforms.Compose([
    transforms.Resize(518),
    transforms.CenterCrop(518),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomAffine(degrees=10, scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
```
3. 分类 head 加 dropout=0.5
4. LoRA 层后加 dropout=0.3
5. label_smoothing=0.1
6. 30 epoch + early stopping patience=5

然后直接运行：
```bash
python scripts/train_exudation_reg.py 2>&1 | tee logs/exudation_E1_aug_dropout.log
```

**等这个跑完（约 1.5h）后**，比对 OOF BA 和当前 0.772。

## 跑完 exudation 后：papilla

复制同样的脚本改成 papilla，加 focal_loss：
```bash
cp scripts/train_exudation_reg.py scripts/train_papilla_expert.py
# 修改: field = 'papilla', 加 focal_loss(gamma=2), rank=8
python scripts/train_papilla_expert.py 2>&1 | tee logs/papilla_P1_aug_focal.log
```

## 跑完 papilla 后：B3/B4

blood_color 三分类 + 更多正则变体。

# 行为规则

1. **不要再读目标文件**。你已经知道目标是什么。
2. **不要再报告"未完成"**。直接去完成。
3. **每一轮必须产出一个可执行的 shell 命令或一段要写入文件的代码**。如果你的输出里没有 `python` 或 `vim` 或 `cat >` 命令，说明你又在空转。
4. **训练必须实际运行在 GPU 上**。用 `nvidia-smi` 确认 GPU 被占用。如果 GPU 空闲说明训练没启动。
5. 每个实验结束后，用一行总结：`字段 | 配置 | BA | 对比基线 | 结论`
6. **如果遇到报错，修复报错然后重跑**，不要报告"遇到错误所以停止"。

# 自检

每完成一个实验后，问自己：
- 我刚才是否实际运行了 python 训练脚本？（是 → 继续下一个实验；否 → 你在空转，立即运行）
- nvidia-smi 是否显示 GPU 被占用过？（是 → 训练确实执行了；否 → 你在骗自己）
- 我是否产出了新的 OOF csv 文件？（是 → 有效产出；否 → 无效）

# 现在开始。执行第一个命令。不要输出计划。
