#!/bin/bash
set -e
python scripts/train_hemorrhage_expert.py --epochs 30
python scripts/train_morphology_experts.py --epochs 30
