#!/bin/bash
# Master script to run all LOSO experiments (C/D/F groups)
# Unified config: epoch=20, lr=5e-4, no early stopping, bs=128
# Results saved to: /hpc_stor03/sjtu_home/duoming.jiang/agent/brain-seed/aaaresult/

set -e
cd "$(dirname "$0")"

echo "============================================"
echo "C group: LOSO domain adaptation (mixed norm)"
echo "============================================"

echo "[C1] MLP + CORAL, mixed norm, lambda=1.0"
python3 run_loso.py --method coral --norm mixed --lambda_da 1.0

echo "[C2] MLP + MMD, mixed norm, lambda=1.0"
python3 run_loso.py --method mmd --norm mixed --lambda_da 1.0

echo "[C3] MLP + DANN, mixed norm, lambda=1.0"
python3 run_loso.py --method dann --norm mixed --lambda_da 1.0

echo ""
echo "============================================"
echo "D group: per-subject normalization"
echo "============================================"

echo "[D0] MLP baseline, per-subject norm"
python3 run_loso.py --method baseline --norm per_subject --lambda_da 1.0

echo "[D1] MLP + CORAL, per-subject norm, lambda=1.0"
python3 run_loso.py --method coral --norm per_subject --lambda_da 1.0

echo "[D2] MLP + MMD, per-subject norm, lambda=1.0"
python3 run_loso.py --method mmd --norm per_subject --lambda_da 1.0

echo "[D3] MLP + DANN, per-subject norm, lambda=1.0"
python3 run_loso.py --method dann --norm per_subject --lambda_da 1.0

echo ""
echo "============================================"
echo "F group: lambda sensitivity (per-subject norm)"
echo "============================================"

for LAMBDA in 0.1 0.5 2.0 5.0; do
    echo "[F1] DANN, per-subject, lambda=$LAMBDA"
    python3 run_loso.py --method dann --norm per_subject --lambda_da $LAMBDA
done

for LAMBDA in 0.1 0.5 2.0 5.0; do
    echo "[F2] MMD, per-subject, lambda=$LAMBDA"
    python3 run_loso.py --method mmd --norm per_subject --lambda_da $LAMBDA
done

for LAMBDA in 0.1 0.5 2.0 5.0; do
    echo "[F3] CORAL, per-subject, lambda=$LAMBDA"
    python3 run_loso.py --method coral --norm per_subject --lambda_da $LAMBDA
done

echo ""
echo "============================================"
echo "All experiments completed!"
echo "Results saved to: /hpc_stor03/sjtu_home/duoming.jiang/agent/brain-seed/aaaresult/"
echo "============================================"