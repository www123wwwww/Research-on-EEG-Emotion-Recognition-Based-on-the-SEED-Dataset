#!/bin/bash
# Extended lambda sensitivity experiments on GPU
# lambda values: 1.0, 2.0, 5.0, 10.0, 20.0
# methods: coral, mmd, dann
# norm: per_subject
# Results saved to /hpc_stor03/sjtu_home/duoming.jiang/agent/brain-seed/aaaresult/

export CUDA_VISIBLE_DEVICES=2
cd /aistor/hpc_stor03/sjtu_home/duoming.jiang/agent/brain-seed/Research-on-EEG-Emotion-Recognition-Based-on-the-SEED-Dataset

echo "Starting extended lambda sweep at $(date)"

for METHOD in coral mmd dann; do
    for LAMBDA in 1.0 2.0 5.0 10.0 20.0; do
        echo "=== $METHOD lambda=$LAMBDA ==="
        python3 run_loso.py --method $METHOD --norm per_subject --lambda_da $LAMBDA --output_dir /hpc_stor03/sjtu_home/duoming.jiang/agent/brain-seed/aaaresult
        echo "Finished $METHOD lambda=$LAMBDA at $(date)"
    done
done

echo "All experiments done at $(date)"