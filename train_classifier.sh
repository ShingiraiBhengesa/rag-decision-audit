#!/bin/bash
#SBATCH --job-name=adaptive_classifier
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=adaptive_classifier_%j.out
#SBATCH --error=adaptive_classifier_%j.err

module load anaconda/3
module load cuda12.8/toolkit/12.8.1
eval "$(conda shell.bash hook)"
conda activate adaptiverag
export LD_LIBRARY_PATH=/home/sbhengesa/.conda/envs/adaptiverag/lib:$LD_LIBRARY_PATH

cd ~/adaptive-rag/repos/Adaptive-RAG/classifier

DATASET_NAME=musique_hotpot_wiki2_nq_tqa_sqd
LLM_NAME=flan_t5_xxl
MODEL=t5-large

for EPOCH in 15 20 25 30 35
do
    echo "=========================================="
    echo "TRAINING EPOCH: ${EPOCH}"
    echo "=========================================="

    TRAIN_OUTPUT_DIR=./outputs/${DATASET_NAME}/model/${MODEL}/${LLM_NAME}/epoch/${EPOCH}
    mkdir -p ${TRAIN_OUTPUT_DIR}

    python run_classifier.py \
        --model_name_or_path ${MODEL} \
        --train_file ./data/${DATASET_NAME}/${LLM_NAME}/binary_silver/train.json \
        --question_column question \
        --answer_column answer \
        --learning_rate 3e-5 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_train_batch_size 32 \
        --output_dir ${TRAIN_OUTPUT_DIR} \
        --overwrite_cache \
        --train_column 'train' \
        --do_train \
        --num_train_epochs ${EPOCH}

    VALID_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}/valid
    mkdir -p ${VALID_OUTPUT_DIR}

    python run_classifier.py \
        --model_name_or_path ${TRAIN_OUTPUT_DIR} \
        --validation_file ./data/${DATASET_NAME}/${LLM_NAME}/silver/valid.json \
        --question_column question \
        --answer_column answer \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_eval_batch_size 100 \
        --output_dir ${VALID_OUTPUT_DIR} \
        --overwrite_cache \
        --val_column 'validation' \
        --do_eval

    PREDICT_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}/predict
    mkdir -p ${PREDICT_OUTPUT_DIR}

    python run_classifier.py \
        --model_name_or_path ${TRAIN_OUTPUT_DIR} \
        --validation_file ./data/${DATASET_NAME}/predict.json \
        --question_column question \
        --answer_column answer \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_eval_batch_size 100 \
        --output_dir ${PREDICT_OUTPUT_DIR} \
        --overwrite_cache \
        --val_column 'validation' \
        --do_eval
done

echo "DONE!"
