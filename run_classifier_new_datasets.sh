#!/bin/bash
#SBATCH --job-name=classify_new
#SBATCH --output=/home/sbhengesa/adaptive-rag/logs/classify_new_%j.out
#SBATCH --error=/home/sbhengesa/adaptive-rag/logs/classify_new_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

module load anaconda/3
module load cuda12.8/toolkit/12.8.1
eval "$(conda shell.bash hook)"
conda activate adaptiverag
export LD_LIBRARY_PATH=/home/sbhengesa/.conda/envs/adaptiverag/lib:$LD_LIBRARY_PATH

cd ~/adaptive-rag/repos/Adaptive-RAG/classifier

MODEL_PATH=./outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xxl/epoch/30

echo "=== StrategyQA ==="
OUTDIR=./outputs/new_datasets/strategyqa
mkdir -p $OUTDIR
python run_classifier.py \
    --model_name_or_path ${MODEL_PATH} \
    --validation_file ./data/new_datasets/strategyqa_predict.json \
    --question_column question \
    --answer_column answer \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_eval_batch_size 100 \
    --output_dir ${OUTDIR} \
    --overwrite_cache \
    --val_column 'validation' \
    --do_eval

echo "=== BoolQ ==="
OUTDIR=./outputs/new_datasets/boolq
mkdir -p $OUTDIR
python run_classifier.py \
    --model_name_or_path ${MODEL_PATH} \
    --validation_file ./data/new_datasets/boolq_predict.json \
    --question_column question \
    --answer_column answer \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_eval_batch_size 100 \
    --output_dir ${OUTDIR} \
    --overwrite_cache \
    --val_column 'validation' \
    --do_eval

if [ -f ./data/new_datasets/complexweb_predict.json ]; then
    echo "=== ComplexWebQuestions ==="
    OUTDIR=./outputs/new_datasets/complexweb
    mkdir -p $OUTDIR
    python run_classifier.py \
        --model_name_or_path ${MODEL_PATH} \
        --validation_file ./data/new_datasets/complexweb_predict.json \
        --question_column question \
        --answer_column answer \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_eval_batch_size 100 \
        --output_dir ${OUTDIR} \
        --overwrite_cache \
        --val_column 'validation' \
        --do_eval
fi

echo "=== DONE ==="
