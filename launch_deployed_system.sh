#!/bin/bash
#SBATCH --job-name=deployed_rag
#SBATCH --output=/home/sbhengesa/adaptive-rag/logs/deployed_%j.out
#SBATCH --error=/home/sbhengesa/adaptive-rag/logs/deployed_%j.err
#SBATCH --partition=machinelearning
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00

module load anaconda/3
eval "$(conda shell.bash hook)"
conda activate adaptiverag
export LD_LIBRARY_PATH=/home/sbhengesa/.conda/envs/adaptiverag/lib:$LD_LIBRARY_PATH

export JAVA_HOME=/home/sbhengesa/adaptive-rag/elasticsearch/elasticsearch-7.10.2/jdk
export PATH=$JAVA_HOME/bin:$PATH

export TRANSFORMERS_CACHE=/home/sbhengesa/.cache/huggingface/hub
export HF_HOME=/home/sbhengesa/.cache/huggingface

NODE_HOSTNAME=$(hostname)
echo "=========================================="
echo "Deployed Adaptive-RAG starting"
echo "Node: $NODE_HOSTNAME"
echo "Time: $(date)"
echo "=========================================="

echo "$NODE_HOSTNAME" > /home/sbhengesa/adaptive-rag/deployed_hostname.txt
echo "$SLURM_JOB_ID"  > /home/sbhengesa/adaptive-rag/deployed_jobid.txt

# Step 1: Elasticsearch
echo ""
echo "[1/3] Starting Elasticsearch..."
cd /home/sbhengesa/adaptive-rag/elasticsearch/elasticsearch-7.10.2
./bin/elasticsearch -d -p /tmp/es_pid_$SLURM_JOB_ID

ES_READY=0
for i in $(seq 1 90); do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9200/_cluster/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "  Elasticsearch ready after ${i}s"
        ES_READY=1
        break
    fi
    sleep 2
done

if [ $ES_READY -eq 0 ]; then
    echo "ERROR: Elasticsearch failed to start"
    tail -30 /home/sbhengesa/adaptive-rag/elasticsearch/logs/adaptiverag-cluster.log
    exit 1
fi

curl -s http://localhost:9200/_cat/indices?v
echo ""

# Step 2: Retriever server
echo ""
echo "[2/3] Starting retriever server on port 8000..."
cd /home/sbhengesa/adaptive-rag/repos/Adaptive-RAG
nohup uvicorn serve:app --host 0.0.0.0 --port 8000 --app-dir retriever_server \
    > /home/sbhengesa/adaptive-rag/logs/retriever_${SLURM_JOB_ID}.log 2>&1 &

for i in $(seq 1 30); do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "  Retriever server ready after ${i}s"
        break
    fi
    sleep 2
done

# Step 3: LLM server
echo ""
echo "[3/3] Starting LLM server with flan-t5-xl on port 8010..."
echo "  (First launch downloads ~11GB, can take 15-20 min)"
cd /home/sbhengesa/adaptive-rag/repos/Adaptive-RAG
MODEL_NAME=flan-t5-xl nohup uvicorn serve:app --host 0.0.0.0 --port 8010 --app-dir llm_server \
    > /home/sbhengesa/adaptive-rag/logs/llm_${SLURM_JOB_ID}.log 2>&1 &

for i in $(seq 1 900); do
    if curl -s http://localhost:8010/ > /dev/null 2>&1; then
        echo "  LLM server ready after ${i}s"
        break
    fi
    sleep 5
done

echo ""
echo "=========================================="
echo "All services running on $NODE_HOSTNAME"
echo "Started: $(date)"
echo "=========================================="

while true; do
    sleep 300
    ES_OK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9200/_cluster/health)
    RT_OK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/)
    LM_OK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8010/)
    if [ "$ES_OK" != "200" ] || [ "$RT_OK" != "200" ] || [ "$LM_OK" != "200" ]; then
        echo "$(date): Health ES:$ES_OK RT:$RT_OK LLM:$LM_OK"
    fi
done
