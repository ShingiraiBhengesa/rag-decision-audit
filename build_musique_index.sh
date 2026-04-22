#!/bin/bash
#SBATCH --job-name=build_idx
#SBATCH --output=/home/sbhengesa/adaptive-rag/logs/build_idx_%j.out
#SBATCH --error=/home/sbhengesa/adaptive-rag/logs/build_idx_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

module load anaconda/3
eval "$(conda shell.bash hook)"
conda activate adaptiverag
export LD_LIBRARY_PATH=/home/sbhengesa/.conda/envs/adaptiverag/lib:$LD_LIBRARY_PATH

export JAVA_HOME=/home/sbhengesa/adaptive-rag/elasticsearch/elasticsearch-7.10.2/jdk
export PATH=$JAVA_HOME/bin:$PATH

NODE=$(hostname)
echo "Building MuSiQue index on $NODE"
echo "Start: $(date)"

cd /home/sbhengesa/adaptive-rag/elasticsearch/elasticsearch-7.10.2
./bin/elasticsearch -d -p /tmp/es_build_$SLURM_JOB_ID

echo "Waiting for Elasticsearch..."
ES_READY=0
for i in $(seq 1 90); do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9200/_cluster/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "Elasticsearch ready after ${i}s"
        ES_READY=1
        break
    fi
    sleep 2
done

if [ $ES_READY -eq 0 ]; then
    echo "ERROR: Elasticsearch failed to start"
    tail -30 /home/sbhengesa/adaptive-rag/elasticsearch/logs/adaptiverag-cluster.log 2>/dev/null
    exit 1
fi

curl -s http://localhost:9200/_cluster/health
echo ""

echo ""
echo "Building MuSiQue index..."
cd /home/sbhengesa/adaptive-rag/repos/Adaptive-RAG
python retriever_server/build_index.py musique
BUILD_EXIT=$?

if [ $BUILD_EXIT -ne 0 ]; then
    echo "ERROR: build_index.py failed with exit $BUILD_EXIT"
    pkill -f elasticsearch 2>/dev/null
    exit $BUILD_EXIT
fi

echo ""
echo "Verifying index..."
curl -s "http://localhost:9200/_cat/indices?v"
echo ""

python << 'PYEOF'
from elasticsearch import Elasticsearch
es = Elasticsearch("http://localhost:9200")
indices = list(es.indices.get_alias("*").keys())
print(f"Indices: {indices}")
for idx_name in indices:
    if not idx_name.startswith("."):
        count = es.count(index=idx_name)['count']
        print(f"Index {idx_name}: {count} documents")
        result = es.search(index=idx_name, body={
            "query": {"match": {"paragraph_text": "Danish Football Union"}},
            "size": 2
        })
        print(f"Sample query returned {len(result['hits']['hits'])} hits")
        if result['hits']['hits']:
            print(f"  First hit: {result['hits']['hits'][0]['_source'].get('title', 'N/A')}")
PYEOF

echo ""
echo "Stopping Elasticsearch..."
pkill -f elasticsearch 2>/dev/null
sleep 3
echo "End: $(date)"
echo "Done."
