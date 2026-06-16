#!/usr/bin/env bash
# Serve Llama-3.1-8B-Instruct via vLLM on a FLAIR cluster node.
#
# Usage (on the cluster node):
#   bash serve_llama.sh [GPU_INDEX] [HF_MODEL_ID]
#
# Examples:
#   bash serve_llama.sh 0 Qwen/Qwen2.5-7B-Instruct      # open, no approval needed
#   bash serve_llama.sh 0 meta-llama/Llama-3.1-8B-Instruct  # gated, needs HF token
#
# Then on your laptop, port-forward and run experiments:
#   ssh -L 8000:localhost:8000 flair6
#   VLLM_BASE_URL=http://localhost:8000/v1 python main.py --models qwen2.5-7b ...

GPU=${1:-0}
MODEL=${2:-"Qwen/Qwen2.5-7B-Instruct"}

# Build the image if it doesn't exist
if ! docker images --format '{{.Repository}}' | grep -q "^${USER}_vllm$"; then
    echo "Building vLLM Docker image..."
    docker build \
        --build-arg UID=$(id -u) \
        --build-arg GID=$(id -g) \
        -t "${USER}_vllm" \
        -f Dockerfile.vllm .
fi

echo "Starting vLLM server on GPU ${GPU} at port 8000..."
docker run --rm \
    --gpus "\"device=${GPU}\"" \
    -p 8000:8000 \
    -v "$HOME/.cache/huggingface:/app/cache" \
    -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
    --name "${USER}_llama_server" \
    "${USER}_vllm" \
    --model "${MODEL}" \
    --dtype bfloat16 \
    --port 8000
