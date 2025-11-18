#!/bin/bash
# Agent2 測試腳本 - 純 Re-ranking 模式（已捨棄 FAISS）

echo "======================================"
echo "Agent2 測試 - 純 Re-ranking 模式"
echo "======================================"
echo ""
echo "配置："
echo "  Layer 2: Cross-Encoder Re-ranking (FAISS 已移除)"
echo "  模型: qllama/bce-reranker-base_v1:latest"
echo "  存儲: JSONL (人類可讀格式)"
echo ""

# 設定環境變數
export LAYER1_K_DOCUMENTS=10
export LAYER2_K_DOCUMENTS=10
export LAYER1_THRESHOLD=0.6
export LAYER2_THRESHOLD=0.8
export CACHE_SIZE=10
export ENABLE_EXPANSION=true

echo "環境變數已設定："
echo "  LAYER1_K_DOCUMENTS=$LAYER1_K_DOCUMENTS"
echo "  LAYER2_K_DOCUMENTS=$LAYER2_K_DOCUMENTS"
echo ""

# 檢查 Ollama 服務
echo "檢查 Ollama 服務..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama 服務運行中"
else
    echo "✗ Ollama 服務未運行，請先啟動 Ollama"
    echo "  執行: ollama serve"
    exit 1
fi

# 檢查 re-ranking 模型
echo ""
echo "檢查 Re-ranking 模型..."
if ollama list | grep -q "bce-reranker-base_v1"; then
    echo "✓ Re-ranking 模型已安裝"
else
    echo "⚠ Re-ranking 模型未安裝"
    echo "  安裝命令: ollama pull qllama/bce-reranker-base_v1:latest"
    read -p "是否現在安裝? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ollama pull qllama/bce-reranker-base_v1:latest
    else
        echo "跳過安裝，繼續啟動..."
    fi
fi

echo ""
echo "啟動 Agent2 (Re-ranking 模式)..."
echo "訪問: http://localhost:5001"
echo ""
python agent2.py
