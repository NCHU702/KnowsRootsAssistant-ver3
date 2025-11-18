#!/bin/bash
# 啟用 Summarization 模式並重建索引

echo "=============================================="
echo "  Agent2 Summarization 模式啟用腳本"
echo "=============================================="
echo ""

# 檢查 Ollama 是否運行
echo "🔍 檢查 Ollama 服務..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama 正在運行"
else
    echo "✗ Ollama 未運行！請先啟動 Ollama:"
    echo "  請執行: ollama serve"
    exit 1
fi

# 檢查模型
echo ""
echo "🔍 檢查 llama3:8b 模型..."
if ollama list | grep -q "llama3:8b"; then
    echo "✓ llama3:8b 模型已安裝"
else
    echo "⚠ llama3:8b 模型未安裝"
    echo "  正在下載模型 (這可能需要幾分鐘)..."
    ollama pull llama3:8b
    if [ $? -eq 0 ]; then
        echo "✓ llama3:8b 模型下載完成"
    else
        echo "✗ 模型下載失敗！"
        exit 1
    fi
fi

# 設定環境變數
echo ""
echo "🔧 設定環境變數..."
export CHUNKING_MODE=summarization
export SUMMARIZATION_MODEL=llama3:8b
export TARGET_SUMMARY_LENGTH=300
export MIN_SECTIONS=3
export MAP_REDUCE_THRESHOLD=1500

echo "✓ 環境變數已設定:"
echo "  CHUNKING_MODE=$CHUNKING_MODE"
echo "  SUMMARIZATION_MODEL=$SUMMARIZATION_MODEL"
echo "  TARGET_SUMMARY_LENGTH=$TARGET_SUMMARY_LENGTH"

# 備份舊索引
if [ -d "vectorstore" ]; then
    echo ""
    echo "📦 備份現有索引..."
    timestamp=$(date +%Y%m%d_%H%M%S)
    mv vectorstore "vectorstore_backup_$timestamp"
    echo "✓ 舊索引已備份到: vectorstore_backup_$timestamp"
fi

# 提示用戶
echo ""
echo "=============================================="
echo "  準備完成！"
echo "=============================================="
echo ""
echo "📝 接下來請執行:"
echo ""
echo "  python agent2.py"
echo ""
echo "系統將使用 SUMMARIZATION 模式重建索引。"
echo ""
echo "💡 提示:"
echo "  - 首次建立索引需要較長時間 (依論文數量而定)"
echo "  - 可以在啟動訊息中確認 'Chunking Mode: SUMMARIZATION'"
echo "  - 建立完成後，RAG 質量將顯著提升！"
echo ""
echo "=============================================="
