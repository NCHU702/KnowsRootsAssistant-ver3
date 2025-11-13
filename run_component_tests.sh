#!/bin/bash

# Hierarchical RAG 組件測試腳本

echo "=========================================="
echo "  Hierarchical RAG 組件測試"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安裝"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "⚠️  Ollama 服務未運行，嘗試啟動..."
    ollama serve &
    sleep 3
fi

# Check required models
echo "檢查所需模型..."
if ! ollama list | grep -q "gemma3:12b"; then
    echo "❌ LLM 模型未安裝: gemma3:12b"
    echo "請執行: ollama pull gemma3:12b"
    exit 1
fi

if ! ollama list | grep -q "embeddinggemma"; then
    echo "⚠️  Embedding 模型未安裝: embeddinggemma:latest"
    echo "請執行: ollama pull embeddinggemma:latest"
fi

echo "✓ 環境檢查完成"
echo ""

# Create test directories
mkdir -p vectorstore_test/layer1
mkdir -p vectorstore_test/layer2

# Run tests
echo "開始執行測試..."
echo ""

python3 test_hierarchical_components.py

# Capture exit code
EXIT_CODE=$?

# Cleanup
echo ""
echo "清理測試資料..."
# rm -rf vectorstore_test  # Uncomment to auto-cleanup

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ 測試完成"
else
    echo ""
    echo "❌ 測試失敗 (退出碼: $EXIT_CODE)"
fi

exit $EXIT_CODE
