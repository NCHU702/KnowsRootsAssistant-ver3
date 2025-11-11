#!/bin/bash

echo "======================================"
echo "串流輸出功能測試指南"
echo "======================================"
echo ""

# 檢查虛擬環境
if [ ! -d ".venv" ]; then
    echo "❌ 找不到虛擬環境！請先運行："
    echo "   python -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

echo "✅ 虛擬環境檢查通過"
echo ""

# 檢查服務器是否運行
echo "正在檢查服務器狀態..."
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ 服務器正在運行"
    echo ""
    
    echo "開始測試串流功能..."
    echo "======================================"
    .venv/bin/python test_streaming.py
    echo ""
    echo "======================================"
    echo "測試完成！"
    echo ""
    echo "💡 提示："
    echo "1. 在瀏覽器開啟 http://localhost:5000"
    echo "2. 在 General Assistant 輸入問題"
    echo "3. 觀察文字是否逐字顯示"
    
else
    echo "❌ 服務器未運行"
    echo ""
    echo "請先啟動服務器："
    echo "   .venv/bin/python agent2.py"
    echo ""
    echo "服務器啟動後，再次運行此腳本："
    echo "   bash test_streaming_feature.sh"
fi
