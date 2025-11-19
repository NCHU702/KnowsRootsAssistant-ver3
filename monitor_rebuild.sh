#!/bin/bash
# 監控 agent2.py 重建進度

echo "🔍 監控索引重建進度..."
echo "按 Ctrl+C 停止監控"
echo ""

while true; do
    clear
    echo "========================================"
    echo "索引重建進度監控"
    echo "時間: $(date '+%H:%M:%S')"
    echo "========================================"
    echo ""
    
    # 檢查是否在處理 PDF
    if grep -q "Processing:" rebuild_log.txt 2>/dev/null; then
        echo "📄 處理 PDF:"
        tail -500 rebuild_log.txt | grep "Processing:" | tail -1
    fi
    
    # 檢查是否在生成摘要
    if grep -q "Summarizing" rebuild_log.txt 2>/dev/null; then
        echo ""
        echo "✍️  生成摘要進度:"
        tail -500 rebuild_log.txt | grep -E "(\[.*Processing paper|Summarizing)" | tail -5
    fi
    
    # 檢查是否完成
    if grep -q "Build Complete" rebuild_log.txt 2>/dev/null; then
        echo ""
        echo "✅ 索引重建完成！"
        tail -20 rebuild_log.txt | grep -E "(Build Complete|Papers:|Chunks:|server started)"
        break
    fi
    
    # 檢查是否啟動
    if grep -q "Running on" rebuild_log.txt 2>/dev/null; then
        echo ""
        echo "🚀 Agent2 已啟動！"
        tail -10 rebuild_log.txt | grep "Running on"
        break
    fi
    
    # 檢查錯誤
    if tail -50 rebuild_log.txt 2>/dev/null | grep -qi "error"; then
        echo ""
        echo "❌ 發現錯誤:"
        tail -50 rebuild_log.txt | grep -i "error" | tail -3
    fi
    
    echo ""
    echo "最新日誌:"
    tail -3 rebuild_log.txt 2>/dev/null
    
    sleep 3
done
