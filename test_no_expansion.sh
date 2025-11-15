#!/bin/bash

# 測試純語義策略（無查詢擴展）

echo "========================================="
echo "啟動測試: 純語義策略（無查詢擴展）"
echo "========================================="
echo ""

# 啟動 agent2.py 在背景
echo "啟動 agent2.py..."
python agent2.py > /tmp/agent2_test.log 2>&1 &
AGENT_PID=$!

# 等待啟動
echo "等待系統啟動 (15秒)..."
sleep 15

# 測試查詢
echo ""
echo "發送測試查詢: 深度學習在醫療的應用是什麼？"
echo ""

curl -X POST http://localhost:4000/query_stream \
  -H "Content-Type: application/json" \
  -d '{"query": "深度學習在醫療的應用是什麼？"}' \
  --no-buffer 2>/dev/null | head -100

echo ""
echo ""
echo "========================================="
echo "測試完成"
echo "========================================="

# 停止 agent2
kill $AGENT_PID 2>/dev/null

echo ""
echo "查看完整日誌:"
echo "tail -100 /tmp/agent2_test.log"
