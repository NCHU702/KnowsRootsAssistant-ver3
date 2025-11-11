#!/bin/bash

echo "╔════════════════════════════════════════════════════════╗"
echo "║     簡化版 Agent 串流 - 測試指南                      ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${YELLOW}📝 改進說明：${NC}"
echo "  ✅ 移除複雜的推理框"
echo "  ✅ 只顯示當前狀態（可覆蓋）"
echo "  ✅ 開始輸出時移除狀態"
echo "  ✅ 最終只保留 LLM 輸出內容"
echo ""

echo -e "${YELLOW}🎬 視覺效果：${NC}"
echo "  1. 🤔 Thinking..."
echo "  2. 💭 Analyzing..."
echo "  3. 🔧 Using AssistantCall..."
echo "  4. ⚙️  Executing AssistantCall..."
echo "  5. [狀態消失] 深度學習是..."
echo "  6. [最終] 只剩下完整輸出內容"
echo ""

echo "════════════════════════════════════════════════════════"
echo -e "${GREEN}🚀 啟動測試${NC}"
echo "════════════════════════════════════════════════════════"
echo ""

if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ 服務器正在運行${NC}"
    echo ""
    echo -e "${BLUE}請在瀏覽器測試：${NC}"
    echo "  1. 開啟 http://localhost:5000"
    echo "  2. 輸入問題，例如："
    echo "     • 什麼是深度學習？"
    echo "     • Summarize papers about neural networks"
    echo ""
    echo -e "${YELLOW}觀察重點：${NC}"
    echo "  ✅ 狀態文字會逐步更新（覆蓋）"
    echo "  ✅ 開始輸出時，狀態消失"
    echo "  ✅ 最終畫面乾淨，只有內容"
    echo ""
else
    echo -e "${YELLOW}⚠️  服務器未運行${NC}"
    echo ""
    echo "請先啟動服務器："
    echo "  ${BLUE}.venv/bin/python agent2.py${NC}"
    echo ""
fi
