#!/bin/bash

echo "╔════════════════════════════════════════════════════════╗"
echo "║    Agent 串流化功能 - 快速測試指南                    ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# 顏色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 檢查虛擬環境
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ 找不到虛擬環境！${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 虛擬環境檢查通過${NC}"
echo ""

# 檢查服務器狀態
echo -e "${BLUE}🔍 檢查服務器狀態...${NC}"
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ 服務器正在運行${NC}"
    echo ""
    
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║            開始測試 Agent 推理過程串流               ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${YELLOW}📝 測試項目：${NC}"
    echo "  1. 網路搜尋查詢 (WebSearchCall)"
    echo "  2. 論文摘要任務 (AssistantCall)"
    echo "  3. RAG 查詢 (AssistantCall)"
    echo ""
    echo -e "${YELLOW}💡 觀察重點：${NC}"
    echo "  • 💭 Thought - Agent 的推理過程"
    echo "  • 🔧 Action - 選擇的工具"
    echo "  • ⚙️  Tool Execution - 工具執行過程"
    echo "  • 👁️ Observation - 執行結果"
    echo "  • ✅ Final Answer - 最終答案"
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo ""
    
    # 運行測試
    .venv/bin/python test_agent_streaming.py
    
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo -e "${GREEN}✅ 測試完成！${NC}"
    echo ""
    echo -e "${YELLOW}📊 下一步：瀏覽器測試${NC}"
    echo "  1. 開啟 ${BLUE}http://localhost:5000${NC}"
    echo "  2. 在 General Assistant 輸入問題"
    echo "  3. 觀察黃色推理框即時顯示 Agent 思考過程"
    echo ""
    echo -e "${YELLOW}📝 測試建議：${NC}"
    echo "  • \"Search for latest AI papers in 2024\""
    echo "  • \"Summarize papers about neural networks\""
    echo "  • \"什麼是深度學習？請詳細說明\""
    echo ""
    
else
    echo -e "${RED}❌ 服務器未運行${NC}"
    echo ""
    echo -e "${YELLOW}請先啟動服務器：${NC}"
    echo "  ${BLUE}.venv/bin/python agent2.py${NC}"
    echo ""
    echo "服務器啟動後，再次運行此腳本："
    echo "  ${BLUE}bash quick_test_agent_streaming.sh${NC}"
    echo ""
fi
