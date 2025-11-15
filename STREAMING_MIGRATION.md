# 串流模式遷移完成

## ✅ 已完成的變更

### 1. 移除非串流端點
- ❌ 已移除：`POST /query` (非串流模式)
- ✅ 保留：`POST /query_stream` (串流模式)

### 2. 系統架構簡化

#### 移除的代碼
```python
# 已移除整個 /query 端點函數 (~100 行代碼)
@app.route('/query', methods=['POST'])
def query():
    # ... 非串流處理邏輯
```

#### 保留的組件
```python
# assistant_call - 保留作為工具定義和緊急備援
def assistant_call(user_input: str) -> str:
    # 在串流模式下，此函數不會被直接調用
    # Agent 會在 query_stream 端點中直接使用 rag_system.query_stream()
    return rag_system.query(user_input)  # 非串流備援

# fallback_routing - 保留作為緊急備援
def fallback_routing(user_input):
    # 此函數現在不會被使用（/query 端點已移除）
    # 保留以防未來需要
```

## 🎯 新的請求流程

### 用戶查詢流程（100% 串流）

```
前端發送請求
    ↓
POST /query_stream
    ↓
Agent 決策
    ├─ AssistantCall → rag_system.query_stream() → 串流輸出論文答案
    └─ WebSearchCall → web_searcher.search() → 串流輸出網頁結果
    ↓
前端即時顯示（打字機效果）
```

### 詳細流程

```python
# 1. 前端發送請求
fetch('/query_stream', {
    method: 'POST',
    body: JSON.stringify({ input: "深度學習的應用" })
})

# 2. 後端串流處理
@app.route('/query_stream', methods=['POST'])
def query_stream():
    # Agent 判斷使用 AssistantCall
    if event['tool'] == 'AssistantCall':
        # 直接串流 RAG 輸出
        for chunk in rag_system.query_stream(action_input):
            yield f"data: {json.dumps({'type': 'tool_token', 'content': chunk})}\n\n"

# 3. RAG 系統串流生成
def query_stream(self, query: str):
    # 先輸出參考論文
    yield "📚 參考論文：\n"
    for paper in papers:
        yield f"- {paper}\n"
    
    # 串流 LLM 回答
    for chunk in self.llm.stream(prompt):
        yield chunk  # "深", "度", "學", "習", ...

# 4. 前端即時顯示
eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'tool_token') {
        outputDiv.innerText += data.content;  // 逐字累加
    }
}
```

## 📊 代碼簡化統計

| 項目 | 變更前 | 變更後 | 減少 |
|------|--------|--------|------|
| API 端點 | 2 個 | 1 個 | -50% |
| 查詢處理邏輯 | 雙軌（串流+非串流） | 單軌（純串流） | -50% |
| 代碼行數 | ~1241 行 | ~1158 行 | -83 行 |
| 維護複雜度 | 高（兩種模式） | 低（單一模式） | ⬇️ |

## 🎨 用戶體驗

### 之前（雙模式）
```
用戶請求 → 等待 5-30 秒 → 突然顯示完整答案
          (無反饋)        (可能看起來像卡住)
```

### 現在（純串流）
```
用戶請求 → 0.5s 顯示 "Thinking..." 
        → 1s 顯示 "📚 參考論文：..."
        → 2s 開始逐字顯示答案 "深度學習..."
        → 3s "深度學習的主要應用..."
        (持續即時反饋，流暢體驗)
```

## 🔧 技術細節

### 串流協議
使用 **Server-Sent Events (SSE)**：
```
Content-Type: text/event-stream

data: {"type": "start", "mode": "agent"}

data: {"type": "thought", "content": "分析中..."}

data: {"type": "action", "tool": "AssistantCall"}

data: {"type": "tool_token", "content": "深"}

data: {"type": "tool_token", "content": "度"}

data: {"type": "done"}
```

### 停止生成功能
用戶可隨時停止生成：
```javascript
// 前端
stopButton.onclick = () => {
    if (currentStreamReader) {
        currentStreamReader.cancel();  // 取消串流
    }
}

// 後端
if stop_event.is_set():
    logger.info("Stop signal detected")
    rag_stream.close()  // 關閉生成器
    break
```

## 📝 API 文檔更新

### 唯一的查詢端點

**`POST /query_stream`**

請求：
```json
{
    "input": "深度學習的主要應用有哪些？"
}
```

回應（串流）：
```
data: {"type": "start", "mode": "agent"}

data: {"type": "thought", "content": "..."}

data: {"type": "action", "tool": "AssistantCall", "input": "..."}

data: {"type": "tool_start", "tool": "AssistantCall"}

data: {"type": "tool_token", "content": "📚 參考論文：\n"}

data: {"type": "tool_token", "content": "- Deep Learning (LeCun, 2015)\n"}

data: {"type": "tool_token", "content": "\n"}

data: {"type": "tool_token", "content": "深"}

data: {"type": "tool_token", "content": "度"}

data: {"type": "tool_token", "content": "學"}

data: {"type": "tool_token", "content": "習"}

...

data: {"type": "done"}
```

## ⚠️ 注意事項

### 保留的非串流代碼
以下函數仍使用非串流 API，但僅作為緊急備援：

1. **`assistant_call()`**
   - 用途：Agent 工具定義
   - 實際調用：在串流模式下不會被調用
   - 保留原因：工具定義需要，緊急備援

2. **`fallback_routing()`**
   - 用途：關鍵字備援路由
   - 實際調用：不會被調用（/query 已移除）
   - 保留原因：向後兼容，緊急備援

### 這些函數何時會被調用？
**正常情況：從不**
- 所有請求都通過 `/query_stream`
- Agent 在串流端點中直接使用 `rag_system.query_stream()`

**異常情況：極少數**
- 如果未來需要添加非串流 API
- 如果需要在終端直接測試功能

## 🚀 優勢

### 1. 代碼更簡潔
- 移除重複邏輯
- 單一維護路徑
- 更容易理解

### 2. 更好的用戶體驗
- 即時反饋
- 流暢顯示
- 可中斷生成

### 3. 更好的可維護性
- 只需維護一種模式
- bug 修復更簡單
- 功能新增更容易

### 4. 現代化
- 符合 ChatGPT 等現代 AI 應用標準
- 提供專業級用戶體驗

## 📦 向後兼容性

### ❌ 不兼容的變更
如果有外部系統使用 `POST /query`，需要遷移到 `POST /query_stream`。

### 遷移指南
```javascript
// 舊代碼（不再支援）
fetch('/query', {
    method: 'POST',
    body: JSON.stringify({ input: query })
})
.then(r => r.json())
.then(data => {
    console.log(data.output);  // 完整輸出
});

// 新代碼（推薦）
const response = await fetch('/query_stream', {
    method: 'POST',
    body: JSON.stringify({ input: query })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');
    
    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            if (data.type === 'tool_token' || data.type === 'chunk') {
                console.log(data.content);  // 串流輸出
            }
        }
    }
}
```

## ✅ 測試建議

1. **正常查詢測試**
   ```bash
   curl -X POST http://localhost:4000/query_stream \
     -H "Content-Type: application/json" \
     -d '{"input": "深度學習的應用"}'
   ```

2. **中斷生成測試**
   - 前端點擊停止按鈕
   - 確認生成立即停止

3. **中文/英文測試**
   - 中文查詢 → 中文回答
   - 英文查詢 → 英文回答

4. **論文參考顯示測試**
   - 確認回答前顯示參考論文列表
   - 檢查論文格式正確

---

**遷移完成日期**：2025-01-14  
**狀態**：✅ 已完成並測試  
**影響範圍**：前後端統一使用串流模式
