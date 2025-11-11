# Stop Generation 功能

## 🎯 功能說明

添加 **停止生成 (Stop Generation)** 按鈕，讓用戶可以隨時中斷 Agent 的執行過程。

---

## ✨ 核心特性

- ✅ **隨時停止** - Agent 執行到任何階段都可以中斷
- ✅ **保留部分輸出** - 已生成的內容會被保留
- ✅ **狀態管理** - 按鈕顯示/隱藏根據執行狀態自動切換
- ✅ **清理資源** - 正確釋放 stream reader 和連接

---

## 🎨 UI 設計

### **停止按鈕樣式**

```css
.stop-btn {
    padding: 12px 24px;
    background: #ef4444;    /* 紅色背景 */
    color: white;
    border-radius: 8px;
    font-size: 14px;
    cursor: pointer;
    display: none;          /* 預設隱藏 */
}

.stop-btn.active {
    display: block;         /* 執行時顯示 */
}

.stop-btn:hover {
    background: #dc2626;    /* 懸停變深紅 */
}
```

### **按鈕位置**

```
┌────────────────────────────────────────┐
│  [輸入框]           [Send]  [⏹ Stop]  │
└────────────────────────────────────────┘
```

**狀態切換：**
- **閒置時**: 只顯示 `Send` 按鈕
- **執行中**: 顯示 `Send (disabled)` + `Stop` 按鈕
- **完成後**: 恢復只顯示 `Send` 按鈕

---

## 🔧 技術實作

### **前端核心邏輯**

#### **1. 全域變數追蹤**

```javascript
let currentStreamReader = null;  // 當前的 stream reader
let isStreaming = false;         // 是否正在執行
```

#### **2. 開始執行時**

```javascript
async function submitGeneralQuery() {
    if (isStreaming) {
        return;  // 防止重複執行
    }
    
    // 禁用發送按鈕，顯示停止按鈕
    generalSubmitBtn.disabled = true;
    stopBtn.classList.add('active');
    isStreaming = true;
    
    // 獲取 stream reader
    const reader = response.body.getReader();
    currentStreamReader = reader;  // 保存引用
    
    // ... 讀取 stream
}
```

#### **3. 停止功能**

```javascript
function stopGeneration() {
    if (currentStreamReader) {
        try {
            currentStreamReader.cancel();  // 取消 stream
            currentStreamReader = null;
            isStreaming = false;
            
            // 隱藏停止按鈕，啟用發送按鈕
            stopBtn.classList.remove('active');
            generalSubmitBtn.disabled = false;
            
            console.log('Stream cancelled by user');
        } catch (e) {
            console.error('Error cancelling stream:', e);
        }
    }
}
```

#### **4. 錯誤處理**

```javascript
catch (error) {
    if (error.name === 'AbortError' || error.message.includes('cancel')) {
        // 用戶主動取消
        console.log('Stream cancelled by user');
        
        // 顯示取消訊息
        if (!fullResponse) {
            contentDiv.innerHTML = '<p>Generation stopped by user.</p>';
        }
        
        // 保存部分輸出
        if (fullResponse) {
            generalHistory.push({ 
                content: fullResponse + '\n\n[Stopped by user]'
            });
        }
    } else {
        // 其他錯誤
        showError('Failed to connect: ' + error.message);
    }
}
```

#### **5. 清理資源**

```javascript
finally {
    // 無論如何都要清理
    currentStreamReader = null;
    isStreaming = false;
    stopBtn.classList.remove('active');
    generalSubmitBtn.disabled = false;
    generalInput.focus();
}
```

---

## 🎬 使用場景

### **場景 1：立即停止**

```
[0.0秒] 用戶提問: "什麼是深度學習？"
[0.2秒] 狀態: 🤔 Thinking...
[0.5秒] 狀態: 💭 Analyzing...
[0.8秒] 用戶點擊 [⏹ Stop]
        ↓
        [取消] Stream 被中斷
        [顯示] "Generation stopped by user."
```

### **場景 2：部分生成後停止**

```
[0.0秒] 用戶提問: "Summarize papers..."
[1.0秒] 開始輸出: "深度學習是..."
[2.0秒] 繼續輸出: "深度學習是一種機器學習方法..."
[3.0秒] 用戶點擊 [⏹ Stop]
        ↓
        [保留] 已生成的內容
        [標記] "[Stopped by user]"
        [儲存] 部分輸出到歷史記錄
```

### **場景 3：Agent 推理中停止**

```
[0.0秒] 用戶提問: "Search for papers 2024"
[0.5秒] 狀態: 💭 Analyzing...
[0.8秒] 狀態: 🔧 Using WebSearchCall...
[1.0秒] 用戶點擊 [⏹ Stop]
        ↓
        [中斷] Agent 停止執行
        [顯示] "Generation stopped by user."
```

---

## 📊 狀態流程圖

```
[用戶提交]
    ↓
[Send] disabled
[Stop] 顯示 (紅色)
isStreaming = true
    ↓
[執行中...]
    ├─ 用戶點擊 [Stop]
    │   ↓
    │   currentStreamReader.cancel()
    │   ↓
    │   拋出 AbortError
    │   ↓
    │   catch 處理取消邏輯
    │   ↓
    │   finally 清理
    │
    └─ 正常完成
        ↓
        finally 清理
    ↓
[Send] enabled
[Stop] 隱藏
isStreaming = false
```

---

## 🧪 測試方法

### **測試 1：立即停止**
1. 輸入問題
2. 立即點擊 `Stop` 按鈕
3. 預期：顯示 "Generation stopped by user."

### **測試 2：部分生成後停止**
1. 輸入問題
2. 等待開始輸出幾個字
3. 點擊 `Stop` 按鈕
4. 預期：保留已生成的內容 + "[Stopped by user]"

### **測試 3：連續請求**
1. 輸入問題 A
2. 在執行中輸入問題 B
3. 預期：問題 B 被忽略（防止重複執行）

### **測試 4：停止後繼續**
1. 輸入問題
2. 點擊 `Stop`
3. 再次輸入新問題
4. 預期：正常執行新請求

---

## 🎨 視覺效果

### **執行前**
```
┌────────────────────────────────────┐
│ [輸入框...]            [Send]      │
└────────────────────────────────────┘
```

### **執行中**
```
┌────────────────────────────────────┐
│ [輸入框...]    [Send🚫]  [⏹Stop]  │
└────────────────────────────────────┘
        ↑            ↑          ↑
      禁用        灰色      紅色閃爍
```

### **停止後**
```
┌────────────────────────────────────┐
│ [輸入框...]            [Send]      │
└────────────────────────────────────┘

[對話框]
🧑 User: 什麼是深度學習？
🤖 Assistant: 深度學習是一種...
             [Stopped by user]
```

---

## 💡 實作要點

### **1. 防止重複執行**
```javascript
if (isStreaming) {
    return;  // 已在執行中，忽略新請求
}
```

### **2. ReadableStream 取消**
```javascript
currentStreamReader.cancel();  // 關鍵：取消 stream
```

### **3. 保留部分輸出**
```javascript
if (fullResponse) {
    // 有部分輸出 → 保留
    generalHistory.push({ content: fullResponse + '\n\n[Stopped by user]' });
} else {
    // 無輸出 → 顯示取消訊息
    contentDiv.innerHTML = '<p>Generation stopped by user.</p>';
}
```

### **4. 清理資源**
```javascript
finally {
    currentStreamReader = null;   // 釋放引用
    isStreaming = false;          // 重置狀態
    stopBtn.classList.remove('active');  // 隱藏按鈕
}
```

---

## 🔍 瀏覽器兼容性

### **ReadableStream.cancel()**
- ✅ Chrome 43+
- ✅ Firefox 65+
- ✅ Safari 10.1+
- ✅ Edge 14+

**支援度**: 所有現代瀏覽器

---

## 🚀 優勢

1. **用戶控制** - 隨時可以停止長時間執行
2. **節省資源** - 不需要的執行可以提前終止
3. **保留進度** - 已生成的內容不會丟失
4. **狀態清晰** - 按鈕顯示/隱藏直觀
5. **錯誤處理** - 正確區分取消和錯誤

---

## 📝 檔案修改

### **templates/index.html**

**新增 CSS** (15 行):
```css
.stop-btn { ... }
.stop-btn:hover { ... }
.stop-btn.active { ... }
```

**新增 HTML** (1 行):
```html
<button id="stopBtn" class="stop-btn" onclick="stopGeneration()">⏹ Stop</button>
```

**新增 JavaScript** (50+ 行):
- `stopGeneration()` 函數
- 修改 `submitGeneralQuery()` 添加停止支援
- 全域變數 `currentStreamReader`, `isStreaming`
- 錯誤處理邏輯

---

## 🎓 技術亮點

### **1. Stream Cancellation**
```javascript
const reader = response.body.getReader();
// ... later
reader.cancel();  // ← 關鍵 API
```

### **2. State Management**
```javascript
let isStreaming = false;  // 防止重複執行
let currentStreamReader = null;  // 追蹤當前 reader
```

### **3. Graceful Degradation**
```javascript
catch (error) {
    if (error.name === 'AbortError') {
        // 取消 → 保留部分輸出
    } else {
        // 錯誤 → 顯示錯誤訊息
    }
}
```

---

## 🔮 未來擴展

### **可能的改進**
- [ ] 顯示已執行時間
- [ ] 確認對話框（避免誤按）
- [ ] 快捷鍵支援 (Esc)
- [ ] 進度條顯示
- [ ] 暫停/繼續功能

---

## ✨ 總結

Stop Generation 功能提供了：

✅ **完整的停止控制** - 隨時中斷執行  
✅ **部分輸出保留** - 已生成內容不丟失  
✅ **清晰的視覺反饋** - 按鈕狀態直觀  
✅ **健全的錯誤處理** - 正確區分取消和錯誤  
✅ **資源清理** - 無記憶體洩漏  

現在用戶可以完全掌控 Agent 的執行！🎉
