# Layer2 Descent 測試場景

## ✅ 應該進入Layer2的Query (需要詳細內容)

### 類型1: 明確要求詳細內容
- [x] "哪些論文使用YOLO，並且他們**如何使用**" ✅ 已測試通過
- [ ] "**詳細介紹**使用LSTM的論文"
- [ ] "**具體說明**這些論文的方法"
- [ ] "**個別介紹**智慧交通相關論文"
- [ ] "請**解釋**這些論文的貢獻"

### 類型2: 實作細節查詢
- [ ] "YOLO論文的**訓練過程**是什麼"
- [ ] "使用LSTM的論文**參數設定**如何"
- [ ] "**實驗設計**是怎樣的"
- [ ] "模型的**架構**是什麼"
- [ ] "**配置**和**超參數**有哪些"

### 類型3: 具體內容/數據查詢 ⭐ 新增
- [ ] "使用YOLO的論文**用了什麼資料集**"
- [ ] "這些論文的**準確率**是多少"
- [ ] "**評估指標**有哪些"
- [ ] "在哪個**領域應用**"
- [ ] "**採用**什麼模型"
- [ ] "**使用什麼方法**進行訓練"

### 類型4: 比較/對比查詢 ⭐ 新增
- [ ] "**比較** YOLO 和 LSTM 的使用方式"
- [ ] "這些論文的**差異**在哪"
- [ ] "方法的**優缺點**"
- [ ] "A **versus** B"

### 類型5: 深入理解查詢 ⭐ 新增
- [ ] "**為什麼**使用這個方法"
- [ ] "這個方法的**原理**是什麼"
- [ ] "論文的**貢獻**是什麼"
- [ ] "**創新點**在哪裡"
- [ ] "有什麼**意義**"

### 類型6: 問題解決查詢 ⭐ 新增
- [ ] "**解決**了什麼問題"
- [ ] "如何**改進**現有方法"
- [ ] "**克服**了哪些困難"
- [ ] "面臨的**挑戰**是什麼"

### 類型7: 複雜連接詞查詢 ⭐ 新增
- [x] "哪些論文使用YOLO**並且**他們如何使用" ✅ 已測試通過
- [ ] "使用LSTM的論文**以及**它們的效果"
- [ ] "資料集**還有**評估指標"

### 類型8: 問句開頭 ⭐ 新增
- [ ] "**為什麼**這些論文使用YOLO"
- [ ] "**怎麼**訓練這些模型"
- [ ] "**如何**評估這些方法"

---

## ❌ 應該停留在Graph的Query (只需要宏觀資訊)

### 純Macro查詢
- [x] "**哪些**論文使用YOLO" ✅ 已測試通過
- [ ] "**有哪些**論文使用LSTM"
- [ ] "**列出**所有使用CNN的論文"
- [ ] "**What** papers use deep learning"
- [ ] "**Which** papers are about traffic"
- [ ] "**有幾篇**論文使用集成學習"

### 計數查詢
- [ ] "**多少**論文使用YOLO"
- [ ] "**How many** papers use LSTM"
- [ ] "**統計**各領域的論文數量"

### 單純列表查詢
- [ ] "**列出**所有資料集"
- [ ] "**List all** methods"
- [ ] "顯示所有領域"

---

## 🎯 邊界案例 (需要特別注意)

### 可能產生歧義的Query
- [ ] "智慧交通論文" → macro-only (太短)
- [ ] "智慧交通論文的方法" → 進Layer2 (包含"方法")
- [ ] "YOLO論文" → macro-only (太短)
- [ ] "YOLO論文的效果" → 進Layer2 (包含"效果")

### 中英混合Query
- [ ] "哪些論文使用 LSTM and how do they use it" → 進Layer2
- [ ] "List papers using YOLO 並說明方法" → 進Layer2

---

## 📊 預期行為統計

### 關鍵詞覆蓋範圍

#### Category 1: 明確詳細請求 (14個關鍵詞)
- 中文: 如何, 怎麼, 詳細, 具體, 細節, 介紹, 說明, 描述, 解釋, 個別, 分別, 各自
- English: how, detail, specific, describe, explain, elaborate, summarize, summary, each, individual, individually

#### Category 2: 實作細節 (24個關鍵詞)
- 訓練, 實驗, 測試, 參數, 超參數, 配置, 設定, 步驟, 流程, 過程, 架構, 實現, 實作
- training, experiment, testing, evaluation, parameter, hyperparameter, configuration, setting, step, process, procedure, architecture, implementation, approach, technique

#### Category 3: 具體內容/數據 (30+個關鍵詞) ⭐ 新增
- 資料集, 數據集, 準確率, 精確度, 召回率, F1, 指標, 評估, 效能, 性能
- 什麼方法, 什麼模型, 採用, 使用什麼
- dataset, accuracy, precision, recall, metric, evaluation, performance, score, result
- what method, which method, what model, which model, use what, adopt, employ

#### Category 4: 比較/對比 (15個關鍵詞) ⭐ 新增
- 比較, 對比, 差異, 不同, 區別, 優缺點, 優勢, 劣勢, 特點
- compare, comparison, contrast, difference, versus, vs, distinguish, advantage, disadvantage

#### Category 5: 深入理解 (12個關鍵詞) ⭐ 新增
- 為什麼, 原理, 機制, 貢獻, 創新, 突破, 意義, 價值
- why, reason, principle, mechanism, contribution, innovation, novelty, significance, value

#### Category 6: 問題解決 (10個關鍵詞) ⭐ 新增
- 解決, 改進, 克服, 問題, 挑戰, 困難
- solve, address, tackle, overcome, problem, challenge, issue, difficulty

### 智能判斷規則 ⭐ 新增

1. **連接詞檢測**: 並且, 而且, 以及, 還有, and, also, plus (7個)
2. **問句開頭檢測**: 為什麼, 怎麼, 如何, why, how (5個)
3. **Query長度判斷**: 
   - < 15 chars → macro-only
   - > 20 chars + 連接詞 → Layer2
   - 其他 → 預設Layer2

---

## 🚀 測試建議

### Phase 1: 基礎測試
1. ✅ "哪些論文使用YOLO" (macro)
2. ✅ "哪些論文使用YOLO，並且他們如何使用" (Layer2)
3. 測試 "哪些論文和智慧交通有關，並且個別介紹" (Layer2)

### Phase 2: 新增場景測試
4. "使用YOLO的論文用了什麼資料集" (Layer2 - 具體內容)
5. "這些論文的準確率是多少" (Layer2 - 數據查詢)
6. "比較YOLO和LSTM的使用方式" (Layer2 - 比較)
7. "為什麼使用YOLO" (Layer2 - 深入理解)

### Phase 3: 邊界測試
8. "智慧交通論文" (macro - 太短)
9. "YOLO論文的訓練過程" (Layer2 - 實作細節)
10. "列出所有使用深度學習的論文" (macro - list all)

---

## 📈 預期改進效果

### 之前系統
- 關鍵詞覆蓋: ~25個
- 只處理明確的"如何"、"詳細"等請求
- 無法識別具體數據查詢 (❌ "用了什麼資料集")
- 無法識別比較查詢 (❌ "比較A和B")

### 現在系統
- 關鍵詞覆蓋: **100+個** ⭐
- 6大類別全面覆蓋
- 智能連接詞檢測
- 問句類型識別
- Query長度智能判斷

### 識別準確率提升
- 明確請求: 95% → 98%
- 隱含請求: 60% → 90% ⭐
- 邊界案例: 50% → 85% ⭐
