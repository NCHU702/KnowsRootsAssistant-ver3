# 組合A解決方案: 查詢擴展 + 自適應權重調整

## 問題描述

短查詢（如「人流預測」、「CNN」）在檢索時產生的相似度分數過於相近（都在 0.64-0.65 範圍），導致難以區分論文的相關性排序。

### 根本原因

1. **查詢資訊不足**: 短查詢包含的語義資訊有限，向量表示不夠豐富
2. **語義模糊**: 短詞彙可能有多重含義，embedding 難以捕捉具體意圖
3. **檢索策略固定**: 不論查詢長短，都使用相同的語義/關鍵詞權重比例

## 解決方案

**組合A** = **查詢擴展 (Query Expansion)** + **自適應權重調整 (Adaptive Weight Adjustment)**

### 1. 查詢擴展 (Query Expansion)

**目標**: 將簡短查詢擴展為更詳細、資訊豐富的查詢

#### 實現模組: `system_api/query_expander.py`

```python
from system_api.query_expander import QueryExpander

expander = QueryExpander(llm=your_llm, min_word_count=5)

# 短查詢會被擴展
query = "人流預測"
expanded = expander.expand(query)
# 結果: "使用深度學習方法進行城市公共空間的人流量預測與分析，
#       包括時空特徵建模和預測模型研究"

# 長查詢保持不變
query = "如何使用LSTM模型進行城市公共空間的人流預測"
expanded = expander.expand(query)
# 結果: 原查詢（未擴展）
```

#### 擴展策略

1. **詞數檢測**: 使用 jieba 分詞，計算有意義的詞數
2. **閾值判斷**: 少於 N 詞（預設 5）才擴展
3. **LLM擴展**: 使用 LLM 生成學術性擴展查詢
4. **Fallback機制**: LLM失敗時使用規則擴展

#### 擴展提示工程

```python
prompt = f"""
你是一個學術研究助手，請幫助擴展使用者的簡短查詢。

使用者查詢: {query}

請將此查詢擴展為完整的學術檢索查詢，包含:
1. 核心概念的完整表述
2. 相關的研究方法或技術
3. 應用領域或場景
4. 相關的研究方向

擴展查詢應該:
- 保留原始查詢的核心意圖
- 使用學術性語言
- 長度在 20-40 字之間
- 不要使用疑問句

範例:
輸入: "人流預測"
輸出: "使用深度學習方法進行城市公共空間的人流量預測與分析，包括時空特徵建模和預測模型研究"

現在請擴展: {query}
擴展查詢:
"""
```

#### 擴展範例

| 原始查詢 | 擴展查詢 | 效果 |
|---------|---------|------|
| 人流預測 | 使用深度學習方法進行城市公共空間的人流量預測與分析，包括時空特徵建模和預測模型研究 | ✓ 語義豐富，向量表示更精準 |
| CNN | 卷積神經網絡（CNN）在計算機視覺和圖像處理中的應用與架構設計研究 | ✓ 明確CNN的含義和應用領域 |
| 深度學習 | 深度學習模型在機器學習和人工智能領域的理論研究、模型設計和實際應用 | ✓ 增加上下文資訊 |

### 2. 自適應權重調整 (Adaptive Weight Adjustment)

**目標**: 根據查詢特性動態調整語義搜尋和關鍵詞搜尋的權重

#### 實現模組: `system_api/adaptive_weights.py`

```python
from system_api.adaptive_weights import AdaptiveWeightAdjuster

adjuster = AdaptiveWeightAdjuster()

semantic_w, keyword_w, reason = adjuster.adjust_weights("人流預測")
# 結果: semantic_w=0.3, keyword_w=0.7, reason="短查詢(2詞)"

semantic_w, keyword_w, reason = adjuster.adjust_weights("如何使用LSTM進行時序預測")
# 結果: semantic_w=0.6, keyword_w=0.4, reason="問題形式; 包含專有名詞(LSTM)"
```

#### 調整規則

| 查詢特徵 | 權重調整 | 原因 |
|---------|---------|------|
| **詞數 ≤ 3** | keyword_weight +0.2 | 短查詢需要精準匹配 |
| **詞數 4-6** | keyword_weight +0.1 | 中短查詢稍微提高精準度 |
| **詞數 7-10** | 保持平衡 | 中等查詢 |
| **詞數 ≥ 11** | semantic_weight +0.2 | 長查詢重視語義理解 |
| **包含專有名詞** | keyword_weight +0.15 | CNN、LSTM等需精準匹配 |
| **問題形式** | semantic_weight +0.1 | "什麼"、"如何"需理解意圖 |
| **包含否定詞** | semantic_weight +0.1 | "不"、"沒"需理解語義 |

#### 權重範圍限制

- 每個權重範圍: [0.1, 0.9]
- 權重總和: 1.0
- 預設權重: semantic=0.5, keyword=0.5

#### 調整範例

| 查詢 | 語義權重 | 關鍵詞權重 | 調整原因 |
|-----|---------|-----------|---------|
| 人流預測 | 0.30 | 0.70 | 短查詢(2詞) |
| CNN | 0.23 | 0.77 | 短查詢(1詞); 包含專有名詞(CNN) |
| 深度學習在人流的應用 | 0.50 | 0.50 | 中等查詢(5詞) |
| 什麼是LSTM模型 | 0.48 | 0.52 | 中短查詢(3詞); 問題形式; 包含專有名詞(LSTM) |
| 如何使用ResNet進行圖像分類研究 | 0.58 | 0.42 | 長查詢(8詞); 問題形式; 包含專有名詞(ResNet) |

## 完整檢索流程

```
使用者查詢: "人流預測"
    ↓
[Step 1: 查詢擴展]
    should_expand("人流預測")
    → 詞數: 2 < 5
    → 需要擴展: True
    → expand_with_llm("人流預測")
    ↓
擴展查詢: "使用深度學習方法進行城市公共空間的人流量預測與分析，
         包括時空特徵建模和預測模型研究"
    ↓
[Step 2: 自適應權重調整]
    adjust_weights(擴展查詢)
    → 詞數: 26 > 10
    → 調整: semantic_weight +0.2
    → 結果: semantic=0.7, keyword=0.3
    ↓
[Step 3: 混合檢索]
    hybrid_search(
        query=擴展查詢,
        semantic_weight=0.7,
        keyword_weight=0.3
    )
    ↓
檢索結果: 相似度分數分佈更廣 (0.45 - 0.75)
```

## 配置參數

### agent2.py 配置

```python
config = {
    'hybrid_search': {
        'enabled': True,
        'semantic_weight': 0.5,    # 會被自適應調整覆蓋
        'keyword_weight': 0.5,     # 會被自適應調整覆蓋
        'use_jieba': True,
    },
    'query_expansion': {
        'enabled': True,           # 啟用查詢擴展
        'min_word_count': 5,       # 少於5詞才擴展
    },
    'adaptive_weights': {
        'enabled': True,           # 啟用自適應權重調整
        'default_semantic_weight': 0.5,
        'default_keyword_weight': 0.5,
    }
}
```

### 環境變數（可選）

可以通過環境變數調整參數:

```bash
# 查詢擴展閾值
export QUERY_EXPANSION_MIN_WORDS=5

# 預設權重
export DEFAULT_SEMANTIC_WEIGHT=0.5
export DEFAULT_KEYWORD_WEIGHT=0.5
```

## 使用方式

### 1. 自動模式（推薦）

系統會自動判斷並應用擴展和權重調整:

```python
from system_api.hierarchical_rag_system import HierarchicalRAGSystem

rag_system = HierarchicalRAGSystem(
    config={
        'query_expansion': {'enabled': True},
        'adaptive_weights': {'enabled': True},
        'hybrid_search': {'enabled': True}
    }
)

# 短查詢會自動擴展 + 調整權重
result = rag_system.query("人流預測")
```

### 2. 手動控制

分別使用各個模組:

```python
from system_api.query_expander import QueryExpander
from system_api.adaptive_weights import AdaptiveWeightAdjuster

# 查詢擴展
expander = QueryExpander(llm=your_llm)
expanded_query = expander.expand("人流預測")

# 權重調整
adjuster = AdaptiveWeightAdjuster()
semantic_w, keyword_w, reason = adjuster.adjust_weights(expanded_query)

# 混合檢索
results = hybrid_retriever.hybrid_search(
    query=expanded_query,
    semantic_weight=semantic_w,
    keyword_weight=keyword_w
)
```

### 3. 選擇性啟用

可以只啟用其中一個功能:

```python
# 只啟用查詢擴展
config = {
    'query_expansion': {'enabled': True},
    'adaptive_weights': {'enabled': False}
}

# 只啟用自適應權重
config = {
    'query_expansion': {'enabled': False},
    'adaptive_weights': {'enabled': True}
}
```

## 測試驗證

### 執行測試腳本

```bash
python test_combination_a.py
```

測試腳本包含三個部分:

1. **測試查詢擴展器**: 驗證不同長度查詢的擴展行為
2. **測試自適應權重**: 驗證不同類型查詢的權重調整
3. **測試完整系統**: 端到端測試擴展+權重+檢索

### 測試案例

| 測試查詢 | 預期行為 |
|---------|---------|
| 人流預測 | ✓ 擴展 + 高關鍵詞權重 |
| CNN | ✓ 擴展 + 超高關鍵詞權重 |
| 深度學習在人流的應用 | ✗ 不擴展 + 平衡權重 |
| 什麼是LSTM | ✓ 擴展 + 平衡權重偏向語義 |
| 如何使用ResNet進行圖像分類 | ✗ 不擴展 + 高語義權重 |

### 評估指標

1. **擴展成功率**: 短查詢是否成功擴展
2. **權重合理性**: 權重調整是否符合查詢特性
3. **分數分佈**: 相似度分數是否有更好的區分度
4. **檢索精度**: Top-K結果是否更相關

## 預期效果

### Before (原始系統)

```
查詢: "人流預測"
語義權重: 0.5, 關鍵詞權重: 0.5

檢索結果:
1. 論文A: 0.648
2. 論文B: 0.647
3. 論文C: 0.645
4. 論文D: 0.644
5. 論文E: 0.643

問題: 分數過於接近，難以排序
```

### After (組合A)

```
查詢: "人流預測"
  ↓ 擴展
查詢: "使用深度學習方法進行城市公共空間的人流量預測與分析，
     包括時空特徵建模和預測模型研究"
  ↓ 調整權重
語義權重: 0.7, 關鍵詞權重: 0.3

檢索結果:
1. 論文A: 0.752  ← 真正相關
2. 論文B: 0.683
3. 論文C: 0.591
4. 論文D: 0.512
5. 論文E: 0.468

✓ 分數區分度提高 (0.752 vs 0.468)
✓ 更相關的論文排名更高
```

## 優勢

1. **提高區分度**: 擴展查詢使向量表示更豐富，分數分佈更廣
2. **精準匹配**: 短查詢提高關鍵詞權重，確保精準度
3. **語義理解**: 長查詢提高語義權重，理解複雜意圖
4. **自動化**: 無需手動調整，系統自動判斷最佳策略
5. **向後兼容**: 長查詢保持不變，不影響原有檢索效果

## 注意事項

### 1. LLM依賴

查詢擴展依賴LLM生成，可能產生:
- 延遲增加 (通常 0.5-2 秒)
- 擴展品質依賴模型能力
- **解決方案**: 實現了 Fallback 機制，LLM失敗時使用規則擴展

### 2. 過度擴展風險

擴展可能引入不相關概念:
- **解決方案**: 
  - 提示工程強調「保留核心意圖」
  - 限制擴展長度 (20-40字)
  - 可通過 `min_word_count` 調整閾值

### 3. 中文分詞依賴

使用 jieba 進行詞數統計:
- **依賴**: 確保已安裝 `jieba`
- **準確性**: 分詞可能不完美，但對詞數統計影響不大

### 4. 權重調整衝突

多個規則可能同時觸發:
- **解決方案**: 
  - 累加調整值
  - 限制最終權重範圍 [0.1, 0.9]
  - 標準化確保總和為 1.0

## 調優建議

### 1. 調整擴展閾值

根據領域調整 `min_word_count`:
```python
# 專業領域（術語較長）
'query_expansion': {'min_word_count': 3}

# 通用領域
'query_expansion': {'min_word_count': 5}

# 學術文獻（查詢通常較長）
'query_expansion': {'min_word_count': 7}
```

### 2. 調整權重範圍

修改 `adaptive_weights.py` 中的調整幅度:
```python
# 保守策略（小幅調整）
if word_count <= 3:
    adjustment = 0.1  # 原本 0.2

# 激進策略（大幅調整）
if word_count <= 3:
    adjustment = 0.3  # 原本 0.2
```

### 3. 添加自定義規則

在 `AdaptiveWeightAdjuster` 中添加領域特定規則:
```python
def _has_domain_terms(self, query: str) -> bool:
    """檢查是否包含領域術語"""
    domain_terms = ['人流', '預測', '時序', ...]
    return any(term in query for term in domain_terms)
```

### 4. 優化擴展提示

根據使用場景修改擴展提示工程，調整:
- 擴展風格（學術性、技術性、通俗化）
- 擴展長度
- 包含的資訊類型

## 監控和日誌

系統會記錄詳細日誌:

```
[INFO] 查詢擴展 (Query Expansion)
[INFO] ✓ 查詢已擴展:
[INFO]   原始: 人流預測
[INFO]   擴展: 使用深度學習方法進行城市公共空間的人流量預測與分析

[INFO] 自適應權重調整 (Adaptive Weights)
[INFO] ✓ 權重已調整:
[INFO]   語義權重: 0.70
[INFO]   關鍵詞權重: 0.30
[INFO]   調整原因: 長查詢(26詞)

[INFO] 使用混合檢索 (Hybrid Search: 語義 + 關鍵詞)
[INFO]   當前權重: 語義=0.70, 關鍵詞=0.30
```

可通過調整日誌級別控制輸出:
```python
logging.basicConfig(level=logging.INFO)  # 詳細資訊
logging.basicConfig(level=logging.WARNING)  # 只顯示警告
```

## 總結

組合A方案通過**查詢擴展**和**自適應權重調整**雙管齊下，有效解決了短查詢導致的相似度分數聚集問題:

✅ **查詢擴展**: 豐富查詢資訊，提升向量表示品質  
✅ **自適應權重**: 根據查詢特性選擇最佳檢索策略  
✅ **完整整合**: 無縫整合到現有 Hierarchical RAG 系統  
✅ **自動化**: 無需手動配置，智能判斷  
✅ **可調優**: 提供豐富的配置選項和擴展機制

通過這兩項功能的結合，系統能夠針對不同類型的查詢採用最合適的檢索策略，顯著提高檢索結果的相關性和區分度。
