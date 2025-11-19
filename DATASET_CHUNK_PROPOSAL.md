# 数据集专用 Chunk 增强方案

## 📋 目标
在建立索引时，自动识别并创建"数据集"专用 chunk，提高数据集相关查询的准确性。

## 🔍 实现策略

### **方案 A：增强 Section Parser（推荐）**

#### 1. 新增数据集章节识别模式
在 `pdf_section_parser.py` 的 `_load_patterns()` 中添加：

```python
'dataset': [
    # English patterns
    r'^Dataset\s*$',
    r'^Data\s*$',
    r'^Datasets?\s*$',
    r'^[234]\.?\d*\.?\s*Dataset',     # 3.1 Dataset
    r'^[234]\.?\d*\.?\s*Data\s*$',    # 3.2 Data
    r'^\d+\.\d+\s+Dataset Description',
    r'^\d+\.\d+\s+Experimental Data',
    r'^\d+\.\d+\s+Data Collection',
    r'^Experimental Setup\s*$',
    r'^Data Collection\s*$',
    r'^Data Preparation\s*$',
    
    # Chinese patterns  
    r'^資料集\s*$',
    r'^数据集\s*$',
    r'^第[三四]章\s*資料集',
    r'^\d+\.\d+\s+資料集',
    r'^\d+\.\d+\s+数据集',
    r'^資料來源\s*$',
    r'^數據來源\s*$',
    r'^實驗資料\s*$',
    r'^实验数据\s*$',
    r'^資料蒐集\s*$',
    r'^数据收集\s*$',
],
```

#### 2. 添加数据集信息提取器
创建新文件 `system_api/dataset_extractor.py`：

```python
"""
Dataset Information Extractor

从论文文本中提取数据集相关信息
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class DatasetInfo:
    """数据集信息结构"""
    name: Optional[str] = None              # 数据集名称
    sample_count: Optional[int] = None      # 样本数量
    source: Optional[str] = None            # 数据来源
    description: Optional[str] = None       # 描述
    split_ratio: Optional[Dict] = None      # 训练/测试分割
    preprocessing: List[str] = None         # 预处理步骤

class DatasetExtractor:
    """提取数据集信息"""
    
    # 常见数据集名称模式
    DATASET_NAME_PATTERNS = [
        r'\b(MNIST|CIFAR-?10|CIFAR-?100|ImageNet|COCO|Pascal VOC)\b',
        r'\b(MS\s*COCO|VOC\s*2007|VOC\s*2012)\b',
        r'\b([A-Z][A-Za-z]+\s+Dataset)\b',
        # 中文
        r'(鼻咽癌|肺癌|乳癌).*?(數據集|资料集|数据集)',
    ]
    
    # 样本数量模式
    SAMPLE_COUNT_PATTERNS = [
        r'(\d{1,3}(?:,\d{3})*)\s*(samples?|images?|instances?|examples?)',
        r'(\d+)[kKmM]?\s*(樣本|样本|資料|数据)',
        r'(training|test|validation)\s+set.*?(\d+)',
    ]
    
    # 训练/测试分割模式
    SPLIT_PATTERNS = [
        r'(\d+)%.*?(training|train)',
        r'(\d+)%.*?(test|testing)',
        r'(\d+):(\d+)\s*(train|test)\s*split',
        r'(\d+)/(\d+)/(\d+)\s*(train|val|test)',
    ]
    
    def extract(self, text: str) -> DatasetInfo:
        """
        从文本中提取数据集信息
        
        Args:
            text: 论文章节文本
            
        Returns:
            DatasetInfo 对象
        """
        info = DatasetInfo(preprocessing=[])
        
        # 提取数据集名称
        info.name = self._extract_dataset_name(text)
        
        # 提取样本数量
        info.sample_count = self._extract_sample_count(text)
        
        # 提取来源
        info.source = self._extract_source(text)
        
        # 提取分割比例
        info.split_ratio = self._extract_split_ratio(text)
        
        # 提取预处理步骤
        info.preprocessing = self._extract_preprocessing(text)
        
        # 生成描述（前300字）
        info.description = text[:300].strip()
        
        return info
    
    def _extract_dataset_name(self, text: str) -> Optional[str]:
        """提取数据集名称"""
        for pattern in self.DATASET_NAME_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_sample_count(self, text: str) -> Optional[int]:
        """提取样本数量"""
        for pattern in self.SAMPLE_COUNT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                count_str = match.group(1).replace(',', '')
                # Handle K/M suffixes
                if 'k' in text[match.start():match.end()].lower():
                    return int(count_str) * 1000
                elif 'm' in text[match.start():match.end()].lower():
                    return int(count_str) * 1000000
                else:
                    return int(count_str)
        return None
    
    def _extract_source(self, text: str) -> Optional[str]:
        """提取数据来源"""
        source_patterns = [
            r'(collected from|obtained from|downloaded from|available at)\s+([^\.\n]+)',
            r'(來源|来源|來自|来自)[：:]\s*([^\.\n]+)',
        ]
        for pattern in source_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(2).strip()
        return None
    
    def _extract_split_ratio(self, text: str) -> Optional[Dict]:
        """提取训练/测试分割比例"""
        for pattern in self.SPLIT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # 简化处理：返回找到的比例
                groups = match.groups()
                if len(groups) >= 2:
                    return {'raw': match.group(0)}
        return None
    
    def _extract_preprocessing(self, text: str) -> List[str]:
        """提取预处理步骤"""
        preprocessing_keywords = [
            'normalization', 'resizing', 'augmentation', 
            'preprocessing', 'cleaning', 'filtering',
            '標準化', '正規化', '預處理', '前處理'
        ]
        
        found = []
        for keyword in preprocessing_keywords:
            if re.search(r'\b' + keyword + r'\b', text, re.IGNORECASE):
                found.append(keyword)
        
        return found
```

#### 3. 修改 Layer2 VectorStore 的分块逻辑
在 `layer2_vectorstore.py` 的 `_build_index_with_summarization()` 中：

```python
# 处理每个章节
for section_idx, section in enumerate(sections, 1):
    # ... 现有过滤逻辑 ...
    
    # ✨ 特殊处理：数据集章节
    if section.name.lower() in ['dataset', 'data', 'experimental setup']:
        logger.info(f"  🔍 检测到数据集章节: {section.name}")
        
        # 提取数据集信息
        from system_api.dataset_extractor import DatasetExtractor
        dataset_extractor = DatasetExtractor()
        dataset_info = dataset_extractor.extract(section.text)
        
        # 创建数据集专用摘要
        summary = self.summarizer.summarize_dataset_section(
            section.text,
            section.name,
            dataset_info=dataset_info,  # 传入提取的结构化信息
            paper_context=paper_title
        )
        
        # 创建 Document（标记为 dataset_chunk）
        chunk_doc = Document(
            page_content=summary,
            metadata={
                'paper_id': paper_id,
                'chunk_id': f"{paper_id}_dataset",
                'chunk_type': 'dataset_chunk',  # ✨ 关键标记
                'section_name': section.name,
                
                # ✨ 结构化数据集信息
                'dataset_name': dataset_info.name,
                'dataset_sample_count': dataset_info.sample_count,
                'dataset_source': dataset_info.source,
                'dataset_split_ratio': dataset_info.split_ratio,
                'dataset_preprocessing': dataset_info.preprocessing,
                
                'original_length': section.char_count,
                'summary_length': len(summary),
                'start_page': section.start_page,
                'end_page': section.end_page,
                **{k: v for k, v in paper_docs[0].metadata.items() 
                   if k not in ['chunk_id', 'chunk_type']}
            }
        )
        
        summarized_chunks.append(chunk_doc)
        logger.info(f"  ✓ 创建数据集 chunk (dataset_name={dataset_info.name})")
        continue
    
    # ... 其他章节的正常处理 ...
```

#### 4. 修改 LLM Summarizer 添加数据集专用摘要
在 `llm_summarizer.py` 中添加：

```python
def summarize_dataset_section(
    self,
    text: str,
    section_name: str,
    dataset_info: 'DatasetInfo',
    paper_context: str = ""
) -> str:
    """
    为数据集章节生成专用摘要
    
    Args:
        text: 章节文本
        section_name: 章节名称
        dataset_info: 提取的结构化数据集信息
        paper_context: 论文上下文（标题等）
        
    Returns:
        数据集摘要文本
    """
    # 构建 prompt（强调数据集关键信息）
    prompt = f"""請為以下學術論文的數據集章節生成一個結構化摘要（200-300字）。

論文: {paper_context}
章節: {section_name}

章節內容:
{text[:2000]}

已識別的結構化資訊:
- 數據集名稱: {dataset_info.name or '未識別'}
- 樣本數量: {dataset_info.sample_count or '未識別'}
- 數據來源: {dataset_info.source or '未識別'}

請在摘要中務必包含以下資訊（如果文中有提到）:
1. **數據集名稱**和類型
2. **樣本數量**（訓練集、測試集分別多少）
3. **數據來源**（公開數據集或自行收集）
4. **數據特徵**（圖像大小、類別數等）
5. **預處理步驟**（標準化、增強等）
6. **訓練/測試分割比例**

請只返回摘要文本，使用繁體中文，不要包含其他內容。"""
    
    try:
        summary = self.llm.invoke(prompt)
        return summary.strip()
    except Exception as e:
        logger.error(f"數據集摘要生成失敗: {e}")
        # Fallback: 使用通用摘要
        return self.summarize_section(text, section_name, paper_context)
```

#### 5. 修改检索逻辑（可选）
在 `hierarchical_rag_system.py` 中，当检测到数据集查询时：

```python
# 在 Layer2 检索时，优先返回 dataset_chunk
if 'dataset' in query.lower() or '資料集' in query or '数据集' in query:
    logger.info("检测到数据集查询，优先检索 dataset_chunk")
    
    # 过滤出 dataset_chunk
    dataset_chunks = [
        doc for doc in layer2_docs 
        if doc.metadata.get('chunk_type') == 'dataset_chunk'
    ]
    
    if dataset_chunks:
        logger.info(f"找到 {len(dataset_chunks)} 个数据集专用 chunk")
        # 优先使用 dataset_chunks，但也保留其他相关 chunks
        layer2_docs = dataset_chunks + [
            doc for doc in layer2_docs 
            if doc.metadata.get('chunk_type') != 'dataset_chunk'
        ][:2]  # 补充最多2个其他chunks
```

---

## 📊 预期效果

### **Before（现状）:**
```
用户问: "鼻咽癌的資料集介紹"
→ Layer1: 找到论文（摘要层）
→ 决策: overview 查询，停在 Layer1
→ LLM: 只看到摘要，没有数据集详细信息
→ 回答: 泛泛介绍论文内容 ❌
```

### **After（改进后）:**
```
用户问: "鼻咽癌的資料集介紹"
→ Layer1: 找到论文
→ 决策: 检测到 "資料集" 关键词 → detail_inquiry → 强制 Layer2
→ Layer2: 检索到 chunk_type='dataset_chunk' 的专用 chunk
→ LLM 看到:
   - 数据集名称: NPC-2023-Dataset
   - 样本数量: 训练集 800 张，测试集 200 张
   - 数据来源: 台大医院内视镜影像
   - 预处理: 去噪、色彩空间转换、增强
→ 回答: 精确回答数据集信息 ✅
```

---

## 🚀 实施步骤

### Phase 1: 基础实现（核心功能）
1. ✅ 在 `pdf_section_parser.py` 添加 dataset patterns
2. ✅ 创建 `dataset_extractor.py` 提取结构化信息
3. ✅ 修改 `layer2_vectorstore.py` 创建 dataset_chunk
4. ✅ 修改 `llm_summarizer.py` 添加数据集专用摘要

### Phase 2: 检索优化（提升准确性）
5. ✅ 修改 `layer2_trigger.py` 确保数据集查询进入 Layer2
6. ✅ 修改 `hierarchical_rag_system.py` 优先返回 dataset_chunk
7. ✅ 更新 prompt 指导 LLM 识别结构化数据集信息

### Phase 3: 测试验证
8. 重建索引测试现有论文
9. 验证数据集查询准确性
10. 调整 pattern 和 prompt

---

## 🔧 配置选项

在 `agent2.py` 的 config 中添加：

```python
'chunking': {
    'mode': 'summarization',
    'summarization': {
        'model': 'llama3.2:latest',
        'enable_dataset_extraction': True,  # ✨ 新增开关
        'dataset_extractor': {
            'enabled': True,
            'extract_structured_info': True,  # 提取结构化信息
            'min_confidence': 0.5,
        }
    }
}
```

---

## 💡 优势

1. **精准回答**: 数据集信息集中存储，不会被淹没在大段文字中
2. **结构化**: metadata 中保存关键字段，可直接查询
3. **向后兼容**: 不影响现有功能，只是增强
4. **可扩展**: 未来可以添加更多专用 chunk 类型（如 "evaluation_metrics_chunk"）
5. **LLM友好**: 专用摘要 prompt 确保关键信息不遗漏

---

## ⚠️ 注意事项

1. **Summarization 模式专属**: 此功能只在 chunking_mode='summarization' 时生效
2. **重建索引**: 需要重新运行 `build_indices()` 才能生效
3. **Pattern 覆盖**: 可能需要根据实际论文调整 dataset section patterns
4. **LLM 成本**: 每个数据集章节需要一次额外的 LLM 调用

---

这个方案如何？我可以立即开始实施吗？
