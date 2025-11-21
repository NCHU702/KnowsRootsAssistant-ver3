"""
GraphRAG Element Extraction Module (Chunk-Level)

Extracts Entities and Relationships from individual text chunks using LLM.
This is the foundation for building a knowledge graph from documents.
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import json
import re

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Pydantic Data Models (Strict Schema)
# ============================================================================

class Entity(BaseModel):
    """
    Represents a single entity (node) extracted from text.
    
    Attributes:
        name: Unique identifier/name of the entity (e.g., "LSTM", "PeMSD7")
        type: High-level category (METHOD, DATASET, METRIC, PERSON, etc.)
        description: Contextual summary of the entity's role in THIS chunk
    """
    name: str = Field(description="Entity name (capitalized, cleaned)")
    type: str = Field(description="Entity type (e.g., METHOD, DATASET, DOMAIN, METRIC)")
    description: str = Field(description="Summary of the entity within this text context")
    
    @validator('name')
    def clean_name(cls, v):
        """Remove extra whitespace and standardize format"""
        return ' '.join(v.strip().split())
    
    @validator('type')
    def uppercase_type(cls, v):
        """Ensure type is uppercase for consistency"""
        return v.upper()


class Relationship(BaseModel):
    """
    Represents a relationship (edge) between two entities.
    
    Attributes:
        source: Name of the source entity (must match an Entity.name)
        target: Name of the target entity (must match an Entity.name)
        relation_type: Verb phrase describing the relationship (e.g., USES, EVALUATED_ON)
        description: Contextual explanation of WHY this relationship exists
        strength: Importance score (1-10), where 10 = critical/primary
    """
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation_type: str = Field(description="Relationship type (uppercase verb phrase)")
    description: str = Field(description="Contextual explanation of the relationship")
    strength: int = Field(default=5, ge=1, le=10, description="Relationship strength (1-10)")
    
    @validator('relation_type')
    def uppercase_relation(cls, v):
        """Ensure relation type is uppercase"""
        return v.upper().replace(' ', '_')


class GraphElements(BaseModel):
    """
    Complete extraction result containing entities and relationships.
    """
    entities: List[Entity] = Field(default_factory=list, description="List of extracted entities")
    relationships: List[Relationship] = Field(default_factory=list, description="List of extracted relationships")


# ============================================================================
# 2. LLM Prompt Template (Few-Shot with Strict Instructions)
# ============================================================================

EXTRACTION_PROMPT_TEMPLATE = """你是一個專業的知識圖譜分析專家。你的任務是從學術論文的文本片段中提取**實體（Entities）**和**關係（Relationships）**。

## 實體類型（ENTITY TYPES）：
- **METHOD**: 演算法、模型、技術（如：LSTM, CNN, Random Forest）
- **DATASET**: 實驗用資料集（如：PeMSD7, COCO, ImageNet）
- **METRIC**: 評估指標（如：RMSE, Accuracy, F1-Score）
- **DOMAIN**: 應用領域（如：Smart Transportation, Healthcare）
- **CONCEPT**: 抽象概念或研究目標（如：Traffic Prediction, Object Detection）

## 提取規則：
1. **實體描述（description）**：總結實體在**這段文字中**的角色（不是通用定義）
2. **關係必填欄位**：每個關係必須包含以下5個欄位，缺一不可：
   - **source**: 源實體名稱
   - **target**: 目標實體名稱
   - **relation_type**: 關係類型（必填！使用清晰動詞，如：USES, EVALUATED_ON, COMPARED_WITH, AIMS_TO）
   - **description**: 關係描述
   - **strength**: 強度（1-10）
3. **關係強度（strength）**：
   - 10: 核心關係（如：「論文使用 LSTM 作為主要模型」）
   - 5-7: 輔助關係（如：「與基線方法比較」）
   - 1-3: 僅提及（如：「相關研究引用 X」）
4. **實體名稱**：清理並標準化（如：「lstm network」→「LSTM」）

## 範例 1：
**輸入文本**：
"本研究使用 LSTM 模型預測交通流量，並在 PeMSD7 資料集上進行評估。實驗結果顯示，RMSE 為 3.45，優於傳統 ARIMA 模型。"

**輸出**：
```json
{{
  "entities": [
    {{
      "name": "LSTM",
      "type": "METHOD",
      "description": "本段落的主要預測模型，用於交通流量預測"
    }},
    {{
      "name": "PeMSD7",
      "type": "DATASET",
      "description": "用於評估 LSTM 模型的交通流量資料集"
    }},
    {{
      "name": "RMSE",
      "type": "METRIC",
      "description": "評估指標，本段落中 LSTM 達到 3.45"
    }},
    {{
      "name": "ARIMA",
      "type": "METHOD",
      "description": "作為基準比較的傳統模型"
    }}
  ],
  "relationships": [
    {{
      "source": "LSTM",
      "target": "PeMSD7",
      "relation_type": "EVALUATED_ON",
      "description": "LSTM 模型在 PeMSD7 資料集上進行性能評估",
      "strength": 9
    }},
    {{
      "source": "LSTM",
      "target": "RMSE",
      "relation_type": "MEASURED_BY",
      "description": "使用 RMSE 指標評估 LSTM 的預測準確度",
      "strength": 8
    }},
    {{
      "source": "LSTM",
      "target": "ARIMA",
      "relation_type": "OUTPERFORMS",
      "description": "LSTM 的性能優於傳統 ARIMA 模型",
      "strength": 7
    }}
  ]
}}
```

## 範例 2：
**輸入文本**：
"We propose a hybrid CNN-LSTM architecture. The CNN extracts spatial features from images, while LSTM captures temporal dependencies."

**輸出**：
```json
{{
  "entities": [
    {{
      "name": "CNN-LSTM",
      "type": "METHOD",
      "description": "Proposed hybrid architecture in this text"
    }},
    {{
      "name": "CNN",
      "type": "METHOD",
      "description": "Component for spatial feature extraction"
    }},
    {{
      "name": "LSTM",
      "type": "METHOD",
      "description": "Component for temporal modeling"
    }}
  ],
  "relationships": [
    {{
      "source": "CNN-LSTM",
      "target": "CNN",
      "relation_type": "CONTAINS",
      "description": "CNN is a component of the hybrid architecture",
      "strength": 9
    }},
    {{
      "source": "CNN-LSTM",
      "target": "LSTM",
      "relation_type": "CONTAINS",
      "description": "LSTM is a component of the hybrid architecture",
      "strength": 9
    }}
  ]
}}
```

---

## 你的任務：
從以下文本片段中提取實體和關係。

**重要提醒**：
- 每個關係必須包含：source, target, **relation_type** (必填！), description, strength
- **只返回有效的 JSON**（不要有任何解釋或 markdown 格式）

**文本片段**：
{text_chunk}

**JSON 輸出**（確保每個關係都有 relation_type）：
"""


# ============================================================================
# 3. Extraction Function with Retry Logic
# ============================================================================

class GraphElementExtractor:
    """
    Extracts entities and relationships from text chunks using LLM.
    
    Features:
    - Pydantic-based structured output validation
    - Retry mechanism for failed parsing
    - Fallback JSON parsing from markdown-wrapped responses
    """
    
    def __init__(self, llm: OllamaLLM, max_retries: int = 2):
        """
        Args:
            llm: LangChain LLM instance (Ollama)
            max_retries: Number of retry attempts for failed extractions
        """
        self.llm = llm
        self.max_retries = max_retries
        self.prompt = ChatPromptTemplate.from_template(EXTRACTION_PROMPT_TEMPLATE)
    
    def extract_graph_elements(self, text_chunk: str, chunk_id: str = "unknown") -> Dict[str, Any]:
        """
        Extract entities and relationships from a single text chunk.
        
        Args:
            text_chunk: Text content to analyze (600-1000 tokens recommended)
            chunk_id: Identifier for tracking (e.g., "paper_123_chunk_001")
        
        Returns:
            Dictionary with:
            {
                "chunk_id": str,
                "entities": List[Dict],
                "relationships": List[Dict],
                "status": "success" | "partial" | "failed",
                "error": str (if failed)
            }
        """
        if not text_chunk or len(text_chunk.strip()) < 50:
            logger.warning(f"Chunk {chunk_id} too short, skipping extraction")
            return self._empty_result(chunk_id, "Text too short")
        
        for attempt in range(self.max_retries + 1):
            try:
                # Generate prompt
                formatted_prompt = self.prompt.format(text_chunk=text_chunk)
                
                # Call LLM
                logger.debug(f"Extracting from chunk {chunk_id} (attempt {attempt + 1})")
                response = self.llm.invoke(formatted_prompt)
                
                # Parse response
                graph_data = self._parse_response(response)
                
                # Validate with Pydantic
                validated = GraphElements(**graph_data)
                
                logger.info(f"✓ Extracted {len(validated.entities)} entities, {len(validated.relationships)} relationships from {chunk_id}")
                
                return {
                    "chunk_id": chunk_id,
                    "entities": [e.dict() for e in validated.entities],
                    "relationships": [r.dict() for r in validated.relationships],
                    "status": "success"
                }
                
            except Exception as e:
                logger.warning(f"Extraction attempt {attempt + 1} failed for {chunk_id}: {e}")
                if attempt == self.max_retries:
                    logger.error(f"✗ All extraction attempts failed for {chunk_id}")
                    return self._empty_result(chunk_id, str(e))
                continue
        
        return self._empty_result(chunk_id, "Max retries exceeded")
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured format.
        Handles both raw JSON and markdown-wrapped JSON.
        """
        # Remove markdown code blocks if present
        cleaned = re.sub(r'```json\s*|\s*```', '', response).strip()
        
        # Try direct JSON parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass
            
            raise ValueError(f"Could not parse JSON from response: {response[:200]}...")
    
    def _empty_result(self, chunk_id: str, error: str) -> Dict[str, Any]:
        """Return empty result with error info"""
        return {
            "chunk_id": chunk_id,
            "entities": [],
            "relationships": [],
            "status": "failed",
            "error": error
        }


# ============================================================================
# 4. Batch Extraction for Multiple Chunks
# ============================================================================

def extract_from_chunks(
    chunks: List[Dict[str, str]], 
    llm: OllamaLLM,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Extract graph elements from multiple text chunks.
    
    Args:
        chunks: List of {"id": str, "text": str} dictionaries
        llm: LangChain LLM instance
        progress_callback: Optional function(current, total) for progress tracking
    
    Returns:
        {
            "total_chunks": int,
            "successful": int,
            "failed": int,
            "entities": List[Dict],  # Deduplicated
            "relationships": List[Dict],
            "chunk_results": List[Dict]  # Per-chunk details
        }
    """
    extractor = GraphElementExtractor(llm)
    
    all_entities = []
    all_relationships = []
    chunk_results = []
    successful = 0
    failed = 0
    
    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get('id', f'chunk_{i}')
        text = chunk.get('text', '')
        
        result = extractor.extract_graph_elements(text, chunk_id)
        chunk_results.append(result)
        
        if result['status'] == 'success':
            all_entities.extend(result['entities'])
            all_relationships.extend(result['relationships'])
            successful += 1
        else:
            failed += 1
        
        if progress_callback:
            progress_callback(i + 1, len(chunks))
    
    # Deduplicate entities by name (case-insensitive)
    seen_entities = {}
    for entity in all_entities:
        name_lower = entity['name'].lower()
        if name_lower not in seen_entities:
            seen_entities[name_lower] = entity
        else:
            # Merge descriptions if duplicate found
            existing = seen_entities[name_lower]
            if entity['description'] not in existing['description']:
                existing['description'] += f" | {entity['description']}"
    
    deduplicated_entities = list(seen_entities.values())
    
    logger.info(f"Batch extraction complete: {successful}/{len(chunks)} successful")
    logger.info(f"Total entities (deduplicated): {len(deduplicated_entities)}")
    logger.info(f"Total relationships: {len(all_relationships)}")
    
    return {
        "total_chunks": len(chunks),
        "successful": successful,
        "failed": failed,
        "entities": deduplicated_entities,
        "relationships": all_relationships,
        "chunk_results": chunk_results
    }
