"""
Chunk Classifier Module

Classifies text chunks into semantic categories based on their content.
Categories: summary, method, experiment, results, other

Uses heuristic rules first (fast), with optional LLM fallback for uncertain cases.
"""

import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ChunkClassifier:
    """
    分類文本塊的語義類別，對應 GraphRAG 實體屬性
    
    Categories (aligned with GraphRAG entities):
    - background: 背景、摘要、導論 (Paper context)
    - method: 方法、演算法、架構 (Method entity)
    - dataset: 數據集、實驗數據 (Dataset entity)
    - metric: 評估指標、性能指標 (Metric entity)
    - domain: 應用領域、使用場景 (Domain entity)
    - results: 結果、實驗結果、性能表現 (Performance outcomes)
    - other: 其他內容
    """
    
    # 關鍵詞模式（中英文）- 對應 GraphRAG 實體類型
    PATTERNS = {
        'background': {
            'keywords': [
                # English - higher priority for section markers
                r'\babstract\b', r'\bintroduction\b', r'\boverview\b', r'\bsummary\b',
                r'\bbackground\b', r'\brelated\s+work\b', r'\bprior\s+work\b',
                r'\bmotivation\b', r'\bcontext\b', r'\bthis\s+paper\b', r'\bwe\s+introduce\b',
                # Chinese
                r'摘要', r'概述', r'導論', r'簡介', r'前言', r'背景', r'相關工作', 
                r'先前研究', r'動機', r'脈絡', r'本文', r'本研究'
            ],
            'weight': 1.2  # Higher weight for background sections
        },
        'method': {
            'keywords': [
                # English
                r'\bmethod\b', r'\bmethodology\b', r'\bapproach\b', r'\balgorithm\b',
                r'\barchitecture\b', r'\bimplementation\b', r'\bmodel\b', r'\bframework\b',
                r'\btechnique\b', r'\bprocedure\b', r'\bdesign\b', r'\bproposed\b',
                # Chinese
                r'方法', r'實作', r'實現', r'演算法', r'架構', r'模型', r'框架', 
                r'設計', r'技術', r'程序', r'提出', r'提議'
            ],
            'weight': 1.0
        },
        'dataset': {
            'keywords': [
                # English
                r'\bdataset\b', r'\bdata\s+set\b', r'\bcorpus\b', r'\bbenchmark\b',
                r'\btraining\s+data\b', r'\btest\s+data\b', r'\bvalidation\s+set\b',
                r'\bexample\b', r'\bsample\b', r'\bcollection\b',
                # Chinese
                r'數據集', r'資料集', r'語料', r'基準', r'測試集', r'訓練集',
                r'驗證集', r'樣本', r'範例', r'數據', r'資料'
            ],
            'weight': 1.0
        },
        'metric': {
            'keywords': [
                # English
                r'\bmetric\b', r'\baccuracy\b', r'\bprecision\b', r'\brecall\b',
                r'\bf1[-\s]score\b', r'\bbleu\b', r'\brouge\b', r'\bperplexity\b',
                r'\bmeasure\b', r'\bscore\b', r'\bevaluation\s+metric\b',
                # Chinese
                r'指標', r'準確率', r'精確率', r'召回率', r'分數', r'評估指標',
                r'性能指標', r'測量', r'度量'
            ],
            'weight': 1.0
        },
        'domain': {
            'keywords': [
                # English
                r'\bdomain\b', r'\bapplication\b', r'\buse\s+case\b', r'\bscenario\b',
                r'\bfield\b', r'\barea\b', r'\btask\b', r'\bproblem\b',
                r'\bNLP\b', r'\bcomputer\s+vision\b', r'\bspeech\b', r'\bmedical\b',
                # Chinese
                r'領域', r'應用', r'場景', r'任務', r'問題', r'範疇',
                r'自然語言', r'電腦視覺', r'語音', r'醫療', r'金融', r'生物'
            ],
            'weight': 1.0
        },
        'results': {
            'keywords': [
                # English
                r'\bresult\b', r'\bperformance\b', r'\bevaluation\b', r'\boutcome\b',
                r'\bfinding\b', r'\bconclusion\b', r'\bachievement\b', r'\bimprovement\b',
                r'\bcomparison\b', r'\boutperform\b',
                # Chinese
                r'結果', r'性能', r'效能', r'評估', r'表現', r'發現', r'結論',
                r'改善', r'提升', r'比較', r'優於', r'勝過'
            ],
            'weight': 1.0
        }
    }
    
    def __init__(self, llm=None, use_llm_fallback: bool = False):
        """
        初始化分類器
        
        Args:
            llm: Optional LLM for uncertain cases
            use_llm_fallback: Whether to use LLM when heuristics are uncertain
        """
        self.llm = llm
        self.use_llm_fallback = use_llm_fallback
        
        # Compile regex patterns
        self.compiled_patterns = {}
        for category, config in self.PATTERNS.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in config['keywords']
            ]
        
        logger.info(f"✓ ChunkClassifier initialized (LLM fallback: {use_llm_fallback})")
    
    def classify_chunk(self, text: str, chunk_id: str = None) -> Dict[str, Any]:
        """
        分類單個 chunk
        
        Args:
            text: Chunk text content
            chunk_id: Optional chunk identifier for logging
            
        Returns:
            {
                'chunk_type': str,  # Category name
                'confidence': float,  # Classification confidence (0-1)
                'method': str,  # 'heuristic' or 'llm'
                'scores': Dict[str, float]  # Score for each category
            }
        """
        # Step 1: Heuristic classification
        scores = self._compute_heuristic_scores(text)
        
        # Find best category
        best_category = max(scores.items(), key=lambda x: x[1])
        category_name, category_score = best_category
        
        # Compute confidence based on score distribution
        total_score = sum(scores.values())
        if total_score > 0:
            confidence = category_score / total_score
        else:
            confidence = 0.0
        
        # Use 'other' if no strong match
        if category_score == 0:
            category_name = 'other'
            confidence = 1.0  # We're confident it doesn't match known patterns
        
        result = {
            'chunk_type': category_name,
            'confidence': confidence,
            'method': 'heuristic',
            'scores': scores
        }
        
        # Step 2: LLM fallback if confidence is low and LLM is available
        if (self.use_llm_fallback and 
            self.llm and 
            confidence < 0.6 and 
            category_score > 0):
            
            logger.debug(f"Low confidence ({confidence:.2f}), using LLM fallback for {chunk_id}")
            llm_result = self._classify_with_llm(text, chunk_id)
            if llm_result:
                result = llm_result
        
        logger.debug(f"Classified {chunk_id or 'chunk'}: {result['chunk_type']} "
                    f"(confidence: {result['confidence']:.2f}, method: {result['method']})")
        
        return result
    
    def classify_chunks_batch(self, chunks: list) -> list:
        """
        批次分類多個 chunks
        
        Args:
            chunks: List of Document objects with .page_content and .metadata
            
        Returns:
            List of classification results (same order as input)
        """
        results = []
        
        for chunk in chunks:
            chunk_id = chunk.metadata.get('chunk_id', 'unknown')
            text = chunk.page_content
            
            classification = self.classify_chunk(text, chunk_id)
            results.append(classification)
        
        # Log summary
        category_counts = {}
        for result in results:
            cat = result['chunk_type']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        logger.info(f"✓ Classified {len(chunks)} chunks: {category_counts}")
        
        return results
    
    def _compute_heuristic_scores(self, text: str) -> Dict[str, float]:
        """
        使用啟發式規則計算各類別分數
        
        Returns:
            Dictionary mapping category name to score
        """
        scores = {category: 0.0 for category in self.PATTERNS.keys()}
        
        # Normalize text for better matching
        text_lower = text.lower()
        
        # Count pattern matches
        for category, patterns in self.compiled_patterns.items():
            match_count = 0
            for pattern in patterns:
                matches = pattern.findall(text_lower)
                match_count += len(matches)
            
            # Score is match count weighted by pattern weight
            weight = self.PATTERNS[category]['weight']
            scores[category] = match_count * weight
        
        return scores
    
    def _classify_with_llm(self, text: str, chunk_id: str = None) -> Optional[Dict[str, Any]]:
        """
        使用 LLM 分類（當啟發式不確定時）
        
        Returns:
            Classification result or None if LLM fails
        """
        try:
            # Truncate text if too long (avoid excessive tokens)
            max_chars = 800
            text_truncated = text[:max_chars] + "..." if len(text) > max_chars else text
            
            prompt = f"""Classify the following research paper text chunk into ONE of these categories:
- summary: Abstract, introduction, overview
- method: Methodology, algorithm, implementation details
- experiment: Experimental setup, datasets, configuration
- results: Results, performance metrics, evaluation
- other: None of the above

Text chunk:
{text_truncated}

Respond with ONLY the category name (one word).
Category:"""
            
            response = self.llm.invoke(prompt).strip().lower()
            
            # Parse response
            valid_categories = list(self.PATTERNS.keys()) + ['other']
            category = None
            
            for valid_cat in valid_categories:
                if valid_cat in response:
                    category = valid_cat
                    break
            
            if not category:
                logger.warning(f"LLM returned invalid category: {response}")
                return None
            
            return {
                'chunk_type': category,
                'confidence': 0.8,  # LLM classifications get medium-high confidence
                'method': 'llm',
                'scores': {}
            }
            
        except Exception as e:
            logger.warning(f"LLM classification failed for {chunk_id}: {e}")
            return None
    
    def get_category_keywords(self, category: str) -> list:
        """獲取某個類別的關鍵詞列表（用於調試）"""
        if category in self.PATTERNS:
            return self.PATTERNS[category]['keywords']
        return []


# Convenience function for quick classification
def classify_chunk(text: str, llm=None) -> str:
    """
    快速分類單個 chunk（僅返回類別名稱）
    
    Returns:
        Category name as string
    """
    classifier = ChunkClassifier(llm=llm, use_llm_fallback=False)
    result = classifier.classify_chunk(text)
    return result['chunk_type']
