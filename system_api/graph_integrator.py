"""
Graph Integrator Module

Integrates graph retrieval results with LLM to produce preliminary answers.
Decides whether to descend to Layer2 for more detailed retrieval.
"""

import logging
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class GraphIntegrator:
    """
    整合 Graph 查詢結果並生成初步答案
    
    Strategy:
    - 使用 LLM 整合 Graph facts（論文、實體、關係）
    - 生成初步答案
    - 評估是否需要進入 Layer2 獲取更詳細資訊
    - 返回信心分數和建議
    """
    
    def __init__(self, llm):
        """
        初始化 Graph Integrator
        
        Args:
            llm: LLM instance for integration
        """
        self.llm = llm
        logger.info("✓ GraphIntegrator initialized")
    
    def integrate_graph_results(
        self,
        query: str,
        graph_results: List[Dict[str, Any]],
        max_papers: int = 10
    ) -> Dict[str, Any]:
        """
        整合 Graph 查詢結果
        
        Args:
            query: 用戶查詢
            graph_results: GraphRetriever 返回的論文列表
            max_papers: 最多使用幾篇論文的資訊
            
        Returns:
            {
                'text': str,  # 整合後的答案
                'confidence': float,  # 0-1
                'should_descend': bool,  # 是否需要進入 Layer2
                'notes': List[str],  # 補充說明
                'referenced_papers': List[str],  # 引用的論文 IDs
                'missing_info': List[str]  # 缺少的資訊類型
            }
        """
        try:
            logger.info(f"🔄 Integrating graph results for query: '{query[:60]}...'")
            
            if not graph_results:
                logger.warning("  No graph results to integrate")
                return {
                    'text': '抱歉，在知識圖譜中沒有找到相關的論文。',
                    'confidence': 0.0,
                    'should_descend': True,  # 嘗試 Layer2
                    'notes': ['No graph matches found'],
                    'referenced_papers': [],
                    'missing_info': ['all']
                }
            
            # 限制論文數量
            papers = graph_results[:max_papers]
            logger.info(f"  Using top {len(papers)} papers from graph")
            
            # 構建 prompt
            prompt = self._build_integration_prompt(query, papers)
            
            # 調用 LLM
            logger.info("  Calling LLM for integration...")
            response = self.llm.invoke(prompt)
            
            # 解析 LLM 回應
            result = self._parse_llm_response(response, papers)
            
            logger.info(f"  ✓ Integration complete:")
            logger.info(f"     Confidence: {result['confidence']:.2f}")
            logger.info(f"     Should descend: {result['should_descend']}")
            logger.info(f"     Answer preview: {result['text'][:100]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Integration failed: {e}", exc_info=True)
            return {
                'text': '整合圖譜資訊時發生錯誤。',
                'confidence': 0.0,
                'should_descend': True,
                'notes': [f'Error: {str(e)}'],
                'referenced_papers': [],
                'missing_info': ['error']
            }
    
    def _build_integration_prompt(
        self,
        query: str,
        papers: List[Dict[str, Any]]
    ) -> str:
        """
        構建整合 prompt
        
        Returns:
            Prompt string for LLM
        """
        # 準備論文摘要
        paper_summaries = []
        for i, paper in enumerate(papers, 1):
            summary = f"""
論文 {i}: {paper['title']}
- ID: {paper['paper_id']}
- 相關實體: {', '.join(paper['matched_entities'][:5])}
- 實體類型: {', '.join(set(paper['entity_types']))}
- 匹配理由: {paper['evidence']}
""".strip()
            paper_summaries.append(summary)
        
        papers_text = '\n\n'.join(paper_summaries)
        
        # 檢測查詢語言
        is_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)
        
        if is_chinese:
            prompt = f"""你是一個學術研究助手。請根據知識圖譜中的論文資訊回答問題。

問題: {query}

知識圖譜中找到的相關論文:
{papers_text}

請完成以下任務:
1. 根據以上論文資訊，直接回答用戶的問題
2. 評估這些資訊是否足以完整回答問題
3. 如果需要更詳細的資訊（例如：具體參數、實驗細節、完整方法描述），請說明缺少什麼

請以 JSON 格式回答（不要用 markdown 代碼塊包裝）:
{{
    "answer": "根據圖譜資訊的答案（繁體中文，2-4 句話）",
    "confidence": 0.7,
    "needs_details": true,
    "missing": ["缺少的資訊類型，例如: 具體參數, 實驗設定, 數據集詳情"]
}}

JSON:"""
        else:
            prompt = f"""You are an academic research assistant. Answer the question based on knowledge graph information.

Question: {query}

Related papers from knowledge graph:
{papers_text}

Please:
1. Answer the user's question directly based on the paper information above
2. Assess if this information is sufficient to fully answer the question
3. If more details are needed (e.g., specific parameters, experimental details, full methodology), state what is missing

Respond in JSON format (do NOT wrap in markdown code blocks):
{{
    "answer": "Answer based on graph information (2-4 sentences)",
    "confidence": 0.7,
    "needs_details": true,
    "missing": ["types of missing information, e.g.: specific parameters, experimental setup, dataset details"]
}}

JSON:"""
        
        return prompt
    
    def _parse_llm_response(
        self,
        response: str,
        papers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        解析 LLM 回應
        
        Returns:
            Integrated result dictionary
        """
        try:
            # 清理回應（移除可能的 markdown 標記）
            cleaned = response.strip()
            if cleaned.startswith('```'):
                # 移除 markdown 代碼塊
                lines = cleaned.split('\n')
                cleaned = '\n'.join(lines[1:-1] if len(lines) > 2 else lines)
            
            # 嘗試解析 JSON
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                # 如果 JSON 解析失敗，嘗試提取關鍵資訊
                logger.warning("Failed to parse JSON, extracting manually")
                data = self._extract_from_text(cleaned)
            
            # 提取欄位
            answer = data.get('answer', cleaned)
            confidence = float(data.get('confidence', 0.6))
            needs_details = data.get('needs_details', True)
            missing = data.get('missing', [])
            
            # 準備引用列表
            referenced_papers = [p['paper_id'] for p in papers[:5]]
            
            # 構建完整答案（添加論文引用）
            paper_refs = '\n\n📚 參考論文:\n' + '\n'.join([
                f"  {i}. {p['title']} (ID: {p['paper_id'][:30]}...)"
                for i, p in enumerate(papers[:5], 1)
            ])
            
            full_answer = answer + paper_refs
            
            # 決定是否需要下降到 Layer2
            should_descend = needs_details or confidence < 0.7
            
            return {
                'text': full_answer,
                'confidence': confidence,
                'should_descend': should_descend,
                'notes': [
                    f"Graph provided {len(papers)} papers",
                    f"Needs details: {needs_details}"
                ],
                'referenced_papers': referenced_papers,
                'missing_info': missing if isinstance(missing, list) else [str(missing)]
            }
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Fallback: 使用原始回應
            return {
                'text': response,
                'confidence': 0.5,
                'should_descend': True,
                'notes': ['Failed to parse structured response'],
                'referenced_papers': [p['paper_id'] for p in papers[:5]],
                'missing_info': ['unknown']
            }
    
    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        從文本中提取資訊（當 JSON 解析失敗時）
        
        Returns:
            Dictionary with extracted fields
        """
        # 簡單的啟發式提取
        confidence = 0.5
        needs_details = True
        
        # 檢查是否提到"缺少"、"需要"等
        if any(keyword in text.lower() for keyword in ['缺少', '需要', 'need', 'missing', 'require']):
            needs_details = True
            confidence = 0.4
        
        # 檢查是否有明確答案
        if any(keyword in text for keyword in ['使用了', '採用了', '包括', 'used', 'includes', 'contains']):
            confidence = 0.6
        
        return {
            'answer': text,
            'confidence': confidence,
            'needs_details': needs_details,
            'missing': ['details']
        }
    
    def determine_chunk_types_for_query(
        self,
        query: str,
        missing_info: List[str]
    ) -> List[str]:
        """
        根據查詢和缺失資訊，決定應該檢索哪些 section names
        
        注意：此方法現在返回 section names（如 "Method", "Results"）而非抽象的 chunk types，
        以適應基於 section_summary 的 chunking 模式
        
        Args:
            query: 用戶查詢
            missing_info: 缺失的資訊類型
            
        Returns:
            List of section names to search (e.g., ['Method', 'Results', 'Dataset'])
        """
        query_lower = query.lower()
        section_names = set()
        
        # 根據查詢關鍵詞映射到論文 section names（基於實際 JSONL 中的命名）
        # Dataset queries → Dataset section (實際存在的 section)
        if any(kw in query_lower for kw in ['dataset', '資料集', '數據集', 'corpus', 'benchmark', '語料', 'data']):
            section_names.update(['Dataset', 'Methodology'])  # Dataset section 存在，Methodology 通常也包含數據描述
        
        # Method queries → Methodology section (實際存在，最常見 49 個)
        if any(kw in query_lower for kw in ['method', 'approach', 'algorithm', '方法', '演算法', 'architecture', 'model', '模型', 'technique']):
            section_names.update(['Methodology'])  # 使用實際存在的 Methodology
        
        # Metric/Result queries → Results section (實際存在 29 個)
        if any(kw in query_lower for kw in ['metric', 'accuracy', 'precision', 'recall', 'f1', 'bleu', 'rouge', '指標', '準確率', 
                                             'result', 'performance', 'evaluation', 'outcome', '結果', '性能', '效能', '表現']):
            section_names.update(['Results', 'Dataset'])  # Results 存在，Dataset 可能包含實驗設置
        
        # Background/overview queries → Abstract, Introduction sections (實際存在)
        if any(kw in query_lower for kw in ['overview', 'summary', 'abstract', 'introduction', '概述', '摘要', '導論', '背景', 'background']):
            section_names.update(['Abstract', 'Introduction', 'Related_Work'])  # 使用實際存在的命名
        
        # Conclusion queries → Conclusion section (實際存在 29 個)
        if any(kw in query_lower for kw in ['conclusion', 'future', 'discussion', '結論', '未來', '討論']):
            section_names.update(['Conclusion'])
        
        # 根據缺失資訊補充（使用實際存在的 section names）
        for missing in missing_info:
            missing_lower = str(missing).lower()
            
            if 'parameter' in missing_lower or '參數' in missing_lower or 'config' in missing_lower:
                section_names.update(['Methodology', 'Dataset'])
            
            if 'dataset' in missing_lower or '資料集' in missing_lower or 'data' in missing_lower:
                section_names.update(['Dataset', 'Methodology'])
            
            if 'metric' in missing_lower or '指標' in missing_lower or 'measure' in missing_lower:
                section_names.update(['Results', 'Dataset'])
            
            if 'result' in missing_lower or '結果' in missing_lower or 'performance' in missing_lower:
                section_names.update(['Results', 'Methodology'])
            
            if 'detail' in missing_lower or '詳細' in missing_lower:
                # Generic details → include methodology + results
                section_names.update(['Methodology', 'Results'])
        
        # 如果沒有匹配，返回核心 sections（使用實際存在的命名）
        if not section_names:
            section_names = {'Methodology', 'Results', 'Dataset', 'Introduction'}
        
        result = list(section_names)
        logger.info(f"  Determined section names for Layer2: {result}")
        
        return result
