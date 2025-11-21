"""
Graph Retriever Module

Queries Neo4j graph database to find papers and entities relevant to user queries.
This is the first step in the Graph-first retrieval flow.
"""

import logging
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


class GraphRetriever:
    """
    從 Neo4j Graph 檢索相關論文和實體
    
    Strategy:
    - 解析查詢關鍵詞
    - 在 Graph 中查找匹配的實體（Dataset, Method, Metric 等）
    - 返回與這些實體相關的論文
    - 包含匹配理由和信心分數
    """
    
    def __init__(self, graph_manager):
        """
        初始化 Graph Retriever
        
        Args:
            graph_manager: GraphManager instance with Neo4j connection
        """
        self.graph_manager = graph_manager
        logger.info("✓ GraphRetriever initialized")
    
    def query_graph_for_papers(
        self, 
        query: str, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        查詢 Graph 以找到相關論文
        
        Args:
            query: 用戶查詢
            top_k: 返回最多幾篇論文
            
        Returns:
            List of {
                'paper_id': str,
                'title': str,
                'score': float,  # 0-1
                'matched_entities': List[str],  # 匹配到的實體名稱
                'entity_types': List[str],  # 實體類型
                'evidence': str,  # 匹配理由
                'relationship_count': int  # 關係數量
            }
        """
        try:
            logger.info(f"🔍 Querying Graph for: '{query}'")
            
            # Step 1: 提取查詢關鍵詞
            keywords = self._extract_keywords(query)
            logger.info(f"  Extracted keywords: {keywords}")
            
            if not keywords:
                logger.warning("  No keywords extracted from query")
                return []
            
            # Step 2: 在 Graph 中搜尋匹配的實體
            matched_papers = []
            
            # 2.1: 搜尋 Dataset entities
            dataset_papers = self._search_by_entity_type(keywords, 'Dataset', top_k)
            matched_papers.extend(dataset_papers)
            
            # 2.2: 搜尋 Method entities
            method_papers = self._search_by_entity_type(keywords, 'Method', top_k)
            matched_papers.extend(method_papers)
            
            # 2.3: 搜尋 Metric entities
            metric_papers = self._search_by_entity_type(keywords, 'Metric', top_k)
            matched_papers.extend(metric_papers)
            
            # 2.4: 搜尋 Domain entities
            domain_papers = self._search_by_domain(keywords, top_k)
            matched_papers.extend(domain_papers)
            
            # Step 3: 去重並合併結果
            papers_dict = {}
            for paper in matched_papers:
                paper_id = paper['paper_id']
                if paper_id in papers_dict:
                    # 合併相同論文的實體和分數
                    papers_dict[paper_id]['matched_entities'].extend(paper['matched_entities'])
                    papers_dict[paper_id]['entity_types'].extend(paper['entity_types'])
                    papers_dict[paper_id]['score'] = max(papers_dict[paper_id]['score'], paper['score'])
                    papers_dict[paper_id]['relationship_count'] += paper['relationship_count']
                else:
                    papers_dict[paper_id] = paper
            
            # 去重實體列表
            for paper in papers_dict.values():
                paper['matched_entities'] = list(set(paper['matched_entities']))
                paper['entity_types'] = list(set(paper['entity_types']))
            
            # Step 4: 排序並返回 top-k
            results = sorted(
                papers_dict.values(),
                key=lambda x: (x['score'], x['relationship_count']),
                reverse=True
            )[:top_k]
            
            logger.info(f"  ✓ Found {len(results)} papers in Graph")
            for i, paper in enumerate(results[:5], 1):
                logger.info(f"    {i}. {paper['paper_id'][:50]} (score: {paper['score']:.2f}, "
                           f"entities: {len(paper['matched_entities'])})")
            
            return results
            
        except Exception as e:
            logger.error(f"Graph query failed: {e}", exc_info=True)
            return []
    
    def query_graph_for_entities(
        self, 
        query: str, 
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        查詢 Graph 以找到相關實體
        
        Args:
            query: 用戶查詢
            top_k: 返回最多幾個實體
            
        Returns:
            List of {
                'name': str,
                'type': str,  # Dataset, Method, Metric, Domain
                'paper_count': int,  # 有多少論文使用此實體
                'papers': List[str]  # paper_ids
            }
        """
        try:
            keywords = self._extract_keywords(query)
            
            if not keywords:
                return []
            
            entities = []
            
            # 搜尋各類型實體
            for entity_type in ['Dataset', 'Method', 'Metric', 'Domain']:
                cypher_query = f"""
                MATCH (p:Paper)-[r]->(e:{entity_type})
                WHERE toLower(e.name) CONTAINS $keyword
                RETURN e.name as name, '{entity_type}' as type, 
                       count(DISTINCT p) as paper_count,
                       collect(DISTINCT p.paper_id) as papers
                ORDER BY paper_count DESC
                LIMIT $limit
                """
                
                for keyword in keywords[:3]:  # 只用前 3 個關鍵詞
                    results = self.graph_manager.graph.query(
                        cypher_query,
                        {"keyword": keyword.lower(), "limit": top_k}
                    )
                    
                    for row in results:
                        entities.append({
                            'name': row['name'],
                            'type': row['type'],
                            'paper_count': row['paper_count'],
                            'papers': row['papers']
                        })
            
            # 去重並排序
            entities_dict = {}
            for entity in entities:
                key = (entity['name'], entity['type'])
                if key not in entities_dict:
                    entities_dict[key] = entity
            
            results = sorted(
                entities_dict.values(),
                key=lambda x: x['paper_count'],
                reverse=True
            )[:top_k]
            
            logger.info(f"  ✓ Found {len(results)} entities")
            
            return results
            
        except Exception as e:
            logger.error(f"Entity query failed: {e}")
            return []
    
    def get_papers_for_entities(
        self, 
        entity_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        根據實體名稱獲取相關論文
        
        Args:
            entity_names: 實體名稱列表
            
        Returns:
            Dict mapping paper_id to {
                'title': str,
                'entities': List[str],
                'entity_types': List[str]
            }
        """
        try:
            if not entity_names:
                return {}
            
            # 構建 Cypher 查詢
            cypher_query = """
            MATCH (p:Paper)-[r]->(e)
            WHERE e.name IN $entity_names
            RETURN p.paper_id as paper_id,
                   p.title as title,
                   collect(DISTINCT e.name) as entities,
                   collect(DISTINCT labels(e)[0]) as entity_types
            """
            
            results = self.graph_manager.graph.query(
                cypher_query,
                {"entity_names": entity_names}
            )
            
            papers = {}
            for row in results:
                papers[row['paper_id']] = {
                    'title': row['title'],
                    'entities': row['entities'],
                    'entity_types': row['entity_types']
                }
            
            logger.info(f"  ✓ Found {len(papers)} papers for {len(entity_names)} entities")
            
            return papers
            
        except Exception as e:
            logger.error(f"Failed to get papers for entities: {e}")
            return {}
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        從查詢中提取關鍵詞
        
        Simple strategy:
        - Split by spaces
        - Remove stopwords
        - Keep words with 2+ characters
        """
        # 中英文停用詞
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'can', 'could', 'may', 'might', 'must', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
            'when', 'where', 'why', 'how', 'of', 'in', 'on', 'at', 'to', 'for',
            'with', 'by', 'from', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'all', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just',
            # Chinese
            '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一',
            '一個', '上', '也', '很', '到', '說', '要', '去', '你', '會', '著', '沒有',
            '看', '好', '自己', '這', '什麼', '怎麼', '哪', '為什麼'
        }
        
        # 清理並分割
        query_lower = query.lower()
        
        # 保留英文字母、數字、中文字符
        cleaned = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', query_lower)
        
        # 分割
        words = cleaned.split()
        
        # 過濾
        keywords = [
            word for word in words
            if len(word) >= 2 and word not in stopwords
        ]
        
        return keywords[:10]  # 最多保留 10 個關鍵詞
    
    def _search_by_entity_type(
        self, 
        keywords: List[str], 
        entity_type: str, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        根據實體類型搜尋論文
        
        Args:
            keywords: 關鍵詞列表
            entity_type: 實體類型（Dataset, Method, Metric 等）
            limit: 最多返回幾筆
            
        Returns:
            List of paper dictionaries
        """
        try:
            papers = []
            
            for keyword in keywords:
                # 使用 CONTAINS 進行模糊匹配
                cypher_query = f"""
                MATCH (p:Paper)-[r]->(e:{entity_type})
                WHERE toLower(e.name) CONTAINS $keyword
                RETURN p.paper_id as paper_id,
                       p.title as title,
                       collect(DISTINCT e.name) as entities,
                       count(r) as rel_count
                ORDER BY rel_count DESC
                LIMIT $limit
                """
                
                results = self.graph_manager.graph.query(
                    cypher_query,
                    {"keyword": keyword.lower(), "limit": limit}
                )
                
                for row in results:
                    papers.append({
                        'paper_id': row['paper_id'],
                        'title': row['title'] or row['paper_id'],
                        'score': 0.8,  # 高信心（直接匹配）
                        'matched_entities': row['entities'],
                        'entity_types': [entity_type],
                        'evidence': f"Matched {entity_type.lower()}: {', '.join(row['entities'][:3])}",
                        'relationship_count': row['rel_count']
                    })
            
            return papers
            
        except Exception as e:
            logger.debug(f"Search by {entity_type} failed: {e}")
            return []
    
    def _search_by_domain(
        self, 
        keywords: List[str], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        根據領域搜尋論文
        """
        try:
            papers = []

            # 搜尋中英文領域（使用 COALESCE 避免缺少屬性時的警告）
            for keyword in keywords:
                cypher_query = """
                MATCH (p:Paper)
                WHERE toLower(coalesce(p.domain, '')) CONTAINS $keyword
                   OR toLower(coalesce(p.domain_zh, '')) CONTAINS $keyword
                RETURN p.paper_id as paper_id,
                       p.title as title,
                       coalesce(p.domain, '') as domain,
                       coalesce(p.domain_zh, '') as domain_zh
                LIMIT $limit
                """

                results = self.graph_manager.graph.query(
                    cypher_query,
                    {"keyword": keyword.lower(), "limit": limit}
                )

                for row in results:
                    domain = row.get('domain_zh') or row.get('domain') or 'Unknown'
                    papers.append({
                        'paper_id': row['paper_id'],
                        'title': row.get('title') or row['paper_id'],
                        'score': 0.6,  # 中信心（領域匹配）
                        'matched_entities': [domain],
                        'entity_types': ['Domain'],
                        'evidence': f"Domain: {domain}",
                        'relationship_count': 1
                    })

            return papers

        except Exception as e:
            logger.debug(f"Search by domain failed: {e}")
            return []
