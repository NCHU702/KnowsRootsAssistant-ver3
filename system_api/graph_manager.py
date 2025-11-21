"""
Graph Manager Module

Manages interactions with Neo4j Graph Database.
Handles schema definition, data ingestion, and natural language querying (Text2Cypher).
"""

import logging
import re
from typing import List, Dict, Any, Optional
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)


class SanitizingGraphCypherQAChain(GraphCypherQAChain):
    """
    Custom GraphCypherQAChain that sanitizes LLM-generated Cypher before execution.
    """
    
    def _call(self, inputs: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """Override to sanitize generated Cypher before Neo4j execution."""
        # Let parent class generate the Cypher
        try:
            # Access the intermediate Cypher generation step
            from langchain_core.callbacks import CallbackManagerForChainRun
            
            # Get the question
            question = inputs[self.input_key]
            
            # Generate Cypher using parent's logic
            intermediate_steps = []
            _run_manager = kwargs.get("run_manager")
            
            # Generate cypher with prompt
            cypher = self.cypher_generation_chain.predict(
                question=question,
                callbacks=_run_manager.get_child() if _run_manager else None
            )
            
            # Log original
            logger.debug(f"Original LLM Cypher:\n{cypher}")
            
            # Sanitize it
            cypher = sanitize_cypher_query(cypher)
            logger.info(f"Sanitized Cypher:\n{cypher}")
            
            # Execute sanitized query
            context = self.graph.query(cypher)
            
            # Generate final answer
            result = self.qa_chain.predict(
                question=question,
                context=context,
                callbacks=_run_manager.get_child() if _run_manager else None
            )
            
            return {self.output_key: result}
            
        except Exception as e:
            logger.error(f"Error in sanitizing chain: {e}")
            # Fallback to parent implementation
            return super()._call(inputs, *args, **kwargs)


def sanitize_cypher_query(raw_cypher: str) -> str:
    """
    Strip common LLM-generated prefixes/labels from Cypher queries.
    Examples: "A:", "1.", "Query:", "```cypher", etc.
    
    Returns the cleaned Cypher or the original if no cleaning patterns match.
    """
    if not raw_cypher:
        return raw_cypher
    
    # Remove leading whitespace
    cleaned = raw_cypher.strip()
    
    # Pattern 1: Remove "A:", "Q:", numbered items like "1.", "(a)", etc.
    cleaned = re.sub(r'^[A-Za-z]:\s*', '', cleaned)
    cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
    cleaned = re.sub(r'^\([a-z]\)\s*', '', cleaned, flags=re.IGNORECASE)
    
    # Pattern 2: Remove code fence markers
    cleaned = re.sub(r'^```(?:cypher)?\s*\n?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned)
    
    # Pattern 3: Remove common label words
    cleaned = re.sub(r'^(?:Query|Cypher|Answer):\s*', '', cleaned, flags=re.IGNORECASE)
    
    # Ensure it starts with a valid Cypher keyword
    valid_keywords = ['MATCH', 'OPTIONAL', 'WITH', 'CALL', 'UNWIND', 'RETURN', 'CREATE', 'MERGE', 'DELETE', 'DETACH']
    first_word = cleaned.split()[0].upper() if cleaned.split() else ''
    
    if first_word not in valid_keywords:
        # Try to find the first valid keyword
        for keyword in valid_keywords:
            pattern = rf'\b{keyword}\b'
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                cleaned = cleaned[match.start():]
                logger.info(f"Sanitized Cypher: removed prefix before '{keyword}'")
                break
    
    return cleaned.strip()

class GraphManager:
    """
    管理 Neo4j 圖資料庫的連接與操作
    
    Schema 定義:
    - Nodes: 
        - Paper (屬性: paper_id, title, year)
        - ResearchGoal (屬性: summary)
        - Method (屬性: name)
        - Dataset (屬性: name)
        - Domain (屬性: name) [新增]
        - Metric (屬性: name) [新增]
    - Relationships:
        - (:Paper)-[:AIMS_TO]->(:ResearchGoal)
        - (:Paper)-[:USES_METHOD]->(:Method)
        - (:Paper)-[:EVALUATED_ON]->(:Dataset)
        - (:Paper)-[:APPLIED_IN]->(:Domain) [新增]
        - (:Paper)-[:EVALUATED_WITH]->(:Metric) [新增]
    """
    
    def __init__(
        self, 
        llm: OllamaLLM, 
        url: str = "bolt://localhost:7687", 
        username: str = "neo4j", 
        password: str = "password"
    ):
        self.llm = llm
        self.url = url
        self.username = username
        self.password = password
        self.graph = None
        
        try:
            self.graph = Neo4jGraph(
                url=url, 
                username=username, 
                password=password
            )
            self._init_schema()
            logger.info("✓ GraphManager initialized and connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            logger.warning("Graph capabilities will be disabled.")

    def _init_schema(self):
        """建立索引"""
        if not self.graph: return
        
        schema_queries = [
            "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",  # [Chunk-Level]
            "CREATE INDEX paper_title_index IF NOT EXISTS FOR (p:Paper) ON (p.title)",
            "CREATE INDEX chunk_paper_index IF NOT EXISTS FOR (c:Chunk) ON (c.paper_id)",  # [Chunk-Level]
            "CREATE INDEX method_name_index IF NOT EXISTS FOR (m:Method) ON (m.name)",
            "CREATE INDEX dataset_name_index IF NOT EXISTS FOR (d:Dataset) ON (d.name)",
            "CREATE INDEX domain_name_index IF NOT EXISTS FOR (dom:Domain) ON (dom.name)", # [新增]
            "CREATE INDEX metric_name_index IF NOT EXISTS FOR (met:Metric) ON (met.name)"   # [新增]
        ]
        
        try:
            for query in schema_queries:
                self.graph.query(query)
            self.graph.refresh_schema()
            logger.debug("Graph schema constraints initialized")
        except Exception as e:
            logger.warning(f"Schema initialization warning: {e}")

    def add_paper_metadata(
        self, 
        paper_id: str, 
        title: str, 
        year: str, 
        research_goal: str, 
        methods: List[str], 
        datasets: List[str],
        domain: str,         # [新增]
        metrics: List[str],   # [新增]
        domain_zh: str = None  # [雙語支援]
    ) -> bool:
        """
        將論文的結構化資訊寫入圖資料庫 (包含 Domain 和 Metrics)
        支援雙語域名 (domain 英文, domain_zh 中文)
        """
        if not self.graph:
            return False

        cypher = """
        MERGE (p:Paper {paper_id: $paper_id})
        SET p.title = $title, 
            p.year = $year,
            p.last_updated = datetime()

        // 1. Research Goal
        MERGE (g:ResearchGoal {summary: $research_goal})
        MERGE (p)-[:AIMS_TO]->(g)

        // 2. Methods
        FOREACH (m_name IN $methods | 
            MERGE (m:Method {name: m_name})
            MERGE (p)-[:USES_METHOD]->(m)
        )

        // 3. Datasets
        FOREACH (d_name IN $datasets | 
            MERGE (d:Dataset {name: d_name})
            MERGE (p)-[:EVALUATED_ON]->(d)
        )

        // 4. Domain [雙語支援]
        MERGE (dom:Domain {name: $domain})
        SET dom.name_zh = $domain_zh
        MERGE (p)-[:APPLIED_IN]->(dom)

        // 5. Metrics [新增]
        FOREACH (met_name IN $metrics | 
            MERGE (met:Metric {name: met_name})
            MERGE (p)-[:EVALUATED_WITH]->(met)
        )
        """
        
        try:
            clean_methods = [m.strip() for m in methods if m and m.strip()]
            clean_datasets = [d.strip() for d in datasets if d and d.strip()]
            clean_metrics = [met.strip() for met in metrics if met and met.strip()]
            clean_domain = domain.strip() if domain else "General"
            clean_domain_zh = domain_zh.strip() if domain_zh else "通用"
            
            params = {
                "paper_id": paper_id,
                "title": title,
                "year": str(year) if year else "Unknown",
                "research_goal": research_goal if research_goal else "Unknown Goal",
                "methods": clean_methods,
                "datasets": clean_datasets,
                "domain": clean_domain,
                "domain_zh": clean_domain_zh,
                "metrics": clean_metrics
            }
            
            self.graph.query(cypher, params=params)
            logger.info(f"✓ Added graph metadata (incl. Domain/Metrics) for: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write to graph: {e}")
            return False

    def query_graph(self, user_query: str) -> str:
        """
        將自然語言轉換為 Cypher 查詢 (with fallback strategy)
        """
        if not self.graph:
            return "Error: Graph database not connected."

        cypher_generation_template = """
        Task: Generate a Cypher statement to query the graph database.
        
        Strict Output Rules (very important):
        0. ONLY output the Cypher query itself. Do NOT output any commentary, labels, prefixes, numbered items, or code fences.
           Examples of disallowed prefixes: "A:", "1.", "(a)", "Query:", "```cypher```, etc.
        1. Use ONLY the provided schema.
        2. Use case-insensitive matching (e.g. toLower(n.name) CONTAINS toLower("keyword")).
        3. Return specific values (e.g. p.title, m.name), not just nodes.
        4. Do NOT include explanations or any surrounding text. Output MUST start with a valid Cypher keyword such as MATCH, OPTIONAL MATCH, CALL, WITH, UNWIND.
        5. IMPORTANT: Domain nodes have BILINGUAL names:
           - dom.name: English name (e.g., "Healthcare", "Smart Manufacturing")
           - dom.name_zh: Chinese name (e.g., "智慧醫療", "智慧製造")
           When querying domains, search BOTH properties with OR:
           WHERE toLower(dom.name) CONTAINS "keyword" OR toLower(dom.name_zh) CONTAINS "keyword"
        6. BROAD SEARCH STRATEGY: When the user asks "which papers are related to X" or "find papers about X", 
           you MUST search in multiple places to ensure high recall:
           - Paper title (p.title)
           - ResearchGoal summary (g.summary)
           - Domain name (dom.name AND dom.name_zh)
           - Dataset name (d.name)
           Use OPTIONAL MATCH and OR conditions to catch all relevant papers.
        
        Schema:
          Nodes: 
            - Paper (paper_id, title, year)
            - ResearchGoal (summary)
            - Method (name)
            - Dataset (name)
            - Domain (name [English], name_zh [Chinese])
            - Metric (name)
          Relationships:
            - (:Paper)-[:AIMS_TO]->(:ResearchGoal)
            - (:Paper)-[:USES_METHOD]->(:Method)
            - (:Paper)-[:EVALUATED_ON]->(:Dataset)
            - (:Paper)-[:APPLIED_IN]->(:Domain)
            - (:Paper)-[:EVALUATED_WITH]->(:Metric)
            
        Examples (IMPORTANT - follow these patterns):
        
        Q: "Which papers use CNN?" or "哪些論文使用 CNN?"
        A: MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
           WHERE toLower(m.name) CONTAINS "cnn" OR toLower(m.name) CONTAINS "卷積"
           RETURN DISTINCT p.title, p.year
        
        Q: "哪些論文和集成式學習有關?" (Which papers are related to ensemble learning?)
        A: MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
           WHERE toLower(m.name) CONTAINS "ensemble" OR toLower(m.name) CONTAINS "集成"
           RETURN DISTINCT p.title, p.year
        
        Q: "Papers using Mask R-CNN"
        A: MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
           WHERE toLower(m.name) CONTAINS "mask" AND toLower(m.name) CONTAINS "cnn"
           RETURN DISTINCT p.title, p.year
        
        Q: "Which papers use LSTM in Healthcare?"
        A: MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
           WHERE toLower(m.name) CONTAINS "lstm"
           OPTIONAL MATCH (p)-[:APPLIED_IN]->(dom:Domain)
           WHERE toLower(dom.name) CONTAINS "healthcare" OR toLower(dom.name_zh) CONTAINS "醫療"
           RETURN DISTINCT p.title, p.year
        
        Q: "List papers evaluated with RMSE."
        A: MATCH (p:Paper)-[:EVALUATED_WITH]->(met:Metric) 
           WHERE toLower(met.name) CONTAINS "rmse" 
           RETURN DISTINCT p.title, p.year
        
        The question is:
        {question}
        """
        
        cypher_prompt = PromptTemplate(
            input_variables=["question"], 
            template=cypher_generation_template
        )
        
        # Custom QA prompt to interpret results more positively
        qa_template = """You are an assistant that helps interpret database query results.

Given the original question and the database query results, provide a clear and helpful answer.

IMPORTANT RULES:
1. If the results contain ANY data (even a single row), that means we FOUND relevant information.
2. Do NOT say "I don't know" or "no information" if results are present.
3. Format paper information clearly with title and year.
4. If results are empty (truly no rows), then you can say no matching papers were found.

Question: {question}

Database Results: {context}

Answer (be direct and informative):"""

        qa_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template=qa_template
        )

        try:
            chain = SanitizingGraphCypherQAChain.from_llm(
                self.llm,
                graph=self.graph,
                verbose=True,
                cypher_prompt=cypher_prompt,
                qa_prompt=qa_prompt,
                allow_dangerous_requests=True
            )
            
            result = chain.invoke(user_query)
            result_text = result['result']
            
            # Check if result is empty or indicates no data found
            if self._is_empty_result(result_text):
                logger.warning("⚠️  First query returned empty results, trying fallback strategy...")
                fallback_result = self._query_graph_fallback(user_query)
                if not self._is_empty_result(fallback_result):
                    logger.info("✓ Fallback query succeeded")
                    return fallback_result
                else:
                    logger.info("Fallback also returned empty, returning original result")
            
            return result_text
            
        except Exception as e:
            # Log the original error
            logger.error(f"Graph query failed: {e}")

            # Attempt to salvage a Cypher statement embedded in the error message
            try:
                import re
                msg = str(e)
                # Find the first occurrence of a Cypher keyword and take everything after it
                m = re.search(r"(MATCH|OPTIONAL MATCH|WITH|CALL|UNWIND|RETURN)[\s\S]*", msg, re.IGNORECASE)
                if m:
                    candidate_cypher = m.group(0).strip()
                    # Remove any leading non-cypher tokens like 'A:' or numbering
                    candidate_cypher = re.sub(r"^[^A-Za-z0-9]*(?=(MATCH|OPTIONAL MATCH|WITH|CALL|UNWIND|RETURN))", "", candidate_cypher, flags=re.IGNORECASE)
                    logger.info(f"Attempting to run sanitized Cypher extracted from error: {candidate_cypher[:200]}")
                    try:
                        rows = self.graph.query(candidate_cypher)
                        # Return the raw rows as string to keep behavior consistent with earlier returns
                        return str(rows)
                    except Exception as e2:
                        logger.error(f"Sanitized cypher execution also failed: {e2}")
            except Exception:
                # If anything in the salvage attempt fails, fall through to return original error
                pass

            return f"無法執行圖譜查詢: {str(e)}"
    
    def _is_empty_result(self, result: str) -> bool:
        """Check if a query result is empty or indicates no data"""
        if not result or result.strip() == "":
            return True
        
        # Check for common "no data" indicators
        empty_indicators = [
            "i don't know",
            "no information",
            "not found",
            "empty",
            "no results",
            "no papers",
            "no data",
            "無資料",
            "找不到",
            "沒有",
            "不知道",      # "don't know"
            "不清楚",      # "not clear"
            "未找到",      # "not found"
            "查無",        # "no results"
        ]
        
        result_lower = result.lower()
        return any(indicator in result_lower for indicator in empty_indicators)
    
    def _query_graph_fallback(self, user_query: str) -> str:
        """
        Fallback strategy with looser Cypher generation.
        Uses OPTIONAL MATCH and removes strict AND conditions.
        """
        if not self.graph:
            return "Error: Graph database not connected."
        
        # More relaxed prompt that favors OPTIONAL MATCH and OR conditions
        fallback_template = """
        Task: Generate a BROAD Cypher query that maximizes recall (find as many relevant papers as possible).
        
        Critical Rules for High Recall:
        1. PREFER "OPTIONAL MATCH" over strict "MATCH" to avoid filtering out papers
        2. Use OR conditions liberally - papers may match ANY of the criteria
        3. Search across MULTIPLE node types: Paper, Method, Dataset, Domain, ResearchGoal
        4. Use case-insensitive CONTAINS (not exact matches)
        5. For domain queries, ALWAYS check BOTH dom.name AND dom.name_zh with OR
        6. Do NOT use multiple AND conditions that all must be satisfied
        7. Return DISTINCT results to avoid duplicates
        
        Schema:
          Nodes: Paper (paper_id, title, year), ResearchGoal (summary), 
                 Method (name), Dataset (name), Domain (name, name_zh), Metric (name)
          Relationships: [:AIMS_TO], [:USES_METHOD], [:EVALUATED_ON], [:APPLIED_IN], [:EVALUATED_WITH]
        
        Examples of BROAD queries (CRITICAL - WHERE must come AFTER WITH when using OPTIONAL MATCH):
        
        Q: "Papers using ensemble learning" or "哪些論文和集成式學習有關?"
        A: MATCH (p:Paper)
           OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
           WITH p, m
           WHERE toLower(coalesce(p.title, '')) CONTAINS "ensemble" OR toLower(coalesce(p.title, '')) CONTAINS "集成"
              OR toLower(coalesce(m.name, '')) CONTAINS "ensemble" OR toLower(coalesce(m.name, '')) CONTAINS "集成"
           RETURN DISTINCT p.title, p.year
        
        Q: "Papers using CNN"
        A: MATCH (p:Paper)
           OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
           WITH p, m
           WHERE toLower(coalesce(p.title, '')) CONTAINS "cnn" 
              OR toLower(coalesce(m.name, '')) CONTAINS "cnn"
              OR toLower(coalesce(m.name, '')) CONTAINS "卷積"
           RETURN DISTINCT p.title, p.year
        
        Q: "Transportation domain research"
        A: MATCH (p:Paper)
           OPTIONAL MATCH (p)-[:APPLIED_IN]->(dom:Domain)
           OPTIONAL MATCH (p)-[:AIMS_TO]->(g:ResearchGoal)
           WITH p, dom, g
           WHERE toLower(coalesce(p.title, '')) CONTAINS "transportation" OR toLower(coalesce(p.title, '')) CONTAINS "交通"
              OR toLower(coalesce(dom.name, '')) CONTAINS "transportation" OR toLower(coalesce(dom.name_zh, '')) CONTAINS "交通"
              OR toLower(coalesce(g.summary, '')) CONTAINS "transportation" OR toLower(coalesce(g.summary, '')) CONTAINS "交通"
           RETURN DISTINCT p.title, p.year
        
        Q: "What datasets are used?"
        A: MATCH (p:Paper)-[:EVALUATED_ON]->(d:Dataset)
           RETURN DISTINCT d.name
        
        The question is:
        {question}
        
        Generate a Cypher query that will find ALL potentially relevant papers (maximize recall).
        """
        
        fallback_prompt = PromptTemplate(
            input_variables=["question"],
            template=fallback_template
        )
        
        # Use same QA prompt as main query
        fallback_qa_template = """You are an assistant that helps interpret database query results.

Given the original question and the database query results, provide a clear and helpful answer.

IMPORTANT RULES:
1. If the results contain ANY data (even a single row), that means we FOUND relevant information.
2. Do NOT say "I don't know" or "no information" if results are present.
3. Format paper information clearly with title and year.
4. If results are empty (truly no rows), then you can say no matching papers were found.

Question: {question}

Database Results: {context}

Answer (be direct and informative):"""

        fallback_qa_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template=fallback_qa_template
        )
        
        try:
            fallback_chain = SanitizingGraphCypherQAChain.from_llm(
                self.llm,
                graph=self.graph,
                verbose=True,
                cypher_prompt=fallback_prompt,
                qa_prompt=fallback_qa_prompt,
                allow_dangerous_requests=True
            )
            
            result = fallback_chain.invoke(user_query)
            return result['result']
            
        except Exception as e:
            logger.error(f"Fallback graph query failed: {e}")
            return f"備用查詢也失敗: {str(e)}"

    def get_visualization_data(self, limit: int = 100) -> Dict[str, Any]:
        """獲取前端視覺化數據"""
        if not self.graph: return {}

        query = """
        MATCH (n)-[r]->(m)
        RETURN 
            id(n) as source_id, labels(n) as source_labels, properties(n) as source_props,
            type(r) as rel_type,
            id(m) as target_id, labels(m) as target_labels, properties(m) as target_props
        LIMIT $limit
        """
        
        try:
            data = self.graph.query(query, params={"limit": limit})
            
            nodes = []
            edges = []
            seen_node_ids = set()

            def process_node(node_id, labels, props):
                str_id = str(node_id)
                if str_id in seen_node_ids: return
                
                label_text = props.get('title') or props.get('name') or props.get('summary') or "Unknown"
                display_label = label_text[:20] + "..." if len(label_text) > 20 else label_text
                node_type = labels[0] if labels else "Node"
                
                # 根據類型給定顏色 (前端可用)
                color_map = {
                    "Paper": "#4ea8de",       # 藍色
                    "Method": "#f72585",      # 粉紅
                    "Dataset": "#4cc9f0",     # 淺藍
                    "ResearchGoal": "#fee440",# 黃色
                    "Domain": "#7209b7",      # 紫色
                    "Metric": "#3a0ca3"       # 深藍
                }
                
                nodes.append({
                    "data": {
                        "id": str_id,
                        "label": display_label,
                        "full_label": label_text,
                        "type": node_type,
                        "color": color_map.get(node_type, "#cccccc")
                    }
                })
                seen_node_ids.add(str_id)

            for row in data:
                process_node(row['source_id'], row['source_labels'], row['source_props'])
                process_node(row['target_id'], row['target_labels'], row['target_props'])
                edges.append({
                    "data": {
                        "source": str(row['source_id']),
                        "target": str(row['target_id']),
                        "relationship": row['rel_type']
                    }
                })
            
            return {"elements": {"nodes": nodes, "edges": edges}}

        except Exception as e:
            logger.error(f"Visualization data fetch failed: {e}")
            return {"error": str(e)}
    
    def get_paper_count(self) -> int:
        """
        獲取 Neo4j 中 Paper 節點的總數
        
        Returns:
            論文數量
        """
        try:
            query = "MATCH (p:Paper) RETURN count(p) as count"
            result = self.graph.query(query)
            if result and len(result) > 0:
                return result[0].get('count', 0)
            return 0
        except Exception as e:
            logger.error(f"Failed to get paper count: {e}")
            return 0
    
    def get_all_paper_ids(self) -> List[str]:
        """
        獲取 Neo4j 中所有 Paper 節點的 paper_id
        
        Returns:
            paper_id 列表
        """
        try:
            query = "MATCH (p:Paper) RETURN p.paper_id as paper_id"
            result = self.graph.query(query)
            return [row.get('paper_id', '') for row in result if row.get('paper_id')]
        except Exception as e:
            logger.error(f"Failed to get paper IDs: {e}")
            return []
    
    def delete_paper(self, paper_id: str) -> bool:
        """
        從 Neo4j 刪除指定的 Paper 節點及其所有關聯關係
        
        Args:
            paper_id: 要刪除的論文 ID
            
        Returns:
            是否成功刪除
        """
        if not self.graph:
            return False
        
        try:
            # 刪除 Paper 節點及其所有關聯關係（包括 Chunks）
            query = """
            MATCH (p:Paper {paper_id: $paper_id})
            OPTIONAL MATCH (p)-[:CONTAINS]->(c:Chunk)
            DETACH DELETE p, c
            """
            self.graph.query(query, params={"paper_id": paper_id})
            logger.info(f"✓ Deleted paper from Graph: {paper_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete paper {paper_id} from Graph: {e}")
            return False
    
    def add_chunk_entities(
        self,
        paper_id: str,
        chunks: List[Any],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> bool:
        """
        將 chunk-level 實體和關係寫入圖資料庫（Chunk-Level GraphRAG）
        
        這是標準 GraphRAG 的核心功能：為每個 chunk 提取實體，並記錄 source_chunks。
        
        Args:
            paper_id: 論文 ID
            chunks: Chunk Document 列表（包含 chunk_id, text 等）
            entities: 提取的實體列表 [{"name": str, "type": str, "description": str, "source_chunks": [str]}]
            relationships: 提取的關係列表 [{"source": str, "target": str, "relation_type": str, ...}]
            
        Returns:
            是否成功
        """
        if not self.graph:
            return False
        
        try:
            # Step 1: 創建 Chunk 節點並連接到 Paper
            for chunk in chunks:
                chunk_id = chunk.metadata.get('chunk_id')
                chunk_text = chunk.page_content[:500]  # 儲存前 500 字元（避免過大）
                chunk_index = chunk.metadata.get('chunk_index', 0)
                
                cypher = """
                MATCH (p:Paper {paper_id: $paper_id})
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.paper_id = $paper_id,
                    c.text = $text,
                    c.chunk_index = $chunk_index,
                    c.created_at = datetime()
                MERGE (p)-[:CONTAINS]->(c)
                """
                
                self.graph.query(cypher, params={
                    "paper_id": paper_id,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "chunk_index": chunk_index
                })
            
            logger.info(f"  ✓ Created {len(chunks)} Chunk nodes")
            
            # Step 2: 創建 Entity 節點並記錄 source_chunks
            entity_count = 0
            for entity in entities:
                entity_name = entity.get('name')
                entity_type = entity.get('type', 'UNKNOWN')
                entity_desc = entity.get('description', '')
                source_chunks = entity.get('source_chunks', [])
                
                if not entity_name:
                    continue
                
                cypher = """
                MERGE (e:{entity_type} {{name: $name}})
                ON CREATE SET e.description = $description,
                              e.created_at = datetime()
                ON MATCH SET e.description = 
                    CASE 
                        WHEN e.description IS NULL THEN $description
                        WHEN NOT e.description CONTAINS $description THEN e.description + ' | ' + $description
                        ELSE e.description
                    END
                
                // 連接到 source chunks
                WITH e
                UNWIND $source_chunks AS chunk_id
                MATCH (c:Chunk {{chunk_id: chunk_id}})
                MERGE (e)-[:MENTIONED_IN]->(c)
                """.format(entity_type=entity_type)
                
                self.graph.query(cypher, params={
                    "name": entity_name,
                    "description": entity_desc,
                    "source_chunks": source_chunks if source_chunks else []
                })
                
                entity_count += 1
            
            logger.info(f"  ✓ Created/Updated {entity_count} Entity nodes")
            
            # Step 3: 創建 Relationship
            rel_count = 0
            for rel in relationships:
                source = rel.get('source')
                target = rel.get('target')
                rel_type = rel.get('relation_type', 'RELATED_TO')
                description = rel.get('description', '')
                strength = rel.get('strength', 5)
                
                if not source or not target:
                    continue
                
                # 動態構建 Cypher（因為 relation_type 是變數）
                cypher = """
                MATCH (s {{name: $source}})
                MATCH (t {{name: $target}})
                MERGE (s)-[r:{rel_type}]->(t)
                SET r.description = $description,
                    r.strength = $strength,
                    r.updated_at = datetime()
                """.format(rel_type=rel_type)
                
                try:
                    self.graph.query(cypher, params={
                        "source": source,
                        "target": target,
                        "description": description,
                        "strength": strength
                    })
                    rel_count += 1
                except Exception as e:
                    logger.warning(f"Failed to create relationship {source}-[{rel_type}]->{target}: {e}")
            
            logger.info(f"  ✓ Created {rel_count} Relationships")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add chunk entities: {e}", exc_info=True)
            return False