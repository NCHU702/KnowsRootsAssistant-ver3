"""
Graph Manager Module

Manages interactions with Neo4j Graph Database.
Handles schema definition, data ingestion, and natural language querying (Text2Cypher).
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

class GraphManager:
    """
    管理 Neo4j 圖資料庫的連接與操作
    
    Schema 定義 (根據您的需求客製化):
    - Nodes: 
        - Paper (屬性: paper_id, title, year)
        - ResearchGoal (屬性: summary) - 論文的主要目標
        - Method (屬性: name) - 使用的方法
        - Dataset (屬性: name) - 使用的數據集
    - Relationships:
        - (:Paper)-[:AIMS_TO]->(:ResearchGoal)
        - (:Paper)-[:USES_METHOD]->(:Method)
        - (:Paper)-[:EVALUATED_ON]->(:Dataset)
    """
    
    def __init__(
        self, 
        llm: OllamaLLM, 
        url: str = "bolt://localhost:7687", 
        username: str = "neo4j", 
        password: str = "password"
    ):
        """
        初始化 Graph Manager
        
        Args:
            llm: 用於 Text2Cypher 的 LLM 實例 (Ollama)
            url: Neo4j Bolt URL
            username: 資料庫使用者名稱
            password: 資料庫密碼
        """
        self.llm = llm
        self.url = url
        self.username = username
        self.password = password
        self.graph = None
        
        try:
            # 初始化 LangChain 的 Neo4jGraph 包裝器
            self.graph = Neo4jGraph(
                url=url, 
                username=username, 
                password=password
            )
            
            # 建立索引與約束 (Schema Initialization)
            self._init_schema()
            
            logger.info("✓ GraphManager initialized and connected to Neo4j")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            logger.warning("Graph capabilities will be disabled.")

    def _init_schema(self):
        """建立索引以加速查詢並確保數據唯一性"""
        if not self.graph: return
        
        # 定義 Cypher 語句來建立約束 (Constraints)
        # 注意: 若已存在不會報錯
        schema_queries = [
            # 確保 paper_id 唯一
            "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
            
            # 為常用搜尋欄位建立索引 (加速搜尋)
            "CREATE INDEX paper_title_index IF NOT EXISTS FOR (p:Paper) ON (p.title)",
            "CREATE INDEX method_name_index IF NOT EXISTS FOR (m:Method) ON (m.name)",
            "CREATE INDEX dataset_name_index IF NOT EXISTS FOR (d:Dataset) ON (d.name)"
        ]
        
        try:
            for query in schema_queries:
                self.graph.query(query)
            
            # 刷新 Schema 讓 LangChain 知道最新的結構
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
        datasets: List[str]
    ) -> bool:
        """
        將論文的結構化資訊寫入圖資料庫
        
        Args:
            paper_id: 論文唯一 ID
            title: 標題
            year: 年份
            research_goal: 研究目標摘要
            methods: 方法列表
            datasets: 數據集列表
        """
        if not self.graph:
            logger.error("Graph not initialized, skipping write")
            return False

        # Cypher 查詢：使用 MERGE 避免重複建立節點
        # 邏輯：找到或建立 Paper，然後建立它與 Goal, Method, Dataset 的關係
        cypher = """
        MERGE (p:Paper {paper_id: $paper_id})
        SET p.title = $title, 
            p.year = $year,
            p.last_updated = datetime()

        // 處理研究目標 (AIMS_TO) - 每個論文通常有一個主要目標
        MERGE (g:ResearchGoal {summary: $research_goal})
        MERGE (p)-[:AIMS_TO]->(g)

        // 處理方法 (USES_METHOD) - 遍歷列表
        FOREACH (m_name IN $methods | 
            MERGE (m:Method {name: m_name})
            MERGE (p)-[:USES_METHOD]->(m)
        )

        // 處理數據集 (EVALUATED_ON) - 遍歷列表
        FOREACH (d_name IN $datasets | 
            MERGE (d:Dataset {name: d_name})
            MERGE (p)-[:EVALUATED_ON]->(d)
        )
        """
        
        try:
            # 清理輸入數據 (去除空白)
            clean_methods = [m.strip() for m in methods if m and m.strip()]
            clean_datasets = [d.strip() for d in datasets if d and d.strip()]
            
            params = {
                "paper_id": paper_id,
                "title": title,
                "year": str(year) if year else "Unknown",
                "research_goal": research_goal if research_goal else "Unknown Goal",
                "methods": clean_methods,
                "datasets": clean_datasets
            }
            
            self.graph.query(cypher, params=params)
            logger.info(f"✓ Added graph metadata for: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write to graph: {e}")
            return False

    def query_graph(self, user_query: str) -> str:
        """
        將自然語言轉換為 Cypher 查詢並執行 (GraphRAG)
        """
        if not self.graph:
            return "Error: Graph database not connected."

        # 定義 Prompt，嚴格限制 LLM 使用正確的 Schema
        # 這對於 Ollama 模型特別重要，避免它產生幻覺 Schema
        cypher_generation_template = """
        Task: Generate a Cypher statement to query the graph database.
        
        Instructions:
        1. Use ONLY the provided schema. Do not assume other properties exist.
        2. Use case-insensitive matching (e.g. toLower(n.name) CONTAINS toLower("keyword")).
        3. Return the actual answer values (e.g. p.title), not just nodes.
        4. Do NOT include any explanations, markdown, or backticks. Just the Cypher query.
        
        Schema:
          Nodes: 
            - Paper (paper_id, title, year)
            - ResearchGoal (summary)
            - Method (name)
            - Dataset (name)
          Relationships:
            - (:Paper)-[:AIMS_TO]->(:ResearchGoal)
            - (:Paper)-[:USES_METHOD]->(:Method)
            - (:Paper)-[:EVALUATED_ON]->(:Dataset)
            
        Examples:
        Q: "Which papers use LSTM?"
        A: MATCH (p:Paper)-[:USES_METHOD]->(m:Method) WHERE toLower(m.name) CONTAINS "lstm" RETURN p.title
        
        Q: "Find papers that evaluate on COCO dataset."
        A: MATCH (p:Paper)-[:EVALUATED_ON]->(d:Dataset) WHERE toLower(d.name) CONTAINS "coco" RETURN p.title
        
        The question is:
        {question}
        """
        
        cypher_prompt = PromptTemplate(
            input_variables=["question"], 
            template=cypher_generation_template
        )

        try:
            # 使用 LangChain 的 GraphCypherQAChain
            chain = GraphCypherQAChain.from_llm(
                self.llm,
                graph=self.graph,
                verbose=True,
                cypher_prompt=cypher_prompt,
                allow_dangerous_requests=True
            )
            
            result = chain.invoke(user_query)
            return result['result']
            
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return f"無法執行圖譜查詢: {str(e)}"

    def get_visualization_data(self, limit: int = 100) -> Dict[str, Any]:
        """
        獲取用於前端視覺化的數據 (Nodes & Edges)
        回傳 Cytoscape.js 相容或通用的 JSON 格式
        """
        if not self.graph: return {}

        # 查詢圖譜中的節點與關係
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
                """Helper to format node data"""
                str_id = str(node_id)
                if str_id in seen_node_ids:
                    return
                
                # 決定顯示的 Label (優先順序: Title > Name > Summary)
                label_text = props.get('title') or props.get('name') or props.get('summary') or "Unknown"
                
                # 截斷太長的文字供顯示
                display_label = label_text[:20] + "..." if len(label_text) > 20 else label_text
                
                node_type = labels[0] if labels else "Node"
                
                nodes.append({
                    "data": {
                        "id": str_id,
                        "label": display_label,
                        "full_label": label_text,
                        "type": node_type,
                        # 可以加入更多顏色或樣式屬性
                    }
                })
                seen_node_ids.add(str_id)

            for row in data:
                # 處理來源節點
                process_node(row['source_id'], row['source_labels'], row['source_props'])
                # 處理目標節點
                process_node(row['target_id'], row['target_labels'], row['target_props'])
                
                # 處理邊 (Edge)
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