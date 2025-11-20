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
            "CREATE INDEX paper_title_index IF NOT EXISTS FOR (p:Paper) ON (p.title)",
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
        metrics: List[str]   # [新增]
    ) -> bool:
        """
        將論文的結構化資訊寫入圖資料庫 (包含 Domain 和 Metrics)
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

        // 4. Domain [新增]
        MERGE (dom:Domain {name: $domain})
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
            
            params = {
                "paper_id": paper_id,
                "title": title,
                "year": str(year) if year else "Unknown",
                "research_goal": research_goal if research_goal else "Unknown Goal",
                "methods": clean_methods,
                "datasets": clean_datasets,
                "domain": clean_domain,
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
        將自然語言轉換為 Cypher 查詢
        """
        if not self.graph:
            return "Error: Graph database not connected."

        cypher_generation_template = """
        Task: Generate a Cypher statement to query the graph database.
        
        Instructions:
        1. Use ONLY the provided schema.
        2. Use case-insensitive matching (e.g. toLower(n.name) CONTAINS toLower("keyword")).
        3. Return specific values (e.g. p.title, m.name), not just nodes.
        4. Do NOT include explanations.
        
        Schema:
          Nodes: 
            - Paper (paper_id, title, year)
            - ResearchGoal (summary)
            - Method (name)
            - Dataset (name)
            - Domain (name)
            - Metric (name)
          Relationships:
            - (:Paper)-[:AIMS_TO]->(:ResearchGoal)
            - (:Paper)-[:USES_METHOD]->(:Method)
            - (:Paper)-[:EVALUATED_ON]->(:Dataset)
            - (:Paper)-[:APPLIED_IN]->(:Domain)
            - (:Paper)-[:EVALUATED_WITH]->(:Metric)
            
        Examples:
        Q: "Which papers use LSTM in Healthcare?"
        A: MATCH (p:Paper)-[:USES_METHOD]->(m:Method), (p)-[:APPLIED_IN]->(dom:Domain) 
           WHERE toLower(m.name) CONTAINS "lstm" AND toLower(dom.name) CONTAINS "healthcare" 
           RETURN p.title
        
        Q: "List papers evaluated with RMSE."
        A: MATCH (p:Paper)-[:EVALUATED_WITH]->(met:Metric) 
           WHERE toLower(met.name) CONTAINS "rmse" 
           RETURN p.title
        
        The question is:
        {question}
        """
        
        cypher_prompt = PromptTemplate(
            input_variables=["question"], 
            template=cypher_generation_template
        )

        try:
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