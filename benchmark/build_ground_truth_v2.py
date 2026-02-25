"""
build_ground_truth_v2.py - 分層抽象化基準查詢 (v2)
=================================================================

15 個查詢，依抽象化程度分三組：
  - Low  (L01–L05): 直接匹配 Graph 中的精確節點/邊
  - Mid  (M01–M05): 語義同義詞、隱含推理、需要理解概念等價
  - High (H01–H05): 多跳推理、跨域聯想、功能性/概念性提問

Ground Truth 從 Neo4j 動態查詢生成，並附上人工判斷依據。
"""
import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR, "benchmark", "ground_truth_v2.json")


def neo4j_session():
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    return driver


def query_paper_ids(session, cypher: str):
    """Execute Cypher and return sorted list of paper_ids."""
    result = session.run(cypher)
    return sorted({r[0] for r in result if r[0]})


def main():
    driver = neo4j_session()

    with driver.session() as session:
        # ── Helper functions ──
        def domain_papers(domain_name):
            return query_paper_ids(session,
                f"MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain {{name: '{domain_name}'}}) RETURN p.paper_id")

        def method_papers(method_name):
            return query_paper_ids(session,
                f"MATCH (p:Paper)-[:USES_METHOD]->(m:Method {{name: '{method_name}'}}) RETURN p.paper_id")

        def metric_papers(metric_name):
            return query_paper_ids(session,
                f"MATCH (p:Paper)-[:EVALUATED_WITH]->(m:Metric {{name: '{metric_name}'}}) RETURN p.paper_id")

        def dataset_papers(dataset_name):
            return query_paper_ids(session,
                f"MATCH (p:Paper)-[:EVALUATED_ON]->(d:Dataset {{name: '{dataset_name}'}}) RETURN p.paper_id")

        def intersect(*lists):
            result = set(lists[0])
            for lst in lists[1:]:
                result &= set(lst)
            return sorted(result)

        def union(*lists):
            result = set()
            for lst in lists:
                result |= set(lst)
            return sorted(result)

        total_papers = session.run("MATCH (n:Paper) RETURN count(n)").single()[0]

        # ================================================================
        # LOW ABSTRACTION (L01–L05): 直接匹配 Graph 邊/節點
        # 問題的關鍵詞直接對應 Graph 中的 node name
        # Graph Oracle: 一條 Cypher 即可精確檢索
        # RAG: 如果摘要包含該關鍵詞，也能找到
        # ================================================================

        # L01: 直接 domain match
        L01 = {
            "query_id": "L01",
            "query_text": "哪些論文屬於農業領域？",
            "abstraction_level": "low",
            "query_type": "domain",
            "cypher": "MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain {name: 'Agriculture'}) RETURN p.paper_id",
            "ground_truth_paper_ids": domain_papers("Agriculture"),
            "reasoning": "直接匹配 Domain 節點 'Agriculture'，無需語義推理",
            "graph_advantage": "單邊精確查詢",
            "rag_challenge": "摘要中應該包含農業/芒果等相關詞彙，RAG 有機會找到"
        }

        # L02: 直接 method match — GAN
        L02 = {
            "query_id": "L02",
            "query_text": "哪些論文使用GAN（生成對抗網路）？",
            "abstraction_level": "low",
            "query_type": "method",
            "cypher": "MATCH (p:Paper)-[:USES_METHOD]->(m:Method {name: 'GAN'}) RETURN p.paper_id",
            "ground_truth_paper_ids": method_papers("GAN"),
            "reasoning": "直接匹配 Method 節點 'GAN'",
            "graph_advantage": "單邊精確查詢",
            "rag_challenge": "摘要中可能包含「生成對抗網路」或「GAN」字樣，RAG 可搜到"
        }

        # L03: 直接 metric match — ACCURACY
        L03 = {
            "query_id": "L03",
            "query_text": "哪些論文使用準確率(Accuracy)作為評估指標？",
            "abstraction_level": "low",
            "query_type": "metric",
            "cypher": "MATCH (p:Paper)-[:EVALUATED_WITH]->(m:Metric {name: 'ACCURACY'}) RETURN p.paper_id",
            "ground_truth_paper_ids": metric_papers("ACCURACY"),
            "reasoning": "直接匹配 Metric 節點 'ACCURACY'",
            "graph_advantage": "單邊精確查詢",
            "rag_challenge": "摘要中通常會提到準確率指標，RAG 有機會找到"
        }

        # L04: 直接 composite (domain + method)
        L04 = {
            "query_id": "L04",
            "query_text": "哪些智慧製造的論文使用CNN？",
            "abstraction_level": "low",
            "query_type": "composite",
            "cypher": "MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain {name: 'Smart Manufacturing'}) "
                      "WHERE (p)-[:USES_METHOD]->(:Method {name: 'CNN'}) RETURN p.paper_id",
            "ground_truth_paper_ids": intersect(
                domain_papers("Smart Manufacturing"), method_papers("CNN")
            ),
            "reasoning": "Smart Manufacturing ∩ CNN — 兩個精確邊的交集",
            "graph_advantage": "雙邊交集，Graph 直接處理",
            "rag_challenge": "RAG 無法做精確交集運算"
        }

        # L05: 直接 dataset match
        L05 = {
            "query_id": "L05",
            "query_text": "哪些論文使用了YouBike資料集？",
            "abstraction_level": "low",
            "query_type": "dataset",
            "cypher": "MATCH (p:Paper)-[:EVALUATED_ON]->(d:Dataset {name: 'YouBike Data'}) RETURN p.paper_id",
            "ground_truth_paper_ids": dataset_papers("YouBike Data"),
            "reasoning": "直接匹配 Dataset 節點 'YouBike Data'",
            "graph_advantage": "單邊精確查詢",
            "rag_challenge": "摘要中應該提及 YouBike，RAG 有機會找到"
        }

        # ================================================================
        # MEDIUM ABSTRACTION (M01–M05): 語義同義詞 / 概念等價 / 隱含推理
        # 問題使用的詞彙與 Graph 節點名稱不同，但語義等價
        # Graph Oracle: 需要知道同義詞映射才能寫出 Cypher
        # RAG: 語義搜尋可能佔優勢（如果 embedding 能捕捉語義）
        # ================================================================

        # M01: "影像辨識" → 需理解 YOLO, CNN, Mask R-CNN 等都是影像辨識方法
        #       且論文主題是辨識影像中的物件
        # GT: 鼻咽癌腫塊辨識(YOLO+CNN), 芒果分類辨識(Mask R-CNN), 車道偵測(CycleGAN)
        #     公車軌跡辨識(CNN) — 這些都是「影像辨識」任務
        M01_ids = union(
            # 鼻咽癌 — 內視鏡影像辨識
            ["基於深度學習之鼻咽癌腫塊辨識_20251106_144224", "基礎3_基於深度學習之鼻咽癌腫塊辨識"],
            # 芒果分類 — 影像辨識
            ["基礎5_應用集成式深度學習模型進行芒果分類辨識"],
            # 車道偵測 — 影像辨識
            ["標準15_基於CycleGAN和特徵融合於挑戰性場景的車道偵測"],
            # 澳門公車軌跡辨識 — GPS + 影像
            ["標準1_基於卷積類神經網路之澳門公車軌跡辨識"],
        )
        M01 = {
            "query_id": "M01",
            "query_text": "哪些論文的研究主題涉及影像辨識？",
            "abstraction_level": "medium",
            "query_type": "semantic_concept",
            "cypher": None,  # 無法用單一 Cypher 表達
            "cypher_approx": (
                "MATCH (p:Paper)-[:USES_METHOD]->(m:Method) "
                "WHERE m.name IN ['YOLO', 'CNN', 'Mask R-CNN', 'CycleGAN'] "
                "AND (p.paper_id CONTAINS '辨識' OR p.paper_id CONTAINS '偵測') "
                "RETURN DISTINCT p.paper_id"
            ),
            "ground_truth_paper_ids": M01_ids,
            "reasoning": "「影像辨識」不是 Graph 中的 Domain 或 Method 節點。"
                         "需要理解：鼻咽癌腫塊「辨識」、芒果「分類辨識」、車道「偵測」、"
                         "公車軌跡「辨識」都屬於影像辨識任務",
            "graph_advantage": "Graph 無直接 '影像辨識' 節點，需要 LLM 推理後組合查詢",
            "rag_challenge": "RAG 語義搜尋可能透過 '辨識'、'偵測'、'影像' 等關鍵字找到相關論文"
        }

        # M02: "預測任務" → 需理解 MAE/RMSE 是回歸預測指標
        #       GT: 所有使用 MAE 或 RMSE 作為評估指標的論文
        M02_ids = union(metric_papers("MAE"), metric_papers("RMSE"))
        M02 = {
            "query_id": "M02",
            "query_text": "哪些論文在做預測類型的研究？",
            "abstraction_level": "medium",
            "query_type": "semantic_concept",
            "cypher": None,
            "cypher_approx": (
                "MATCH (p:Paper)-[:EVALUATED_WITH]->(m:Metric) "
                "WHERE m.name IN ['MAE', 'RMSE'] "
                "RETURN DISTINCT p.paper_id"
            ),
            "ground_truth_paper_ids": M02_ids,
            "reasoning": "「預測」不是直接的 Graph 節點。但使用 MAE/RMSE 指標的論文"
                         "本質上都是在做數值預測（回歸）任務。需要理解指標→任務類型的映射。"
                         "另外，部分論文標題含有「預測」二字也是提示。",
            "graph_advantage": "Graph 需要先推理出 預測→MAE/RMSE 的關聯，再查 EVALUATED_WITH 邊",
            "rag_challenge": "RAG 可能透過摘要中的「預測」一詞語義搜尋，但不一定全面"
        }

        # M03: "生成模型" → 需理解 GAN, CycleGAN, Conditional GAN, Autoencoder based on RNN
        #       都屬於「生成模型」家族
        M03_ids = union(
            method_papers("GAN"),
            method_papers("CycleGAN"),
            method_papers("Conditional GAN"),
            method_papers("Autoencoder based on RNN"),
        )
        M03 = {
            "query_id": "M03",
            "query_text": "哪些論文使用了生成式模型？",
            "abstraction_level": "medium",
            "query_type": "semantic_method_group",
            "cypher": None,
            "cypher_approx": (
                "MATCH (p:Paper)-[:USES_METHOD]->(m:Method) "
                "WHERE m.name IN ['GAN', 'CycleGAN', 'Conditional GAN', 'Autoencoder based on RNN'] "
                "RETURN DISTINCT p.paper_id"
            ),
            "ground_truth_paper_ids": M03_ids,
            "reasoning": "「生成式模型」不是單一 Method 節點。需要理解 GAN 及其變體 "
                         "(CycleGAN, Conditional GAN) 和 Autoencoder (基於RNN的自編碼器) "
                         "都屬於生成式模型家族。",
            "graph_advantage": "Graph 中有個別 Method 節點但無「生成式模型」群組概念",
            "rag_challenge": "RAG 語義搜尋「生成式」可能部分命中，但能否找齊所有變體是挑戰"
        }

        # M04: "處理時序資料的方法" → 需理解 LSTM, GRU, HRNN, STL分解法 都處理時序
        M04_ids = union(
            method_papers("LSTM"),
            method_papers("GRU"),
            method_papers("HRNN"),
            method_papers("STL分解法"),
        )
        M04 = {
            "query_id": "M04",
            "query_text": "哪些論文使用了能處理時間序列的方法？",
            "abstraction_level": "medium",
            "query_type": "semantic_method_group",
            "cypher": None,
            "cypher_approx": (
                "MATCH (p:Paper)-[:USES_METHOD]->(m:Method) "
                "WHERE m.name IN ['LSTM', 'GRU', 'HRNN', 'STL分解法'] "
                "RETURN DISTINCT p.paper_id"
            ),
            "ground_truth_paper_ids": M04_ids,
            "reasoning": "「時間序列」不是 Graph 節點。需要理解 LSTM 和 GRU 是 RNN 變體 "
                         "(天生處理序列)，HRNN 是階層式 RNN，STL分解法是時序分解方法。",
            "graph_advantage": "Graph 需要 LLM 先推理出哪些方法處理時序，再查聯合查詢",
            "rag_challenge": "RAG 可能透過 '時間'、'序列'、'預測' 等語義找到部分論文"
        }

        # M05: "人流相關研究" → 需理解 "人流" 出現在多篇標題中
        #       GT: 標準11(區域人流), 標準14(人流異常預測), 標準4(醫院病患人流),
        #           標準2(捷運進出站人數預測 — "人數"≈"人流"), 標準7(交通站點用量)
        M05_ids = [
            "標準11_應用補償式遷移學習模型於區域人流之強健性預測",
            "標準14_基於集成式生成對抗網路進行人流異常預測",
            "標準4_利用RBF-DNN配合醫院病患人流資料探究",
            "標準2_基於人流資料、土地使用分區圖以及Google趨勢資料進行捷運進出站人數預測-以台北捷運為例",
            "標準7_使用深度學習模型結合二維高斯函數探討交通站點之用量與成因",
        ]
        M05 = {
            "query_id": "M05",
            "query_text": "哪些論文的研究與人流分析有關？",
            "abstraction_level": "medium",
            "query_type": "semantic_concept",
            "cypher": None,
            "cypher_approx": (
                "MATCH (p:Paper) "
                "WHERE p.paper_id CONTAINS '人流' OR p.paper_id CONTAINS '人數' "
                "OR p.paper_id CONTAINS '站點' "
                "RETURN p.paper_id"
            ),
            "ground_truth_paper_ids": sorted(M05_ids),
            "reasoning": "「人流分析」不是 Graph 中的 Domain 或 Method。需要理解：區域人流、"
                         "人流異常預測、醫院病患人流、捷運人數預測、交通站點用量 都與人流相關。"
                         "標準7 雖然標題講「站點用量」但實質上是公共自行車租借量≈人流。",
            "graph_advantage": "Graph 無 '人流' 概念節點，需要 LLM 從 paper_id/ResearchGoal 推理",
            "rag_challenge": "RAG 語義搜尋 '人流' 可能直接命中包含該詞的摘要"
        }

        # ================================================================
        # HIGH ABSTRACTION (H01–H05): 多跳推理 / 跨域概念 / 功能性提問
        # 問題與 Graph 節點完全不對應，需要深度理解論文內容
        # ================================================================

        # H01: "智慧城市" → 跨 Domain 概念：Smart Transportation + Smart Building
        #       + Environmental Monitoring 都是智慧城市的子系統
        H01_ids = union(
            domain_papers("Smart Transportation"),
            domain_papers("Smart Building"),
            domain_papers("Environmental Monitoring"),
        )
        H01 = {
            "query_id": "H01",
            "query_text": "哪些論文的研究可以應用在智慧城市建設？",
            "abstraction_level": "high",
            "query_type": "cross_domain",
            "cypher": None,
            "cypher_approx": (
                "MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain) "
                "WHERE d.name IN ['Smart Transportation', 'Smart Building', 'Environmental Monitoring'] "
                "RETURN DISTINCT p.paper_id"
            ),
            "ground_truth_paper_ids": H01_ids,
            "reasoning": "「智慧城市」不是 Graph 中的 Domain 節點。需要理解智慧城市是一個"
                         "上位概念，涵蓋：智慧交通(10)、智慧建築(1)、環境監測(2)。"
                         "需要跨 Domain 多跳推理。",
            "graph_advantage": "Graph 需要先將 '智慧城市' 分解為多個 Domain，再做聯集",
            "rag_challenge": "RAG 可能透過「智慧」語義找到部分，但很難聯想到所有子領域"
        }

        # H02: "資料不足/資料稀缺問題" → 需理解哪些論文在處理小樣本/資料增強
        #       GT: 基礎4(小資料集), 基礎10(GAN產生模擬資料), 標準11(遷移學習)
        #       這些都是處理資料不足的策略
        H02_ids = sorted([
            "基礎4_應用混合式深度學習開發適應於小資料集之鼻咽癌分類模型",
            "基礎10_利用生成對抗網路產生模擬資料以提昇刀具磨耗預測之準確率",
            "標準11_應用補償式遷移學習模型於區域人流之強健性預測",
        ])
        H02 = {
            "query_id": "H02",
            "query_text": "哪些論文嘗試解決訓練資料不足的問題？",
            "abstraction_level": "high",
            "query_type": "problem_oriented",
            "cypher": None,
            "cypher_approx": None,
            "ground_truth_paper_ids": H02_ids,
            "reasoning": "「資料不足」不是 Graph 中的任何節點。需要深度理解："
                         "基礎4 標題明確提及「小資料集」；"
                         "基礎10 用 GAN 產生模擬資料是典型的資料增強策略；"
                         "標準11 使用遷移學習(Transfer Learning)是處理資料不足的經典方法。"
                         "三篇論文分別用不同策略處理同一個高階問題。",
            "graph_advantage": "Graph 完全無法直接處理此查詢，需要 LLM 深度理解 ResearchGoal",
            "rag_challenge": "RAG 可能找到「小資料集」但很難聯想到 GAN 資料增強和遷移學習"
        }

        # H03: "醫療影像" → 跨 Domain(Healthcare) + 語義概念(影像)
        #       需理解：Healthcare 領域中使用影像相關方法/資料的論文
        #       GT: 鼻咽癌腫塊辨識系列(使用內視鏡影像+YOLO+CNN), 鼻咽癌分類(CNN+LSTM)
        #       不包含：標準4(醫院人流，非影像)
        H03_ids = sorted([
            "基於深度學習之鼻咽癌腫塊辨識_20251106_144224",
            "基礎3_基於深度學習之鼻咽癌腫塊辨識",
            "基礎4_應用混合式深度學習開發適應於小資料集之鼻咽癌分類模型",
        ])
        H03 = {
            "query_id": "H03",
            "query_text": "哪些論文利用深度學習處理醫療影像問題？",
            "abstraction_level": "high",
            "query_type": "cross_concept",
            "cypher": None,
            "cypher_approx": (
                "MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain {name: 'Healthcare'}) "
                "WHERE (p)-[:USES_METHOD]->(:Method {name: 'CNN'}) "
                "OR (p)-[:EVALUATED_ON]->(:Dataset {name: 'Endoscopic Images'}) "
                "RETURN DISTINCT p.paper_id"
            ),
            "ground_truth_paper_ids": H03_ids,
            "reasoning": "需要交叉理解兩個概念：「醫療」(Healthcare Domain) + 「影像」(影像相關方法或資料集)。"
                         "三篇鼻咽癌論文都處理內視鏡影像(Endoscopic Images)。"
                         "標準4 雖在醫療領域但處理的是人流資料非影像，故不計入。",
            "graph_advantage": "Graph 需要組合 Domain + Dataset/Method 兩種邊的推理",
            "rag_challenge": "RAG 需同時理解「醫療」和「影像」兩個語義，且做交集"
        }

        # H04: "公共運輸" → 需跨概念理解
        #       GT: 公車(標準1澳門公車), 捷運(標準2台北捷運), 公共自行車(基礎1 YouBike, 標準7 YouBike)
        #       不包含：路網旅行時間(基礎7，偏路網非公共運輸)、天際線查詢(查詢2/3，偏演算法)
        H04_ids = sorted([
            "標準1_基於卷積類神經網路之澳門公車軌跡辨識",
            "標準2_基於人流資料、土地使用分區圖以及Google趨勢資料進行捷運進出站人數預測-以台北捷運為例",
            "基礎1_以純Youbike資料識別Covid-19對臺北市區活動的影響_20220219",
            "標準7_使用深度學習模型結合二維高斯函數探討交通站點之用量與成因",
        ])
        H04 = {
            "query_id": "H04",
            "query_text": "哪些論文研究公共運輸系統相關的問題？",
            "abstraction_level": "high",
            "query_type": "semantic_aggregation",
            "cypher": None,
            "cypher_approx": None,
            "ground_truth_paper_ids": H04_ids,
            "reasoning": "「公共運輸」不是 Graph 節點。需要理解：公車、捷運、公共自行車(YouBike)"
                         "都屬於公共運輸系統。標準1(公車軌跡)、標準2(捷運人數)、"
                         "基礎1(YouBike 數據)、標準7(YouBike 站點用量)都與公共運輸直接相關。"
                         "不包含路網旅行時間(偏向一般路網)和天際線查詢(偏向演算法理論)。",
            "graph_advantage": "Graph 無 '公共運輸' 概念，需要 LLM 從論文標題/內容推理",
            "rag_challenge": "RAG 語義搜尋「公共運輸」可能找到公車、捷運，但 YouBike 不一定"
        }

        # H05: "深度學習但非神經網路" → 哪些論文使用非傳統深度學習的方法
        #       即: 不使用 CNN/LSTM/GRU/RNN 等神經網路方法的論文
        #       GT: 使用 Fuzzy Neural Network, DBSCAN, random forest, cluster analysis, 
        #           STL分解法, Wavelet Model, Skyline Query 等非 DL 方法
        #       這是一個反向推理問題
        H05_traditional_ids = sorted([
            "基礎1_以純Youbike資料識別Covid-19對臺北市區活動的影響_20220219",  # random forest, cluster analysis
            "基礎2_基於社群網路資料之旅遊區辨識",  # DBSCAN
            "基礎7_基於模糊類神經網路的路網旅行時間預測方法",  # Fuzzy Neural Network
            "基礎8_使用模糊類神經網路預測灌溉用水分配-以濁幹線系統為例",  # Fuzzy Neural Network
        ])
        H05 = {
            "query_id": "H05",
            "query_text": "哪些論文使用的主要方法不是深度學習？",
            "abstraction_level": "high",
            "query_type": "negation_reasoning",
            "cypher": None,
            "cypher_approx": (
                "MATCH (p:Paper)-[:USES_METHOD]->(m:Method) "
                "WHERE NOT m.name IN ['CNN', 'LSTM', 'GRU', 'HRNN', 'GAN', "
                "'CycleGAN', 'Conditional GAN', 'Ensemble Learning', '集成式學習', "
                "'Deep Learning', 'YOLO', 'RBF', 'UNet', 'RBF- DNN', '3D-RCL', "
                "'Mask R-CNN', 'Autoencoder based on RNN', 'SVR', '卷積神經網路', "
                "'Pyramid Structure', 'Feature Fusion', 'Wavelet Model'] "
                "AND NOT (p)-[:USES_METHOD]->(:Method {name: 'CNN'}) "
                "AND NOT (p)-[:USES_METHOD]->(:Method {name: 'LSTM'}) "
                "AND NOT (p)-[:USES_METHOD]->(:Method {name: 'GAN'}) "
                "RETURN DISTINCT p.paper_id"
            ),
            "ground_truth_paper_ids": H05_traditional_ids,
            "reasoning": "反向推理問題：需要先理解什麼是深度學習方法，再找出「不使用」深度學習的論文。"
                         "基礎1 使用 random forest + cluster analysis（傳統 ML）；"
                         "基礎2 使用 DBSCAN（聚類演算法，非 DL）；"
                         "基礎7, 基礎8 使用 Fuzzy Neural Network（模糊系統，非典型深度學習）。"
                         "注意：模糊類神經網路雖然有「神經」字眼但屬於傳統模糊控制系統。",
            "graph_advantage": "Graph 需要先定義 DL 方法集合，再做排除，非常困難",
            "rag_challenge": "RAG 需要理解否定邏輯，這是語義搜尋的弱點"
        }

        # ── Combine all queries ──
        queries = [L01, L02, L03, L04, L05, M01, M02, M03, M04, M05, H01, H02, H03, H04, H05]

        # Add expected_count to all
        for q in queries:
            q["expected_count"] = len(q["ground_truth_paper_ids"])

        # Build output
        output = {
            "_metadata": {
                "description": "Benchmark Ground Truth v2: 分層抽象化 (Low/Medium/High × 5)",
                "version": "2.0-abstraction",
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "Live Neo4j + 人工語義判斷",
                "total_papers_in_graph": total_papers,
                "total_queries": len(queries),
                "abstraction_levels": {
                    "low": "直接匹配 Graph 邊/節點 (L01–L05)",
                    "medium": "語義同義詞、概念等價、隱含推理 (M01–M05)",
                    "high": "多跳推理、跨域聯想、功能性提問 (H01–H05)"
                },
                "notes": "Low 查詢 Ground Truth 來自 Neo4j 精確查詢；"
                         "Medium/High 查詢 Ground Truth 結合 Neo4j 查詢與人工語義判斷。"
            },
            "queries": queries
        }

    driver.close()

    # Write
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Ground Truth v2 已生成: {OUTPUT_PATH}")
    print(f"   共 {len(queries)} 組查詢, {total_papers} 篇 Paper in Graph\n")

    for level in ["low", "medium", "high"]:
        level_qs = [q for q in queries if q["abstraction_level"] == level]
        print(f"  ── {level.upper()} ({len(level_qs)} queries) ──")
        for q in level_qs:
            print(f"    {q['query_id']}: {q['query_text']} → {q['expected_count']} 篇")
        print()


if __name__ == "__main__":
    main()
