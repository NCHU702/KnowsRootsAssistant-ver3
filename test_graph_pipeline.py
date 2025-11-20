import logging
import sys
import os
import time
import json
from pathlib import Path
from langchain_ollama import OllamaLLM
import PyPDF2

# 引入系統模組
from system_api.graph_manager import GraphManager
from system_api.graph_extractor import GraphDataExtractor

# 設定 Logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str, max_pages: int = 5) -> str:
    """提取 PDF 文字 (有錯誤處理)"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            pages_to_read = min(max_pages, total_pages)
            text = ""
            for page_num in range(pages_to_read):
                page = pdf_reader.pages[page_num]
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
    except Exception as e:
        logger.error(f"PDF 讀取失敗 {pdf_path}: {e}")
        return ""

def get_test_pdfs(data_dir: str = "./data", limit: int = 3) -> list:
    """獲取測試用的 PDF 列表"""
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"目錄不存在: {data_dir}")
        return []
    
    pdf_files = list(data_path.glob("*.pdf"))
    selected_files = pdf_files[:limit]
    logger.info(f"已選擇 {len(selected_files)} 篇論文進行測試: {[f.name for f in selected_files]}")
    return selected_files

def verify_neo4j_write(graph_manager, paper_id):
    """
    反查 Neo4j 確認寫入結果
    """
    cypher = f"""
    MATCH (p:Paper {{paper_id: '{paper_id}'}})
    OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
    OPTIONAL MATCH (p)-[:EVALUATED_ON]->(d:Dataset)
    OPTIONAL MATCH (p)-[:AIMS_TO]->(g:ResearchGoal)
    RETURN p.title as title, count(DISTINCT m) as method_count, count(DISTINCT d) as dataset_count, count(DISTINCT g) as goal_count
    """
    try:
        result = graph_manager.graph.query(cypher)
        if result:
            return result[0]
        return None
    except Exception as e:
        logger.error(f"驗證查詢失敗: {e}")
        return None

def run_batch_test():
    print("\n" + "="*60)
    print("🚀 開始 Graph RAG 批量深度測試 (Batch Advanced Test)")
    print("="*60)

    # 1. 初始化
    model_name = "jcai/llama-3-taiwan-8b-instruct:q4_k_m"
    print(f"🔧 初始化 LLM ({model_name})...")
    llm = OllamaLLM(model=model_name, temperature=0, num_ctx=8192) # 增加 context window 以防截斷

    print("🔧 初始化 GraphManager 和 Extractor...")
    graph_manager = GraphManager(llm=llm)
    extractor = GraphDataExtractor(llm=llm)

    if not graph_manager.graph:
        print("❌ Neo4j 連接失敗，測試中止。")
        return

    # 2. 獲取 PDF
    pdf_files = get_test_pdfs("./data", limit=35)
    if not pdf_files:
        print("❌ 沒有找到 PDF 檔案。")
        return

    results_summary = []

    # 3. 迴圈處理
    for pdf_file in pdf_files:
        print(f"\n{'='*40}")
        print(f"📄 正在處理: {pdf_file.name}")
        print(f"{'='*40}")
        
        start_time = time.time()
        step_times = {}

        try:
            # Step A: Text Extraction
            t0 = time.time()
            paper_text = extract_text_from_pdf(str(pdf_file))
            step_times['pdf_read'] = time.time() - t0
            
            if len(paper_text) < 100:
                logger.warning(f"⚠️ 跳過 {pdf_file.name}: 提取內容過少")
                results_summary.append({"file": pdf_file.name, "status": "Skipped (Empty Text)"})
                continue

            # Step B: LLM Extraction
            print("  🧠 LLM 分析中 (提取 Goal, Methods, Datasets)...")
            t0 = time.time()
            extracted_data = extractor.extract(paper_text)
            step_times['llm_extract'] = time.time() - t0
            
            # 打印提取結果供檢查
            print(f"    -> Goal: {extracted_data.get('research_goal')[:50]}...")
            print(f"    -> Methods ({len(extracted_data.get('methods', []))}): {extracted_data.get('methods')}")
            print(f"    -> Datasets ({len(extracted_data.get('datasets', []))}): {extracted_data.get('datasets')}")

            # Step C: Neo4j Ingestion
            paper_id = pdf_file.stem.replace(" ", "_") # 簡單的 ID 生成
            paper_title = pdf_file.stem
            
            print("  💾 寫入 Neo4j 圖資料庫...")
            t0 = time.time()
            success = graph_manager.add_paper_metadata(
                paper_id=paper_id,
                title=paper_title,
                year="2024", # 模擬數據
                research_goal=extracted_data.get('research_goal', ''),
                methods=extracted_data.get('methods', []),
                datasets=extracted_data.get('datasets', [])
            )
            step_times['db_write'] = time.time() - t0

            # Step D: Verification
            verification = verify_neo4j_write(graph_manager, paper_id)
            
            total_time = time.time() - start_time
            
            if success and verification:
                print(f"  ✅ 成功! 耗時: {total_time:.2f}s")
                print(f"  📊 資料庫驗證: Methods={verification['method_count']}, Datasets={verification['dataset_count']}, Goal={verification['goal_count']}")
                results_summary.append({
                    "file": pdf_file.name,
                    "status": "Success",
                    "methods": verification['method_count'],
                    "time": f"{total_time:.2f}s"
                })
            else:
                print("  ❌ 寫入或驗證失敗")
                results_summary.append({"file": pdf_file.name, "status": "Failed"})

        except Exception as e:
            logger.error(f"處理 {pdf_file.name} 時發生錯誤: {e}")
            results_summary.append({"file": pdf_file.name, "status": f"Error: {str(e)[:30]}"})

    # 4. 最終報告
    print("\n" + "="*60)
    print("📑 測試總結報告")
    print("="*60)
    print(f"{'File Name':<40} | {'Status':<15} | {'Methods':<8} | {'Time'}")
    print("-" * 80)
    for res in results_summary:
        print(f"{res['file'][:40]:<40} | {res['status']:<15} | {res.get('methods', 0):<8} | {res.get('time', '-')}")
    print("="*60)

if __name__ == "__main__":
    run_batch_test()