# test_neo4j.py
import logging
from langchain_ollama import OllamaLLM
from system_api.graph_manager import GraphManager

# 設定 logging
logging.basicConfig(level=logging.INFO)

def test_connection():
    print("正在測試連接 Neo4j...")
    
    # 初始化一個簡單的 LLM (這裡只是為了滿足參數需求，不會真的調用)
    llm = OllamaLLM(model="llama3:latest") # 或您使用的任何模型
    
    # 初始化 GraphManager
    gm = GraphManager(llm=llm)
    
    if gm.graph:
        print("✅ 連接成功！GraphManager 已就緒。")
        # 測試簡單寫入
        print("測試寫入一筆假資料...")
        gm.add_paper_metadata(
            paper_id="test_001",
            title="Test Paper for GraphRAG",
            year="2024",
            research_goal="Verify Neo4j Connection",
            methods=["Testing", "Debugging"],
            datasets=["Dummy Dataset"]
        )
        print("✅ 寫入測試完成。請去 Neo4j Browser 確認是否看到節點。")
    else:
        print("❌ 連接失敗，請檢查 Docker 是否運行以及密碼是否正確。")

if __name__ == "__main__":
    test_connection()