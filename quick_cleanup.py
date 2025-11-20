"""
快速清理索引並重新啟動測試

這個腳本會：
1. 備份現有索引
2. 清空 Layer1 和 Layer2
3. 顯示重新啟動指令
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

def backup_and_clean():
    """備份並清理索引"""
    vectorstore_path = "./vectorstore"
    
    # 1. 備份
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{vectorstore_path}_backup_{timestamp}"
    
    if os.path.exists(vectorstore_path):
        print(f"📦 Creating backup: {backup_path}")
        shutil.copytree(vectorstore_path, backup_path)
        print("✓ Backup created")
    
    # 2. 清理
    layer1_path = Path(vectorstore_path) / "layer1"
    layer2_path = Path(vectorstore_path) / "layer2"
    
    files_to_remove = [
        layer1_path / "abstract.faiss",
        layer1_path / "abstract.pkl",
        layer1_path / "metadata.pkl",
        layer2_path / "chunks.jsonl"
    ]
    
    print("\n🧹 Cleaning indices...")
    for file_path in files_to_remove:
        if file_path.exists():
            file_path.unlink()
            print(f"  ✓ Removed: {file_path.name}")
    
    print("\n✅ Cleanup complete!")
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("\n1️⃣  重新啟動 agent2.py（會自動重建索引並執行 Graph RAG）：")
    print("   python agent2.py")
    print("\n2️⃣  等待索引建立完成後，執行驗證腳本：")
    print("   python verify_graph_indexing.py")
    print("\n3️⃣  檢查結果應該顯示：")
    print("   - Layer1: 5 papers")
    print("   - Layer2: ~87 chunks")
    print("   - Neo4j: 5 papers (with bilingual domains)")
    print("\n" + "="*80)

if __name__ == '__main__':
    print("\n" + "="*80)
    print("索引清理腳本")
    print("="*80)
    print("\n⚠️  這會清空現有的索引（會先備份）")
    
    response = input("\n確定要繼續嗎？ (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        backup_and_clean()
    else:
        print("❌ 操作已取消")
