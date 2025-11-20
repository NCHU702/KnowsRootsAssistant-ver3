"""
索引清理和重建腳本

這個腳本會：
1. 備份現有索引
2. 清空 Layer1 和 Layer2
3. 提示用戶重新啟動 agent2.py 以重建索引
4. 提供驗證命令

Usage:
    python cleanup_and_rebuild.py [--skip-backup]
"""

import sys
import os
import shutil
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def backup_vectorstore(vectorstore_path: str = "./vectorstore") -> str:
    """備份現有的 vectorstore"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{vectorstore_path}_backup_{timestamp}"
    
    try:
        if os.path.exists(vectorstore_path):
            logger.info(f"Creating backup: {backup_path}")
            shutil.copytree(vectorstore_path, backup_path)
            logger.info(f"✅ Backup created successfully")
            return backup_path
        else:
            logger.warning(f"Vectorstore path not found: {vectorstore_path}")
            return None
    except Exception as e:
        logger.error(f"❌ Backup failed: {e}")
        return None

def clean_indices(vectorstore_path: str = "./vectorstore"):
    """清空 Layer1 和 Layer2 索引（保留目錄結構）"""
    layer1_path = Path(vectorstore_path) / "layer1"
    layer2_path = Path(vectorstore_path) / "layer2"
    
    files_to_remove = [
        layer1_path / "abstract.faiss",
        layer1_path / "abstract.pkl",
        layer1_path / "metadata.pkl",
        layer2_path / "chunks.jsonl"
    ]
    
    removed_count = 0
    for file_path in files_to_remove:
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"  ✓ Removed: {file_path}")
                removed_count += 1
            except Exception as e:
                logger.error(f"  ✗ Failed to remove {file_path}: {e}")
        else:
            logger.debug(f"  - File not found (skipped): {file_path}")
    
    logger.info(f"✅ Cleaned {removed_count} index files")
    
    # 確保目錄存在
    layer1_path.mkdir(parents=True, exist_ok=True)
    layer2_path.mkdir(parents=True, exist_ok=True)
    
    return removed_count

def verify_cleanup(vectorstore_path: str = "./vectorstore"):
    """驗證清理結果"""
    layer1_path = Path(vectorstore_path) / "layer1"
    layer2_path = Path(vectorstore_path) / "layer2"
    
    remaining_files = []
    for path in [layer1_path, layer2_path]:
        if path.exists():
            for file in path.iterdir():
                if file.is_file():
                    remaining_files.append(str(file))
    
    if remaining_files:
        logger.warning(f"⚠️  Some files still remain:")
        for file in remaining_files:
            logger.warning(f"  - {file}")
    else:
        logger.info("✅ All index files cleaned successfully")
    
    return len(remaining_files) == 0

def print_next_steps():
    """列印後續步驟"""
    print("\n" + "="*80)
    print("🎉 清理完成！")
    print("="*80)
    print("\n📋 下一步操作：")
    print("\n1️⃣  重新啟動 agent2.py（會自動重建索引）：")
    print("   .venv/bin/python agent2.py")
    print("\n2️⃣  等待索引重建完成，觀察日誌確認：")
    print("   - 應該顯示 \"Detected N new PDF(s) - rebuilding indices\"")
    print("   - 最後顯示 \"Total Papers: 5\" (不是 7)")
    print("\n3️⃣  執行 batch_index_to_graph.py 將論文寫入 Neo4j：")
    print("   .venv/bin/python batch_index_to_graph.py --data-dir ./test_data")
    print("\n4️⃣  驗證一致性：")
    print("   .venv/bin/python reconcile_indices.py")
    print("   應該顯示所有數量都是 5")
    print("\n5️⃣  測試查詢：")
    print("   .venv/bin/python test_graph_query.py")
    print("\n" + "="*80)
    print("\n💡 提示：")
    print("   - 如果需要恢復備份，執行：")
    print("     rm -rf ./vectorstore")
    print("     cp -r ./vectorstore_backup_XXXXXX ./vectorstore")
    print("="*80 + "\n")

def main():
    """主程式"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean and prepare for index rebuild')
    parser.add_argument('--skip-backup', action='store_true', help='Skip backup step (not recommended)')
    parser.add_argument('--force', action='store_true', help='Force cleanup without confirmation')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("索引清理和重建腳本")
    print("="*80)
    print("\n⚠️  警告：此操作將清空現有的 Layer1 和 Layer2 索引！")
    print("   系統將需要重新索引所有論文。")
    print("\n當前 PDF 數量：")
    
    # Count PDFs
    test_data_path = Path("./test_data")
    if test_data_path.exists():
        pdf_count = len(list(test_data_path.glob("*.pdf")))
        print(f"   ./test_data: {pdf_count} PDFs")
    else:
        print("   ./test_data: 目錄不存在")
    
    if not args.force:
        print("\n")
        response = input("確定要繼續嗎？ (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ 操作已取消")
            return
    
    print("\n開始清理流程...\n")
    
    # Step 1: Backup
    backup_path = None
    if not args.skip_backup:
        logger.info("Step 1: Creating backup...")
        backup_path = backup_vectorstore()
        if not backup_path:
            logger.error("Backup failed. Aborting.")
            return
    else:
        logger.warning("Step 1: SKIPPED (--skip-backup)")
    
    # Step 2: Clean
    logger.info("Step 2: Cleaning indices...")
    removed = clean_indices()
    
    # Step 3: Verify
    logger.info("Step 3: Verifying cleanup...")
    clean = verify_cleanup()
    
    if not clean:
        logger.error("❌ Cleanup verification failed. Some files could not be removed.")
        if backup_path:
            logger.info(f"💾 Backup is available at: {backup_path}")
        return
    
    # Step 4: Print instructions
    print_next_steps()
    
    if backup_path:
        logger.info(f"💾 Backup saved at: {backup_path}")

if __name__ == '__main__':
    main()
