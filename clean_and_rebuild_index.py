#!/usr/bin/env python3
"""
清理並重建索引

解決舊索引缺少 pdf_path metadata 的問題
"""

import os
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_and_rebuild():
    """清理舊索引並準備重建"""
    
    print("="*80)
    print("清理舊索引（準備重建）")
    print("="*80)
    
    vectorstore_path = "./vectorstore"
    backup_path = "./vectorstore_backup_before_rebuild"
    
    # 1. 備份現有索引
    if os.path.exists(vectorstore_path):
        print(f"\n1️⃣ 備份現有索引到: {backup_path}")
        if os.path.exists(backup_path):
            print(f"   移除舊備份...")
            shutil.rmtree(backup_path)
        shutil.copytree(vectorstore_path, backup_path)
        print(f"   ✅ 備份完成")
    
    # 2. 刪除現有索引
    print(f"\n2️⃣ 刪除現有索引...")
    layer1_path = os.path.join(vectorstore_path, "layer1")
    layer2_path = os.path.join(vectorstore_path, "layer2")
    
    if os.path.exists(layer1_path):
        shutil.rmtree(layer1_path)
        print(f"   ✅ Layer 1 已刪除")
    
    if os.path.exists(layer2_path):
        shutil.rmtree(layer2_path)
        print(f"   ✅ Layer 2 已刪除")
    
    # 3. 清理 backups
    backups_path = os.path.join(vectorstore_path, "backups")
    if os.path.exists(backups_path):
        shutil.rmtree(backups_path)
        print(f"   ✅ Backups 已清理")
    
    print("\n" + "="*80)
    print("✅ 清理完成！")
    print("="*80)
    print("\n下一步: 執行 'python agent2.py' 將自動重建索引")
    print("所有論文將使用正確的 metadata（包含 pdf_path）")
    print("\n如需恢復舊索引，請執行:")
    print(f"  rm -rf {vectorstore_path}")
    print(f"  mv {backup_path} {vectorstore_path}")
    print("="*80)


if __name__ == "__main__":
    try:
        clean_and_rebuild()
    except Exception as e:
        logger.error(f"清理失敗: {e}", exc_info=True)
        exit(1)
