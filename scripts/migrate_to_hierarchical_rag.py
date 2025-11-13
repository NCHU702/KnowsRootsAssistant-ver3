#!/usr/bin/env python3
"""
索引遷移腳本：從 AcademicRAGSystem 遷移到 HierarchicalRAGSystem

功能：
1. 備份現有 vectorstore
2. 重新處理所有 PDF，建立雙層索引
3. 驗證遷移結果
4. 提供回滾選項

使用方式：
    python scripts/migrate_to_hierarchical_rag.py --backup      # 僅備份
    python scripts/migrate_to_hierarchical_rag.py --dry-run     # 模擬運行
    python scripts/migrate_to_hierarchical_rag.py --execute     # 執行遷移
    python scripts/migrate_to_hierarchical_rag.py --verify      # 驗證結果
    python scripts/migrate_to_hierarchical_rag.py --rollback    # 回滾到備份
"""

import argparse
import os
import shutil
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system_api.hierarchical_rag_system import HierarchicalRAGSystem

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGMigrationTool:
    """RAG 索引遷移工具"""
    
    def __init__(
        self,
        pdf_directory: str = "./data",
        old_vectorstore_path: str = "./vectorstore",
        new_vectorstore_path: str = "./vectorstore_hierarchical",
        backup_path: str = "./vectorstore_backup",
        model_name: str = "gemma3:12b",
        embedding_model: str = "embeddinggemma:latest"
    ):
        self.pdf_directory = pdf_directory
        self.old_vectorstore_path = old_vectorstore_path
        self.new_vectorstore_path = new_vectorstore_path
        self.backup_path = backup_path
        self.model_name = model_name
        self.embedding_model = embedding_model
        
        # 確保目錄存在
        os.makedirs(pdf_directory, exist_ok=True)
        
    def backup_old_vectorstore(self) -> bool:
        """備份現有 vectorstore"""
        logger.info("="*70)
        logger.info("Step 1: Backing up old vectorstore")
        logger.info("="*70)
        
        if not os.path.exists(self.old_vectorstore_path):
            logger.warning(f"Old vectorstore not found at {self.old_vectorstore_path}")
            logger.info("No backup needed")
            return True
        
        try:
            # 創建帶時間戳的備份目錄
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{self.backup_path}_{timestamp}"
            
            logger.info(f"Backing up: {self.old_vectorstore_path} -> {backup_dir}")
            shutil.copytree(self.old_vectorstore_path, backup_dir)
            
            # 創建符號連結到最新備份
            latest_link = self.backup_path
            if os.path.islink(latest_link):
                os.unlink(latest_link)
            elif os.path.exists(latest_link):
                shutil.rmtree(latest_link)
            
            os.symlink(backup_dir, latest_link)
            logger.info(f"✓ Backup created successfully: {backup_dir}")
            logger.info(f"✓ Latest backup link: {latest_link}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Backup failed: {e}", exc_info=True)
            return False
    
    def count_pdfs(self) -> int:
        """統計 PDF 數量"""
        if not os.path.exists(self.pdf_directory):
            return 0
        
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        return len(pdf_files)
    
    def dry_run(self) -> dict:
        """模擬運行，預覽遷移結果"""
        logger.info("="*70)
        logger.info("Step 2: Dry Run (Preview)")
        logger.info("="*70)
        
        info = {
            'pdf_count': 0,
            'old_vectorstore_exists': False,
            'old_vectorstore_size': 0,
            'estimated_time': 0,
            'warnings': []
        }
        
        # 統計 PDF
        info['pdf_count'] = self.count_pdfs()
        logger.info(f"Found {info['pdf_count']} PDF files in {self.pdf_directory}")
        
        if info['pdf_count'] == 0:
            info['warnings'].append("No PDF files found")
            logger.warning("⚠ No PDF files found. Nothing to migrate.")
        
        # 檢查舊 vectorstore
        if os.path.exists(self.old_vectorstore_path):
            info['old_vectorstore_exists'] = True
            size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(self.old_vectorstore_path)
                for filename in filenames
            ) / (1024 * 1024)  # MB
            info['old_vectorstore_size'] = round(size, 2)
            logger.info(f"Old vectorstore size: {info['old_vectorstore_size']} MB")
        else:
            logger.info("No old vectorstore found (fresh migration)")
        
        # 估算時間（約 30-40 秒 per PDF + embedding 時間）
        info['estimated_time'] = info['pdf_count'] * 40
        logger.info(f"Estimated migration time: ~{info['estimated_time']}s ({info['estimated_time']//60}min)")
        
        # 檢查目標路徑
        if os.path.exists(self.new_vectorstore_path):
            info['warnings'].append(f"Target path {self.new_vectorstore_path} already exists")
            logger.warning(f"⚠ Target path already exists: {self.new_vectorstore_path}")
            logger.warning("  This will be overwritten during migration")
        
        logger.info("\n📋 Migration Preview:")
        logger.info(f"  Source PDFs: {self.pdf_directory}")
        logger.info(f"  Old Vectorstore: {self.old_vectorstore_path}")
        logger.info(f"  New Vectorstore: {self.new_vectorstore_path}")
        logger.info(f"  Backup Location: {self.backup_path}_<timestamp>")
        logger.info(f"  PDFs to process: {info['pdf_count']}")
        logger.info(f"  Estimated time: ~{info['estimated_time']//60} minutes")
        
        if info['warnings']:
            logger.warning("\n⚠ Warnings:")
            for warning in info['warnings']:
                logger.warning(f"  - {warning}")
        
        return info
    
    def execute_migration(self) -> bool:
        """執行實際遷移"""
        logger.info("="*70)
        logger.info("Step 3: Executing Migration")
        logger.info("="*70)
        
        pdf_count = self.count_pdfs()
        if pdf_count == 0:
            logger.error("✗ No PDF files found. Aborting migration.")
            return False
        
        try:
            # 初始化階層式 RAG 系統
            logger.info(f"Initializing HierarchicalRAGSystem...")
            logger.info(f"  Model: {self.model_name}")
            logger.info(f"  Embedding: {self.embedding_model}")
            logger.info(f"  Target: {self.new_vectorstore_path}")
            
            rag_system = HierarchicalRAGSystem(
                pdf_directory=self.pdf_directory,
                model_name=self.model_name,
                embedding_model=self.embedding_model,
                vectorstore_path=self.new_vectorstore_path,
                chunk_size=800,
                chunk_overlap=100
            )
            
            logger.info("✓ System initialized")
            
            # 獲取所有 PDF
            pdf_files = [
                os.path.join(self.pdf_directory, f)
                for f in os.listdir(self.pdf_directory)
                if f.endswith('.pdf')
            ]
            
            logger.info(f"\nBuilding hierarchical indices for {len(pdf_files)} PDFs...")
            logger.info("This may take several minutes...\n")
            
            # 建立索引
            start_time = datetime.now()
            rag_system.build_indices(pdf_files)
            duration = (datetime.now() - start_time).total_seconds()
            
            # 獲取統計
            stats = rag_system.get_stats()
            
            logger.info("\n" + "="*70)
            logger.info("✓ Migration Completed Successfully!")
            logger.info("="*70)
            logger.info(f"Duration: {duration:.1f}s ({duration//60:.0f}min {duration%60:.0f}s)")
            logger.info(f"\nNew Index Statistics:")
            logger.info(f"  Layer 1 (Abstracts): {stats['layer1']['paper_count']} papers")
            logger.info(f"  Layer 2 (Chunks): {stats['layer2']['chunk_count']} chunks")
            logger.info(f"  Index Location: {self.new_vectorstore_path}")
            
            # 計算大小
            new_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(self.new_vectorstore_path)
                for filename in filenames
            ) / (1024 * 1024)  # MB
            logger.info(f"  Index Size: {new_size:.2f} MB")
            
            logger.info("\n📝 Next Steps:")
            logger.info("  1. Verify the migration with: python scripts/migrate_to_hierarchical_rag.py --verify")
            logger.info("  2. Set environment variable: export RAG_MODE=hierarchical")
            logger.info("  3. Restart agent2.py to use the new system")
            logger.info("  4. If issues occur, rollback with: python scripts/migrate_to_hierarchical_rag.py --rollback")
            
            return True
            
        except Exception as e:
            logger.error(f"\n✗ Migration failed: {e}", exc_info=True)
            logger.error("\nYou can restore from backup with: python scripts/migrate_to_hierarchical_rag.py --rollback")
            return False
    
    def verify_migration(self) -> bool:
        """驗證遷移結果"""
        logger.info("="*70)
        logger.info("Step 4: Verifying Migration")
        logger.info("="*70)
        
        if not os.path.exists(self.new_vectorstore_path):
            logger.error(f"✗ New vectorstore not found at {self.new_vectorstore_path}")
            logger.error("Migration appears to have failed or not been executed.")
            return False
        
        try:
            # 嘗試載入系統
            logger.info("Loading HierarchicalRAGSystem...")
            rag_system = HierarchicalRAGSystem(
                pdf_directory=self.pdf_directory,
                model_name=self.model_name,
                embedding_model=self.embedding_model,
                vectorstore_path=self.new_vectorstore_path,
                chunk_size=800,
                chunk_overlap=100
            )
            
            # 檢查是否就緒
            is_ready = rag_system.is_ready()
            if not is_ready:
                logger.error("✗ System loaded but not ready (indices not properly built)")
                return False
            
            # 獲取統計
            stats = rag_system.get_stats()
            
            logger.info("✓ System loaded successfully")
            logger.info(f"\n📊 Index Statistics:")
            logger.info(f"  Layer 1:")
            logger.info(f"    Papers: {stats['layer1']['paper_count']}")
            logger.info(f"    Initialized: {stats['layer1']['is_initialized']}")
            logger.info(f"  Layer 2:")
            logger.info(f"    Chunks: {stats['layer2']['chunk_count']}")
            logger.info(f"    Papers: {stats['layer2']['paper_count']}")
            logger.info(f"    Initialized: {stats['layer2']['is_initialized']}")
            
            # 執行測試查詢
            logger.info(f"\n🔍 Testing query functionality...")
            test_query = "machine learning"
            try:
                result = rag_system.query(test_query)
                logger.info(f"✓ Test query executed successfully")
                logger.info(f"  Termination: {result.get('termination_layer', 'unknown')}")
                logger.info(f"  Confidence: {result.get('confidence', 0):.2f}")
            except Exception as e:
                logger.warning(f"⚠ Test query failed: {e}")
            
            logger.info("\n" + "="*70)
            logger.info("✓ Verification Passed!")
            logger.info("="*70)
            logger.info("\nThe hierarchical RAG system is ready to use.")
            logger.info("Set RAG_MODE=hierarchical and restart agent2.py")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Verification failed: {e}", exc_info=True)
            return False
    
    def rollback(self) -> bool:
        """回滾到備份"""
        logger.info("="*70)
        logger.info("Rollback: Restoring from backup")
        logger.info("="*70)
        
        if not os.path.exists(self.backup_path):
            logger.error(f"✗ No backup found at {self.backup_path}")
            return False
        
        try:
            # 刪除新的 vectorstore（如果存在）
            if os.path.exists(self.new_vectorstore_path):
                logger.info(f"Removing new vectorstore: {self.new_vectorstore_path}")
                shutil.rmtree(self.new_vectorstore_path)
            
            # 恢復舊的 vectorstore
            logger.info(f"Restoring backup: {self.backup_path} -> {self.old_vectorstore_path}")
            
            # 如果舊路徑存在，先刪除
            if os.path.exists(self.old_vectorstore_path):
                shutil.rmtree(self.old_vectorstore_path)
            
            shutil.copytree(self.backup_path, self.old_vectorstore_path)
            
            logger.info("✓ Rollback completed successfully")
            logger.info(f"Old vectorstore restored to: {self.old_vectorstore_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Rollback failed: {e}", exc_info=True)
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate from AcademicRAGSystem to HierarchicalRAGSystem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 完整遷移流程
  python scripts/migrate_to_hierarchical_rag.py --backup
  python scripts/migrate_to_hierarchical_rag.py --dry-run
  python scripts/migrate_to_hierarchical_rag.py --execute
  python scripts/migrate_to_hierarchical_rag.py --verify
  
  # 如果遇到問題，回滾
  python scripts/migrate_to_hierarchical_rag.py --rollback
        """
    )
    
    parser.add_argument(
        '--backup',
        action='store_true',
        help='僅備份現有 vectorstore'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='模擬運行（預覽遷移計劃）'
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='執行實際遷移'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='驗證遷移結果'
    )
    
    parser.add_argument(
        '--rollback',
        action='store_true',
        help='回滾到備份'
    )
    
    parser.add_argument(
        '--pdf-dir',
        default='./data',
        help='PDF 文件目錄（預設: ./data）'
    )
    
    parser.add_argument(
        '--old-vectorstore',
        default='./vectorstore',
        help='舊 vectorstore 路徑（預設: ./vectorstore）'
    )
    
    parser.add_argument(
        '--new-vectorstore',
        default='./vectorstore_hierarchical',
        help='新 vectorstore 路徑（預設: ./vectorstore_hierarchical）'
    )
    
    parser.add_argument(
        '--backup-path',
        default='./vectorstore_backup',
        help='備份路徑（預設: ./vectorstore_backup）'
    )
    
    args = parser.parse_args()
    
    # 檢查是否至少指定了一個操作
    if not any([args.backup, args.dry_run, args.execute, args.verify, args.rollback]):
        parser.print_help()
        print("\n❌ Error: Please specify at least one operation (--backup, --dry-run, --execute, --verify, or --rollback)")
        sys.exit(1)
    
    # 初始化工具
    tool = RAGMigrationTool(
        pdf_directory=args.pdf_dir,
        old_vectorstore_path=args.old_vectorstore,
        new_vectorstore_path=args.new_vectorstore,
        backup_path=args.backup_path
    )
    
    success = True
    
    # 執行操作
    if args.backup:
        success = tool.backup_old_vectorstore()
    
    if args.dry_run and success:
        tool.dry_run()
    
    if args.execute and success:
        # 執行前先備份（如果還沒備份）
        if not args.backup:
            logger.info("Auto-backup before execution...")
            if not tool.backup_old_vectorstore():
                logger.error("Backup failed. Aborting migration.")
                sys.exit(1)
        
        success = tool.execute_migration()
    
    if args.verify and success:
        success = tool.verify_migration()
    
    if args.rollback:
        success = tool.rollback()
    
    # 退出碼
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
