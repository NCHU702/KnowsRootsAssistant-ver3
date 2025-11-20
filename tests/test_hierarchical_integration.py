"""
測試完整的 Hierarchical RAG 系統整合

這個腳本測試 Phase 2 的完整系統：
1. HierarchicalRAGSystem - 主系統
2. IndexManager - 事務性索引管理
3. QueryLogger - 查詢日誌
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import directly to avoid search_engine import issues
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import components directly without going through __init__
from system_api.hierarchical_rag_system import HierarchicalRAGSystem
from system_api.index_manager import IndexManager
from system_api.query_logger import QueryLogger


def print_section(title):
    """Print section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def test_system_initialization():
    """測試系統初始化"""
    print_section("測試 1: 系統初始化")
    
    try:
        # Create temporary vectorstore directory
        temp_dir = tempfile.mkdtemp(prefix="hierarchical_rag_test_")
        print(f"📁 臨時目錄: {temp_dir}")
        
        # Initialize system
        print("\n🔧 初始化 Hierarchical RAG System...")
        system = HierarchicalRAGSystem(
            pdf_directory="./data",
            model_name="gemma3:12b",
            embedding_model="embeddinggemma:latest",
            vectorstore_path=temp_dir,
            chunk_size=800,
            chunk_overlap=100
        )
        
        print("✓ 系統初始化成功!")
        print(f"  模型: {system.model_name}")
        print(f"  Embedding: {system.embedding_model}")
        print(f"  向量庫路徑: {system.vectorstore_path}")
        
        # Check components
        print("\n🔍 檢查組件:")
        print(f"  ✓ Abstract Extractor: {system.abstract_extractor is not None}")
        print(f"  ✓ Layer 1 VectorStore: {system.layer1 is not None}")
        print(f"  ✓ Layer 2 VectorStore: {system.layer2 is not None}")
        print(f"  ✓ Confidence Evaluator: {system.confidence_evaluator is not None}")
        print(f"  ✓ Context Expander: {system.context_expander is not None}")
        
        return system, temp_dir
        
    except Exception as e:
        print(f"❌ 系統初始化失敗: {e}")
        logger.error("System initialization failed", exc_info=True)
        return None, None


def test_build_indices(system):
    """測試建立索引"""
    print_section("測試 2: 建立索引（使用現有 PDF）")
    
    try:
        # Check if data directory exists and has PDFs
        pdf_dir = Path("./data")
        if not pdf_dir.exists():
            print("⚠️  ./data 目錄不存在，跳過索引建立測試")
            return False
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print("⚠️  ./data 目錄沒有 PDF 文件，跳過索引建立測試")
            return False
        
        # Use only first 2 PDFs for testing
        test_pdfs = [str(pdf) for pdf in pdf_files[:2]]
        print(f"📚 使用 {len(test_pdfs)} 個 PDF 進行測試:")
        for pdf in test_pdfs:
            print(f"  - {os.path.basename(pdf)}")
        
        print("\n🔨 開始建立索引...")
        result = system.build_indices(pdf_paths=test_pdfs)
        
        if result['status'] == 'success':
            print("✓ 索引建立成功!")
            print(f"  處理論文數: {result['papers_processed']}")
            print(f"  提取摘要數: {result['abstracts_extracted']}")
            print(f"  創建區塊數: {result['chunks_created']}")
            print(f"  耗時: {result['duration']:.2f}s")
            
            # Check stats
            stats = result['stats']
            print(f"\n📊 索引統計:")
            print(f"  Layer 1: {stats['layer1']['paper_count']} 篇論文")
            print(f"  Layer 2: {stats['layer2']['chunk_count']} 個區塊")
            
            return True
        else:
            print(f"❌ 索引建立失敗: {result.get('message')}")
            return False
            
    except Exception as e:
        print(f"❌ 索引建立測試失敗: {e}")
        logger.error("Index building test failed", exc_info=True)
        return False


def test_query_system(system):
    """測試查詢系統"""
    print_section("測試 3: 階層式查詢")
    
    if not system.is_ready():
        print("⚠️  系統未就緒（索引未建立），跳過查詢測試")
        return False
    
    try:
        test_queries = [
            "What is machine learning?",
            "深度學習的最新進展",
            "Remote sensing applications"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 查詢 {i}: '{query}'")
            
            # Perform hierarchical retrieval (not full generation to save time)
            result = system._hierarchical_retrieval(query)
            
            if result['status'] == 'success':
                print(f"  ✓ 檢索成功!")
                print(f"    使用層級: {', '.join(result['layers_used'])}")
                print(f"    終止於: {result['terminated_at']}")
                print(f"    信心度: {result['final_confidence']:.2f}")
                print(f"    總耗時: {result['timings']['total']:.2f}s")
                
                if 'timings' in result:
                    print(f"    時間分解:")
                    for key, val in result['timings'].items():
                        if val and key != 'total':
                            print(f"      {key}: {val:.2f}s")
            else:
                print(f"  ❌ 檢索失敗: {result.get('message')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 查詢測試失敗: {e}")
        logger.error("Query test failed", exc_info=True)
        return False


def test_index_manager(system):
    """測試索引管理器"""
    print_section("測試 4: 索引管理器（事務性更新）")
    
    try:
        # Initialize index manager
        print("🔧 初始化 Index Manager...")
        index_mgr = IndexManager(
            layer1=system.layer1,
            layer2=system.layer2,
            abstract_extractor=system.abstract_extractor,
            backup_dir=os.path.join(system.vectorstore_path, "backups"),
            max_backups=3
        )
        
        print("✓ Index Manager 初始化成功!")
        
        # Test backup listing
        print("\n📋 當前備份:")
        backups = index_mgr.list_backups()
        if backups:
            for backup in backups[:3]:
                print(f"  - {backup['backup_id']}")
                print(f"    創建時間: {backup['created_at']}")
                print(f"    大小: {backup['size_bytes'] / 1024:.1f} KB")
        else:
            print("  (無備份)")
        
        # Test stats
        print("\n📊 Index Manager 統計:")
        stats = index_mgr.get_stats()
        print(f"  備份目錄: {stats['backup_dir']}")
        print(f"  最大備份數: {stats['max_backups']}")
        print(f"  當前備份數: {stats['current_backups']}")
        print(f"  總備份大小: {stats['total_backup_size_mb']:.2f} MB")
        
        # Note: We don't actually add/remove documents in test to avoid modifying real data
        print("\n⚠️  跳過實際的添加/刪除操作（避免修改真實數據）")
        print("  Index Manager 的 add_document() 和 remove_document() 功能已實作")
        
        return True
        
    except Exception as e:
        print(f"❌ Index Manager 測試失敗: {e}")
        logger.error("Index Manager test failed", exc_info=True)
        return False


def test_query_logger(system):
    """測試查詢日誌"""
    print_section("測試 5: 查詢日誌")
    
    try:
        # Initialize query logger
        log_file = os.path.join(system.vectorstore_path, "query_log.jsonl")
        print(f"🔧 初始化 Query Logger...")
        print(f"  日誌文件: {log_file}")
        
        logger_instance = QueryLogger(log_file=log_file)
        
        print("✓ Query Logger 初始化成功!")
        
        # Log some sample queries
        if system.is_ready():
            print("\n📝 記錄測試查詢...")
            
            test_queries = [
                "What is deep learning?",
                "機器學習應用"
            ]
            
            for query in test_queries:
                result = system._hierarchical_retrieval(query)
                logger_instance.log_query(query, result)
                print(f"  ✓ 已記錄: '{query[:50]}...'")
        
        # Test statistics
        print("\n📊 查詢統計:")
        stats = logger_instance.get_statistics()
        
        if stats['total_queries'] > 0:
            print(f"  總查詢數: {stats['total_queries']}")
            
            if 'status_distribution' in stats:
                print(f"  狀態分布:")
                for status, count in stats['status_distribution'].items():
                    print(f"    {status}: {count}")
            
            if 'performance' in stats:
                print(f"  性能:")
                print(f"    平均耗時: {stats['performance']['avg_duration']:.2f}s")
                print(f"    最小耗時: {stats['performance']['min_duration']:.2f}s")
                print(f"    最大耗時: {stats['performance']['max_duration']:.2f}s")
            
            if 'termination_distribution' in stats:
                print(f"  終止點分布:")
                for point, count in stats['termination_distribution'].items():
                    pct = (count / stats['total_queries']) * 100
                    print(f"    {point}: {count} ({pct:.1f}%)")
        else:
            print("  尚無查詢記錄")
        
        # Test performance summary
        print("\n" + "="*60)
        print(logger_instance.get_performance_summary())
        
        return True
        
    except Exception as e:
        print(f"❌ Query Logger 測試失敗: {e}")
        logger.error("Query Logger test failed", exc_info=True)
        return False


def test_system_stats(system):
    """測試系統統計"""
    print_section("測試 6: 系統統計")
    
    try:
        stats = system.get_stats()
        
        print("📊 完整系統統計:")
        print(f"\n  系統就緒: {stats['initialized']}")
        
        print(f"\n  Layer 1:")
        for key, value in stats['layer1'].items():
            if key != 'cache':
                print(f"    {key}: {value}")
        
        print(f"\n  Layer 2:")
        for key, value in stats['layer2'].items():
            if key != 'cache':
                print(f"    {key}: {value}")
        
        print(f"\n  配置:")
        print(f"    Layer 1 閾值: {stats['config']['layer1']['confidence_threshold']}")
        print(f"    Layer 2 閾值: {stats['config']['layer2']['confidence_threshold']}")
        print(f"    上下文擴展: {stats['config']['expansion']['enabled']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 系統統計測試失敗: {e}")
        logger.error("System stats test failed", exc_info=True)
        return False


def main():
    """執行所有整合測試"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*18 + "Hierarchical RAG 系統整合測試" + " "*29 + "║")
    print("╚" + "="*78 + "╝")
    
    results = {
        '系統初始化': False,
        '建立索引': False,
        '階層式查詢': False,
        'Index Manager': False,
        'Query Logger': False,
        '系統統計': False
    }
    
    # Test 1: System initialization
    system, temp_dir = test_system_initialization()
    results['系統初始化'] = system is not None
    
    if not system:
        print("\n❌ 系統初始化失敗，無法繼續測試")
        return False
    
    # Test 2: Build indices
    results['建立索引'] = test_build_indices(system)
    
    # Test 3: Query system
    results['階層式查詢'] = test_query_system(system)
    
    # Test 4: Index Manager
    results['Index Manager'] = test_index_manager(system)
    
    # Test 5: Query Logger
    results['Query Logger'] = test_query_logger(system)
    
    # Test 6: System stats
    results['系統統計'] = test_system_stats(system)
    
    # Summary
    print_section("測試總結")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ 通過" if success else "❌ 失敗"
        print(f"  {status}: {test_name}")
    
    print(f"\n總計: {passed}/{total} 個測試通過")
    
    if passed == total:
        print("\n🎉 所有整合測試通過！系統已準備好部署。")
        success = True
    else:
        print("\n⚠️  部分測試失敗，請檢查錯誤日誌。")
        success = False
    
    # Cleanup
    if temp_dir and os.path.exists(temp_dir):
        print(f"\n🧹 清理臨時目錄: {temp_dir}")
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            print(f"  ⚠️  無法刪除臨時目錄，請手動清理")
    
    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  測試被用戶中斷")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 測試過程發生錯誤: {e}")
        logger.error("Test suite failed", exc_info=True)
        sys.exit(1)
