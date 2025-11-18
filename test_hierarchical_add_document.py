#!/usr/bin/env python3
"""
測試 HierarchicalRAGSystem 的 add_document 功能整合
驗證 TextPreprocessor 是否正確整合到增量更新流程中
"""

import os
import sys
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_add_document_with_preprocessing():
    """測試 add_document 是否使用 TextPreprocessor"""
    
    print("="*80)
    print("測試 HierarchicalRAGSystem.add_document() 與 TextPreprocessor 整合")
    print("="*80)
    
    # 1. 設置測試環境
    test_vectorstore = "./test_vectorstore_hierarchical"
    test_data = "./test_data"
    
    # 清理舊的測試環境
    if os.path.exists(test_vectorstore):
        print(f"\n清理舊的測試向量存儲: {test_vectorstore}")
        shutil.rmtree(test_vectorstore)
    
    # 2. 初始化系統
    print("\n初始化 HierarchicalRAGSystem...")
    from system_api.hierarchical_rag_system import HierarchicalRAGSystem
    
    rag_system = HierarchicalRAGSystem(
        pdf_directory=test_data,
        vectorstore_path=test_vectorstore,
        model_name="gemma3:12b",
        embedding_model="embeddinggemma:latest"
    )
    
    # 3. 建立初始索引（使用部分文件）
    print("\n建立初始索引（前 3 篇論文）...")
    pdf_files = sorted([f for f in os.listdir(test_data) if f.endswith('.pdf')])
    initial_pdfs = [os.path.join(test_data, f) for f in pdf_files[:3]]
    
    build_result = rag_system.build_indices(pdf_paths=initial_pdfs)
    
    if build_result.get('status') == 'success':
        print(f"✓ 初始索引建立成功")
        print(f"  - 摘要提取: {build_result['abstracts_extracted']}")
        print(f"  - 區塊建立: {build_result['chunks_created']}")
    else:
        print(f"✗ 初始索引建立失敗")
        return
    
    # 4. 獲取初始統計
    stats_before = rag_system.get_stats()
    print(f"\n初始統計:")
    print(f"  Layer 1 論文數: {stats_before['layer1']['paper_count']}")
    print(f"  Layer 2 區塊數: {stats_before['layer2']['chunk_count']}")
    
    # 5. 測試 add_document（應該使用 TextPreprocessor）
    print("\n\n" + "="*80)
    print("測試增量添加文檔 (應該使用 TextPreprocessor)")
    print("="*80)
    
    if len(pdf_files) > 3:
        new_pdf = os.path.join(test_data, pdf_files[3])
        print(f"\n添加新文檔: {pdf_files[3]}")
        
        result = rag_system.add_document(new_pdf)
        
        if result['status'] == 'success':
            print(f"\n✓ 文檔添加成功！")
            print(f"  Paper ID: {result['paper_id']}")
            print(f"  添加區塊數: {result['chunks_added']}")
            print(f"  處理時間: {result['duration_seconds']:.1f}秒")
            
            # 檢查是否有預處理統計
            if 'preprocess_stats' in result:
                stats = result['preprocess_stats']
                print(f"\n  📊 預處理統計:")
                print(f"    原始長度: {stats['original_length']} 字元")
                print(f"    最終長度: {stats['final_length']} 字元")
                print(f"    移除比例: {stats.get('removed_percentage', 0):.1f}%")
                
                if stats.get('sections_removed'):
                    print(f"    移除章節: {', '.join(stats['sections_removed'])}")
            
        else:
            print(f"\n✗ 文檔添加失敗: {result.get('error')}")
            return
        
        # 6. 驗證統計已更新
        stats_after = rag_system.get_stats()
        print(f"\n更新後統計:")
        print(f"  Layer 1 論文數: {stats_after['layer1']['paper_count']} (增加 {stats_after['layer1']['paper_count'] - stats_before['layer1']['paper_count']})")
        print(f"  Layer 2 區塊數: {stats_after['layer2']['chunk_count']} (增加 {stats_after['layer2']['chunk_count'] - stats_before['layer2']['chunk_count']})")
        
        # 7. 驗證兼容性屬性
        print(f"\n驗證兼容性屬性:")
        print(f"  rag_system.vectorstore: {rag_system.vectorstore is not None}")
        print(f"  rag_system.is_ready(): {rag_system.is_ready()}")
        
    else:
        print("\n沒有足夠的 PDF 文件進行測試")
    
    # 8. 清理
    print("\n清理測試環境...")
    if os.path.exists(test_vectorstore):
        shutil.rmtree(test_vectorstore)
    
    print("\n" + "="*80)
    print("✓ 測試完成！")
    print("="*80)

if __name__ == "__main__":
    try:
        test_add_document_with_preprocessing()
    except Exception as e:
        logger.error(f"測試失敗: {e}", exc_info=True)
        sys.exit(1)
