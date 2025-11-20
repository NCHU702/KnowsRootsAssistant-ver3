"""
測試 Chunk 優化效果

隨機抽取一篇論文，比較 naive 和 summarization 兩種分塊模式的結果
"""

import os
import random
import json
import logging
from system_api.hierarchical_rag_system import HierarchicalRAGSystem
from datetime import datetime

# 啟用詳細日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 模型配置
model_name = "llama3.2:latest"

def get_random_pdf():
    """從 test_data 中隨機選擇一篇論文"""
    pdf_dir = "./test_data"
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        raise ValueError("No PDF files found in test_data directory")
    
    selected_pdf = random.choice(pdf_files)
    return os.path.join(pdf_dir, selected_pdf), selected_pdf

def test_chunking_mode(pdf_path, pdf_name, mode='naive'):
    """測試指定的分塊模式"""
    print(f"\n{'='*80}")
    print(f"測試模式: {mode.upper()}")
    print(f"論文: {pdf_name}")
    print(f"{'='*80}\n")
    
    # 配置
    config = {
        'layer1': {
            'k_documents': 10,
            'confidence_threshold': 0.6,
            'similarity_threshold': 0.5
        },
        'layer2': {
            'k_documents': 2,
            'confidence_threshold': 0.6
        },
        'expansion': {
            'enabled': True
        },
        'hybrid_search': {
            'enabled': True,
            'semantic_weight': 0.9,
            'keyword_weight': 0.1,
            'use_jieba': True,
            'enable_smart_weighting': True,
            'use_llm_analyzer': True,
        },
        'query_expansion': {
            'enabled': False,
            'min_word_count': 15,
        },
        'adaptive_weights': {
            'enabled': False,
            'default_semantic_weight': 0.9,
            'default_keyword_weight': 0.1,
        },
        'performance': {
            'cache_size': 10
        },
        'layer2_reranking': {
            'enabled': True,
            'model': 'qllama/bce-reranker-base_v1:latest',
            'ollama_base_url': 'http://localhost:11434',
            'max_length': 512,
            'timeout': 30,
            'max_candidates': 300,
            'cache_limit_mb': 100,
        },
        'chunking': {
            'mode': mode,
            'summarization': {
                'model': model_name,
                'section_parser': 'pymupdf_regex',
                'min_sections': 3,
                'map_reduce_threshold': 1500,
                'target_summary_length': 300,
                'ollama_base_url': 'http://localhost:11434',
                'store_original': False
            }
        }
    }
    
    # 創建測試用的向量存儲路徑
    vectorstore_path = f"./test_vectorstore_{mode}"
    
    # 清理舊的測試資料
    if os.path.exists(vectorstore_path):
        import shutil
        shutil.rmtree(vectorstore_path)
    
    # 創建臨時測試目錄（只包含選定的論文）
    temp_test_dir = f"./temp_test_data_{mode}"
    if os.path.exists(temp_test_dir):
        import shutil
        shutil.rmtree(temp_test_dir)
    os.makedirs(temp_test_dir)
    
    # 複製選定的論文到臨時目錄
    import shutil
    temp_pdf_path = os.path.join(temp_test_dir, pdf_name)
    shutil.copy2(pdf_path, temp_pdf_path)
    
    # 初始化 RAG 系統
    start_time = datetime.now()
    rag_system = HierarchicalRAGSystem(
        pdf_directory=temp_test_dir,  # 使用臨時目錄
        model_name=model_name,
        embedding_model="quentinz/bge-large-zh-v1.5:latest",
        vectorstore_path=vectorstore_path,
        chunk_size=800,
        chunk_overlap=100,
        config=config
    )
    
    # 建立索引（會自動處理臨時目錄中的論文）
    print(f"\n建立索引並處理論文: {pdf_name}")
    check_result = rag_system.check_and_update_indices(force_rebuild=True)
    
    # 清理臨時目錄
    if os.path.exists(temp_test_dir):
        shutil.rmtree(temp_test_dir)
    
    result = check_result
    
    if result['status'] == 'success':
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n✓ 索引建立成功")
        print(f"  處理時間: {duration:.1f}s")
        papers_processed = result.get('details', {}).get('papers_processed', 1)
        print(f"  論文數量: {papers_processed}")
        
        # 獲取 Layer 2 的詳細資訊
        layer2_stats = rag_system.get_stats()['layer2']
        print(f"  總 Chunks: {layer2_stats['chunk_count']}")
        
        # 讀取 JSONL 檔案分析 chunks
        jsonl_path = os.path.join(vectorstore_path, "layer2", "chunks.jsonl")
        if os.path.exists(jsonl_path):
            chunks = []
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    chunks.append(json.loads(line))
            
            print(f"\n📊 Chunk 分析:")
            print(f"  總數: {len(chunks)}")
            
            # 分析 chunk 類型
            chunk_types = {}
            total_original_length = 0
            total_summary_length = 0
            
            for chunk in chunks:
                chunk_type = chunk.get('chunk_type', 'unknown')
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                
                if mode == 'summarization':
                    original_length = chunk.get('original_length', 0)
                    summary_length = len(chunk.get('text', ''))
                    total_original_length += original_length
                    total_summary_length += summary_length
            
            print(f"\n  Chunk 類型分布:")
            for chunk_type, count in chunk_types.items():
                print(f"    {chunk_type}: {count}")
            
            if mode == 'summarization' and total_original_length > 0:
                compression_ratio = (total_summary_length / total_original_length) * 100
                print(f"\n  壓縮效果:")
                print(f"    原始長度: {total_original_length:,} chars")
                print(f"    摘要長度: {total_summary_length:,} chars")
                print(f"    壓縮率: {compression_ratio:.1f}%")
                print(f"    資訊密度提升: {100 - compression_ratio:.1f}%")
            
            # 顯示前 3 個 chunks 的範例
            print(f"\n  前 3 個 Chunks 範例:")
            for i, chunk in enumerate(chunks[:3]):
                print(f"\n  [{i+1}] {chunk.get('chunk_type', 'unknown')}")
                if mode == 'summarization':
                    section_name = chunk.get('section_name', 'N/A')
                    original_length = chunk.get('original_length', 0)
                    summary_length = len(chunk.get('text', ''))
                    print(f"      Section: {section_name}")
                    print(f"      原始: {original_length} chars → 摘要: {summary_length} chars")
                else:
                    text_length = len(chunk.get('text', ''))
                    print(f"      長度: {text_length} chars")
                
                # 顯示部分內容
                text = chunk.get('text', '')
                preview = text[:150] + '...' if len(text) > 150 else text
                print(f"      內容: {preview}")
        
        return {
            'mode': mode,
            'status': 'success',
            'duration': duration,
            'chunks_added': len(chunks) if os.path.exists(jsonl_path) else 0,
            'chunks': chunks if os.path.exists(jsonl_path) else []
        }
    else:
        print(f"\n✗ 索引建立失敗: {result.get('error', 'Unknown error')}")
        return {
            'mode': mode,
            'status': 'error',
            'error': result.get('error', 'Unknown error')
        }

def compare_results(naive_result, summarization_result):
    """比較兩種模式的結果"""
    print(f"\n{'='*80}")
    print("📊 比較結果")
    print(f"{'='*80}\n")
    
    if naive_result['status'] == 'success' and summarization_result['status'] == 'success':
        print(f"模式               | Naive          | Summarization  | 改善")
        print(f"-" * 70)
        print(f"處理時間 (秒)      | {naive_result['duration']:>14.1f} | {summarization_result['duration']:>14.1f} | {((summarization_result['duration'] - naive_result['duration']) / naive_result['duration'] * 100):>+6.1f}%")
        print(f"Chunks 數量        | {naive_result['chunks_added']:>14} | {summarization_result['chunks_added']:>14} | {((summarization_result['chunks_added'] - naive_result['chunks_added']) / naive_result['chunks_added'] * 100):>+6.1f}%")
        
        # 計算壓縮率
        if summarization_result['chunks']:
            total_original = sum(c.get('original_length', 0) for c in summarization_result['chunks'])
            total_summary = sum(len(c.get('text', '')) for c in summarization_result['chunks'])
            if total_original > 0:
                compression = (total_summary / total_original) * 100
                print(f"\nSummarization 模式資訊密度: {100 - compression:.1f}% 提升")
        
        print(f"\n結論:")
        if summarization_result['chunks_added'] < naive_result['chunks_added']:
            reduction = ((naive_result['chunks_added'] - summarization_result['chunks_added']) / naive_result['chunks_added'] * 100)
            print(f"  ✓ Summarization 模式減少了 {reduction:.1f}% 的 chunks")
            print(f"  ✓ 預期可減少 LLM context 使用量，提升檢索效率")
        else:
            print(f"  ⚠ Summarization 模式產生了更多 chunks")
            print(f"    這可能是因為論文段落數量較多")

def main():
    print("\n" + "="*80)
    print("🔬 Chunk 優化效果測試")
    print("="*80)
    
    # 隨機選擇一篇論文
    pdf_path, pdf_name = get_random_pdf()
    print(f"\n📄 隨機選擇論文: {pdf_name}")
    
    try:
        # 測試 Naive 模式
        print(f"\n{'='*80}")
        print("階段 1/2: 測試 Naive 模式")
        print(f"{'='*80}")
        naive_result = test_chunking_mode(pdf_path, pdf_name, mode='naive')
        
        # 測試 Summarization 模式
        print(f"\n{'='*80}")
        print("階段 2/2: 測試 Summarization 模式")
        print(f"{'='*80}")
        summarization_result = test_chunking_mode(pdf_path, pdf_name, mode='summarization')
        
        # 比較結果
        compare_results(naive_result, summarization_result)
        
        print(f"\n{'='*80}")
        print("✓ 測試完成")
        print(f"{'='*80}\n")
        
        # 清理提示
        print("💡 提示:")
        print("  - Naive 模式結果存儲在: ./test_vectorstore_naive")
        print("  - Summarization 模式結果存儲在: ./test_vectorstore_summarization")
        print("  - 如需清理，請執行: rm -rf test_vectorstore_*")
        
    except Exception as e:
        print(f"\n✗ 測試過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
