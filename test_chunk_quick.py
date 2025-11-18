"""
快速測試 Chunk 優化效果

隨機抽取一篇論文，快速比較兩種分塊模式
"""

import os
import random
import json
import logging
from datetime import datetime

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_random_pdf():
    """從 test_data 中隨機選擇一篇論文"""
    pdf_dir = "./test_data"
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        raise ValueError("No PDF files found in test_data directory")
    
    selected_pdf = random.choice(pdf_files)
    return os.path.join(pdf_dir, selected_pdf), selected_pdf

def test_naive_chunking(pdf_path, pdf_name):
    """測試 Naive 分塊"""
    print(f"\n{'='*80}")
    print("測試 NAIVE 模式")
    print(f"{'='*80}\n")
    
    from system_api.pdf_loader import PDFLoader
    
    # 載入 PDF
    logger.info(f"載入 PDF: {pdf_name}")
    loader = PDFLoader()
    documents = loader.load(pdf_path)
    
    # Naive 分塊 (800 chars)
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"✓ Naive 分塊完成")
    print(f"  總 Chunks: {len(chunks)}")
    
    # 分析 chunk 長度
    chunk_lengths = [len(chunk.page_content) for chunk in chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    
    print(f"  平均長度: {avg_length:.0f} chars")
    print(f"  最短: {min(chunk_lengths)} chars")
    print(f"  最長: {max(chunk_lengths)} chars")
    
    # 顯示前 3 個 chunks
    print(f"\n前 3 個 Chunks 範例:")
    for i, chunk in enumerate(chunks[:3]):
        text = chunk.page_content
        preview = text[:100] + '...' if len(text) > 100 else text
        print(f"\n  [{i+1}] 長度: {len(text)} chars")
        print(f"      內容: {preview}")
    
    return {
        'mode': 'naive',
        'chunk_count': len(chunks),
        'avg_length': avg_length,
        'total_chars': sum(chunk_lengths)
    }

def test_summarization_chunking(pdf_path, pdf_name):
    """測試 Summarization 分塊"""
    print(f"\n{'='*80}")
    print("測試 SUMMARIZATION 模式")
    print(f"{'='*80}\n")
    
    from system_api.pdf_section_parser import PDFSectionParser
    from system_api.llm_summarizer import LLMSummarizer
    
    # 1. 解析段落
    logger.info("步驟 1/2: 解析 PDF 段落...")
    parser = PDFSectionParser(method='pymupdf_regex', min_sections=3)
    sections = parser.parse(pdf_path)
    
    print(f"✓ 段落解析完成")
    print(f"  段落數: {len(sections)}")
    
    for i, section in enumerate(sections[:5]):  # 只顯示前 5 個
        print(f"    {i+1}. {section.name}: {section.char_count} chars")
    
    if len(sections) > 5:
        print(f"    ... 還有 {len(sections) - 5} 個段落")
    
    # 2. 生成摘要
    logger.info("步驟 2/2: 生成段落摘要...")
    summarizer = LLMSummarizer(
        model_name="llama3.2:latest",
        target_summary_length=200,  # 較短的摘要加快速度
        timeout=60  # 每個摘要最多 60 秒
    )
    
    summaries = []
    total_original = 0
    total_summary = 0
    
    print(f"\n正在生成摘要...")
    for i, section in enumerate(sections):
        try:
            logger.info(f"  處理 [{i+1}/{len(sections)}] {section.name}...")
            
            summary = summarizer.summarize_section(
                section.text,
                section.name
            )
            
            summaries.append({
                'section': section.name,
                'original_length': section.char_count,
                'summary_length': len(summary),
                'summary': summary
            })
            
            total_original += section.char_count
            total_summary += len(summary)
            
            compression = (len(summary) / section.char_count * 100) if section.char_count > 0 else 0
            print(f"  ✓ {section.name}: {section.char_count} → {len(summary)} chars ({compression:.1f}%)")
            
        except Exception as e:
            logger.warning(f"  ✗ {section.name} 失敗: {e}")
            # 使用 extractive fallback
            summary = section.text[:200]
            summaries.append({
                'section': section.name,
                'original_length': section.char_count,
                'summary_length': len(summary),
                'summary': summary
            })
            total_original += section.char_count
            total_summary += len(summary)
    
    print(f"\n✓ 摘要生成完成")
    print(f"  總段落: {len(summaries)}")
    
    if total_original > 0:
        compression_ratio = (total_summary / total_original) * 100
        print(f"  原始長度: {total_original:,} chars")
        print(f"  摘要長度: {total_summary:,} chars")
        print(f"  壓縮率: {compression_ratio:.1f}%")
        print(f"  資訊密度提升: {100 - compression_ratio:.1f}%")
    
    # 顯示前 3 個摘要
    print(f"\n前 3 個摘要範例:")
    for i, item in enumerate(summaries[:3]):
        preview = item['summary'][:100] + '...' if len(item['summary']) > 100 else item['summary']
        print(f"\n  [{i+1}] {item['section']}")
        print(f"      原始: {item['original_length']} → 摘要: {item['summary_length']} chars")
        print(f"      內容: {preview}")
    
    return {
        'mode': 'summarization',
        'chunk_count': len(summaries),
        'avg_length': total_summary / len(summaries) if summaries else 0,
        'total_chars': total_summary,
        'original_chars': total_original,
        'compression_ratio': (total_summary / total_original * 100) if total_original > 0 else 0
    }

def compare_results(naive_result, summ_result):
    """比較結果"""
    print(f"\n{'='*80}")
    print("📊 比較結果")
    print(f"{'='*80}\n")
    
    print(f"{'指標':<20} | {'Naive':>15} | {'Summarization':>15} | {'變化':>10}")
    print("-" * 70)
    print(f"{'Chunks 數量':<20} | {naive_result['chunk_count']:>15} | {summ_result['chunk_count']:>15} | {((summ_result['chunk_count'] - naive_result['chunk_count']) / naive_result['chunk_count'] * 100):>+9.1f}%")
    print(f"{'平均長度 (chars)':<20} | {naive_result['avg_length']:>15.0f} | {summ_result['avg_length']:>15.0f} | {((summ_result['avg_length'] - naive_result['avg_length']) / naive_result['avg_length'] * 100):>+9.1f}%")
    print(f"{'總字元數':<20} | {naive_result['total_chars']:>15,} | {summ_result['total_chars']:>15,} | {((summ_result['total_chars'] - naive_result['total_chars']) / naive_result['total_chars'] * 100):>+9.1f}%")
    
    if 'compression_ratio' in summ_result:
        print(f"\nSummarization 模式:")
        print(f"  原始文本: {summ_result['original_chars']:,} chars")
        print(f"  壓縮後: {summ_result['total_chars']:,} chars")
        print(f"  壓縮率: {summ_result['compression_ratio']:.1f}%")
        print(f"  資訊密度提升: {100 - summ_result['compression_ratio']:.1f}%")
    
    print(f"\n結論:")
    chunk_reduction = ((naive_result['chunk_count'] - summ_result['chunk_count']) / naive_result['chunk_count'] * 100)
    if chunk_reduction > 0:
        print(f"  ✓ Summarization 減少了 {chunk_reduction:.1f}% 的 chunks")
    else:
        print(f"  • Summarization 產生了 {abs(chunk_reduction):.1f}% 更多的 chunks")
    
    char_reduction = ((naive_result['total_chars'] - summ_result['total_chars']) / naive_result['total_chars'] * 100)
    if char_reduction > 0:
        print(f"  ✓ 總字元數減少 {char_reduction:.1f}%，節省 LLM context")
    else:
        print(f"  • 總字元數增加 {abs(char_reduction):.1f}%")

def main():
    print("\n" + "="*80)
    print("🔬 Chunk 優化快速測試")
    print("="*80)
    
    # 隨機選擇論文
    pdf_path, pdf_name = get_random_pdf()
    print(f"\n📄 隨機選擇: {pdf_name}\n")
    
    try:
        # 測試 Naive 模式
        naive_result = test_naive_chunking(pdf_path, pdf_name)
        
        # 測試 Summarization 模式
        summ_result = test_summarization_chunking(pdf_path, pdf_name)
        
        # 比較結果
        compare_results(naive_result, summ_result)
        
        print(f"\n{'='*80}")
        print("✓ 測試完成")
        print(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
