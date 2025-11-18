#!/usr/bin/env python3
"""
Test Text Preprocessor

Tests the removal of references and appendix sections from PDF text
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from system_api.text_preprocessor import TextPreprocessor
from PyPDF2 import PdfReader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_preprocessor_with_sample_text():
    """Test with sample text"""
    
    print("="*70)
    print("Test 1: Sample Text with References")
    print("="*70)
    
    sample_text = """
Introduction
This is the main content of the paper.
We discuss various methods and approaches.

Methodology
Our approach uses deep learning techniques.
The model architecture is based on transformers.

Results
We achieved 95% accuracy on the test set.
The results show significant improvement.

Conclusion
This paper demonstrates the effectiveness of our approach.
Future work will explore additional applications.

References

[1] Smith, J. (2020). Deep Learning Basics.
[2] Jones, A. (2021). Transformer Architecture.
[3] Brown, B. (2022). Advanced Neural Networks.
[4] Lee, C. (2023). Model Optimization Techniques.
"""
    
    preprocessor = TextPreprocessor()
    cleaned_text, stats = preprocessor.preprocess(sample_text)
    
    print(f"\n原始長度: {stats['original_length']} chars")
    print(f"清理後長度: {stats['final_length']} chars")
    print(f"移除比例: {stats.get('removed_percentage', 0):.1f}%")
    print(f"移除的段落: {stats['removed_sections']}")
    print(f"\n清理後的文本:")
    print("-"*70)
    print(cleaned_text)
    print("-"*70)


def test_preprocessor_with_chinese_text():
    """Test with Chinese text"""
    
    print("\n\n")
    print("="*70)
    print("Test 2: Chinese Text with Multiple Sections")
    print("="*70)
    
    sample_text = """
目錄

第一章 緒論 .......................... 1
第二章 文獻探討 ...................... 5
第三章 研究方法 ...................... 10

表目錄

表1 實驗參數設定 ..................... 15
表2 結果比較 ......................... 20

圖目錄

圖1 系統架構圖 ....................... 8
圖2 實驗流程圖 ....................... 12

摘要
本研究探討深度學習在自然語言處理中的應用。

前言
深度學習技術近年來在各個領域都取得了顯著的進展。
本文將介紹我們的方法。

方法論
我們採用Transformer架構來處理文本數據。
模型訓練使用大規模語料庫。

實驗結果
實驗結果顯示我們的方法達到了95%的準確率。
與現有方法相比有顯著提升。

結論
本研究證明了深度學習方法的有效性。
未來將探索更多應用場景。

誌謝

感謝指導教授的悉心指導。
感謝實驗室同學的協助。

參考文獻

[1] 張三. (2020). 深度學習基礎. 
[2] 李四. (2021). Transformer架構詳解.
[3] 王五. (2022). 自然語言處理進展.
[4] 趙六. (2023). 模型優化技術.

附錄

附錄A: 實驗數據詳細表格
附錄B: 模型超參數設定
"""
    
    preprocessor = TextPreprocessor()
    cleaned_text, stats = preprocessor.preprocess(sample_text)
    
    print(f"\n原始長度: {stats['original_length']} chars")
    print(f"清理後長度: {stats['final_length']} chars")
    print(f"移除比例: {stats.get('removed_percentage', 0):.1f}%")
    print(f"移除的段落: {stats['removed_sections']}")
    print(f"\n清理後的文本:")
    print("-"*70)
    print(cleaned_text)
    print("-"*70)


def test_preprocessor_with_real_pdf():
    """Test with real PDF from data directory"""
    
    print("\n\n")
    print("="*70)
    print("Test 3: Real PDF File")
    print("="*70)
    
    # Get first PDF from data directory
    data_dir = Path("./data")
    if not data_dir.exists():
        print("⚠️  ./data directory not found, skipping this test")
        return
    
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("⚠️  No PDF files found in ./data, skipping this test")
        return
    
    pdf_path = pdf_files[0]
    print(f"\n測試文件: {pdf_path.name}")
    
    # Extract text
    reader = PdfReader(pdf_path)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text() + "\n"
    
    # Preprocess
    preprocessor = TextPreprocessor()
    cleaned_text, stats = preprocessor.preprocess(pdf_text)
    
    print(f"\n原始長度: {stats['original_length']} chars")
    print(f"清理後長度: {stats['final_length']} chars")
    print(f"移除長度: {stats.get('removed_length', 0)} chars")
    print(f"移除比例: {stats.get('removed_percentage', 0):.1f}%")
    print(f"移除的段落: {stats['removed_sections']}")
    
    # Show first 500 chars of original and cleaned
    print(f"\n原始文本前500字符:")
    print("-"*70)
    print(pdf_text[:500])
    print("-"*70)
    
    print(f"\n清理後文本前500字符:")
    print("-"*70)
    print(cleaned_text[:500])
    print("-"*70)
    
    print(f"\n原始文本最後500字符:")
    print("-"*70)
    print(pdf_text[-500:])
    print("-"*70)
    
    print(f"\n清理後文本最後500字符:")
    print("-"*70)
    print(cleaned_text[-500:])
    print("-"*70)
    
    # Validate
    is_valid = preprocessor.validate_preprocessing(pdf_text, cleaned_text)
    print(f"\n驗證結果: {'✓ 通過' if is_valid else '✗ 失敗'}")
    
    # Get statistics
    section_stats = preprocessor.get_statistics(pdf_text)
    print(f"\n文檔段落分析:")
    for section, info in section_stats.get('sections_found', {}).items():
        print(f"  {section}: 位置 {info['position']} ({info['position_percentage']:.1f}%)")


def test_multiple_pdfs():
    """Test with multiple PDFs"""
    
    print("\n\n")
    print("="*70)
    print("Test 4: Multiple PDF Files")
    print("="*70)
    
    data_dir = Path("./data")
    if not data_dir.exists():
        print("⚠️  ./data directory not found, skipping this test")
        return
    
    pdf_files = list(data_dir.glob("*.pdf"))[:5]  # Test first 5 PDFs
    if not pdf_files:
        print("⚠️  No PDF files found in ./data, skipping this test")
        return
    
    print(f"\n測試 {len(pdf_files)} 個 PDF 文件...")
    
    preprocessor = TextPreprocessor()
    results = []
    
    for pdf_path in pdf_files:
        try:
            # Extract text
            reader = PdfReader(pdf_path)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text() + "\n"
            
            # Preprocess
            cleaned_text, stats = preprocessor.preprocess(pdf_text)
            
            results.append({
                'filename': pdf_path.name,
                'original_length': stats['original_length'],
                'final_length': stats['final_length'],
                'removed_percentage': stats.get('removed_percentage', 0),
                'removed_sections': stats['removed_sections']
            })
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")
    
    # Print summary
    print(f"\n處理結果摘要:")
    print("-"*70)
    print(f"{'檔案名稱':<30} {'原始':<10} {'清理後':<10} {'移除%':<8} {'移除段落'}")
    print("-"*70)
    
    for result in results:
        sections = ', '.join(result['removed_sections']) if result['removed_sections'] else 'None'
        print(f"{result['filename']:<30} "
              f"{result['original_length']:<10} "
              f"{result['final_length']:<10} "
              f"{result['removed_percentage']:<8.1f} "
              f"{sections}")
    
    print("-"*70)
    
    # Calculate averages
    if results:
        avg_removed = sum(r['removed_percentage'] for r in results) / len(results)
        print(f"\n平均移除比例: {avg_removed:.1f}%")
        
        sections_count = {}
        for r in results:
            for section in r['removed_sections']:
                sections_count[section] = sections_count.get(section, 0) + 1
        
        print(f"段落出現次數:")
        for section, count in sections_count.items():
            print(f"  {section}: {count}/{len(results)} ({count/len(results)*100:.0f}%)")


if __name__ == "__main__":
    print("\n🚀 Starting Text Preprocessor Tests...\n")
    
    # Run all tests
    test_preprocessor_with_sample_text()
    test_preprocessor_with_chinese_text()
    test_preprocessor_with_real_pdf()
    test_multiple_pdfs()
    
    print("\n\n✅ All tests completed!")
