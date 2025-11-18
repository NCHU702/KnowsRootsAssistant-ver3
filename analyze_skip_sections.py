#!/usr/bin/env python3
"""
分析 data 中所有論文的低價值章節格式

專門分析：
- 參考文獻 (References)
- 附錄 (Appendix)
- 致謝 (Acknowledgements)
- 目錄 (Table of Contents)
- 表目錄 (List of Tables)
- 圖目錄 (List of Figures)
"""

import os
import re
import fitz  # PyMuPDF
from collections import Counter
from pathlib import Path

def extract_skip_section_headers(pdf_path):
    """提取 PDF 中低價值章節的標題"""
    doc = fitz.open(pdf_path)
    
    potential_headers = []
    
    # 關鍵詞列表
    keywords = {
        'references': ['reference', 'bibliography', '參考文獻', '参考文献', '引用文獻', '文獻'],
        'appendix': ['appendix', 'appendices', '附錄', '附录'],
        'acknowledgements': ['acknowledgement', 'acknowledgment', '致謝', '誌謝', '謝辭', '謝誌'],
        'toc': ['table of contents', 'contents', '目錄', '目录'],
        'lot': ['list of tables', '表目錄', '表目录'],
        'lof': ['list of figures', '圖目錄', '图目录']
    }
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")
        
        for block in blocks:
            if block[6] == 0:  # Text block
                text = block[4].strip()
                lines = text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    
                    # 檢查是否包含關鍵詞
                    if line and len(line) < 100:  # 標題通常較短
                        line_lower = line.lower()
                        
                        for category, kw_list in keywords.items():
                            if any(kw in line_lower or kw in line for kw in kw_list):
                                # 排除明顯不是標題的情況
                                if not any(exclude in line_lower for exclude in [
                                    'http', 'www', 'doi', '@', '.com', '.org',
                                    '等人', '學者', '研究', '使用', '透過', '方法',
                                    '圖表', '如圖', '如表', '見圖', '見表'
                                ]):
                                    potential_headers.append({
                                        'text': line,
                                        'category': category,
                                        'page': page_num,
                                        'pdf': os.path.basename(pdf_path)
                                    })
                                    break  # 找到一個分類就跳出
    
    doc.close()
    return potential_headers

def analyze_all_pdfs():
    """分析所有 PDF"""
    search_dirs = [Path("./test_data"), Path("./data")]
    pdf_files = []
    
    for search_dir in search_dirs:
        if search_dir.exists():
            found = list(search_dir.glob("*.pdf"))
            pdf_files.extend(found)
            print(f"  {search_dir}: {len(found)} 個 PDF")
    
    print(f"\n總共找到 {len(pdf_files)} 個 PDF 檔案\n")
    print("="*80)
    
    all_headers = []
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\r[{i}/{len(pdf_files)}] 分析：{pdf_path.name[:60]}", end='', flush=True)
        try:
            headers = extract_skip_section_headers(str(pdf_path))
            all_headers.extend(headers)
        except Exception as e:
            print(f"\n  ❌ 錯誤：{e}")
    
    print("\n")
    return all_headers

def categorize_and_print(headers):
    """分類並輸出結果"""
    categories = {
        'references': [],
        'appendix': [],
        'acknowledgements': [],
        'toc': [],
        'lot': [],
        'lof': []
    }
    
    for h in headers:
        categories[h['category']].append(h)
    
    print("\n" + "="*80)
    print("低價值章節格式分析報告")
    print("="*80)
    
    category_names = {
        'references': '【參考文獻 / References】',
        'appendix': '【附錄 / Appendix】',
        'acknowledgements': '【致謝 / Acknowledgements】',
        'toc': '【目錄 / Table of Contents】',
        'lot': '【表目錄 / List of Tables】',
        'lof': '【圖目錄 / List of Figures】'
    }
    
    for category, headers in categories.items():
        if not headers:
            continue
        
        print(f"\n{category_names[category]} ({len(headers)} 個)")
        print("-"*80)
        
        # 統計獨特格式
        unique_formats = {}
        for h in headers:
            text = h['text']
            if text not in unique_formats:
                unique_formats[text] = []
            unique_formats[text].append(h['pdf'])
        
        # 按出現次數排序
        sorted_formats = sorted(unique_formats.items(), key=lambda x: len(x[1]), reverse=True)
        
        for text, pdfs in sorted_formats[:30]:  # 顯示前 30 個
            count = len(pdfs)
            print(f"  [{count:2d}x] {text}")
            if count <= 3:  # 如果少於等於 3 次，顯示來源
                for pdf in pdfs[:3]:
                    print(f"        └─ {pdf}")

def generate_pattern_recommendations(headers):
    """生成 Pattern 建議"""
    print("\n" + "="*80)
    print("建議的 skip_section_keywords Pattern")
    print("="*80)
    
    categories = {
        'references': [],
        'appendix': [],
        'acknowledgements': [],
        'toc': [],
        'lot': [],
        'lof': []
    }
    
    for h in headers:
        categories[h['category']].append(h['text'])
    
    print("\n```python")
    print("skip_section_keywords = [")
    
    # References
    refs = set(categories['references'])
    print("    # References (參考文獻)")
    common_refs = ['references', 'reference', 'bibliography', '參考文獻', '参考文献', '引用文獻']
    for ref in sorted(common_refs):
        print(f"    '{ref}',")
    
    # Appendix
    print("    \n    # Appendix (附錄)")
    common_app = ['appendix', 'appendices', '附錄', '附录']
    for app in sorted(common_app):
        print(f"    '{app}',")
    
    # Acknowledgements
    print("    \n    # Acknowledgements (致謝)")
    common_ack = ['acknowledgement', 'acknowledgements', 'acknowledgment', '致謝', '誌謝', '謝辭', '謝誌']
    for ack in sorted(common_ack):
        print(f"    '{ack}',")
    
    # Table of Contents
    print("    \n    # Table of Contents (目錄)")
    common_toc = ['table of contents', 'contents', '目錄', '目录']
    for toc in sorted(common_toc):
        print(f"    '{toc}',")
    
    # List of Tables
    print("    \n    # List of Tables (表目錄)")
    common_lot = ['list of tables', '表目錄', '表目录']
    for lot in sorted(common_lot):
        print(f"    '{lot}',")
    
    # List of Figures
    print("    \n    # List of Figures (圖目錄)")
    common_lof = ['list of figures', '圖目錄', '图目录']
    for lof in sorted(common_lof):
        print(f"    '{lof}',")
    
    print("]")
    print("```")

if __name__ == "__main__":
    print("\n🔍 開始分析低價值章節格式...")
    
    # 分析所有 PDF
    all_headers = analyze_all_pdfs()
    
    print(f"\n總共找到 {len(all_headers)} 個低價值章節標題")
    
    # 分類並輸出
    categorize_and_print(all_headers)
    
    # 生成建議
    generate_pattern_recommendations(all_headers)
    
    print("\n" + "="*80)
    print("✅ 分析完成！")
    print("="*80)
