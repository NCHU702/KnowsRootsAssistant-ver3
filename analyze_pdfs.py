#!/usr/bin/env python3
"""
Analyze PDF structure to improve text preprocessing
"""

import sys
from pathlib import Path
from PyPDF2 import PdfReader

# Setup
test_data_dir = Path("./test_data")
pdf_files = list(test_data_dir.glob("*.pdf"))[:3]  # First 3 PDFs

print("="*80)
print("PDF Structure Analysis")
print("="*80)

for pdf_path in pdf_files:
    print(f"\n\n{'='*80}")
    print(f"檔案: {pdf_path.name}")
    print(f"{'='*80}\n")
    
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        # Extract first 10 pages
        for i, page in enumerate(reader.pages[:10]):
            text += page.extract_text() + "\n"
        
        print(f"總頁數: {len(reader.pages)}")
        print(f"前10頁字符數: {len(text)}")
        print(f"\n前3000字符:")
        print("-"*80)
        print(text[:3000])
        print("-"*80)
        
        # Find key sections
        sections_to_find = [
            ("目錄", r"目錄"),
            ("表目錄", r"表目錄"),
            ("圖目錄", r"圖目錄"),
            ("摘要", r"摘要"),
            ("Abstract", r"Abstract"),
            ("前言/緒論", r"(前言|緒論|第一章)"),
            ("參考文獻", r"參考文獻"),
            ("致謝", r"(致謝|誌謝)"),
        ]
        
        print(f"\n\n段落位置分析:")
        print("-"*80)
        
        import re
        for section_name, pattern in sections_to_find:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                pos = match.start()
                percentage = (pos / len(text)) * 100
                print(f"{section_name:<15} 位置: {pos:>6} ({percentage:>5.1f}%)")
            else:
                print(f"{section_name:<15} 未找到")
        
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()

print("\n\n" + "="*80)
print("分析完成")
print("="*80)
