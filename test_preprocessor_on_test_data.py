#!/usr/bin/env python3
"""
Test preprocessor on test_data PDFs
"""

import sys
from pathlib import Path
from PyPDF2 import PdfReader

sys.path.insert(0, str(Path(__file__).parent))

from system_api.text_preprocessor import TextPreprocessor

# Setup
test_data_dir = Path("./test_data")
pdf_files = sorted(list(test_data_dir.glob("*.pdf")))

preprocessor = TextPreprocessor()

print("="*80)
print("Testing TextPreprocessor on test_data PDFs")
print("="*80)

results = []

for pdf_path in pdf_files:
    try:
        # Extract text
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Preprocess
        cleaned_text, stats = preprocessor.preprocess(text)
        
        results.append({
            'filename': pdf_path.name,
            'original_length': stats['original_length'],
            'final_length': stats['final_length'],
            'removed_percentage': stats.get('removed_percentage', 0),
            'removed_sections': stats['removed_sections']
        })
        
        print(f"\n{pdf_path.name}")
        print(f"  原始: {stats['original_length']} chars")
        print(f"  清理後: {stats['final_length']} chars")
        print(f"  移除: {stats.get('removed_percentage', 0):.1f}%")
        if stats['removed_sections']:
            print(f"  移除段落: {', '.join(stats['removed_sections'])}")
        else:
            print(f"  移除段落: None")
        
    except Exception as e:
        print(f"\n{pdf_path.name}")
        print(f"  ✗ 錯誤: {e}")

# Summary
print("\n" + "="*80)
print("摘要")
print("="*80)
print(f"{'檔案名稱':<50} {'原始':<10} {'清理後':<10} {'移除%':<8}")
print("-"*80)

for result in results:
    print(f"{result['filename']:<50} "
          f"{result['original_length']:<10} "
          f"{result['final_length']:<10} "
          f"{result['removed_percentage']:<8.1f}")

print("-"*80)

if results:
    avg_removed = sum(r['removed_percentage'] for r in results) / len(results)
    print(f"平均移除比例: {avg_removed:.1f}%")
    
    # Count sections
    section_counts = {}
    for r in results:
        for section in r['removed_sections']:
            section_counts[section] = section_counts.get(section, 0) + 1
    
    if section_counts:
        print(f"\n段落出現次數:")
        for section, count in sorted(section_counts.items()):
            print(f"  {section}: {count}/{len(results)} ({count/len(results)*100:.0f}%)")
