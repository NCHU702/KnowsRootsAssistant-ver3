#!/usr/bin/env python3
"""
驗證實際的 PDF parsing 是否過濾掉圖表標題等誤報
"""

import sys
sys.path.append('./system_api')

from pdf_section_parser import PDFSectionParser

# 測試鼻咽癌論文（測試腳本報告有很多 "遺漏"）
pdf_path = './data/基於深度學習之鼻咽癌腫塊辨識_20251106_144224.pdf'

parser = PDFSectionParser()
sections = parser.parse(pdf_path)

print("=" * 80)
print(f"實際檢測到的章節數量: {len(sections)}")
print("=" * 80)

for i, section in enumerate(sections, 1):
    print(f"\n[{i}] {section.name}")
    print(f"    頁面: {section.start_page}-{section.end_page}")
    print(f"    字數: {section.char_count}")
    print(f"    內容預覽: {section.text[:100]}...")

print("\n" + "=" * 80)
print("驗證結果:")
print("=" * 80)

# 檢查是否包含測試腳本報告的 "遺漏"（誤報）
false_positives = [
    "Figure 1. Dataset shape",
    "表4.2 資料集比較",
    "由於資料集為股票交易的歷史資料",
    "在此也要感謝實驗室的學長"
]

found_false_positives = []
for section in sections:
    for fp in false_positives:
        if fp in section.name:
            found_false_positives.append((section.name, fp))

if found_false_positives:
    print("❌ 發現誤報章節標題:")
    for name, fp in found_false_positives:
        print(f"   - {name} (包含: {fp})")
else:
    print("✅ 沒有發現圖表標題等誤報！")
    print("✅ 所有章節標題都是真正的 section headers")

print("\n檢測到的 Dataset 章節:")
dataset_sections = [s for s in sections if 'dataset' in s.name.lower()]
for ds in dataset_sections:
    print(f"  • {ds.name} (頁 {ds.start_page}-{ds.end_page}, {ds.char_count} 字)")
