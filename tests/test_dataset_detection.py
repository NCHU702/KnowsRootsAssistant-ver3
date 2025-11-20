"""
測試修正後的 dataset 章節檢測
驗證能否正確識別「第三章 資料集與初步整理」
"""
import sys
sys.path.append('./system_api')

from pdf_section_parser import PDFSectionParser

# 測試鼻咽癌論文
pdf_path = './data/基於深度學習之鼻咽癌腫塊辨識_20251106_144224.pdf'
parser = PDFSectionParser()

print("=" * 100)
print("測試：鼻咽癌論文的章節解析")
print("=" * 100)

sections = parser.parse(pdf_path)

print(f"\n✓ 共解析出 {len(sections)} 個章節\n")

# 找出 dataset 相關章節（使用檢測邏輯）
dataset_sections = []
for s in sections:
    name_lower = s.name.lower()
    is_dataset = (
        '資料集' in name_lower or
        '数据集' in name_lower or
        'dataset' in name_lower or
        'data collection' in name_lower or
        ('data' in name_lower and
         (name_lower.strip() == 'data' or
          name_lower.startswith('data ') or
          name_lower.endswith(' data')))
    )
    if is_dataset:
        dataset_sections.append(s)

print(f"🔍 找到 {len(dataset_sections)} 個 Dataset 章節：\n")

for section in dataset_sections:
    print(f"  📍 章節名稱: {section.name}")
    print(f"     頁數: {section.start_page + 1} - {section.end_page + 1}")  # +1 因為 0-indexed
    print(f"     字數: {section.char_count}")
    print(f"     內容預覽: {section.text[:200].replace(chr(10), ' ')}...")
    print()

# 測試檢測邏輯
print("\n" + "=" * 100)
print("測試：檢測邏輯")
print("=" * 100)

test_section_names = [
    "第三章 資料集與初步整理",
    "5.1 資料集介紹",
    "6.1 資料集與實驗參數介紹",
    "4.1 Dataset",
    "Chapter 3 Dataset",
    "Data Collection",
    "第二章 方法",  # 應該不匹配
    "Introduction",  # 應該不匹配
]

for name in test_section_names:
    name_lower = name.lower()
    is_dataset = (
        '資料集' in name_lower or
        '数据集' in name_lower or
        'dataset' in name_lower or
        'data collection' in name_lower or
        ('data' in name_lower and
         (name_lower.strip() == 'data' or
          name_lower.startswith('data ') or
          name_lower.endswith(' data')))
    )
    result = "✅ 匹配" if is_dataset else "❌ 不匹配"
    print(f"{result} - {name}")
