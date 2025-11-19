"""
直接讀取鼻咽癌論文的「第三章 資料集」內容
"""
import fitz

pdf_path = './data/基於深度學習之鼻咽癌腫塊辨識_20251106_144224.pdf'
doc = fitz.open(pdf_path)

print("=" * 100)
print("📄 基於深度學習之鼻咽癌腫塊辨識")
print("=" * 100)

# 讀取第 31 頁開始的內容（章節：資料集）
for page_num in range(30, 35):  # 第 31-35 頁 (0-indexed: 30-34)
    page = doc[page_num]
    text = page.get_text()
    print(f"\n{'='*100}")
    print(f"第 {page_num + 1} 頁")
    print(f"{'='*100}")
    print(text)

doc.close()
