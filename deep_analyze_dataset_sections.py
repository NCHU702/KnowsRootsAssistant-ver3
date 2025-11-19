"""
深度分析所有論文中 Dataset 介紹的真實位置
不依賴 section parser，直接讀取 PDF 原文
"""
import fitz
import os
import re

def extract_text_with_pages(pdf_path):
    """提取 PDF 每一頁的文本"""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        pages.append({
            'page_num': page_num + 1,
            'text': text
        })
    doc.close()
    return pages

def find_dataset_sections(pages):
    """
    在所有頁面中找出包含 dataset 資訊的區域
    不限定標題，而是找實際內容
    """
    dataset_mentions = []
    
    # 更精確的模式：找實際的數據集描述內容
    patterns = [
        # 明確的數據集介紹
        r'(?:使用|採用|收集|來自|取自|選用).*?(?:資料集|數據集|dataset|data)',
        r'(?:資料集|數據集|dataset).*?(?:包含|共有|總共|包括).*?(?:\d+|samples|images|筆|張)',
        r'(?:訓練|測試|驗證).*?(?:集|set).*?(?:\d+|samples)',
        r'(?:split|分割|劃分).*?(?:訓練|測試|training|test)',
        
        # 數據來源
        r'(?:公開|open|public).*?(?:資料集|數據集|dataset)',
        r'(?:資料|數據|data).*?(?:來源|source|collected from)',
        
        # 數據規模
        r'(?:共|總共|total).*?\d+.*?(?:筆|張|個|samples|images|instances)',
        r'\d+\s*(?:training|test|validation)\s*(?:samples|images|instances)',
        
        # 預處理
        r'(?:預處理|preprocessing|augmentation|增強)',
        r'(?:normalize|標準化|正規化)',
        r'(?:resize|調整大小|裁剪|crop)',
        
        # 具體數據集名稱
        r'(?:ImageNet|COCO|MNIST|CIFAR|Pascal|VOC|KITTI|CityScapes|NPC|MRI|CT)',
        r'(?:自行|自己|manually).*?(?:收集|標註|label)',
    ]
    
    for page in pages:
        page_num = page['page_num']
        text = page['text']
        
        # 檢查每個模式
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # 獲取匹配上下文（前後各 100 字）
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end].replace('\n', ' ').strip()
                
                dataset_mentions.append({
                    'page': page_num,
                    'pattern': pattern,
                    'matched_text': match.group(),
                    'context': context[:200]  # 限制顯示長度
                })
    
    return dataset_mentions

def analyze_section_structure(pages):
    """
    分析論文的章節結構
    找出 Dataset 相關章節的標題和位置
    """
    section_headers = []
    
    # 章節標題模式
    header_patterns = [
        # 中文章節
        r'^第[一二三四五六七八九十\d]+章\s+(.+)$',
        r'^\d+\.\s+(.+)$',
        r'^\d+\.\d+\s+(.+)$',
        
        # 英文章節
        r'^[IVX]+\.\s+(.+)$',
        r'^Chapter\s+\d+\s+(.+)$',
        r'^\d+\s+[A-Z][a-z]+',
    ]
    
    for page in pages:
        lines = page['text'].split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # 檢查是否為章節標題
            for pattern in header_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    section_headers.append({
                        'page': page['page_num'],
                        'line_num': i,
                        'header': line,
                        'next_lines': ' '.join(lines[i+1:i+4]).strip()[:150]
                    })
                    break
            
            # 檢查是否包含 dataset 關鍵字的標題（獨立一行，字數少）
            if len(line) < 50 and re.search(r'(?:資料集|數據集|dataset|data\s+collection|experimental\s+setup)', line, re.IGNORECASE):
                section_headers.append({
                    'page': page['page_num'],
                    'line_num': i,
                    'header': line,
                    'type': 'dataset_related',
                    'next_lines': ' '.join(lines[i+1:i+4]).strip()[:150]
                })
    
    return section_headers

def main():
    data_dir = './data'
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    print(f"📚 找到 {len(pdf_files)} 個 PDF 檔案\n")
    print("=" * 100)
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        print(f"\n📄 論文: {pdf_file}")
        print("=" * 100)
        
        try:
            # 提取頁面文本
            pages = extract_text_with_pages(pdf_path)
            print(f"✓ 共 {len(pages)} 頁")
            
            # 分析章節結構
            print("\n【章節結構】")
            section_headers = analyze_section_structure(pages)
            dataset_related_sections = [s for s in section_headers if 'dataset' in s.get('header', '').lower() or '資料集' in s.get('header', '') or '數據集' in s.get('header', '')]
            
            if dataset_related_sections:
                print(f"找到 {len(dataset_related_sections)} 個 Dataset 相關章節：")
                for section in dataset_related_sections:
                    print(f"  📍 第 {section['page']} 頁: {section['header']}")
                    print(f"     內容預覽: {section['next_lines']}")
            else:
                print("  ⚠️  沒有找到明確的 Dataset 章節標題")
            
            # 找出包含 dataset 資訊的段落
            print("\n【Dataset 資訊出現位置】")
            dataset_mentions = find_dataset_sections(pages)
            
            # 按頁面分組統計
            page_counts = {}
            for mention in dataset_mentions:
                page_counts[mention['page']] = page_counts.get(mention['page'], 0) + 1
            
            # 顯示出現最多的前 3 頁
            top_pages = sorted(page_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"找到 {len(dataset_mentions)} 處 Dataset 相關內容")
            print(f"主要集中在:")
            for page, count in top_pages:
                print(f"  📍 第 {page} 頁: {count} 處提及")
            
            # 顯示一些具體例子
            print("\n【具體內容範例】（前 5 個）")
            for i, mention in enumerate(dataset_mentions[:5], 1):
                print(f"\n  例子 {i} - 第 {mention['page']} 頁:")
                print(f"  匹配: {mention['matched_text']}")
                print(f"  上下文: ...{mention['context']}...")
            
            print("\n" + "-" * 100)
            
        except Exception as e:
            print(f"❌ 讀取失敗: {e}")
            continue

if __name__ == '__main__':
    main()
