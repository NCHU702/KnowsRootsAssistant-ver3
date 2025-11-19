"""
精確分析：找出所有論文中真實的「資料集」章節
不是簡單的關鍵字匹配，而是找整章節
"""
import fitz
import os
import re

def find_dataset_chapter(pdf_path):
    """
    找出論文中的「資料集」章節
    返回：章節標題、起始頁、結束頁、實際內容
    """
    doc = fitz.open(pdf_path)
    results = []
    
    # 第一步：找出所有可能的章節標題
    chapter_pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 檢查是否為資料集相關章節標題
            # 格式1: "第三章 資料集"
            # 格式2: "3.1 資料集介紹"
            # 格式3: "Dataset"、"Data Collection"
            if re.match(r'^第[一二三四五六七八九十]+章\s*資料集', line):
                chapter_pages.append({
                    'page': page_num + 1,
                    'title': line,
                    'type': '章節標題（中文）',
                    'pattern': '第X章 資料集'
                })
            elif re.match(r'^\d+\.?\d*\.?\s*資料集', line):
                chapter_pages.append({
                    'page': page_num + 1,
                    'title': line,
                    'type': '節標題（中文）',
                    'pattern': 'X.X 資料集'
                })
            elif re.match(r'^Chapter\s+\d+.*Dataset', line, re.IGNORECASE):
                chapter_pages.append({
                    'page': page_num + 1,
                    'title': line,
                    'type': '章節標題（英文）',
                    'pattern': 'Chapter X Dataset'
                })
            elif re.match(r'^\d+\.?\d*\.?\s*Dataset', line, re.IGNORECASE):
                chapter_pages.append({
                    'page': page_num + 1,
                    'title': line,
                    'type': '節標題（英文）',
                    'pattern': 'X.X Dataset'
                })
            elif re.match(r'^Data\s+Collection', line, re.IGNORECASE):
                chapter_pages.append({
                    'page': page_num + 1,
                    'title': line,
                    'type': '節標題（英文）',
                    'pattern': 'Data Collection'
                })
    
    # 第二步：對於找到的章節，提取實際內容
    for chapter in chapter_pages:
        page_num = chapter['page'] - 1
        page = doc[page_num]
        text = page.get_text()
        
        # 提取章節標題後的 500 字
        title_pos = text.find(chapter['title'])
        if title_pos != -1:
            content_preview = text[title_pos:title_pos+800].replace('\n', ' ').strip()
            chapter['content_preview'] = content_preview
        else:
            chapter['content_preview'] = "（未找到內容）"
        
        results.append(chapter)
    
    doc.close()
    return results

def main():
    data_dir = './data'
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    print(f"📚 分析 {len(pdf_files)} 個 PDF 論文")
    print("=" * 120)
    
    total_found = 0
    papers_with_dataset = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        
        try:
            chapters = find_dataset_chapter(pdf_path)
            
            if chapters:
                total_found += len(chapters)
                papers_with_dataset.append({
                    'file': pdf_file,
                    'chapters': chapters
                })
        except Exception as e:
            print(f"❌ 讀取失敗 {pdf_file}: {e}")
            continue
    
    # 顯示結果
    print(f"\n✅ 找到 {len(papers_with_dataset)} 篇論文包含明確的「資料集」章節")
    print(f"✅ 總共 {total_found} 個資料集相關章節\n")
    print("=" * 120)
    
    for paper in papers_with_dataset:
        print(f"\n📄 {paper['file']}")
        print("=" * 120)
        for chapter in paper['chapters']:
            print(f"  📍 第 {chapter['page']} 頁")
            print(f"     標題: {chapter['title']}")
            print(f"     類型: {chapter['type']}")
            print(f"     模式: {chapter['pattern']}")
            print(f"     內容預覽:")
            # 分段顯示，避免太長
            preview = chapter['content_preview'][:500]
            for i in range(0, len(preview), 100):
                print(f"       {preview[i:i+100]}")
            print()
    
    # 統計分析
    print("\n" + "=" * 120)
    print("📊 統計分析")
    print("=" * 120)
    
    pattern_count = {}
    for paper in papers_with_dataset:
        for chapter in paper['chapters']:
            pattern = chapter['pattern']
            pattern_count[pattern] = pattern_count.get(pattern, 0) + 1
    
    print("\n章節標題模式分布：")
    for pattern, count in sorted(pattern_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count} 次")
    
    print(f"\n沒有找到資料集章節的論文數: {len(pdf_files) - len(papers_with_dataset)}")

if __name__ == '__main__':
    main()
