#!/usr/bin/env python3
"""
分析 test_data 中論文的實際章節格式

目的：
1. 提取所有論文的章節標題
2. 統計常見格式
3. 找出當前 pattern 無法匹配的標題
"""

import os
import re
import fitz  # PyMuPDF
from collections import Counter
from pathlib import Path

def extract_potential_headers(pdf_path):
    """提取 PDF 中可能是章節標題的行"""
    doc = fitz.open(pdf_path)
    
    potential_headers = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")
        
        for block in blocks:
            if block[6] == 0:  # Text block
                text = block[4].strip()
                lines = text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    
                    # 篩選條件：可能是標題的行
                    if line and len(line) < 100:  # 標題通常較短
                        # 可能是標題的特徵
                        is_numbered = bool(re.match(r'^[\d一二三四五六七八九十IVX]+[\.、\s]', line))
                        is_short = len(line) < 50
                        is_caps = line.isupper() or (line[0].isupper() if line else False)
                        has_keywords = any(kw in line.lower() for kw in [
                            'abstract', 'introduction', 'method', 'result', 'conclusion',
                            'experiment', 'discussion', 'related', 'background', 'approach',
                            '摘要', '導論', '緒論', '方法', '結果', '結論', '實驗', '討論',
                            '相關', '文獻', '研究', '系統', '模型', '分析', '評估'
                        ])
                        
                        if (is_numbered or has_keywords) and is_short:
                            potential_headers.append({
                                'text': line,
                                'page': page_num,
                                'pdf': os.path.basename(pdf_path)
                            })
    
    doc.close()
    return potential_headers

def analyze_all_pdfs():
    """分析所有 PDF"""
    # 搜尋 test_data 和 data 目錄
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
        print(f"\n[{i}/{len(pdf_files)}] 分析：{pdf_path.name}")
        try:
            headers = extract_potential_headers(str(pdf_path))
            all_headers.extend(headers)
            print(f"  找到 {len(headers)} 個潛在標題")
        except Exception as e:
            print(f"  ❌ 錯誤：{e}")
    
    return all_headers

def categorize_headers(headers):
    """分類章節標題"""
    categories = {
        'abstract': [],
        'introduction': [],
        'related_work': [],
        'methodology': [],
        'results': [],
        'discussion': [],
        'conclusion': [],
        'references': [],
        'other': []
    }
    
    for h in headers:
        text_lower = h['text'].lower()
        
        # 分類邏輯
        if 'abstract' in text_lower or '摘要' in h['text']:
            categories['abstract'].append(h)
        elif 'introduction' in text_lower or '導論' in h['text'] or '緒論' in h['text']:
            categories['introduction'].append(h)
        elif 'related' in text_lower or 'literature' in text_lower or 'background' in text_lower or '相關' in h['text'] or '文獻' in h['text']:
            categories['related_work'].append(h)
        elif 'method' in text_lower or 'approach' in text_lower or '方法' in h['text']:
            categories['methodology'].append(h)
        elif 'result' in text_lower or 'experiment' in text_lower or '結果' in h['text'] or '實驗' in h['text']:
            categories['results'].append(h)
        elif 'discussion' in text_lower or '討論' in h['text']:
            categories['discussion'].append(h)
        elif 'conclusion' in text_lower or '結論' in h['text']:
            categories['conclusion'].append(h)
        elif 'reference' in text_lower or 'bibliography' in text_lower or '參考文獻' in h['text']:
            categories['references'].append(h)
        else:
            categories['other'].append(h)
    
    return categories

def print_report(categories):
    """輸出分析報告"""
    print("\n" + "="*80)
    print("章節標題格式分析報告")
    print("="*80)
    
    for category, headers in categories.items():
        if not headers:
            continue
        
        print(f"\n【{category.upper()}】({len(headers)} 個)")
        print("-"*80)
        
        # 統計獨特的格式
        unique_formats = {}
        for h in headers:
            text = h['text']
            if text not in unique_formats:
                unique_formats[text] = []
            unique_formats[text].append(h['pdf'])
        
        # 按出現次數排序
        sorted_formats = sorted(unique_formats.items(), key=lambda x: len(x[1]), reverse=True)
        
        for text, pdfs in sorted_formats[:20]:  # 只顯示前 20 個
            count = len(pdfs)
            print(f"  [{count:2d}x] {text}")
            if count <= 3:  # 如果少於 3 次，顯示來源
                for pdf in pdfs:
                    print(f"        └─ {pdf}")

def extract_numbering_patterns(headers):
    """提取編號格式"""
    print("\n" + "="*80)
    print("章節編號格式統計")
    print("="*80)
    
    patterns = []
    
    for h in headers:
        text = h['text']
        
        # 提取編號格式
        if re.match(r'^\d+\.', text):
            patterns.append(('數字+點', re.match(r'^(\d+\.)', text).group(1)))
        elif re.match(r'^\d+\s', text):
            patterns.append(('數字+空格', re.match(r'^(\d+)\s', text).group(1)))
        elif re.match(r'^[IVX]+\.', text):
            patterns.append(('羅馬數字+點', re.match(r'^([IVX]+\.)', text).group(1)))
        elif re.match(r'^[一二三四五六七八九十]+[、．]', text):
            patterns.append(('中文數字+、或．', re.match(r'^([一二三四五六七八九十]+[、．])', text).group(1)))
        elif re.match(r'^第[一二三四五六七八九十]+章', text):
            patterns.append(('第X章', re.match(r'^(第[一二三四五六七八九十]+章)', text).group(1)))
    
    pattern_counter = Counter([p[0] for p in patterns])
    
    print("\n編號格式統計：")
    for pattern_type, count in pattern_counter.most_common():
        print(f"  {pattern_type}: {count} 次")
        # 顯示範例
        examples = [p[1] for p in patterns if p[0] == pattern_type][:5]
        print(f"    範例：{', '.join(set(examples))}")

if __name__ == "__main__":
    print("\n🔍 開始分析 test_data 中的論文章節格式...")
    
    # 分析所有 PDF
    all_headers = analyze_all_pdfs()
    
    print(f"\n\n總共找到 {len(all_headers)} 個潛在章節標題")
    
    # 分類標題
    categories = categorize_headers(all_headers)
    
    # 輸出報告
    print_report(categories)
    
    # 編號格式分析
    extract_numbering_patterns(all_headers)
    
    print("\n" + "="*80)
    print("✅ 分析完成！")
    print("="*80)
