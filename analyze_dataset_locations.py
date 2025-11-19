#!/usr/bin/env python3
"""
分析论文中 dataset 相关信息出现的位置
"""
import os
import re
from collections import defaultdict
import fitz  # PyMuPDF

def analyze_pdf_for_dataset(pdf_path):
    """分析单个 PDF 中 dataset 相关信息的位置"""
    
    results = {
        'filename': os.path.basename(pdf_path),
        'sections_found': [],
        'dataset_mentions': []
    }
    
    try:
        doc = fitz.open(pdf_path)
        
        # 定义 dataset 相关关键词
        dataset_keywords = [
            'dataset', 'data set', 'data collection', 'experimental data',
            '資料集', '数据集', '資料來源', '数据来源', '實驗資料', '实验数据',
            'training data', 'test data', 'validation data'
        ]
        
        # 定义章节标题模式
        section_patterns = [
            (r'^(Abstract|摘要)\s*$', 'Abstract'),
            (r'^(\d+\.?\s*)?Introduction', 'Introduction'),
            (r'^(\d+\.?\s*)?Related Work', 'Related Work'),
            (r'^(\d+\.?\s*)?Method', 'Methodology'),
            (r'^(\d+\.?\s*)?Experiment', 'Experiments'),
            (r'^(\d+\.?\s*)?(Result|Performance)', 'Results'),
            (r'^(\d+\.?\s*)?Dataset', 'Dataset Section'),
            (r'^(\d+\.?\s*)?Data\s', 'Data Section'),
        ]
        
        current_section = 'Unknown'
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            lines = text.split('\n')
            
            for line_num, line in enumerate(lines):
                line_lower = line.strip().lower()
                
                # 检测章节标题
                for pattern, section_name in section_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        current_section = section_name
                        if section_name not in [s['name'] for s in results['sections_found']]:
                            results['sections_found'].append({
                                'name': section_name,
                                'page': page_num + 1,
                                'line': line.strip()[:80]
                            })
                        break
                
                # 检测 dataset 关键词
                for keyword in dataset_keywords:
                    if keyword.lower() in line_lower:
                        context_start = max(0, line_num - 1)
                        context_end = min(len(lines), line_num + 2)
                        context = '\n'.join(lines[context_start:context_end])
                        
                        results['dataset_mentions'].append({
                            'page': page_num + 1,
                            'section': current_section,
                            'keyword': keyword,
                            'context': context[:200]
                        })
                        break
        
        doc.close()
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def main():
    """主函数：分析所有 PDF"""
    pdf_dir = './test_data'
    
    if not os.path.exists(pdf_dir):
        print(f"❌ Directory not found: {pdf_dir}")
        return
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    print(f"\n{'='*80}")
    print(f"📊 分析 {len(pdf_files)} 篇論文中的 Dataset 資訊位置")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"🔍 分析: {pdf_file[:60]}...")
        
        result = analyze_pdf_for_dataset(pdf_path)
        all_results.append(result)
        
        if 'error' in result:
            print(f"   ❌ 錯誤: {result['error']}\n")
            continue
        
        print(f"   找到章節: {len(result['sections_found'])}")
        for section in result['sections_found']:
            print(f"      • {section['name']} (第 {section['page']} 頁)")
        
        print(f"   Dataset 提及次數: {len(result['dataset_mentions'])}")
        if result['dataset_mentions']:
            section_counts = defaultdict(int)
            for mention in result['dataset_mentions']:
                section_counts[mention['section']] += 1
            
            print(f"   出現位置分佈:")
            for section, count in sorted(section_counts.items(), key=lambda x: -x[1]):
                print(f"      • {section}: {count} 次")
        
        print()
    
    # 統計總結
    print(f"\n{'='*80}")
    print("📈 總體統計")
    print(f"{'='*80}\n")
    
    section_distribution = defaultdict(int)
    total_mentions = 0
    
    for result in all_results:
        for mention in result.get('dataset_mentions', []):
            section_distribution[mention['section']] += 1
            total_mentions += 1
    
    print(f"總 Dataset 提及次數: {total_mentions}")
    print(f"\n最常出現的章節:")
    for section, count in sorted(section_distribution.items(), key=lambda x: -x[1])[:10]:
        percentage = (count / total_mentions * 100) if total_mentions > 0 else 0
        print(f"   {section:25s}: {count:3d} 次 ({percentage:5.1f}%)")
    
    # 詳細範例
    print(f"\n{'='*80}")
    print("📝 典型範例 (前5個)")
    print(f"{'='*80}\n")
    
    for result in all_results[:2]:
        if result.get('dataset_mentions'):
            print(f"📄 {result['filename'][:60]}")
            for mention in result['dataset_mentions'][:2]:
                print(f"\n   📍 位置: 第 {mention['page']} 頁 - {mention['section']}")
                print(f"   🔑 關鍵詞: {mention['keyword']}")
                print(f"   📝 內容: {mention['context'][:150]}...")
            print()

if __name__ == '__main__':
    main()
