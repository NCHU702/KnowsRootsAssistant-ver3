"""
全面測試 _load_patterns 中所有 pattern 的準確性
對 ./data 中所有 35 篇論文進行測試
"""
import sys
sys.path.append('./system_api')

from pdf_section_parser import PDFSectionParser
import os
import fitz
from collections import defaultdict

def get_real_section_titles(pdf_path):
    """
    手動提取 PDF 中的所有章節標題（用於對比）
    找出所有看起來像標題的文本
    """
    doc = fitz.open(pdf_path)
    potential_titles = []
    
    for page_num in range(min(len(doc), 70)):  # 只看前 70 頁（避免參考文獻）
        page = doc[page_num]
        blocks = page.get_text("blocks")
        
        for block in blocks:
            if block[6] == 0:  # 文字區塊
                text = block[4].strip()
                lines = text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    # 排除太長或太短的（不太像標題）
                    if 3 <= len(line) <= 80:
                        # 可能是標題的特徵
                        is_potential_title = (
                            # 章節編號
                            line.startswith('第') and '章' in line or
                            line.startswith('Chapter') or
                            # 數字編號
                            line[0].isdigit() and ('.' in line[:5] or line[1:3] == '  ') or
                            # 全大寫英文
                            line.isupper() and len(line) > 5 or
                            # 常見關鍵字
                            any(kw in line for kw in ['Abstract', 'Introduction', 'Method', 'Result', 
                                                       'Conclusion', 'Reference', 'Dataset', 
                                                       '摘要', '緒論', '方法', '實驗', '結論', 
                                                       '參考文獻', '資料集'])
                        )
                        
                        if is_potential_title:
                            potential_titles.append({
                                'page': page_num + 1,
                                'text': line
                            })
    
    doc.close()
    return potential_titles

def analyze_pattern_coverage(pdf_path, parser, real_titles):
    """
    分析 pattern 的覆蓋率
    """
    # 用 parser 解析
    try:
        detected_sections = parser.parse(pdf_path)
    except Exception as e:
        return None, f"解析失敗: {e}"
    
    # 統計各類型章節
    detected_by_type = defaultdict(list)
    for section in detected_sections:
        detected_by_type[section.name].append({
            'name': section.name,
            'start_page': section.start_page + 1,
            'end_page': section.end_page + 1,
            'char_count': section.char_count,
            'preview': section.text[:100].replace('\n', ' ')
        })
    
    # 對比真實標題
    real_keywords = {
        'Abstract': ['Abstract', 'ABSTRACT', '摘要'],
        'Introduction': ['Introduction', 'INTRODUCTION', '緒論', '導論', '第一章'],
        'Related_work': ['Related Work', 'Literature Review', 'Background', '文獻', '相關研究'],
        'Methodology': ['Method', 'Methodology', 'Approach', '研究方法', '方法'],
        'Results': ['Result', 'Experiment', 'Simulation', '實驗', '結果'],
        'Discussion': ['Discussion', '討論'],
        'Conclusion': ['Conclusion', '結論'],
        'Dataset': ['Dataset', 'Data Collection', '資料集', '数据集', 'Data'],
        'References': ['Reference', 'Bibliography', '參考文獻']
    }
    
    # 檢查是否有遺漏
    missed_sections = defaultdict(list)
    for title_info in real_titles:
        title = title_info['text']
        matched_type = None
        
        for section_type, keywords in real_keywords.items():
            if any(kw in title for kw in keywords):
                matched_type = section_type
                break
        
        if matched_type:
            # 檢查是否被 parser 檢測到
            detected_names = [s.name.lower() for s in detected_sections]
            if matched_type.lower() not in detected_names:
                missed_sections[matched_type].append({
                    'page': title_info['page'],
                    'title': title
                })
    
    return {
        'detected': detected_by_type,
        'missed': missed_sections,
        'total_detected': len(detected_sections),
        'real_titles_count': len(real_titles)
    }, None

def main():
    data_dir = './data'
    pdf_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pdf')])
    
    parser = PDFSectionParser(min_sections=3)
    
    print("=" * 120)
    print("全面 Pattern 準確性測試")
    print(f"測試論文數量: {len(pdf_files)}")
    print("=" * 120)
    
    # 統計資料
    total_stats = {
        'total_papers': len(pdf_files),
        'successful_parses': 0,
        'failed_parses': 0,
        'section_type_counts': defaultdict(int),
        'missed_by_type': defaultdict(int),
        'papers_with_issues': []
    }
    
    for idx, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(data_dir, pdf_file)
        
        print(f"\n[{idx}/{len(pdf_files)}] 📄 {pdf_file}")
        print("-" * 120)
        
        # 提取真實標題
        print("  🔍 提取真實章節標題...")
        real_titles = get_real_section_titles(pdf_path)
        print(f"  ✓ 找到 {len(real_titles)} 個潛在標題")
        
        # 分析 pattern 覆蓋率
        print("  🔍 測試 pattern 檢測...")
        result, error = analyze_pattern_coverage(pdf_path, parser, real_titles)
        
        if error:
            print(f"  ❌ {error}")
            total_stats['failed_parses'] += 1
            continue
        
        total_stats['successful_parses'] += 1
        
        # 顯示檢測結果
        print(f"  ✓ 檢測到 {result['total_detected']} 個章節：")
        for section_type, sections in sorted(result['detected'].items()):
            count = len(sections)
            total_stats['section_type_counts'][section_type] += count
            print(f"     • {section_type}: {count} 個")
            for sec in sections[:2]:  # 只顯示前 2 個
                print(f"       - 頁 {sec['start_page']}-{sec['end_page']}: {sec['preview'][:80]}...")
        
        # 顯示遺漏的章節
        if result['missed']:
            print(f"  ⚠️  可能遺漏的章節:")
            has_issues = False
            for section_type, missed_list in result['missed'].items():
                if missed_list:
                    has_issues = True
                    total_stats['missed_by_type'][section_type] += len(missed_list)
                    print(f"     • {section_type}: {len(missed_list)} 個")
                    for missed in missed_list[:2]:  # 只顯示前 2 個
                        print(f"       - 頁 {missed['page']}: {missed['title']}")
            
            if has_issues:
                total_stats['papers_with_issues'].append({
                    'file': pdf_file,
                    'missed': dict(result['missed'])
                })
    
    # 總結報告
    print("\n" + "=" * 120)
    print("📊 總結報告")
    print("=" * 120)
    
    print(f"\n✅ 成功解析: {total_stats['successful_parses']}/{total_stats['total_papers']}")
    print(f"❌ 解析失敗: {total_stats['failed_parses']}/{total_stats['total_papers']}")
    
    print(f"\n📈 各類型章節檢測統計:")
    for section_type, count in sorted(total_stats['section_type_counts'].items(), 
                                      key=lambda x: x[1], reverse=True):
        print(f"  • {section_type}: {count} 次")
    
    if total_stats['missed_by_type']:
        print(f"\n⚠️  遺漏章節統計（需要優化 pattern）:")
        for section_type, count in sorted(total_stats['missed_by_type'].items(), 
                                          key=lambda x: x[1], reverse=True):
            print(f"  • {section_type}: {count} 次遺漏")
    
    if total_stats['papers_with_issues']:
        print(f"\n⚠️  有遺漏問題的論文 ({len(total_stats['papers_with_issues'])} 篇):")
        for paper_info in total_stats['papers_with_issues'][:10]:  # 只顯示前 10 篇
            print(f"\n  📄 {paper_info['file']}")
            for section_type, missed_list in paper_info['missed'].items():
                if missed_list:
                    print(f"     • 遺漏 {section_type}:")
                    for missed in missed_list[:2]:
                        print(f"       - 頁 {missed['page']}: {missed['title']}")
    
    # 計算準確率
    if total_stats['successful_parses'] > 0:
        avg_sections = sum(total_stats['section_type_counts'].values()) / total_stats['successful_parses']
        print(f"\n📊 平均每篇論文檢測章節數: {avg_sections:.1f}")
        
        total_missed = sum(total_stats['missed_by_type'].values())
        total_detected = sum(total_stats['section_type_counts'].values())
        if total_detected + total_missed > 0:
            accuracy = (total_detected / (total_detected + total_missed)) * 100
            print(f"📊 整體檢測準確率: {accuracy:.1f}%")
    
    print("\n" + "=" * 120)
    print("✅ 測試完成！")
    print("=" * 120)

if __name__ == '__main__':
    main()
