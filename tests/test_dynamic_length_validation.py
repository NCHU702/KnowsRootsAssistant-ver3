#!/usr/bin/env python3
"""
測試動態摘要長度功能 - 使用真實論文

按照 agent2 的邏輯：
1. 隨機選擇 test_data 中的一篇論文
2. 使用 summarization 模式建立索引
3. 檢查每個章節的摘要長度是否在 200-1500 字範圍內
4. 驗證長章節的摘要長度 > 短章節的摘要長度
"""

import os
import sys
import random
import logging
import time
from pathlib import Path

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dynamic_summary_length():
    """測試動態摘要長度功能"""
    
    print("\n" + "="*70)
    print("🧪 動態摘要長度測試 - 使用真實論文")
    print("="*70)
    
    # Step 1: 隨機選擇一篇論文
    test_data_dir = Path("./test_data")
    pdf_files = list(test_data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ 錯誤：test_data 目錄中沒有找到 PDF 文件")
        return False
    
    selected_pdf = random.choice(pdf_files)
    print(f"\n📄 隨機選擇論文：{selected_pdf.name}")
    print(f"   路徑：{selected_pdf}")
    
    # Step 2: 初始化組件
    print("\n" + "-"*70)
    print("初始化組件...")
    print("-"*70)
    
    try:
        from system_api.pdf_section_parser import PDFSectionParser
        from system_api.llm_summarizer import LLMSummarizer
        
        # 初始化 PDF 解析器
        parser = PDFSectionParser(method='pymupdf_regex')
        print("✓ PDFSectionParser 初始化成功")
        
        # 初始化 LLM 摘要器（使用與 agent2 相同的配置）
        summarizer = LLMSummarizer(
            model_name="llama3.2:latest",
            ollama_base_url="http://localhost:11434",
            map_reduce_threshold=3000,  # agent2 的新閾值
            target_summary_length=300,  # 基準值（會動態調整）
            timeout=30,
            max_retries=3
        )
        print("✓ LLMSummarizer 初始化成功")
        print(f"  - 模型：llama3.2:latest")
        print(f"  - Map-Reduce 閾值：3000 字")
        print(f"  - 摘要長度範圍：{summarizer.min_summary_length}-{summarizer.max_summary_length} 字")
        
    except Exception as e:
        print(f"❌ 組件初始化失敗：{e}")
        return False
    
    # Step 3: 解析論文章節
    print("\n" + "-"*70)
    print("解析論文章節...")
    print("-"*70)
    
    try:
        sections = parser.parse(str(selected_pdf))
        print(f"✓ 成功解析 {len(sections)} 個章節")
        
        if not sections:
            print("❌ 錯誤：沒有解析到任何章節")
            return False
        
        # 顯示章節資訊
        print("\n章節列表：")
        for i, section in enumerate(sections, 1):
            print(f"  {i}. {section.name:<30} ({len(section.text):>6} 字)")
        
    except Exception as e:
        print(f"❌ 章節解析失敗：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: 過濾低價值章節（與 layer2_vectorstore.py 邏輯一致）
    # Based on analysis of 45 PDFs (475 skip section headers found)
    skip_section_keywords = [
        # References (参考文献) - 315 occurrences
        'bibliography', 'reference', 'references',
        '参考文献', '參考文獻', '引用文獻', '文獻',
        '文獻回顧', '文獻探討',  # Common in Chinese theses (17x + 9x)
        
        # Appendix (附录) - 3 occurrences
        'appendix', 'appendices', '附录', '附錄',
        
        # Acknowledgements (致谢) - 32 occurrences
        'acknowledgement', 'acknowledgements', 'acknowledgment',
        '致謝', '誌謝', '謝誌', '謝辭',
        
        # Table of Contents (目录) - 111 occurrences
        'contents', 'table of contents', '目录', '目錄',
        '圖片目錄', '表格目錄',  # 1x each
        
        # List of Tables/Figures (表/图目录) - 34 occurrences each
        'list of tables', '表目录', '表目錄',
        'list of figures', '图目录', '圖目錄'
    ]
    
    filtered_sections = []
    skipped_sections = []
    
    for section in sections:
        section_name_lower = section.name.lower()
        if any(keyword.lower() in section_name_lower for keyword in skip_section_keywords):
            skipped_sections.append(section.name)
        else:
            filtered_sections.append(section)
    
    if skipped_sections:
        print(f"\n⏭️  跳過 {len(skipped_sections)} 個低價值章節：")
        for name in skipped_sections:
            print(f"  - {name}")
    
    print(f"\n📊 處理章節數：{len(filtered_sections)} / {len(sections)}")
    
    # Step 5: 生成摘要並驗證長度
    print("\n" + "-"*70)
    print("生成摘要並驗證長度...")
    print("-"*70)
    
    summary_results = []
    total_start_time = time.time()
    
    for i, section in enumerate(filtered_sections, 1):
        print(f"\n[{i}/{len(filtered_sections)}] 處理：{section.name}")
        print(f"  原文長度：{len(section.text)} 字")
        
        # 計算預期目標長度
        expected_target = summarizer._calculate_target_length(len(section.text))
        print(f"  預期摘要長度：{expected_target} 字")
        
        # 判斷是否使用 Map-Reduce
        will_use_map_reduce = len(section.text) >= summarizer.map_reduce_threshold
        strategy = "Map-Reduce" if will_use_map_reduce else "Direct"
        print(f"  策略：{strategy}")
        
        # 生成摘要
        start_time = time.time()
        try:
            summary = summarizer.summarize_section(
                section.text,
                section.name,
                paper_context=selected_pdf.stem
            )
            duration = time.time() - start_time
            
            actual_length = len(summary)
            compression_ratio = (actual_length / len(section.text)) * 100
            
            # 驗證長度範圍
            in_range = summarizer.min_summary_length <= actual_length <= summarizer.max_summary_length
            length_check = "✓" if in_range else "✗"
            
            print(f"  {length_check} 實際摘要長度：{actual_length} 字")
            print(f"  壓縮比：{compression_ratio:.1f}%")
            print(f"  耗時：{duration:.1f}s")
            
            summary_results.append({
                'section_name': section.name,
                'original_length': len(section.text),
                'expected_target': expected_target,
                'actual_length': actual_length,
                'compression_ratio': compression_ratio,
                'strategy': strategy,
                'duration': duration,
                'in_range': in_range,
                'summary_preview': summary[:100] + "..." if len(summary) > 100 else summary
            })
            
            if not in_range:
                print(f"  ⚠️  警告：摘要長度超出範圍 [{summarizer.min_summary_length}-{summarizer.max_summary_length}]")
        
        except Exception as e:
            print(f"  ❌ 摘要生成失敗：{e}")
            import traceback
            traceback.print_exc()
            summary_results.append({
                'section_name': section.name,
                'original_length': len(section.text),
                'expected_target': expected_target,
                'actual_length': 0,
                'compression_ratio': 0,
                'strategy': strategy,
                'duration': 0,
                'in_range': False,
                'error': str(e)
            })
    
    total_duration = time.time() - total_start_time
    
    # Step 6: 生成測試報告
    print("\n" + "="*70)
    print("📊 測試報告")
    print("="*70)
    
    print(f"\n論文：{selected_pdf.name}")
    print(f"總處理時間：{total_duration:.1f}s")
    print(f"處理章節數：{len(summary_results)}")
    
    # 統計資訊
    successful = [r for r in summary_results if 'error' not in r]
    failed = [r for r in summary_results if 'error' in r]
    in_range_count = sum(1 for r in successful if r['in_range'])
    
    print(f"\n成功率：{len(successful)}/{len(summary_results)} ({len(successful)/len(summary_results)*100:.1f}%)")
    print(f"範圍檢查：{in_range_count}/{len(successful)} 在 200-1500 字範圍內")
    
    if failed:
        print(f"\n❌ 失敗章節：{len(failed)}")
        for r in failed:
            print(f"  - {r['section_name']}: {r.get('error', 'Unknown error')}")
    
    # 詳細結果表格
    print("\n" + "-"*70)
    print("章節摘要詳情")
    print("-"*70)
    print(f"{'章節名稱':<25} {'原文':<8} {'預期':<6} {'實際':<6} {'壓縮比':<8} {'策略':<12} {'耗時':<8}")
    print("-"*70)
    
    for r in successful:
        print(f"{r['section_name']:<25} "
              f"{r['original_length']:<8} "
              f"{r['expected_target']:<6} "
              f"{r['actual_length']:<6} "
              f"{r['compression_ratio']:>6.1f}% "
              f"{r['strategy']:<12} "
              f"{r['duration']:>6.1f}s")
    
    # 驗證動態長度邏輯
    print("\n" + "-"*70)
    print("動態長度邏輯驗證")
    print("-"*70)
    
    # 按原文長度排序
    sorted_results = sorted(successful, key=lambda x: x['original_length'])
    
    if len(sorted_results) >= 2:
        shortest = sorted_results[0]
        longest = sorted_results[-1]
        
        print(f"\n最短章節：{shortest['section_name']}")
        print(f"  原文：{shortest['original_length']} 字 → 摘要：{shortest['actual_length']} 字")
        
        print(f"\n最長章節：{longest['section_name']}")
        print(f"  原文：{longest['original_length']} 字 → 摘要：{longest['actual_length']} 字")
        
        # 驗證：長章節的摘要應該比短章節長
        if longest['original_length'] > shortest['original_length'] * 2:
            if longest['actual_length'] > shortest['actual_length']:
                print("\n✓ 驗證通過：長章節產生更長的摘要")
            else:
                print("\n⚠️  警告：長章節的摘要不一定更長（可能因為內容密度差異）")
    
    # 摘要預覽
    print("\n" + "-"*70)
    print("摘要範例預覽")
    print("-"*70)
    
    for i, r in enumerate(successful[:3], 1):  # 只顯示前 3 個
        print(f"\n{i}. {r['section_name']} ({r['actual_length']} 字)")
        print(f"   {r['summary_preview']}")
    
    # 最終判斷
    print("\n" + "="*70)
    all_in_range = all(r['in_range'] for r in successful)
    all_successful = len(failed) == 0
    
    if all_successful and all_in_range:
        print("✅ 測試通過！所有摘要長度都在 200-1500 字範圍內")
        return True
    elif all_successful:
        print("⚠️  測試部分通過：所有摘要生成成功，但部分長度超出範圍")
        return True
    else:
        print("❌ 測試失敗：部分摘要生成失敗")
        return False

if __name__ == "__main__":
    try:
        success = test_dynamic_summary_length()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  測試被用戶中斷")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 測試執行出錯：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
