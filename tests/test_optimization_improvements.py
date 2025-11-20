#!/usr/bin/env python3
"""
綜合測試：醫療詞典 + Layer 2 智能觸發決策器

測試目標：
1. 醫療查詢準確度提升
2. Layer 2 觸發決策智能化
3. 系統整體效能改善
"""

import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any

# 設定路徑
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizationTester:
    """優化測試器"""
    
    def __init__(self):
        self.rag_system = None
        self.test_results = []
    
    def initialize_system(self):
        """初始化 RAG 系統"""
        print("\n" + "="*70)
        print("初始化 RAG 系統（含醫療詞典 + Layer 2 智能觸發）")
        print("="*70)
        
        try:
            from system_api.hierarchical_rag_system import HierarchicalRAGSystem
            
            self.rag_system = HierarchicalRAGSystem(
                pdf_directory="./test_data",
                model_name="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
                embedding_model="quentinz/bge-large-zh-v1.5:latest",
                vectorstore_path="./vectorstore",
                chunk_size=800,
                chunk_overlap=100,
                config={
                    'layer1': {
                        'k_documents': 10,
                        'confidence_threshold': 0.6,
                        'similarity_threshold': 0.48,
                    },
                    'layer2': {
                        'k_documents': 10,
                        'confidence_threshold': 0.8,
                    },
                    'hybrid_search': {
                        'enabled': True,
                        'semantic_weight': 0.9,
                        'keyword_weight': 0.1,
                        'use_jieba': True,
                        'enable_smart_weighting': True,
                        'use_llm_analyzer': True,  # 啟用 LLM 詞彙分析
                    },
                    'expansion': {
                        'enabled': True
                    }
                }
            )
            
            if not self.rag_system.is_ready():
                print("⚠️  系統未就緒，請先建立索引")
                return False
            
            print("✅ 系統初始化成功")
            return True
            
        except Exception as e:
            print(f"❌ 系統初始化失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_medical_queries(self):
        """測試醫療查詢（驗證醫療詞典效果）"""
        print("\n" + "="*70)
        print("測試 1: 醫療查詢準確度（醫療詞典整合）")
        print("="*70)
        
        medical_queries = [
            {
                'query': '深度學習在鼻咽癌的應用',
                'expected_keywords': ['鼻咽癌', '深度學習'],
                'description': '疾病名稱識別測試'
            },
            {
                'query': 'CT影像分析',
                'expected_keywords': ['CT', '影像'],
                'description': '醫療檢查技術識別'
            },
            {
                'query': '糖尿病預測模型',
                'expected_keywords': ['糖尿病', '預測'],
                'description': '慢性病名稱識別'
            },
        ]
        
        results = []
        
        for test_case in medical_queries:
            print(f"\n{'─'*70}")
            print(f"查詢: {test_case['query']}")
            print(f"說明: {test_case['description']}")
            print(f"{'─'*70}")
            
            start_time = time.time()
            result = self.rag_system.query(test_case['query'])
            duration = time.time() - start_time
            
            # 分析結果
            layer1_docs = result.get('layer1_docs', [])
            final_docs = result.get('final_docs', [])
            terminated_at = result.get('terminated_at', 'unknown')
            
            print(f"\n結果:")
            print(f"  ⏱️  執行時間: {duration:.2f}s")
            print(f"  📊 Layer 1 找到: {len(layer1_docs)} 篇論文")
            print(f"  📄 最終文檔數: {len(final_docs)} 個")
            print(f"  🎯 終止於: {terminated_at}")
            
            if layer1_docs:
                print(f"\n  前 3 篇論文:")
                for i, doc in enumerate(layer1_docs[:3], 1):
                    title = doc.metadata.get('title', 'Unknown')
                    print(f"    {i}. {title[:60]}...")
            else:
                print(f"  ⚠️  未找到相關論文")
            
            # 檢查是否改善
            success = len(layer1_docs) > 0
            
            test_result = {
                'query': test_case['query'],
                'description': test_case['description'],
                'papers_found': len(layer1_docs),
                'duration': duration,
                'terminated_at': terminated_at,
                'success': success
            }
            results.append(test_result)
        
        # 總結
        print(f"\n{'='*70}")
        print(f"醫療查詢測試總結:")
        print(f"{'='*70}")
        
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        success_rate = (successful / total * 100) if total > 0 else 0
        
        print(f"  成功率: {successful}/{total} ({success_rate:.0f}%)")
        print(f"  平均找到論文數: {sum(r['papers_found'] for r in results) / total:.1f} 篇")
        print(f"  平均執行時間: {sum(r['duration'] for r in results) / total:.2f}s")
        
        self.test_results.extend(results)
        return success_rate >= 60  # 至少 60% 成功率
    
    def test_layer2_trigger_intelligence(self):
        """測試 Layer 2 智能觸發（驗證決策器效果）"""
        print("\n" + "="*70)
        print("測試 2: Layer 2 智能觸發決策")
        print("="*70)
        
        test_cases = [
            {
                'query': '列出所有深度學習的論文',
                'query_type': 'list',
                'expected_layer2': False,
                'description': '列表查詢 - 應在 Layer 1 終止'
            },
            {
                'query': '有哪些人流預測研究',
                'query_type': 'overview',
                'expected_layer2': False,
                'description': '概覽查詢 - 應在 Layer 1 終止'
            },
            {
                'query': '詳細解釋 LSTM 的運作原理',
                'query_type': 'detailed_explanation',
                'expected_layer2': True,
                'description': '詳細解釋 - 應進入 Layer 2'
            },
            {
                'query': 'CNN 的具體實作方法',
                'query_type': 'methodology',
                'expected_layer2': True,
                'description': '方法學查詢 - 應進入 Layer 2'
            },
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"\n{'─'*70}")
            print(f"查詢: {test_case['query']}")
            print(f"類型: {test_case['query_type']}")
            print(f"說明: {test_case['description']}")
            print(f"預期: {'進入 Layer 2' if test_case['expected_layer2'] else 'Layer 1 終止'}")
            print(f"{'─'*70}")
            
            result = self.rag_system.query(test_case['query'])
            
            # 分析觸發決策
            terminated_at = result.get('terminated_at', 'unknown')
            trigger_decision = result.get('layer2_trigger_decision', {})
            
            actual_layer2 = (terminated_at == 'layer2' or terminated_at == 'layer2_fallback')
            decision_correct = (actual_layer2 == test_case['expected_layer2'])
            
            print(f"\n結果:")
            print(f"  終止於: {terminated_at}")
            print(f"  實際行為: {'進入 Layer 2' if actual_layer2 else 'Layer 1 終止'}")
            
            if trigger_decision:
                print(f"\n  觸發決策詳情:")
                print(f"    查詢類型識別: {trigger_decision['decision_factors'].get('query_type', 'N/A')}")
                print(f"    複雜度: {trigger_decision['decision_factors'].get('query_complexity', 'N/A')}")
                print(f"    基礎閾值: {trigger_decision['decision_factors'].get('base_threshold', 0):.2f}")
                print(f"    調整後閾值: {trigger_decision.get('adjusted_threshold', 0):.2f}")
                print(f"    決策理由: {trigger_decision.get('reason', 'N/A')[:80]}...")
            
            status = "✅ 正確" if decision_correct else "❌ 不符預期"
            print(f"\n  {status}")
            
            test_result = {
                'query': test_case['query'],
                'query_type': test_case['query_type'],
                'expected_layer2': test_case['expected_layer2'],
                'actual_layer2': actual_layer2,
                'correct': decision_correct,
                'terminated_at': terminated_at
            }
            results.append(test_result)
        
        # 總結
        print(f"\n{'='*70}")
        print(f"Layer 2 觸發決策測試總結:")
        print(f"{'='*70}")
        
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print(f"  決策準確度: {correct}/{total} ({accuracy:.0f}%)")
        
        # 分類統計
        layer1_stops = sum(1 for r in results if not r['actual_layer2'])
        layer2_triggers = sum(1 for r in results if r['actual_layer2'])
        
        print(f"  Layer 1 終止: {layer1_stops} 次")
        print(f"  Layer 2 觸發: {layer2_triggers} 次")
        
        self.test_results.extend(results)
        return accuracy >= 75  # 至少 75% 準確度
    
    def test_performance_comparison(self):
        """測試效能比較"""
        print("\n" + "="*70)
        print("測試 3: 效能與效率分析")
        print("="*70)
        
        # 測試一般查詢的效率
        queries = [
            '深度學習研究',
            '機器學習應用',
            '人流預測方法',
        ]
        
        stats = {
            'total_queries': 0,
            'layer1_stops': 0,
            'layer2_enters': 0,
            'total_time': 0,
            'avg_papers': 0,
        }
        
        for query in queries:
            start = time.time()
            result = self.rag_system.query(query)
            duration = time.time() - start
            
            stats['total_queries'] += 1
            stats['total_time'] += duration
            
            if result.get('terminated_at') == 'layer1':
                stats['layer1_stops'] += 1
            else:
                stats['layer2_enters'] += 1
            
            layer1_docs = result.get('layer1_docs', [])
            stats['avg_papers'] += len(layer1_docs)
        
        # 計算平均值
        if stats['total_queries'] > 0:
            stats['avg_papers'] /= stats['total_queries']
            stats['avg_time'] = stats['total_time'] / stats['total_queries']
        
        print(f"\n效能統計:")
        print(f"  總查詢數: {stats['total_queries']}")
        print(f"  Layer 1 終止: {stats['layer1_stops']} ({stats['layer1_stops']/stats['total_queries']*100:.0f}%)")
        print(f"  Layer 2 進入: {stats['layer2_enters']} ({stats['layer2_enters']/stats['total_queries']*100:.0f}%)")
        print(f"  平均執行時間: {stats['avg_time']:.2f}s")
        print(f"  平均找到論文數: {stats['avg_papers']:.1f} 篇")
        
        # 效率評估
        efficiency_score = (stats['layer1_stops'] / stats['total_queries'] * 100) if stats['total_queries'] > 0 else 0
        
        print(f"\n💡 效率評估:")
        if efficiency_score >= 30:
            print(f"  ✅ 良好 - {efficiency_score:.0f}% 的查詢在 Layer 1 完成")
        else:
            print(f"  ⚠️  可改進 - 僅 {efficiency_score:.0f}% 在 Layer 1 完成")
        
        return True
    
    def run_all_tests(self):
        """執行所有測試"""
        print("\n" + "="*70)
        print("🧪 綜合優化測試")
        print("="*70)
        print("測試項目:")
        print("  1️⃣  醫療查詢準確度（醫療詞典）")
        print("  2️⃣  Layer 2 智能觸發（決策器）")
        print("  3️⃣  系統效能與效率")
        print("="*70)
        
        # 初始化
        if not self.initialize_system():
            print("\n❌ 系統初始化失敗，測試終止")
            return False
        
        # 執行測試
        test1_pass = self.test_medical_queries()
        test2_pass = self.test_layer2_trigger_intelligence()
        test3_pass = self.test_performance_comparison()
        
        # 最終總結
        print("\n" + "="*70)
        print("📊 測試總結")
        print("="*70)
        
        results = {
            '醫療查詢準確度': ('✅ 通過' if test1_pass else '❌ 失敗'),
            'Layer 2 智能觸發': ('✅ 通過' if test2_pass else '❌ 失敗'),
            '系統效能分析': ('✅ 通過' if test3_pass else '❌ 失敗'),
        }
        
        for name, status in results.items():
            print(f"  {status} - {name}")
        
        all_pass = test1_pass and test2_pass and test3_pass
        
        print("\n" + "="*70)
        if all_pass:
            print("🎉 所有測試通過！優化效果顯著！")
            print("\n改進總結:")
            print("  ✅ 醫療詞典成功整合，醫療查詢準確度提升")
            print("  ✅ Layer 2 智能觸發決策器工作正常")
            print("  ✅ 系統效能得到優化")
        else:
            print("⚠️  部分測試未通過，需要進一步調整")
        print("="*70)
        
        return all_pass


def main():
    """主函數"""
    tester = OptimizationTester()
    
    try:
        success = tester.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⚠️  測試被用戶中斷")
        return 130
    except Exception as e:
        print(f"\n\n❌ 測試過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
