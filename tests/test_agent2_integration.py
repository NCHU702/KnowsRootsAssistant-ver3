#!/usr/bin/env python3
"""
Agent2.py 整合測試

測試 agent2.py 與 HierarchicalRAGSystem 的整合，確保：
1. Hierarchical RAG 系統正常初始化
2. 監控端點正常工作
3. 查詢功能正常
4. API 正常運作
"""

import requests
import time
import sys
import json
from typing import Dict, Any

BASE_URL = "http://localhost:4000"

class Color:
    """終端顏色"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """打印標題"""
    print(f"\n{Color.BOLD}{'='*70}{Color.END}")
    print(f"{Color.BOLD}{Color.BLUE}{text}{Color.END}")
    print(f"{Color.BOLD}{'='*70}{Color.END}\n")

def print_success(text: str):
    """打印成功訊息"""
    print(f"{Color.GREEN}✓{Color.END} {text}")

def print_warning(text: str):
    """打印警告訊息"""
    print(f"{Color.YELLOW}⚠{Color.END} {text}")

def print_error(text: str):
    """打印錯誤訊息"""
    print(f"{Color.RED}✗{Color.END} {text}")

def print_info(text: str):
    """打印資訊訊息"""
    print(f"{Color.BLUE}ℹ{Color.END} {text}")


def test_server_connection() -> bool:
    """測試伺服器連線"""
    print_header("測試 1: 伺服器連線")
    
    try:
        response = requests.get(f"{BASE_URL}/rag/health", timeout=5)
        if response.status_code == 200:
            print_success("伺服器連線成功")
            return True
        else:
            print_error(f"伺服器返回錯誤碼: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error("無法連接到伺服器")
        print_info("請確保 agent2.py 正在運行：python agent2.py")
        return False
    except Exception as e:
        print_error(f"連線測試失敗: {e}")
        return False


def test_rag_mode() -> Dict[str, Any]:
    """測試 RAG 模式檢測"""
    print_header("測試 2: RAG 模式檢測")
    
    try:
        response = requests.get(f"{BASE_URL}/rag/mode", timeout=5)
        if response.status_code == 200:
            mode_info = response.json()
            mode = mode_info.get('mode', 'unknown')
            type_name = mode_info.get('type', 'unknown')
            features = mode_info.get('features', [])
            
            print_success(f"模式檢測成功")
            print(f"  當前模式: {mode}")
            print(f"  系統類型: {type_name}")
            print(f"  支援功能: {', '.join(features)}")
            
            if mode == 'hierarchical':
                config = mode_info.get('config', {})
                print(f"  配置:")
                print(f"    Layer 1 閾值: {config.get('layer1_threshold', 'N/A')}")
                print(f"    Layer 2 閾值: {config.get('layer2_threshold', 'N/A')}")
                print(f"    擴展啟用: {config.get('expansion_enabled', 'N/A')}")
            
            return mode_info
        else:
            print_error(f"模式檢測失敗: {response.status_code}")
            return {}
    except Exception as e:
        print_error(f"模式檢測異常: {e}")
        return {}


def test_health_check() -> bool:
    """測試健康檢查"""
    print_header("測試 3: 系統健康檢查")
    
    try:
        response = requests.get(f"{BASE_URL}/rag/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            status = health.get('status', 'unknown')
            components = health.get('components', {})
            issues = health.get('issues', [])
            
            if status == 'healthy':
                print_success(f"系統狀態: {status.upper()}")
            elif status == 'degraded':
                print_warning(f"系統狀態: {status.upper()}")
            else:
                print_error(f"系統狀態: {status.upper()}")
            
            print(f"\n  組件狀態:")
            for component, comp_status in components.items():
                symbol = "✓" if comp_status in ['up', 'ready'] else "✗"
                print(f"    {symbol} {component}: {comp_status}")
            
            if issues:
                print(f"\n  ⚠ 問題:")
                for issue in issues:
                    print(f"    - {issue}")
            
            return status in ['healthy', 'degraded']
        else:
            print_error(f"健康檢查失敗: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"健康檢查異常: {e}")
        return False


def test_stats_endpoint(mode: str) -> bool:
    """測試統計端點"""
    print_header("測試 4: 系統統計")
    
    try:
        response = requests.get(f"{BASE_URL}/rag/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            
            print_success("統計資訊獲取成功")
            
            # 系統統計
            system_stats = stats.get('system_stats', {})
            layer1 = system_stats.get('layer1', {})
            layer2 = system_stats.get('layer2', {})
            
            print(f"\n  Layer 1 (摘要):")
            print(f"    論文數: {layer1.get('paper_count', 0)}")
            print(f"    已初始化: {layer1.get('is_initialized', False)}")
            
            print(f"\n  Layer 2 (區塊):")
            print(f"    區塊數: {layer2.get('chunk_count', 0)}")
            print(f"    論文數: {layer2.get('paper_count', 0)}")
            print(f"    已初始化: {layer2.get('is_initialized', False)}")
            
            # 查詢統計（如果有）
            query_stats = stats.get('query_stats')
            if query_stats:
                total = query_stats.get('total_queries', 0)
                termination = query_stats.get('termination_distribution', {})
                
                print(f"\n  查詢統計:")
                print(f"    總查詢數: {total}")
                if termination:
                    layer1_count = termination.get('layer1', 0)
                    layer2_count = termination.get('layer2', 0)
                    layer1_pct = (layer1_count / total * 100) if total > 0 else 0
                    print(f"    Layer 1 終止: {layer1_count} ({layer1_pct:.1f}%)")
                    print(f"    Layer 2 終止: {layer2_count}")
            
            return True
        else:
            print_error(f"統計獲取失敗: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"統計獲取異常: {e}")
        return False


def test_query_endpoint(mode: str) -> bool:
    """測試查詢端點"""
    print_header("測試 5: 查詢功能")
    
    test_query = "What is machine learning?"
    
    try:
        print_info(f"發送查詢: '{test_query}'")
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/query",
            json={"input": test_query},
            timeout=30
        )
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('answer', '')
            
            print_success(f"查詢成功 (耗時: {duration:.1f}s)")
            print(f"\n  回答長度: {len(answer)} 字符")
            
            # Hierarchical RAG 統計資訊
            retrieval_stats = result.get('retrieval_stats', {})
            if retrieval_stats:
                termination = retrieval_stats.get('termination_layer', 'unknown')
                confidence = retrieval_stats.get('confidence', 0)
                timing = retrieval_stats.get('retrieval_time', {})
                
                print(f"  終止層: {termination}")
                print(f"  信心度: {confidence:.2f}")
                if timing:
                    print(f"  時間分解:")
                    for key, value in timing.items():
                        print(f"    {key}: {value:.2f}s")
            
            # 顯示部分回答
            preview = answer[:200] + "..." if len(answer) > 200 else answer
            print(f"\n  回答預覽:\n  {preview}")
            
            return True
        else:
            print_error(f"查詢失敗: {response.status_code}")
            print(f"  錯誤: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print_error(f"查詢超時（> 30s）")
        return False
    except Exception as e:
        print_error(f"查詢異常: {e}")
        return False


def test_query_stream_endpoint(mode: str) -> bool:
    """測試流式查詢端點"""
    print_header("測試 6: 流式查詢功能")
    
    test_query = "深度學習"
    
    try:
        print_info(f"發送流式查詢: '{test_query}'")
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/query_stream",
            json={"input": test_query},
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            chunk_count = 0
            first_chunk_time = None
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        try:
                            data = json.loads(line_str[6:])
                            chunk_type = data.get('type')
                            
                            if chunk_type == 'start':
                                print_info(f"  開始流式輸出")
                            elif chunk_type == 'chunk':
                                if first_chunk_time is None:
                                    first_chunk_time = time.time() - start_time
                                chunk_count += 1
                            elif chunk_type == 'done':
                                duration = time.time() - start_time
                                print_success(f"流式查詢完成")
                                print(f"  總耗時: {duration:.1f}s")
                                print(f"  首字節時間: {first_chunk_time:.1f}s" if first_chunk_time else "")
                                print(f"  區塊數: {chunk_count}")
                                return True
                        except json.JSONDecodeError:
                            pass
            
            print_warning("流式查詢結束但未收到 'done' 訊號")
            return True
        else:
            print_error(f"流式查詢失敗: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print_error(f"流式查詢超時（> 30s）")
        return False
    except Exception as e:
        print_error(f"流式查詢異常: {e}")
        return False


def test_api_compatibility() -> bool:
    """測試 API 響應格式"""
    print_header("測試 7: API 響應格式")
    
    # 測試標準查詢格式
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json={"input": "test query"},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # 檢查必需字段
            required_fields = ['answer']
            missing_fields = [f for f in required_fields if f not in result]
            
            if not missing_fields:
                print_success("API 響應格式正確")
                
                # 檢查 Hierarchical RAG 字段
                new_fields = ['rag_mode', 'retrieval_stats']
                present_new_fields = [f for f in new_fields if f in result]
                if present_new_fields:
                    print_info(f"  Hierarchical 字段: {', '.join(present_new_fields)}")
                
                return True
            else:
                print_error(f"缺少必需字段: {', '.join(missing_fields)}")
                return False
        else:
            print_error(f"API 測試失敗: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"API 響應格式測試異常: {e}")
        return False


def main():
    """主測試流程"""
    print("\n")
    print(Color.BOLD + "="*70 + Color.END)
    print(Color.BOLD + Color.BLUE + "Agent2.py + HierarchicalRAGSystem 整合測試" + Color.END)
    print(Color.BOLD + "="*70 + Color.END)
    
    results = {
        '伺服器連線': False,
        'RAG 模式': False,
        '健康檢查': False,
        '系統統計': False,
        '查詢功能': False,
        '流式查詢': False,
        'API 響應': False
    }
    
    # 測試 1: 連線
    if not test_server_connection():
        print_error("\n❌ 伺服器未運行，終止測試")
        sys.exit(1)
    results['伺服器連線'] = True
    
    # 測試 2: 模式檢測
    mode_info = test_rag_mode()
    mode = mode_info.get('mode', 'unknown')
    if mode == 'hierarchical':
        results['RAG 模式'] = True
    
    # 測試 3: 健康檢查
    if test_health_check():
        results['健康檢查'] = True
    
    # 測試 4: 統計
    if test_stats_endpoint(mode):
        results['系統統計'] = True
    
    # 測試 5: 查詢
    if test_query_endpoint(mode):
        results['查詢功能'] = True
    
    # 測試 6: 流式查詢
    if test_query_stream_endpoint(mode):
        results['流式查詢'] = True
    
    # 測試 7: API 響應
    if test_api_compatibility():
        results['API 響應'] = True
    
    # 總結
    print_header("測試總結")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")
    
    print(f"\n{Color.BOLD}通過率: {passed}/{total} ({passed/total*100:.1f}%){Color.END}")
    
    if passed == total:
        print(f"\n{Color.GREEN}{Color.BOLD}🎉 所有測試通過！系統已準備好使用。{Color.END}")
        sys.exit(0)
    else:
        print(f"\n{Color.YELLOW}{Color.BOLD}⚠ 部分測試失敗，請檢查問題並修復。{Color.END}")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Color.YELLOW}測試被用戶中斷{Color.END}")
        sys.exit(130)
