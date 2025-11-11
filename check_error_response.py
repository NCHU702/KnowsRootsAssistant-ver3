#!/usr/bin/env python3
"""
捕獲 Ollama 500 錯誤的完整響應體
"""

import requests

# 測試導致失敗的幾個案例
test_cases = [
    # 案例1: 封面頁 - 大量換行
    "國立成功大學 \n測量及空間資訊學系 \n碩士論文 \n\n\n\n\n基於自動編碼器與條件生成對抗網路之旅程推薦應用",
    
    # 案例2: 目錄頁 - 大量點號
    "Chapter 2 Related Work ........................................................................................... 5",
    
    # 案例3: 包含中文冒號
    "研 究 生：賴元淯 \n指導教授：呂學展",
]

url = "http://localhost:11434/api/embed"

for i, text in enumerate(test_cases, 1):
    print(f"\n{'='*70}")
    print(f"測試案例 {i}:")
    print(f"長度: {len(text)} 字符, {len(text.encode('utf-8'))} 字節")
    print(f"內容: {text[:100]}...")
    print(f"{'='*70}")
    
    payload = {"model": "nomic-embed-text", "input": text}
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"狀態碼: {response.status_code}")
        
        if response.status_code != 200:
            print(f"錯誤響應體: {response.text}")
            print(f"響應頭: {dict(response.headers)}")
        else:
            print("✓ 成功")
    except Exception as e:
        print(f"異常: {e}")
