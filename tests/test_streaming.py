#!/usr/bin/env python3
"""Test script for streaming functionality"""

import requests
import json
import time

def test_streaming():
    """Test the streaming endpoint"""
    url = "http://localhost:5000/query_stream"
    
    test_query = "什麼是深度學習？"
    
    print(f"Testing streaming with query: {test_query}\n")
    print("=" * 60)
    
    try:
        response = requests.post(
            url,
            json={"input": test_query},
            stream=True,
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"Error: HTTP {response.status_code}")
            print(response.text)
            return
        
        print("Streaming response:\n")
        
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                
                if line_text.startswith('data: '):
                    data = json.loads(line_text[6:])
                    
                    if data['type'] == 'start':
                        print(f"[START] Action: {data['action']}")
                        print("-" * 60)
                        
                    elif data['type'] == 'chunk':
                        print(data['content'], end='', flush=True)
                        
                    elif data['type'] == 'error':
                        print(f"\n[ERROR] {data['message']}")
                        
                    elif data['type'] == 'done':
                        print("\n" + "-" * 60)
                        print("[DONE]")
        
        print("\n" + "=" * 60)
        print("Streaming test completed!")
        
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Is agent2.py running?")
        print("Run: .venv/bin/python agent2.py")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_streaming()
