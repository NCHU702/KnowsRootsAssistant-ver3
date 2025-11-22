"""
Test agent2.py with single paper queries to verify the complete flow
"""

import requests
import json

# Test queries for single paper details
test_queries = [
    "芒果分類那篇論文用什麼方法？",
    "YOLO模型在哪篇論文有討論？具體內容是什麼？",
    "肺癌檢測那篇論文的數據集是什麼？",
]

print("=" * 80)
print("Agent2 Single Paper Query Test")
print("=" * 80)

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*80}")
    print(f"Test {i}: {query}")
    print('='*80)
    
    try:
        response = requests.post(
            'http://localhost:5000/ask',
            json={'query': query},
            stream=True,
            timeout=120
        )
        
        if response.status_code == 200:
            print("\n📝 Streaming Response:")
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        data = line_text[6:]  # Remove 'data: ' prefix
                        if data != '[DONE]':
                            try:
                                chunk_data = json.loads(data)
                                if 'chunk' in chunk_data:
                                    print(chunk_data['chunk'], end='', flush=True)
                                    full_response += chunk_data['chunk']
                            except json.JSONDecodeError:
                                pass
            
            print(f"\n\n✓ Response completed ({len(full_response)} chars)")
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print()

print("\n" + "=" * 80)
print("✓ All tests completed")
print("=" * 80)
