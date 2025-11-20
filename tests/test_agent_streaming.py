#!/usr/bin/env python3
"""Test script for Agent streaming with reasoning process"""

import requests
import json
import time

def test_agent_streaming():
    """Test the agent streaming endpoint with reasoning display"""
    url = "http://localhost:5000/query_stream"
    
    # Test query that should trigger agent reasoning
    test_queries = [
        "Search for the latest deep learning papers in 2024",  # Should trigger WebSearchCall
        "什麼是深度學習？請詳細說明",  # Should trigger AssistantCall
        "Summarize papers about neural networks",  # Should trigger AssistantCall
    ]
    
    for i, test_query in enumerate(test_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}/{len(test_queries)}: {test_query}")
        print('=' * 70)
        
        try:
            response = requests.post(
                url,
                json={"input": test_query},
                stream=True,
                timeout=120
            )
            
            if response.status_code != 200:
                print(f"❌ Error: HTTP {response.status_code}")
                print(response.text)
                continue
            
            print("\n📡 Streaming agent reasoning process:\n")
            
            step_count = 0
            mode = "unknown"
            
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    
                    if line_text.startswith('data: '):
                        try:
                            data = json.loads(line_text[6:])
                            
                            if data['type'] == 'start':
                                mode = data.get('mode', data.get('action', 'unknown'))
                                print(f"🚀 [START] Mode: {mode}")
                                print("-" * 70)
                            
                            elif data['type'] == 'thought':
                                step_count = data['step'] + 1
                                print(f"\n💭 [THOUGHT - Step {step_count}]")
                                print(f"   {data['content']}")
                            
                            elif data['type'] == 'action':
                                print(f"\n🔧 [ACTION - Step {step_count}]")
                                print(f"   Tool: {data['tool']}")
                                print(f"   Input: {data['input'][:100]}{'...' if len(data['input']) > 100 else ''}")
                            
                            elif data['type'] == 'tool_start':
                                print(f"\n⚙️  [TOOL START] {data['tool']} executing...")
                            
                            elif data['type'] == 'tool_token':
                                print(data['content'], end='', flush=True)
                            
                            elif data['type'] == 'observation':
                                print(f"\n\n👁️  [OBSERVATION - Step {step_count}]")
                                obs = data['content']
                                print(f"   {obs[:200]}{'...' if len(obs) > 200 else ''}")
                            
                            elif data['type'] == 'chunk':
                                print(data['content'], end='', flush=True)
                            
                            elif data['type'] == 'final_answer':
                                print(f"\n\n✅ [FINAL ANSWER]")
                                print("-" * 70)
                                answer = data.get('content', '')
                                print(answer[:500] + ('...' if len(answer) > 500 else ''))
                            
                            elif data['type'] == 'error':
                                print(f"\n❌ [ERROR] {data.get('message', data.get('content', 'Unknown error'))}")
                            
                            elif data['type'] == 'done':
                                print("\n" + "-" * 70)
                                print(f"✅ [DONE] Completed in {mode} mode")
                        
                        except json.JSONDecodeError as e:
                            print(f"⚠️  JSON parse error: {e}")
                            continue
            
            print(f"\n{'=' * 70}\n")
            
            # Wait between tests
            if i < len(test_queries):
                print("⏳ Waiting 3 seconds before next test...\n")
                time.sleep(3)
        
        except requests.exceptions.ConnectionError:
            print("❌ Error: Cannot connect to server. Is agent2.py running?")
            print("   Run: .venv/bin/python agent2.py")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("🎉 All tests completed!")
    print("=" * 70)

if __name__ == "__main__":
    print("=" * 70)
    print("Agent Streaming with Reasoning Process - Test Suite")
    print("=" * 70)
    print("\nThis test will demonstrate:")
    print("  1. Agent thinking process (Thought)")
    print("  2. Tool selection (Action)")
    print("  3. Tool execution (Observation)")
    print("  4. Final answer synthesis")
    print("\n" + "=" * 70)
    
    test_agent_streaming()
