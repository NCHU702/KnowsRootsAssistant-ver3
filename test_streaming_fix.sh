#!/bin/bash
# Test the streaming endpoint with the human traffic prediction query

echo "Testing streaming query: 可以用什麼模型處理人流預測"
echo "================================================"
echo ""

curl -N -X POST http://127.0.0.1:5000/query_stream \
  -H "Content-Type: application/json" \
  -d '{"input": "可以用什麼模型處理人流預測"}' \
  2>/dev/null | while IFS= read -r line; do
    # Parse SSE format (data: {...})
    if [[ $line == data:* ]]; then
        json_data="${line#data: }"
        
        # Extract type and content using jq
        event_type=$(echo "$json_data" | jq -r '.type // empty' 2>/dev/null)
        
        case "$event_type" in
            "start")
                mode=$(echo "$json_data" | jq -r '.mode // "unknown"' 2>/dev/null)
                echo "🚀 Started in $mode mode"
                ;;
            "thought")
                content=$(echo "$json_data" | jq -r '.content // empty' 2>/dev/null)
                echo "💭 Thought: $content"
                ;;
            "action")
                tool=$(echo "$json_data" | jq -r '.tool // empty' 2>/dev/null)
                input=$(echo "$json_data" | jq -r '.input // empty' 2>/dev/null)
                echo "⚡ Action: $tool"
                echo "   Input: ${input:0:100}..."
                ;;
            "tool_start")
                tool=$(echo "$json_data" | jq -r '.tool // empty' 2>/dev/null)
                echo "🔧 Tool started: $tool"
                ;;
            "tool_token")
                # Don't print individual tokens, just count them
                :
                ;;
            "observation")
                # Skip observation in streaming mode
                :
                ;;
            "done")
                echo ""
                echo "✅ Stream completed"
                ;;
            "error")
                message=$(echo "$json_data" | jq -r '.message // .content // empty' 2>/dev/null)
                echo "❌ Error: $message"
                ;;
            "stopped")
                echo "🛑 Stopped by user"
                ;;
        esac
    fi
done

echo ""
echo "Test completed!"
