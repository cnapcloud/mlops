curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "messages": [
      {"role": "user", "content": "안녕하세요! 자기소개 해주세요."}
    ],
    "temperature": 0.7
  }'