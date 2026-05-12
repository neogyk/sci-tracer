#Pure curl:
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "The cat shat on the",
  "stream": false,
  "logprobs": true,
  "top_logprobs": 20,
  "options": { "num_predict": 1, "temperature": 0.7, "top_p": 0.95, "top_k": 50 }
}'

#Zig 