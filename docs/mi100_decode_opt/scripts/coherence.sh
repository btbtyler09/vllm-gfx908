#!/usr/bin/env bash
# Usage: coherence.sh SERVED_MODEL_NAME [LABEL]
# Hits http://localhost:8000/v1/chat/completions with 4 canary prompts.
# Exits 0 if all pass, non-zero if any fail.
set -u

SERVED="${1:?served model name required}"
LABEL="${2:-coherence}"
OUT="/tmp/decode_opt/${LABEL}.log"
: > "$OUT"

check() {
  local name="$1"; local prompt="$2"; local regex="$3"
  local body
  body=$(jq -nc --arg m "$SERVED" --arg p "$prompt" '{
    model: $m,
    messages: [{role: "user", content: $p}],
    max_tokens: 256,
    temperature: 0.7,
    chat_template_kwargs: {enable_thinking: false}
  }')
  local resp
  resp=$(curl -s -m 90 -H 'Content-Type: application/json' -d "$body" http://localhost:8000/v1/chat/completions)
  local content
  content=$(echo "$resp" | jq -r '.choices[0].message.content // empty')
  echo "==== $name ====" >> "$OUT"
  echo "$content" >> "$OUT"
  echo "" >> "$OUT"

  if [[ -z "$content" ]]; then
    echo "FAIL: $name (empty response). Raw: $resp"; return 1
  fi
  # Reject pathological repeats (!!!!! or same 4-gram repeated)
  if echo "$content" | grep -qE '!{5,}'; then
    echo "FAIL: $name (5+ consecutive !)"; return 1
  fi
  if echo "$content" | grep -qE '\b([A-Za-z]+( +[A-Za-z]+){3})( +\1){4,}'; then
    echo "FAIL: $name (4-gram repeated 5+ times)"; return 1
  fi
  if ! echo "$content" | tr '\n' ' ' | grep -qiE "$regex"; then
    echo "FAIL: $name (regex '$regex' missed). Content: $(echo "$content" | head -c 200)"; return 1
  fi
  echo "PASS: $name"
}

FAILS=0
check "fibonacci" \
  "Write a Python function that returns the nth Fibonacci number." \
  "def +fib" || FAILS=$((FAILS+1))

check "hash_collisions" \
  "Briefly explain how a hash table handles collisions." \
  "collision.*(bucket|chain|chaining|probe|probing|linear|open addressing|separate)|chain.*collision|probe.*collision" || FAILS=$((FAILS+1))

check "french_translation" \
  "Translate to French: The cat sits on the table." \
  "Le chat" || FAILS=$((FAILS+1))

check "ocean_haiku" \
  "Write a 3-line haiku about the ocean." \
  "ocean|sea|wave|tide|shore|deep|blue|salt|foam|surf" || FAILS=$((FAILS+1))

echo ""
if [[ $FAILS -eq 0 ]]; then
  echo "COHERENCE PASS (log: $OUT)"
  exit 0
else
  echo "COHERENCE FAIL: $FAILS/4 (log: $OUT)"
  exit 1
fi
