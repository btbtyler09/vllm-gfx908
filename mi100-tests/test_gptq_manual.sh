#!/bin/bash
# Manual GPTQ Testing Protocol
# Run this after starting vLLM server to test for regressions

set -e

# Configuration
MODEL="kaitchup/Qwen3-32B-autoround-4bit-gptq"
SERVER_URL="http://localhost:8000"
TIMEOUT=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 GPTQ Manual Testing Protocol${NC}"
echo "=" * 50
echo "Model: $MODEL"
echo "Server: $SERVER_URL"
echo

# Check if server is responding
echo -e "${BLUE}🔍 Checking server status...${NC}"
if ! curl -s --max-time 5 "$SERVER_URL/health" > /dev/null; then
    echo -e "${RED}❌ Server not responding at $SERVER_URL${NC}"
    echo "Please start vLLM server first:"
    echo "  vllm serve $MODEL --tensor-parallel-size 4 --gpu-memory-utilization 0.98"
    exit 1
fi
echo -e "${GREEN}✅ Server is responding${NC}"
echo

# Test prompts with different characteristics
declare -a TEST_PROMPTS=(
    "Hello, how are you?"
    "What is 2+2?"
    "Write a short poem about AI."
    "Explain quantum computing briefly."
    "Hi"  # Edge case - very short prompt
)

declare -a TEST_DESCRIPTIONS=(
    "Simple greeting"
    "Basic math"
    "Creative writing"
    "Technical explanation" 
    "Very short prompt (edge case)"
)

# Test each prompt
TOTAL_TESTS=${#TEST_PROMPTS[@]}
PASSED_TESTS=0
FAILED_TESTS=0
REGRESSION_DETECTED=false

for i in "${!TEST_PROMPTS[@]}"; do
    prompt="${TEST_PROMPTS[$i]}"
    description="${TEST_DESCRIPTIONS[$i]}"
    
    echo -e "${BLUE}[$((i+1))/$TOTAL_TESTS] Testing: $description${NC}"
    echo "Prompt: '$prompt'"
    
    # Prepare JSON payload
    json_payload=$(cat <<EOF
{
    "model": "$MODEL",
    "prompt": "$prompt",
    "max_tokens": 50,
    "temperature": 0.0,
    "stream": false
}
EOF
)
    
    # Send request and capture response
    start_time=$(date +%s.%N)
    response=$(curl -s --max-time $TIMEOUT -X POST "$SERVER_URL/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$json_payload")
    end_time=$(date +%s.%N)
    
    # Calculate response time
    response_time=$(echo "$end_time - $start_time" | bc)
    
    # Check if request succeeded
    if [[ $? -ne 0 ]] || [[ -z "$response" ]]; then
        echo -e "${RED}❌ Request failed or timed out${NC}"
        ((FAILED_TESTS++))
        echo
        continue
    fi
    
    # Extract generated text from JSON response
    generated_text=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'choices' in data and len(data['choices']) > 0:
        print(data['choices'][0]['text'])
    else:
        print('ERROR: No choices in response')
except:
    print('ERROR: Invalid JSON response')
" 2>/dev/null)
    
    # Check for various issues
    has_exclamation_regression=false
    has_error=false
    
    if [[ "$generated_text" == *"!!!!"* ]]; then
        has_exclamation_regression=true
        REGRESSION_DETECTED=true
    fi
    
    if [[ "$generated_text" == *"ERROR"* ]]; then
        has_error=true
    fi
    
    # Determine test result
    if [[ "$has_exclamation_regression" == true ]]; then
        echo -e "${RED}❌ REGRESSION DETECTED: '!!!!' found in output${NC}"
        echo -e "${RED}   Output: $generated_text${NC}"
        ((FAILED_TESTS++))
    elif [[ "$has_error" == true ]]; then
        echo -e "${RED}❌ ERROR in response${NC}"
        echo -e "${RED}   Output: $generated_text${NC}"
        ((FAILED_TESTS++))
    elif [[ -z "$generated_text" ]] || [[ "$generated_text" == "ERROR"* ]]; then
        echo -e "${RED}❌ Empty or invalid response${NC}"
        echo -e "${RED}   Raw response: $response${NC}"
        ((FAILED_TESTS++))
    else
        echo -e "${GREEN}✅ Response looks normal${NC}"
        echo -e "${GREEN}   Response time: ${response_time}s${NC}"
        echo "   Output preview: '$(echo "$generated_text" | head -c 80)...'"
        ((PASSED_TESTS++))
    fi
    
    echo
done

# Summary
echo -e "${BLUE}📊 Test Summary${NC}"
echo "=" * 30
echo "Total tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"

if [[ $FAILED_TESTS -gt 0 ]]; then
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
fi

if [[ "$REGRESSION_DETECTED" == true ]]; then
    echo -e "${RED}⚠️  CRITICAL: '!!!!' REGRESSION DETECTED${NC}"
    echo -e "${RED}This indicates a fundamental issue with GPTQ inference${NC}"
fi

# Final result
if [[ $FAILED_TESTS -eq 0 ]] && [[ "$REGRESSION_DETECTED" == false ]]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED${NC}"
    echo "No regressions detected - inference appears to be working correctly"
    exit 0
else
    echo -e "${RED}❌ TESTS FAILED${NC}"
    if [[ "$REGRESSION_DETECTED" == true ]]; then
        echo "Critical regression detected - kernel changes may have broken inference"
    fi
    exit 1
fi