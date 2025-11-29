#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8004}"

echo "=== YouTube ingest test suite ==="
echo "Base URL: ${BASE_URL}"
echo

# 10 test cases
TEST_URLS=(
  "https://www.youtube.com/watch?v=3IQVNjWH60A"   # 1 – Jeff Nippard upper body (SHOULD PARSE)
  "https://www.youtube.com/watch?v=cbKkB3POqaY"   # 2 – 25-min full body HIIT
  "https://www.youtube.com/watch?v=Jv0Fl11dSWo"   # 3
  "https://www.youtube.com/watch?v=ByUxQODHzes"   # 4
  "https://www.youtube.com/watch?v=d25EPGfefJI"   # 5
  "https://www.youtube.com/watch?v=zEf4pKoKc70"   # 6
  "https://www.youtube.com/watch?v=ykoAF6b3EPE"   # 7
  "https://www.youtube.com/watch?v=fTUYm2Da8GA"   # 8
  "https://www.youtube.com/watch?v=dB0pzbSkh_s"   # 9
  "https://www.youtube.com/watch?v=yE0_AK2zSqM"   # 10
)

TEST_LABELS=(
  "Jeff Nippard upper body – SHOULD PARSE"
  "25-min full body HIIT – may be non-structured"
  "Test 3"
  "Test 4"
  "Test 5"
  "Test 6"
  "Test 7"
  "Test 8"
  "Test 9"
  "Test 10"
)

for i in "${!TEST_URLS[@]}"; do
  url="${TEST_URLS[$i]}"
  label="${TEST_LABELS[$i]}"

  # Skip empty slots (if any)
  if [[ -z "${url}" ]]; then
    continue
  fi

  echo "▶ URL:   ${url}"
  echo "   Note: ${label}"
  echo

  # Call the API
  resp="$(curl -s -S -X POST "${BASE_URL}/ingest/youtube" \
    -H "Content-Type: application/json" \
    -d "{\"url\":\"${url}\"}")" || {
      echo "   ❌ Request failed"
      echo
      continue
    }

  # Compact summary
  echo "${resp}" | jq '{
    title,
    youtube_strategy: (._provenance.youtube_strategy // null),
    blocks_count: (.blocks | length)
  }' 2>/dev/null || {
    echo "   ⚠️ Could not parse JSON with jq. Raw response:"
    echo "${resp}"
  }

  echo "----------------------------------------"
  echo
done
