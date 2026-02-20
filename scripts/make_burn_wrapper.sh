#!/bin/bash
# Wrapper for video burning that uses merged predictions if available

set -e

DECODE_JSON="$1"
VIDEO_DIR="$2"
OUTPUT_DIR="$3"

if [ -z "$DECODE_JSON" ] || [ -z "$VIDEO_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <decode_json> <video_dir> <output_dir>"
    exit 1
fi

# Check if merged output exists
MERGED_JSON="${DECODE_JSON%.json}-merged.json"

USE_MERGED=0
if [ -f "$MERGED_JSON" ] && [ "${OVERLAP_ENABLED:-1}" = "1" ]; then
    # Check if merged JSON has sufficient coverage
    # Count entries in both files
    STANDARD_COUNT=$(python3 -c "import json; d=json.load(open('$DECODE_JSON')); print(len(d.get('utt_id', [])))" 2>/dev/null || echo "0")
    MERGED_COUNT=$(python3 -c "import json; d=json.load(open('$MERGED_JSON')); print(len(d))" 2>/dev/null || echo "0")

    # Use merged only if it has at least 50% of standard entries
    # (merged should have fewer entries since it combines overlapping segments)
    if [ "$MERGED_COUNT" -gt 0 ] && [ "$STANDARD_COUNT" -gt 0 ]; then
        COVERAGE=$(python3 -c "print(100 * $MERGED_COUNT / max(1, $STANDARD_COUNT))")
        echo "Merged coverage: ${MERGED_COUNT}/${STANDARD_COUNT} videos (${COVERAGE}%)"

        # If merged has very few videos compared to standard, it likely failed
        if (( $(echo "$MERGED_COUNT >= 1" | bc -l) )) && (( $(echo "$COVERAGE > 10" | bc -l) )); then
            USE_MERGED=1
        else
            echo "⚠ Merged JSON incomplete (coverage < 10%), falling back to standard decode"
        fi
    fi
fi

if [ "$USE_MERGED" = "1" ]; then
    echo "✓ Using merged predictions for video burning..."
    python3 "$(dirname "$0")/make_burn.py" \
        --jsonl "$MERGED_JSON" \
        --video_dir "$VIDEO_DIR" \
        --out_dir "$OUTPUT_DIR"
    echo "✓ Videos burned with merged predictions"
else
    echo "Using standard segment-level predictions..."
    python3 "$(dirname "$0")/make_burn.py" \
        --jsonl "$DECODE_JSON" \
        --video_dir "$VIDEO_DIR" \
        --out_dir "$OUTPUT_DIR"
fi
