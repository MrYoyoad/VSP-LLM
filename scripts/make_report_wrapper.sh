#!/bin/bash
# Wrapper script that detects merged output and generates appropriate reports

set -e

DECODE_JSON="$1"
OUTPUT_DIR="$2"

if [ -z "$DECODE_JSON" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <decode_json> <output_dir>"
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
    echo "✓ Using merged predictions for reports..."

    # Use merged output for standard report
    python3 "$(dirname "$0")/make_report.py" \
        --jsonl "$MERGED_JSON" \
        --out_dir "$OUTPUT_DIR"

    # Generate conflict report
    python3 "$(dirname "$0")/generate_conflict_report.py" \
        --merged-json "$MERGED_JSON" \
        --output-dir "$OUTPUT_DIR/conflicts"

    echo "✓ Reports generated with merged predictions"
else
    echo "Using standard segment-level reports..."
    python3 "$(dirname "$0")/make_report.py" \
        --jsonl "$DECODE_JSON" \
        --out_dir "$OUTPUT_DIR"
fi
