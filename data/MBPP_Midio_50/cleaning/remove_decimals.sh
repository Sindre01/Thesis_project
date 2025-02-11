#!/bin/bash

# Directory containing files (modify if needed)
DIR="../includes_tests"

# Loop through all text-based files (modify file pattern if necessary)
for file in "$DIR"/*.txt; do
    if [[ -f "$file" ]]; then
        echo "Processing: $file"

        # Use `sed` to remove decimals from numbers
        sed -i -E 's/([0-9]+)\.[0-9]+/\1/g' "$file"
    fi
done

echo "Decimal removal completed!"
