#!/usr/bin/env bash
# Copyright (c) 2025-2026 Kirky.X
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license information.
#
# Check copyright headers in all .rs files.
# Verifies each file contains:
#   - "Copyright (c) 2025" (matches 2025 or 2025-2026)
#   - "Kirky.X"
#   - "MIT License"
# Exits non-zero if any file violates.

set -euo pipefail

violations=0
violated_files=""

# Collect all .rs files from src/, tests/, examples/, and build.rs
mapfile -t files < <({
    find src tests examples -name "*.rs" -type f 2>/dev/null
    [ -f build.rs ] && echo "build.rs"
} | sort)

for file in "${files[@]}"; do
    # Read first 5 lines of the file (copyright header location)
    header=$(head -5 "$file" 2>/dev/null || echo "")

    # Check for Copyright (c) 2025 (matches 2025 or 2025-2026)
    if ! echo "$header" | grep -qE "Copyright \(c\) 2025(-2026)? Kirky\.X"; then
        violated_files="${violated_files}\n  - $file (missing/incorrect copyright)"
        violations=$((violations + 1))
        continue
    fi

    # Check for MIT License
    if ! echo "$header" | grep -q "MIT License"; then
        violated_files="${violated_files}\n  - $file (missing MIT License)"
        violations=$((violations + 1))
        continue
    fi
done

if [ "$violations" -gt 0 ]; then
    echo "=== Copyright Header Check FAILED ==="
    echo "Found $violations file(s) with incorrect copyright headers:"
    echo -e "$violated_files"
    echo ""
    echo "Expected header format:"
    echo "  // Copyright (c) 2025-2026 Kirky.X"
    echo "  //"
    echo "  // Licensed under the MIT License"
    echo "  // See LICENSE file in the project root for full license information."
    echo ""
    echo "Run to fix: See scripts/check_copyright.sh"
    exit 1
fi

echo "=== Copyright Header Check PASSED ==="
echo "All ${#files[@]} .rs files have correct copyright headers."
exit 0
