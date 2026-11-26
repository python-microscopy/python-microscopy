#!/usr/bin/env bash
# Build the end-user installer with a hash-locked dependency set.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_PYTHON=3.13
PACKAGE_NAME=python-microscopy
TEMPLATE="$SCRIPT_DIR/install-python-microscopy.sh.in"
OUTPUT="$SCRIPT_DIR/install-python-microscopy.sh"

if ! command -v uv &>/dev/null; then
    echo "ERROR: uv must be installed and available in PATH to build the installer." >&2
    exit 1
fi

TMP_REQS=$(mktemp)
TMP_OUTPUT=$(mktemp "$SCRIPT_DIR/.install-python-microscopy.XXXXXX")
trap 'rm -f "$TMP_REQS" "$TMP_OUTPUT"' EXIT

echo "==> Compiling universal requirements for Python $TARGET_PYTHON..."
printf '%s\n' "$PACKAGE_NAME" | uv pip compile - \
    --python-version "$TARGET_PYTHON" \
    --universal \
    --generate-hashes \
    --custom-compile-command "printf '%s\\n' '$PACKAGE_NAME' | uv pip compile - --python-version $TARGET_PYTHON --universal --generate-hashes -o requirements.txt" \
    -o "$TMP_REQS"

echo "==> Embedding requirements in installer..."
awk -v reqfile="$TMP_REQS" '
    $0 == "___EMBEDDED_REQUIREMENTS_PLACEHOLDER___" {
        count++
        while ((getline line < reqfile) > 0) print line
        close(reqfile)
        next
    }
    { print }
    END { if (count != 1) exit 1 }
' "$TEMPLATE" > "$TMP_OUTPUT"

chmod 755 "$TMP_OUTPUT"
mv "$TMP_OUTPUT" "$OUTPUT"
echo "==> Built: $OUTPUT"