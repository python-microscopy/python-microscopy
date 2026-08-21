#!/usr/bin/env bash
# Linux/Mac installer for PYME. Also serves as the Mac installer pending a native .app.
# Usage: install-python-microscopy.sh [--dest <path>]   (default: ~/PYME)
#
# Context A (CI/dev): uv is expected to be in PATH already.
# Context B (end-user): uv is bootstrapped via astral.sh if not found.
set -euo pipefail

# --- Configuration ---
TARGET_PYTHON=3.13
PACKAGE_NAME=python-microscopy
ENTRY_POINTS=(PYMEAcquire PYMEImage PYMEVis PYMEClusterOfOne)
DEFAULT_DEST="$HOME/PYME"

DEST="$DEFAULT_DEST"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest)   DEST="$2"; shift 2 ;;
        --dest=*) DEST="${1#--dest=}"; shift ;;
        -h|--help)
            echo "Usage: $0 [--dest <path>]"
            echo "  Installs PYME into a self-contained directory using uv."
            echo "  Default: $DEFAULT_DEST"
            exit 0 ;;
        *) echo "Unknown argument: $1 (run '$0 --help' for usage)" >&2; exit 1 ;;
    esac
done

DEST="${DEST/#\~/$HOME}"
echo "==> Installing PYME to: $DEST"
mkdir -p "$DEST"

# --- Ensure uv is available ---
if ! command -v uv &>/dev/null; then
    echo "==> uv not found; downloading via astral.sh..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add the two common install locations; one will contain the binary.
    export PATH="$HOME/.local/bin:${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        echo "ERROR: uv installation failed or installed to an unexpected location." >&2
        exit 1
    fi
fi

# --- Managed Python ---
uv python install "$TARGET_PYTHON"

# --- Virtual environment ---
uv venv --python "$TARGET_PYTHON" "$DEST/venv"

# --- Install PYME from pip ---
echo "==> Installing $PACKAGE_NAME..."
uv pip install --python "$DEST/venv/bin/python" "$PACKAGE_NAME"

# --- Top-level entry point symlinks ---
for ep in "${ENTRY_POINTS[@]}"; do
    ln -sf "$DEST/venv/bin/$ep" "$DEST/$ep"
done

# --- Activated-shell helper ---
cat > "$DEST/pyme-shell" <<'SHELL_SCRIPT'
#!/usr/bin/env bash
PYME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PYME_DIR/venv/bin/activate"
exec "${SHELL:-bash}"
SHELL_SCRIPT
chmod +x "$DEST/pyme-shell"

echo ""
echo "==> Done."
echo "    Entry points: ${ENTRY_POINTS[*]}"
echo "    Add to PATH:       export PATH=\"$DEST:\$PATH\""
echo "    Activated shell:   $DEST/pyme-shell"
