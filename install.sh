#!/usr/bin/env bash
# Selenita alpha installer (Linux / macOS).
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/lualab-xyz/selenita/alpha-1/install.sh | bash
#   bash install.sh [version]    # e.g. 0.1.0a1 — default: newest release
#
# Behavior:
#   1. Resolves the release tag (explicit version or newest release, prereleases included).
#   2. Finds a local Python >= 3.11.
#   3. Downloads the release wheel, verifies it against SHA256SUMS.
#   4. Installs it into a dedicated venv (~/.selenita/venv) and links the
#      `selenita` command into ~/.local/bin.
#
# Overrides: SELENITA_HOME (install prefix, default ~/.selenita) and
# SELENITA_BIN (command directory, default ~/.local/bin).
set -euo pipefail

REPO="lualab-xyz/selenita"
PREFIX="${SELENITA_HOME:-$HOME/.selenita}"
BIN_DIR="${SELENITA_BIN:-$HOME/.local/bin}"
WANT_VERSION="${1:-}"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || die "curl is required (apt install curl / brew install curl)"

# --- 1. Resolve the release tag (prereleases included via /releases list) ---
if [[ -n "$WANT_VERSION" ]]; then
  TAG="v${WANT_VERSION#v}"
else
  TAG="$(curl -fsSL "https://api.github.com/repos/$REPO/releases?per_page=1" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["tag_name"])' 2>/dev/null)" \
    || die "could not list releases of github.com/$REPO"
fi
VERSION="${TAG#v}"
WHEEL="selenita-${VERSION}-py3-none-any.whl"
log "Installing selenita ${VERSION}"

# --- 2. Locate a Python >= 3.11 ---------------------------------------------
PY=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && \
      "$candidate" -c 'import sys; quit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    PY="$candidate"
    break
  fi
done
[[ -n "$PY" ]] || die "Python >= 3.11 is required (try: apt install python3.11)"
log "Using Python: $($PY --version 2>&1) ($($PY -c 'import sys; print(sys.executable)'))"

# --- 3. Download the wheel and verify its checksum ---------------------------
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
log "Downloading $WHEEL"
curl -fL -o "$TMP/$WHEEL" "https://github.com/$REPO/releases/download/$TAG/$WHEEL" \
  || die "wheel download failed — does the release exist? (github.com/$REPO/releases)"
curl -fL -o "$TMP/SHA256SUMS" "https://github.com/$REPO/releases/download/$TAG/SHA256SUMS" \
  || die "SHA256SUMS download failed"
(cd "$TMP" && grep -F "$WHEEL" SHA256SUMS | sha256sum -c -) >/dev/null \
  || die "checksum verification failed for $WHEEL"

# --- 4. Install into a dedicated venv and link the command -------------------
log "Creating venv at $PREFIX/venv"
"$PY" -m venv "$PREFIX/venv"
"$PREFIX/venv/bin/pip" install --quiet --upgrade "$TMP/$WHEEL"

mkdir -p "$BIN_DIR"
ln -sf "$PREFIX/venv/bin/selenita" "$BIN_DIR/selenita"

log "Installed $($PREFIX/venv/bin/python -c 'import importlib.metadata as m; print("selenita", m.version("selenita"))')"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) log "NOTE: $BIN_DIR is not on your PATH — add it to your shell profile";;
esac
log "Try: selenita --help   (configure the LLM endpoint via SELENITA_* env vars)"