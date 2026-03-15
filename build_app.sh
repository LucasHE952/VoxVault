#!/usr/bin/env bash
# build_app.sh — Build the OnDevice Dictation .app bundle with py2app.
#
# Usage:
#   bash build_app.sh           # full release build → dist/OnDevice Dictation.app
#   bash build_app.sh --alias   # fast alias build for development/testing
#   bash build_app.sh --install # build + copy to /Applications

set -euo pipefail

ALIAS=false
INSTALL=false

for arg in "$@"; do
    case "$arg" in
        --alias)   ALIAS=true ;;
        --install) INSTALL=true ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: bash build_app.sh [--alias] [--install]"
            exit 1
            ;;
    esac
done

# ── Verify we're in the project root ─────────────────────────────────────────
if [[ ! -f "setup.py" ]]; then
    echo "ERROR: Run this script from the project root (where setup.py lives)."
    exit 1
fi

# ── Activate virtualenv ───────────────────────────────────────────────────────
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
else
    echo "WARNING: .venv not found — using system Python. Run setup.sh first."
fi

# ── Install py2app if needed ──────────────────────────────────────────────────
if ! python -c "import py2app" 2>/dev/null; then
    echo "Installing py2app…"
    pip install py2app --quiet
fi

# ── Check model weights are present ──────────────────────────────────────────
MODEL_DIR="$HOME/.cache/ondevice-dictation/models/voxtral-realtime"
if [[ ! -d "$MODEL_DIR" ]]; then
    echo ""
    echo "WARNING: Model weights not found at:"
    echo "  $MODEL_DIR"
    echo ""
    echo "The app will build but will show an error on first launch."
    echo "Run setup.sh to download the model (~2.9 GB) before using the app."
    echo ""
fi

# ── Clean previous build ──────────────────────────────────────────────────────
echo "Cleaning previous build artefacts…"
rm -rf build/ dist/

# ── Build ─────────────────────────────────────────────────────────────────────
if $ALIAS; then
    echo "Building alias bundle (development mode — fast, uses source in place)…"
    python setup.py py2app --alias 2>&1
else
    echo "Building release bundle (this takes a few minutes)…"
    python setup.py py2app 2>&1
fi

echo ""
echo "✓ Build complete: dist/OnDevice Dictation.app"

# ── Copy namespace packages that py2app can't discover ───────────────────────
# mlx and mlx_lm are namespace packages (PEP 420) — py2app's imp_find_module
# can't locate them, so they end up as .pyc stubs in python312.zip without
# the critical C extension (mlx/core.cpython-312-darwin.so). Copy them as
# real directories into the bundle's lib/ tree.
BUNDLE_LIB="dist/OnDevice Dictation.app/Contents/Resources/lib/python3.12"
SITE_PKGS="$(python -c 'import site; print(site.getsitepackages()[0])')"

for pkg in mlx mlx_lm mlx_audio; do
    SRC="$SITE_PKGS/$pkg"
    if [[ -d "$SRC" ]]; then
        echo "Copying namespace package: $pkg"
        rm -rf "$BUNDLE_LIB/$pkg"
        cp -R "$SRC" "$BUNDLE_LIB/$pkg"
    else
        echo "WARNING: $pkg not found in site-packages — skipping"
    fi
done

# ── Remove zipped copies that shadow the real directories ────────────────────
# python312.zip contains .pyc stubs for mlx/mlx_audio/mlx_lm but NOT their
# C extensions (.so). The zip is earlier on sys.path, so Python finds the
# broken zip copy first. Delete them from the zip so Python falls through to
# the real directories we just copied above.
ZIPFILE="$BUNDLE_LIB/../python312.zip"
if [[ -f "$ZIPFILE" ]]; then
    echo "Removing shadowed packages from python312.zip…"
    # -d = delete entries matching the glob patterns
    zip -d "$ZIPFILE" "mlx/*" "mlx_lm/*" "mlx_audio/*" 2>/dev/null || true
fi

# ── Fix @rpath for torchaudio → torch dependency ────────────────────────────
# libtorchaudio.so links to @rpath/libtorch.dylib. In the venv, torch sets up
# the rpath at import time, but in a frozen bundle that doesn't happen.
# Add the torch/lib directory as an rpath to every torchaudio native lib.
# libtorchaudio.so is at torchaudio/lib/, libtorch.dylib is at torch/lib/
# @loader_path is the dir containing the loading binary → torchaudio/lib/
# So ../../torch/lib gets us to the right place regardless of install location.
# torchaudio libs need two rpaths:
#   1. @loader_path — for sibling libs (e.g. _torchaudio.so → libtorchaudio.so)
#   2. @loader_path/../../torch/lib — for torch libs (libtorch.dylib, libc10.dylib)
if [[ -d "$BUNDLE_LIB/torchaudio/lib" ]]; then
    echo "Fixing @rpath for torchaudio…"
    for f in "$BUNDLE_LIB/torchaudio/lib/"*.{so,dylib}; do
        [[ -f "$f" ]] || continue
        install_name_tool -add_rpath "@loader_path" "$f" 2>/dev/null || true
        install_name_tool -add_rpath "@loader_path/../../torch/lib" "$f" 2>/dev/null || true
    done
fi

# ── Re-sign with ad-hoc signature ─────────────────────────────────────────────
# py2app copies .so and .dylib files from the venv but their existing code
# signatures are invalidated by the copy.  macOS dyld kills the process if
# any loaded native file has a bad signature.
#
# Strategy: sign every native file individually first, THEN deep-sign the
# outer bundle.  The --deep flag alone misses files inside nested frameworks
# and lib/ directories.
# ── Fix Python.framework symlink structure ───────────────────────────────────
# macOS requires: Versions/Current → 3.12 (symlink), and top-level Python,
# Resources → Versions/Current/* (symlinks). py2app copies them as real
# files/dirs, which makes the bundle "ambiguous" to codesign.
PYFW="dist/OnDevice Dictation.app/Contents/Frameworks/Python.framework"
if [[ -d "$PYFW" ]]; then
    echo "Fixing Python.framework symlink structure…"
    rm -rf "$PYFW/Versions/Current"
    rm -rf "$PYFW/Python"
    rm -rf "$PYFW/Resources"
    ln -s "3.12" "$PYFW/Versions/Current"
    ln -s "Versions/Current/Python" "$PYFW/Python"
    ln -s "Versions/Current/Resources" "$PYFW/Resources"
fi

# ── Re-sign everything (inside-out) ─────────────────────────────────────────
# Sign order matters: individual binaries → frameworks → outer bundle.
echo "Re-signing native binaries inside the bundle…"
NATIVE_COUNT=0
while IFS= read -r f; do
    codesign --force --sign - "$f" 2>/dev/null && ((NATIVE_COUNT++)) || true
done < <(find "dist/OnDevice Dictation.app" \( -name '*.so' -o -name '*.dylib' \) -not -path "*/Python.framework/*")
echo "  Signed $NATIVE_COUNT native files"

# Sign Frameworks/ dylibs
echo "Re-signing Frameworks…"
for f in "dist/OnDevice Dictation.app/Contents/Frameworks/"*.dylib; do
    [[ -f "$f" ]] && codesign --force --sign - "$f"
done

# Sign Python.framework (binary first, then framework bundle)
if [[ -d "$PYFW" ]]; then
    echo "Re-signing Python.framework…"
    codesign --force --sign - "$PYFW/Versions/3.12/Python"
    codesign --force --sign - "$PYFW"
fi

echo "Re-signing outer bundle…"
codesign --force --sign - "dist/OnDevice Dictation.app" 2>&1
echo "✓ Signed"

# ── Install ───────────────────────────────────────────────────────────────────
if $INSTALL; then
    DEST="/Applications/OnDevice Dictation.app"
    echo "Installing to /Applications…"
    rm -rf "$DEST"
    # -R preserves symlinks; -P prevents dereferencing (required for framework signing)
    cp -RPp "dist/OnDevice Dictation.app" "$DEST"
    echo "✓ Installed: $DEST"
    echo ""
    echo "Launch from Spotlight or Finder, or run:"
    echo "  open '/Applications/OnDevice Dictation.app'"
else
    echo ""
    echo "To test:    open 'dist/OnDevice Dictation.app'"
    echo "To install: bash build_app.sh --install"
    echo "            (or: cp -r 'dist/OnDevice Dictation.app' /Applications/)"
fi
