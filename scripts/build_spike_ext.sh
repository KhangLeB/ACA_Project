#!/usr/bin/env bash
# Phase 4 step 2: install the in-repo Spike extension (src/spike_ext/xqnn.cc) into the Spike
# source tree, register it in the build, rebuild, and install. Idempotent. Run under WSL.
set -euo pipefail

SPIKE_SRC="$HOME/riscv-tools/spike-src"
SPIKE_BUILD="$SPIKE_SRC/build"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT_SRC="$ROOT_DIR/src/spike_ext/xqnn.cc"

[ -f "$EXT_SRC" ] || { echo "missing $EXT_SRC" >&2; exit 1; }
[ -d "$SPIKE_BUILD" ] || { echo "missing Spike build dir $SPIKE_BUILD" >&2; exit 1; }

# 1. Copy the extension source into Spike's customext/ directory.
cp "$EXT_SRC" "$SPIKE_SRC/customext/xqnn.cc"

# 2. Register xqnn.cc in the customext source list (edit the template, then regenerate).
MK_IN="$SPIKE_SRC/customext/customext.mk.in"
if ! grep -q 'xqnn.cc' "$MK_IN"; then
  sed -i 's/\(\s*dummy_rocc.cc \\\)/\1\n\txqnn.cc \\/' "$MK_IN"
  echo "Added xqnn.cc to customext.mk.in"
fi
grep -q 'xqnn.cc' "$MK_IN" || { echo "failed to register xqnn.cc" >&2; exit 1; }

# 2b. Link xqnn.o statically into the spike binary so the extension registers itself at startup.
# This bypasses find_extension()'s dlopen fallback, which in this build double-registers the
# built-in MMIO device plugins (libcustomext.so statically bundles libriscv.a) and aborts with
# 'Plugin "imsic_mmio" already registered'. Mirrors the existing extension.o LDFLAGS hack.
SPIKE_MK_IN="$SPIKE_SRC/spike_main/spike_main.mk.in"
if ! grep -q 'xqnn.o' "$SPIKE_MK_IN"; then
  sed -i 's/^spike_main_LDFLAGS = extension.o$/spike_main_LDFLAGS = extension.o xqnn.o/' "$SPIKE_MK_IN"
  echo "Added xqnn.o to spike_main_LDFLAGS"
fi
grep -q 'xqnn.o' "$SPIKE_MK_IN" || { echo "failed to add xqnn.o to spike link" >&2; exit 1; }

# 3. Regenerate the build Makefiles from the templates (fast, no full reconfigure).
cd "$SPIKE_BUILD"
./config.status customext.mk spike_main.mk Makefile >/dev/null

# 4. Rebuild and install (WSL has limited RAM, keep parallelism modest). Force a relink of the
# spike binary: xqnn.o is injected via LDFLAGS, which the Makefile does not track as a prerequisite,
# so without removing the stale binary make would not re-link it.
rm -f "$SPIKE_BUILD/spike"
make -j4
make install

echo "Spike rebuilt with xqnn extension -> $HOME/riscv-tools/spike-install/bin/spike"
