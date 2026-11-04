# Toolchain Setup (Ubuntu / WSL2 Ubuntu)

Sets up a bare-metal RV64IMC GCC toolchain and the Spike ISA simulator. Works on native Ubuntu
(22.04/24.04, x86_64) or Ubuntu under WSL2. All scripts assume the install location
`$HOME/riscv-tools`, so **run everything as your normal user** (not root), except `apt-get`.

This document covers **environment setup** only. Once set up, build and verify the project by
following the **Build & run** section of the [README](../README.md#build--run--script-purpose-expected-result):
it lists every script (including the Spike custom-extension build), its purpose, and the expected
result.

> **WSL users:** every command below is plain Ubuntu bash. Either open the Ubuntu shell (`wsl -d Ubuntu`)
> and paste them, or run a single command from Windows with
> `wsl -d Ubuntu -- bash -lc "<command>"` (inside PowerShell, escape `$` as `` `$ ``).
> Windows drives are mounted under `/mnt/` (e.g. `D:\` -> `/mnt/d/`), so the project is reachable as
> `cd "/mnt/d/<path to>/Project"`. For best performance and to avoid CRLF issues, a clone inside the
> Linux filesystem (`~/`) is preferable; if scripts fail with `set: pipefail: invalid option name`,
> they have CRLF line endings: run `sed -i 's/\r$//' scripts/*.sh`.

---

## Environment setup

### 1. System packages

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential autoconf automake autotools-dev curl python3 python3-pip \
  libmpc-dev libmpfr-dev libgmp-dev gawk bison flex texinfo gperf libtool \
  patchutils bc zlib1g-dev libexpat1-dev ninja-build cmake libglib2.0-dev \
  libboost-all-dev device-tree-compiler git
```

WSL without a working sudo password: run the same command as root from Windows with
`wsl -d Ubuntu -u root -- bash -lc "apt-get update && apt-get install -y <packages>"`.
(Only use root for `apt-get`; the toolchain itself must be installed as your normal user.)

### 2. RV64IMC bare-metal GCC (prebuilt xPack)

Building GCC from source is slow and RAM-heavy; use the prebuilt xPack release:

```bash
mkdir -p ~/riscv-tools && cd ~/riscv-tools
curl -L -o xpack-gcc.tar.gz \
  https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/download/v14.2.0-3/xpack-riscv-none-elf-gcc-14.2.0-3-linux-x64.tar.gz
tar -xzf xpack-gcc.tar.gz
```

Binary: `~/riscv-tools/xpack-riscv-none-elf-gcc-14.2.0-3/bin/riscv-none-elf-gcc`.
Compile with `-march=rv64imc -mabi=lp64` (no exact rv64imc multilib is bundled, but compiling and
linking work; the linker prints a harmless `LOAD segment with RWX permissions` warning).

### 3. Spike (build from source)

Spike needs its git **submodules** (e.g. `berkeley-softfloat-3`); without them `make` fails with
`No rule to make target '../softfloat/../berkeley-softfloat-3/source/include/softfloat.h'`.
`--recurse-submodules` is therefore required (a plain `--depth 1` clone is not enough).

```bash
cd ~/riscv-tools
rm -rf spike-src
git clone --recurse-submodules https://github.com/riscv-software-src/riscv-isa-sim.git spike-src
cd spike-src && mkdir build && cd build
../configure --prefix=$HOME/riscv-tools/spike-install
make -j4
make install
```

- `--prefix` must be an absolute path (`$HOME/...` expands correctly in a Linux shell).
  If you call this through `wsl ... bash -lc "..."` from PowerShell, write `` `$HOME `` or PowerShell
  expands `$HOME` to the Windows home and `configure` fails with
  `expected an absolute directory name for --prefix`.
- Use `-j4` (or lower) when the machine/WSL has about 4 GB RAM (check `free -h`); the build takes
  several minutes.
- Keep `~/riscv-tools/spike-src` and its `build/` directory: building the Spike custom extension
  (README, script #4) reuses them.

Binary: `~/riscv-tools/spike-install/bin/spike`.

---

## Verification

Once the environment above is set up, verify it by following the **Build & run** section of the
[README](../README.md#build--run--script-purpose-expected-result): it lists every script, its purpose
and the expected result.

Minimum check: `./scripts/build_and_run_smoke.sh` -> `Spike exit code: 0`.

Run scripts from the project root (`Project/`); if needed, `chmod +x scripts/*.sh` first.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `make`: `No rule to make target ... softfloat.h` | Spike cloned without submodules; re-clone with `--recurse-submodules` (step 3). |
| `configure: error: expected an absolute directory name for --prefix` | `$HOME` expanded by PowerShell; run inside the Ubuntu shell or escape `` `$HOME ``. |
| `set: pipefail: invalid option name` | Script has CRLF endings; `sed -i 's/\r$//' scripts/*.sh`. |
| `Permission denied` when running `./scripts/...` | Exec bit lost (repo on `/mnt/<drive>` without `metadata`, or checked out with `core.filemode=false`). Run `bash scripts/<name>.sh`, or `chmod +x scripts/*.sh scripts/*.py`. Verify the repo side with `git ls-files -s scripts` (expect `100755`). |
| Scripts can't find gcc/spike | Toolchain installed as root or elsewhere; scripts expect `$HOME/riscv-tools/...` of the current user. |
| `sudo` prompts fail/loop in WSL | Use `wsl -d Ubuntu -u root -- <cmd>` for `apt-get` only. |
| Build killed / out of memory | Lower parallelism (`make -j2`) or raise WSL memory in `.wslconfig`. |
