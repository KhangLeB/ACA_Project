# Toolchain Setup (WSL2 Ubuntu)

These steps set up a bare-metal RV64IMC GCC toolchain and the Spike ISA simulator inside WSL2 Ubuntu.

## 1. Prerequisites (as root, to avoid sudo password issues)

```bash
wsl -d Ubuntu -u root -- bash -lc "apt-get update && apt-get install -y \
  build-essential autoconf automake autotools-dev curl python3 python3-pip \
  libmpc-dev libmpfr-dev libgmp-dev gawk bison flex texinfo gperf libtool \
  patchutils bc zlib1g-dev libexpat1-dev ninja-build cmake libglib2.0-dev \
  libboost-all-dev device-tree-compiler git"
```

## 2. RV64IMC bare-metal GCC (prebuilt xPack toolchain)

Building GCC from source is slow and RAM-heavy; use the prebuilt xPack release instead:

```bash
mkdir -p ~/riscv-tools && cd ~/riscv-tools
curl -L -o xpack-gcc.tar.gz \
  https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/download/v14.2.0-3/xpack-riscv-none-elf-gcc-14.2.0-3-linux-x64.tar.gz
tar -xzf xpack-gcc.tar.gz
```

Binary: `~/riscv-tools/xpack-riscv-none-elf-gcc-14.2.0-3/bin/riscv-none-elf-gcc`.
Use `-march=rv64imc -mabi=lp64` (no exact rv64imc multilib is bundled, but compilation/linking works).

## 3. Spike (build from source)

```bash
cd ~/riscv-tools
git clone --depth 1 https://github.com/riscv-software-src/riscv-isa-sim.git spike-src
cd spike-src && mkdir build && cd build
../configure --prefix=$HOME/riscv-tools/spike-install
make -j4
make install
```

Binary: `~/riscv-tools/spike-install/bin/spike`. Run with `--isa=rv64imc <elf>`.

## 4. Verify

```bash
./scripts/build_and_run_smoke.sh
```

Builds `src/smoke_test/` (crt0.S + link.ld + main.c, HTIF tohost/fromhost exit protocol) and runs it
under Spike. Expected: `Spike exit code: 0`.

## Notes

- If `sudo` prompts fail/loop in WSL, use `wsl -d Ubuntu -u root -- <cmd>` instead (no password needed).
- WSL RAM may be limited (check with `free -h`); prefer `make -j4` over higher parallelism for builds.
