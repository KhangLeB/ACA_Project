#!/usr/bin/env python3
"""Phase 3: classify a Spike --log-commits trace into an instruction mix.

Spike's commit-log lines (this build) look like:
    core   0: 3 0x000000008000010e (0x00050863)
i.e. no mnemonic is printed, only PC and raw encoding. So we build a PC -> mnemonic map from
`objdump -d` static disassembly of the same ELF, then join it against the dynamic PC trace.

Usage:
    parse_spike_log.py --dis <objdump_txt> --log <commit_log> --name <label> [--csv <path>]
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter

DIS_LINE = re.compile(r"^\s*([0-9a-fA-F]+):\s+[0-9a-fA-F]+\s+(\S+)")
# Capture both the retired PC and the raw instruction encoding (the parenthesized hex) so custom
# instructions, which objdump cannot disassemble, can be classified directly from their opcode.
LOG_LINE = re.compile(r"^core\s+\d+:\s+\d+\s+(0x[0-9a-fA-F]+)\s+\((0x[0-9a-fA-F]+)\)")

# custom-0 opcode (xqnn extension: xqmac8 / xqrequant). Low 7 bits of a 32-bit encoding.
CUSTOM0_OPCODE = 0x0b

LOAD = {"lb", "lbu", "lh", "lhu", "lw", "lwu", "ld",
        "c.lw", "c.ld", "c.lwsp", "c.ldsp"}
STORE = {"sb", "sh", "sw", "sd",
         "c.sw", "c.sd", "c.swsp", "c.sdsp"}
BRANCH = {"beq", "bne", "blt", "bge", "bltu", "bgeu",
          "jal", "jalr", "j", "ret",
          "c.beqz", "c.bnez", "c.j", "c.jal", "c.jr", "c.jalr"}
MUL = {"mul", "mulh", "mulhsu", "mulhu", "mulw",
       "div", "divu", "divw", "divuw", "rem", "remw", "remu", "remuw"}
# everything else integer/immediate/move is bucketed as ALU by default in classify()


def classify(mnemonic):
    if mnemonic in LOAD:
        return "load"
    if mnemonic in STORE:
        return "store"
    if mnemonic in BRANCH:
        return "branch"
    if mnemonic in MUL:
        return "multiply"
    if mnemonic in {"ecall", "ebreak", "fence", "fence.i", "nop", "c.nop", "c.ebreak"} or \
       mnemonic.startswith("csr"):
        return "other"
    return "alu"


def build_pc_map(dis_path):
    pc_to_mnem = {}
    with open(dis_path, "r", errors="replace") as f:
        for line in f:
            m = DIS_LINE.match(line)
            if m:
                pc_to_mnem[int(m.group(1), 16)] = m.group(2)
    return pc_to_mnem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dis", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    pc_to_mnem = build_pc_map(args.dis)

    pc_counts = Counter()
    pc_enc = {}
    total = 0
    with open(args.log, "r", errors="replace") as f:
        for line in f:
            m = LOG_LINE.match(line)
            if not m:
                continue
            pc = int(m.group(1), 16)
            pc_counts[pc] += 1
            if pc not in pc_enc:
                pc_enc[pc] = int(m.group(2), 16)
            total += 1

    cat_counts = Counter()
    unknown = 0
    for pc, cnt in pc_counts.items():
        # Custom instructions are opcode-classified (objdump cannot decode them).
        if (pc_enc.get(pc, 0) & 0x7f) == CUSTOM0_OPCODE:
            cat_counts["custom"] += cnt
            continue
        mnem = pc_to_mnem.get(pc)
        if mnem is None:
            unknown += cnt
            continue
        cat_counts[classify(mnem)] += cnt

    print(f"=== {args.name} ===")
    print(f"total retired instructions: {total}")
    for cat in ("load", "store", "alu", "multiply", "branch", "custom", "other"):
        c = cat_counts.get(cat, 0)
        pct = 100.0 * c / total if total else 0.0
        print(f"  {cat:10s} {c:10d} ({pct:5.1f}%)")
    if unknown:
        pct = 100.0 * unknown / total if total else 0.0
        print(f"  {'unknown':10s} {unknown:10d} ({pct:5.1f}%)  <- PC not found in objdump (e.g. trailing spin loop)")

    if args.csv:
        write_header = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["name", "total", "load", "store", "alu", "multiply", "branch", "custom", "other", "unknown"])
            w.writerow([args.name, total,
                        cat_counts.get("load", 0), cat_counts.get("store", 0),
                        cat_counts.get("alu", 0), cat_counts.get("multiply", 0),
                        cat_counts.get("branch", 0), cat_counts.get("custom", 0),
                        cat_counts.get("other", 0), unknown])


if __name__ == "__main__":
    sys.exit(main())
