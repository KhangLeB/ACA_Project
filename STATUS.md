# Trạng thái dự án — 1 trang tóm tắt (đọc file này trước, không cần đọc file khác)

Dự án: tăng tốc suy luận neural network đã lượng tử hóa (INT8) trên CPU RISC-V RV64IMC,
bằng cách thêm lệnh CPU mới (`Xqmac8`). Model dùng: MobileNetV3-Large, khối `features[3]`.

**GitHub: đã đồng bộ** (commit `5d2a782`, 2026-10-07) — Phase 3 + Phase 4 đã lên remote.

## Bảng trạng thái

| # | Việc | Xong? | Lệnh để chạy lại | Xem chi tiết ở |
|---|---|---|---|---|
| 0 | Cài GCC RISC-V + Spike, test toolchain | ✅ | `scripts/build_and_run_smoke.sh` | docs/toolchain_setup.md |
| 1 | Chọn model, lượng tử hóa INT8 | ✅ | `scripts/model_prep/export_block3.py` → `export_to_c.py` | DESIGN.md |
| 2 | Viết kernel C, kiểm tra khớp 100% với đáp án chuẩn | ✅ | `scripts/build_and_run_block3.sh` | src/kernels/kernels.c |
| 3 | Đo số lệnh CPU, tìm 3 điểm nghẽn | ✅ | `scripts/profile_phase3.sh` | EXPERIMENTS.md (mục 2026-09-12) |
| 4a | Thiết kế `Xqmac8`/`Xqrequant` + mô hình phần mềm, test bit-exact | ✅ | `scripts/build_and_run_xqmac.sh` | DESIGN.md, EXPERIMENTS.md (2026-10-07) |
| 4b | Cài 2 lệnh thật vào Spike C++ + đo X1 vs B0 (**4.13× giảm lệnh conv1x1**) | ✅ | `scripts/build_spike_ext.sh` → `build_and_run_xqmac_insn.sh` → `measure_phase4.sh` | src/spike_ext/xqnn.cc, EXPERIMENTS.md (2026-10-07) |
| 4c | B1 tối ưu phần mềm (depthwise interior/border) + mix đầy đủ | ✅ | `scripts/measure_phase4.sh` | results/phase4_instruction_mix.csv |
| 4d | Push Phase 3 + Phase 4 lên GitHub | ✅ | `git push` (commit 5d2a782) | — |
| 5 | Viết báo cáo giữa kỳ | ⬜ chưa làm | — | — |
| 6 | Port sang CV-Wally RTL, tổng hợp (synthesis) | ⬜ chưa làm | — | — |

## 3 điểm nghẽn đã đo được (Phase 3) — chỉ cần nhớ 3 dòng này

1. **Mỗi phép nhân hữu ích tốn thêm ~7–9 lệnh phụ** (chỉ 11–13% lệnh là phép nhân thật).
2. **Depthwise conv tốn nhiều lệnh rẽ nhánh nhất** (28% so với 13–16% chỗ khác) — do kiểm tra
   biên ảnh lặp lại không cần thiết.
3. **Load bộ nhớ từng byte một, không tái sử dụng** (12–26% số lệnh).

→ Đây là 3 lý do để thiết kế `Xqmac8` ở Phase 4.

## Việc tiếp theo (Phase 4)

✅ Bước 1–4 đã xong. `Xqmac8`/`Xqrequant` là lệnh thật trong Spike (`src/spike_ext/xqnn.cc`,
opcode custom-0 0x0b), verify bit-exact (directed + randomized). Kết quả đo (8×8, kênh thật):
- conv1x1 (GEMM/FC): 970k → 235k = **4.13×** giảm lệnh (X1 Xqmac8) — vượt mục tiêu ≥×2.
- requantize: 85k → 55k = **1.55×** (X1 Xqrequant).
- depthwise: 840k → 695k = **1.21×** (B1 phần mềm, branch 28%→21%).
Bottleneck mới của X1 = ALU 51% (sinh địa chỉ, vòng lặp, bù `zp·colsum`) → động lực cho
 hardware-loop / post-increment load-store ở Phase 7.

Việc còn lại trước nộp giữa kỳ: **push Phase 3 + Phase 4 lên GitHub** (remote mới có tới Phase 1),
rồi viết báo cáo giữa kỳ từ các số liệu này.

## Không cần đọc gì thêm để làm việc tiếp

Không cần đọc lại toàn bộ EXPERIMENTS.md hay DESIGN.md để hiểu — file này đã đủ. Chỉ mở
EXPERIMENTS.md nếu cần số liệu chi tiết (bảng đầy đủ %, số lệnh từng loại) để trích vào báo cáo.
