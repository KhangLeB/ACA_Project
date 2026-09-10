OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
  RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M
}

SECTIONS
{
  . = ORIGIN(RAM);
  .text.init : { *(.text.init) } > RAM
  .text      : { *(.text*) } > RAM
  .rodata    : { *(.rodata*) } > RAM
  .data      : { *(.data*) } > RAM
  .bss       : { *(.bss*) *(COMMON) } > RAM
  .tohost    : { *(.tohost) } > RAM
  .fromhost  : { *(.fromhost) } > RAM

  . = ALIGN(16);
  _stack_top = ORIGIN(RAM) + LENGTH(RAM);
}
