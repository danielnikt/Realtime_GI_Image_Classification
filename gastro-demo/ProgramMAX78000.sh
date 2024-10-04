#!/bin/sh
openocd -s "${TOOLCHAIN_PATH}/OpenOCD/scripts" -f interface/cmsis-dap.cfg -f target/max78000_nsrst.cfg -c "program build/max78000.elf reset exit "
