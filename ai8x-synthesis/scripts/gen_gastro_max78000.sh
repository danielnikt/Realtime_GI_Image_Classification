#!/bin/sh
DEVICE="MAX78000"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir $TARGET --prefix GASTRO_2024.09.30-202109_gastro6_64_noaug --checkpoint-file ../ai8x-training/logs/2024.09.30-202109_gastro6_64_noaug/qat_best_q8.pth.tar --config-file networks/cats-dogs-hwc.yaml --fifo --softmax $COMMON_ARGS "$@"
