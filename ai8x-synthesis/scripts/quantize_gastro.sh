#!/bin/sh
python quantize.py ../ai8x-training/logs/2024.09.30-202109_gastro6_64_noaug/qat_best.pth.tar ../ai8x-training/logs/2024.09.30-202109_gastro6_64_noaug/qat_best_q8.pth.tar --device MAX78000 -v "$@"
