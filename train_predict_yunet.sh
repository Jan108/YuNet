#!/usr/bin/env bash
set -eo pipefail

# Yunet-n
for cls in "all" "bird" ; do
  PYTHONPATH='/mnt/data/afarec/code/face_detection/YuNet/':$PYTHONPATH \
  python /mnt/data/afarec/code/face_detection/YuNet/tools/train.py \
  "$(dirname "$0")/configs/yunet_${cls}_config.py" --seed 0 \
  --work-dir "./work_dir/yunet_${cls}/"

  PYTHONPATH='/mnt/data/afarec/code/face_detection/YuNet/':$PYTHONPATH \
  python /mnt/data/afarec/code/face_detection/YuNet/tools/test_widerface.py \
  "$(dirname "$0")/configs/yunet_${cls}_config.py" \
  "./work_dir/yunet_${cls}/latest.pth" \
  --mode 2 \
  --save-preds \
  --out "./work_dir/yunet_${cls}/"

  PYTHONPATH='/mnt/data/afarec/code/face_detection/YuNet/':$PYTHONPATH \
  python /mnt/data/afarec/code/face_detection/YuNet/tools/test_widerface.py \
  "$(dirname "$0")/configs/yunet_${cls}_config.py" \
  "./work_dir/yunet_${cls}/latest.pth" \
  --mode 2 \
  --out "./work_dir/yunet_${cls}/" \
  --latency_test 1000
done

PYTHONPATH='/mnt/data/afarec/code/face_detection/YuNet/':$PYTHONPATH \
python /mnt/data/afarec/code/face_detection/YuNet/tools/test_widerface.py \
  "$(dirname "$0")/configs/yunet_all_config.py" \
  "$(dirname "$0")/weights/yunet_n.pth" \
  --mode 2 \
  --save-preds \
  --out "./work_dir/yunet_pretrained/"

PYTHONPATH='/mnt/data/afarec/code/face_detection/YuNet/':$PYTHONPATH \
python /mnt/data/afarec/code/face_detection/YuNet/tools/test_widerface.py \
  "$(dirname "$0")/configs/yunet_all_config.py" \
  "$(dirname "$0")/weights/yunet_n.pth" \
  --mode 2 \
  --out "./work_dir/yunet_pretrained/" \
  --latency_test 1000


# Yunet-s
for cls in "all" "bird" ; do
  PYTHONPATH='/mnt/data/afarec/code/face_detection/YuNet/':$PYTHONPATH \
  python /mnt/data/afarec/code/face_detection/YuNet/tools/train.py \
  "$(dirname "$0")/configs/yunet_s_${cls}_config.py" --seed 0 \
  --work-dir "./work_dir/yunet_s_${cls}/"

  PYTHONPATH='/mnt/data/afarec/code/face_detection/YuNet/':$PYTHONPATH \
  python /mnt/data/afarec/code/face_detection/YuNet/tools/test_widerface.py \
  "$(dirname "$0")/configs/yunet_s_${cls}_config.py" \
  "./work_dir/yunet_s_${cls}/latest.pth" \
  --mode 2 \
  --save-preds \
  --out "./work_dir/yunet_s_${cls}/"

  PYTHONPATH='/mnt/data/afarec/code/face_detection/YuNet/':$PYTHONPATH \
  python /mnt/data/afarec/code/face_detection/YuNet/tools/test_widerface.py \
  "$(dirname "$0")/configs/yunet_s_${cls}_config.py" \
  "./work_dir/yunet_s_${cls}/latest.pth" \
  --mode 2 \
  --out "./work_dir/yunet_s_${cls}/" \
  --latency_test 1000
done

PYTHONPATH='/mnt/data/afarec/code/face_detection/YuNet/':$PYTHONPATH \
python /mnt/data/afarec/code/face_detection/YuNet/tools/test_widerface.py \
  "$(dirname "$0")/configs/yunet_s_all_config.py" \
  "$(dirname "$0")/weights/yunet_s.pth" \
  --mode 2 \
  --save-preds \
  --out "./work_dir/yunet_s_pretrained/"

PYTHONPATH='/mnt/data/afarec/code/face_detection/YuNet/':$PYTHONPATH \
python /mnt/data/afarec/code/face_detection/YuNet/tools/test_widerface.py \
  "$(dirname "$0")/configs/yunet_s_all_config.py" \
  "$(dirname "$0")/weights/yunet_s.pth" \
  --mode 2 \
  --out "./work_dir/yunet_s_pretrained/" \
  --latency_test 1000