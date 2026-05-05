#!/bin/bash
# Spot VM bootstrap — runs calibration of all 17 engines on an A100 80GB,
# uploads engine_calibration.json to GCS, then shuts the VM down.
#
# Used as --metadata-from-file=startup-script= on a Deep Learning VM Image
# with PyTorch + CUDA 12.x preinstalled.

set -euo pipefail

LOG=/var/log/calibrate.log
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date) — calibrate startup ==="

cd /home
sudo apt-get -qq update
sudo apt-get -qq install -y git tesseract-ocr tesseract-ocr-nor poppler-utils \
                            ghostscript default-jre rclone

# Clone repo
sudo rm -rf norwegian-ocr-benchmark
git clone https://github.com/sondreskarsten/norwegian-ocr-benchmark.git
cd norwegian-ocr-benchmark

# Use system python (Deep Learning VM has CUDA torch preinstalled)
pip install -q -r requirements.txt
pip install -q google-cloud-storage PyMuPDF

# Calibration uses ADC from the attached service account; no key needed.
echo "=== $(date) — running calibration ==="
python -m scripts.calibrate --n 10 --out /home/engine_calibration.json || true

echo "=== $(date) — calibration finished ==="
ls -la /home/engine_calibration.json || true

# Auto-shutdown to stop billing the spot VM
echo "=== $(date) — shutting down ==="
sudo shutdown -h +1
