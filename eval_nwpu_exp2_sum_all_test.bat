@echo off
setlocal

cd /d C:\Users\14288\OneDrive\Desktop\yolov12\yolov12-main

C:\Users\14288\anaconda3\envs\torch311\python.exe evaluate_nwpu_checkpoints.py ^
  --run-dir C:\Users\14288\OneDrive\Desktop\yolov12\runs\detect\exp2_sum_all ^
  --split test ^
  --batch 16 ^
  --no-plots ^
  --exist-ok ^
  --include-best-last

pause
