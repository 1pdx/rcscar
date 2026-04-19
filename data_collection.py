# -*- coding: utf-8 -*-
"""
数据采集统一入口：ARS40X Cluster_1_General (0x701) CAN + 可选惯导填充。
转发所有命令行参数至 ars40x_cluster_logger.main。

示例:
  python data_collection.py --interface socketcan --channel can0 --ins
  # 可选标定元数据: 先 set/export CLUSTER_CSV_CALIBRATION=-6.710 或加 --calibration -6.71
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent / "ars40x_cluster_logger.py"
    cmd = [sys.executable, str(script), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
