"""
ARS40X Radar - Cluster_1_General (0x701) CAN采集与解析脚本
=============================================================
支持传感器: Continental ARS 404-21 / ARS 408-21
CAN消息:    0x600 Cluster_0_Status  (帧头，含簇数量)
            0x701 Cluster_1_General (位置/速度/RCS)
文档参考:   Standardized ARS Interface V1.8 (2017-10-18)

输出CSV格式（与 DRI Raw 一致，CAN 来源为 Cluster 接口 0x600/0x701）:
  元数据头（Data Type / Data File / Run Number / Calibration + 空行）:
    Data Type,Raw
    Data File,<路径>
    Run Number,<编号>
    Calibration,<可选；默认 0.000。与 DRI 对齐时可在运行前设置环境变量，例如 CLUSTER_CSV_CALIBRATION=-6.710；
               亦可使用命令行 --calibration；显式参数优先于环境变量>
    (空行)

  Calibration 写入规则（均为可选，不设则 CSV 中为 0.000）:
    - 优先级: 代码/命令行传入的 calibration > 环境变量 CLUSTER_CSV_CALIBRATION > 默认 0.000
    - Linux/macOS:  export CLUSTER_CSV_CALIBRATION=-6.710
    - Windows CMD:  set CLUSTER_CSV_CALIBRATION=-6.710
    - Windows PowerShell:  $env:CLUSTER_CSV_CALIBRATION="-6.710"
  表头（紧跟空行后第 1 行）:
    Time, R, ViewAng, Heading, Speed, MeasType, TargetHeading,
    MeasRadOrAng, Aim, DX00, DY00, RCS00, ..., DX19, DY19, RCS19
  数据行:
    每帧一行，最多记录20个簇，不足时填NaN
  惯导补充（--ins）:
    Heading/Speed 等与雷达「强制逐行对齐」：仅在每个雷达帧（0x600 触发）写一行时采样一次惯导，
    行频=雷达帧频；INS 短时缺失可在缓存窗口内沿用上一帧（见 imu_gnss_pose.sample_ins_for_radar_csv_row）。
    R、ViewAng 由本帧首簇（ROI 内）DX/DY 推导。

  空间门（Cluster 写入 CSV 前筛选）：
    仅保留纵向 DX∈[4, 50] m（DistLong）、横向 |DY|≤2 m（DistLat）；超出门限的簇不写入。

  丢帧与 NaN：
    无 0x600 帧头时收到的 0x701 一律丢弃，不生成行。
    若 MeasCounter 相对上一帧不连续，在写入新帧之前插入若干全 NaN 占位行（缺口宽度受 CLUSTER_CSV_MAX_MC_GAP 限制）；
    可设环境变量 CLUSTER_CSV_MEAS_COUNTER_GAP_FILL=0 关闭该填充。

信号解析定义 (Table 33):
  Cluster_ID       Start=0,  Len=8,  Offset=0,      Res=1,    Unit=-
  Cluster_DistLong Start=19, Len=13, Offset=-500,   Res=0.2,  Unit=m   → DX
  Cluster_DistLat  Start=24, Len=10, Offset=-102.3, Res=0.2,  Unit=m   → DY
  Cluster_VrelLong Start=46, Len=10, Offset=-128.0, Res=0.25, Unit=m/s
  Cluster_DynProp  Start=48, Len=3,  Offset=0,      Res=1,    Unit=-
  Cluster_VrelLat  Start=53, Len=9,  Offset=-64.0,  Res=0.25, Unit=m/s
  Cluster_RCS      Start=56, Len=8,  Offset=-64.0,  Res=0.5,  Unit=dBm² → RCS

依赖:
  pip install python-can

数据采集入口（推荐在本目录执行）:
  python ars40x_cluster_logger.py --interface socketcan --channel can0 --ins
  python data_collection.py           # 等价调用上方脚本（可加参数；含 --calibration）

  # 与 DRI 同一标定写入元数据（可选；二选一即可）:
  #   export CLUSTER_CSV_CALIBRATION=-6.710   # 再运行脚本
  #   或  python ... --calibration -6.71

使用方法:
  python ars40x_cluster_logger.py --demo                          # 离线演示
  python ars40x_cluster_logger.py --interface socketcan --channel can0 --ins
  python ars40x_cluster_logger.py --interface pcan --channel PCAN_USBBUS1
  python ars40x_cluster_logger.py --interface kvaser --channel 0
"""

import can
import csv
import time
import argparse
import os
import random
import math
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union


# ─────────────────────────────────────────────
#  CAN ID 定义
# ─────────────────────────────────────────────
CAN_ID_CLUSTER_STATUS  = 0x600   # Cluster_0_Status  (列表头)
CAN_ID_CLUSTER_GENERAL = 0x701   # Cluster_1_General (簇数据)

MAX_CLUSTERS = 20   # 每帧最多记录簇数，与原始CSV保持一致（DX00~DX19）

# MeasCounter 不连续时插入占位的全 NaN 行（视为丢帧）；过大间隔视为计数翻转/异常，不插入以免刷屏。
_MC_GAP_ENV = os.getenv("CLUSTER_CSV_MAX_MC_GAP", "").strip()
try:
    CLUSTER_CSV_MAX_MC_GAP: int = int(_MC_GAP_ENV) if _MC_GAP_ENV else 256
except ValueError:
    CLUSTER_CSV_MAX_MC_GAP = 256

_MFILL_ENV = os.getenv("CLUSTER_CSV_MEAS_COUNTER_GAP_FILL", "").strip().lower()
CLUSTER_CSV_MEAS_COUNTER_GAP_FILL: bool = _MFILL_ENV not in ("0", "false", "no", "off")

# 落盘空间门：DistLong→DX（纵向，前方为正）、DistLat→DY（横向）；仅采集该扇区内的簇。
CLUSTER_ROI_DX_MIN_M = 4.0
CLUSTER_ROI_DX_MAX_M = 50.0
CLUSTER_ROI_ABS_DY_MAX_M = 2.0


def cluster_in_roi(dx: float, dy: float) -> bool:
    """前方纵向 [CLUSTER_ROI_DX_MIN_M, CLUSTER_ROI_DX_MAX_M]，左右 |DY|≤CLUSTER_ROI_ABS_DY_MAX_M。"""
    if not math.isfinite(dx) or not math.isfinite(dy):
        return False
    if dx < CLUSTER_ROI_DX_MIN_M or dx > CLUSTER_ROI_DX_MAX_M:
        return False
    if abs(dy) > CLUSTER_ROI_ABS_DY_MAX_M:
        return False
    return True


# ─────────────────────────────────────────────
#  位域提取（Intel字节序，小端位编址）
# ─────────────────────────────────────────────
def extract_bits(data: bytes, bit_start: int, bit_len: int) -> int:
    value = 0
    for i in range(bit_len):
        byte_idx = (bit_start + i) // 8
        bit_idx  = (bit_start + i) % 8
        if byte_idx < len(data):
            value |= (((data[byte_idx] >> bit_idx) & 1) << i)
    return value


# ─────────────────────────────────────────────
#  0x600 Cluster_0_Status 解析
# ─────────────────────────────────────────────
def parse_cluster_status(data: bytes) -> dict:
    """
    Table 30/31:
      NofClustersNear [0:8]
      NofClustersFar  [8:8]
      MeasCounter     [24:16]
      InterfaceVersion[36:4]
    """
    return {
        "NofClustersNear":  extract_bits(data,  0,  8),
        "NofClustersFar":   extract_bits(data,  8,  8),
        "MeasCounter":      extract_bits(data, 24, 16),
        "InterfaceVersion": extract_bits(data, 36,  4),
    }


# ─────────────────────────────────────────────
#  0x701 Cluster_1_General 解析
# ─────────────────────────────────────────────
def parse_cluster_general(data: bytes) -> dict:
    """
    Continental Standardized ARS Interface **Table 33 Cluster_1_General**（位序 Intel / 小端位编址）:
      Cluster_ID       bit 0..7（8bit）   → 簇标识（界面散点按目标合并时使用）
      Cluster_DistLong bit 19..31（13bit）→ 物理量 = raw×0.2 + (-500)   → DX (m)
      Cluster_DistLat  bit 24..33（10bit）→ 物理量 = raw×0.2 + (-102.3) → DY (m)
      Cluster_DynProp  bit 48..50（3bit） → 整型 0..7（写入 CSV MeasType）
      Cluster_RCS      bit 56..63（8bit） → 物理量 = raw×0.5 + (-64)   → RCS (dBm²)

    与文档 Resolution / Offset 一致；未使用虚拟填充值。
    """
    cluster_id = extract_bits(data, 0, 8)
    dist_long = extract_bits(data, 19, 13) * 0.2 - 500.0
    dist_lat = extract_bits(data, 24, 10) * 0.2 - 102.3
    dyn_prop = extract_bits(data, 48, 3)
    rcs = extract_bits(data, 56, 8) * 0.5 - 64.0
    return {
        "ClusterID": int(cluster_id),
        "DX": round(dist_long, 2),
        "DY": round(dist_lat, 2),
        "DynProp": int(dyn_prop),
        "RCS": round(rcs, 2),
    }


# ─────────────────────────────────────────────
#  CSV 文件初始化
# ─────────────────────────────────────────────
def format_cluster_csv_calibration(
    calibration: Optional[Union[str, float]] = None,
) -> str:
    """
    与 DRI Raw 中 Calibration 列显示形式一致（如 -6.710）。
    本字段完全可选：未指定且未设置环境变量时写入 0.000。
    优先级: 显式 calibration → 环境变量 CLUSTER_CSV_CALIBRATION → 默认字符串 0.000。
    """
    if calibration is not None:
        if isinstance(calibration, str):
            s = calibration.strip()
            if s:
                return s
        else:
            try:
                return f"{float(calibration):.3f}"
            except (TypeError, ValueError):
                pass
    env = os.getenv("CLUSTER_CSV_CALIBRATION", "").strip()
    if env:
        return env
    return "0.000"


def build_columns() -> list:
    """构建与原始CSV完全一致的列名"""
    cols = ["Time", "R", "ViewAng", "Heading", "Speed",
            "MeasType", "TargetHeading", "MeasRadOrAng", "Aim"]
    for i in range(MAX_CLUSTERS):
        cols += [f"DX{i:02d}", f"DY{i:02d}", f"RCS{i:02d}"]
    return cols


def init_csv(
    filepath: str,
    run_number: int = 1,
    calibration: Optional[Union[str, float]] = None,
) -> tuple:
    """
    创建 CSV，写入 DRI 风格元数据（4 行）+ 空行 + 表头。无「CAN Format」行。
    返回 (file_handle, csv_writer)
    """
    f = open(filepath, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    cal_s = format_cluster_csv_calibration(calibration)

    writer.writerow(["Data Type", "Raw"])
    writer.writerow(["Data File", filepath])
    writer.writerow(["Run Number", run_number])
    writer.writerow(["Calibration", cal_s])
    writer.writerow([])
    writer.writerow(build_columns())
    f.flush()
    return f, writer


def build_nan_cluster_csv_row() -> List[str]:
    """CAN 丢帧占位：与 build_columns() 列数一致，全部为字符串 NaN（无惯导、无雷达几何填充）。"""
    return ["NaN"] * len(build_columns())


def build_data_row(
    timestamp: float,
    clusters: List[Dict[str, Any]],
    ins_extras: Optional[Dict[str, float]] = None,
) -> List[Any]:
    """
    [Time, R, ViewAng, Heading, Speed, MeasType, TargetHeading,
     MeasRadOrAng, Aim, DX00..RCS19]

    ins_extras: sample_ins_for_radar_csv_row()（每雷达帧仅调用一次；行频=雷达帧频）。
    R / ViewAng：若本帧有簇，则由首簇 DX/DY 推导平面距离与 atan2(DY,DX)(°)；否则 NaN。
    MeasType：来自 Table 33 首簇 DynProp（CAN）；无簇时为 NaN（不再写死 0）。
    """
    if clusters:
        c0 = clusters[0]
        dx = float(c0["DX"])
        dy = float(c0["DY"])
        r_geom = round(math.hypot(dx, dy), 4)
        view_deg = round(math.degrees(math.atan2(dy, dx)), 4)
        dp = c0.get("DynProp")
        meas_type = int(dp) if dp is not None else "NaN"
    else:
        r_geom = "NaN"
        view_deg = "NaN"
        meas_type = "NaN"

    if ins_extras:
        heading = round(float(ins_extras["heading_deg"]), 4)
        speed = round(float(ins_extras["speed_mps"]), 4)
        tgt_head = heading
    else:
        heading = "NaN"
        speed = "NaN"
        tgt_head = "NaN"

    row = [
        round(timestamp, 4),
        r_geom,
        view_deg,
        heading,
        speed,
        meas_type,
        tgt_head,
        "NaN",
        "NaN",
    ]
    for i in range(MAX_CLUSTERS):
        if i < len(clusters):
            c = clusters[i]
            row += [c["DX"], c["DY"], c["RCS"]]
        else:
            row += ["NaN", "NaN", "NaN"]
    return row


# ─────────────────────────────────────────────
#  主采集器
# ─────────────────────────────────────────────
class ClusterLogger:
    def __init__(
        self,
        bus,
        output_dir=".",
        sensor_id=0,
        run_number=1,
        enable_ins: bool = False,
        calibration: Optional[Union[str, float]] = None,
    ):
        self.bus        = bus
        self.sensor_id  = sensor_id
        self.enable_ins = bool(enable_ins)

        # 根据传感器ID计算实际CAN ID（多传感器时偏移0x10）
        offset = sensor_id * 0x10
        self.id_status  = CAN_ID_CLUSTER_STATUS  + offset
        self.id_general = CAN_ID_CLUSTER_GENERAL + offset

        # 帧缓存
        self.current_status  = {}
        self.frame_clusters  = []    # 当前帧收集的簇列表
        self.frame_start_ts  = None  # 当前帧时间戳（取0x600到达时刻）

        # 统计
        self.total_frames   = 0
        self.total_clusters = 0

        self._last_meas_counter_for_gap: Optional[int] = None

        # 初始化CSV
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"cluster_log_{ts}.csv")
        self.csv_file, self.csv_writer = init_csv(
            self.csv_path, run_number, calibration=calibration
        )
        print(f"[INFO] 输出文件: {self.csv_path}")
        print(
            f"[INFO] Cluster ROI: DX∈[{CLUSTER_ROI_DX_MIN_M}, {CLUSTER_ROI_DX_MAX_M}] m, "
            f"|DY|≤{CLUSTER_ROI_ABS_DY_MAX_M} m（仅写入门内簇）"
        )
        if self.enable_ins:
            try:
                from imu_gnss_pose import reset_ins_cluster_csv_row_hold

                reset_ins_cluster_csv_row_hold()
            except Exception:
                pass

    def _flush_frame(self):
        """将当前帧所有簇写成一行落盘"""
        if self.frame_start_ts is None:
            return
        ins_extras = None
        if self.enable_ins:
            try:
                from imu_gnss_pose import sample_ins_for_radar_csv_row

                ins_extras = sample_ins_for_radar_csv_row()
            except Exception:
                ins_extras = None
        row = build_data_row(self.frame_start_ts, self.frame_clusters, ins_extras)
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        self.total_frames   += 1
        self.total_clusters += len(self.frame_clusters)
        self.frame_clusters  = []
        self.frame_start_ts  = None

    def _write_nan_csv_row(self) -> None:
        """MeasCounter 跳变时表示中间雷达帧缺失，占位一行（无任何 CAN 推导字段）。"""
        if self.csv_writer is None:
            return
        self.csv_writer.writerow(build_nan_cluster_csv_row())
        self.csv_file.flush()
        self.total_frames += 1

    def process_message(self, msg):
        if msg.arbitration_id == self.id_status:
            # 收到新帧头 → 落盘上一帧，开始新帧
            self._flush_frame()
            new_status = parse_cluster_status(msg.data)
            mc = int(new_status["MeasCounter"])
            if CLUSTER_CSV_MEAS_COUNTER_GAP_FILL and self._last_meas_counter_for_gap is not None:
                prev = self._last_meas_counter_for_gap
                if mc != prev:
                    expected_next = (prev + 1) % 65536
                    gap = (mc - expected_next + 65536) % 65536
                    if 0 < gap <= CLUSTER_CSV_MAX_MC_GAP:
                        for _ in range(gap):
                            self._write_nan_csv_row()
            self._last_meas_counter_for_gap = mc
            self.current_status = new_status
            self.frame_start_ts = _message_timestamp(msg)

        elif msg.arbitration_id == self.id_general:
            # 无 0x600 帧头时不解析簇，避免与虚拟时间轴合并
            if self.frame_start_ts is None:
                return
            # 收到簇数据 → 追加到当前帧（最多MAX_CLUSTERS个）；仅 ROI 内落盘
            if len(self.frame_clusters) < MAX_CLUSTERS:
                c = parse_cluster_general(msg.data)
                if cluster_in_roi(float(c["DX"]), float(c["DY"])):
                    self.frame_clusters.append(c)

    def run(self, duration_s=None):
        start_time = time.time()
        print(f"[INFO] 开始采集... (Ctrl+C 停止)")
        print(f"[INFO] 监听 0x{self.id_status:03X}(状态帧) / 0x{self.id_general:03X}(簇数据)")
        print("-" * 60)
        try:
            while True:
                if duration_s and (time.time() - start_time) >= duration_s:
                    break
                msg = self.bus.recv(timeout=1.0)
                if msg is None:
                    continue
                self.process_message(msg)
                if msg.arbitration_id == self.id_status:
                    mc  = self.current_status.get("MeasCounter", "?")
                    nof = (self.current_status.get("NofClustersNear", 0) +
                           self.current_status.get("NofClustersFar",  0))
                    print(f"\r[帧 {mc:5}] 本帧簇数={nof}  "
                          f"已记录帧={self.total_frames}  "
                          f"已记录簇={self.total_clusters}  ",
                          end="", flush=True)
        except KeyboardInterrupt:
            print("\n[INFO] 用户中断")
        finally:
            self._flush_frame()
            self.csv_file.close()
            self._print_summary()

    def _print_summary(self):
        print("\n" + "=" * 60)
        print(f"  采集完成")
        print(f"  总帧数:    {self.total_frames}")
        print(f"  总簇数:    {self.total_clusters}")
        if self.total_frames > 0:
            print(f"  平均簇/帧: {self.total_clusters / self.total_frames:.2f}")
        print(f"  输出文件:  {self.csv_path}")
        print("=" * 60)


# ─────────────────────────────────────────────
#  离线演示模式
# ─────────────────────────────────────────────
class DemoLogger(ClusterLogger):
    """生成虚拟CAN数据，验证脚本逻辑与CSV格式，无需任何硬件"""

    def __init__(
        self,
        output_dir=".",
        run_number=1,
        n_frames=100,
        calibration: Optional[Union[str, float]] = None,
    ):
        self.sensor_id      = 0
        self.enable_ins     = False
        self.id_status      = CAN_ID_CLUSTER_STATUS
        self.id_general     = CAN_ID_CLUSTER_GENERAL
        self.current_status = {}
        self.frame_clusters = []
        self.frame_start_ts = None
        self.total_frames   = 0
        self.total_clusters = 0
        self.n_frames       = n_frames
        self._last_meas_counter_for_gap = None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"cluster_log_demo_{ts}.csv")
        self.csv_file, self.csv_writer = init_csv(
            self.csv_path, run_number, calibration=calibration
        )
        print(f"[DEMO] 输出文件: {self.csv_path}")

    def _make_status_msg(self, meas_cnt: int, n_near: int, t: float):
        data = bytearray(8)
        for i in range(8):
            data[i // 8] |= (((n_near >> i) & 1) << (i % 8))
        for i in range(16):
            byte_idx = (24 + i) // 8
            bit_idx  = (24 + i) % 8
            data[byte_idx] |= (((meas_cnt >> i) & 1) << bit_idx)
        return type("Msg", (), {
            "arbitration_id": CAN_ID_CLUSTER_STATUS,
            "data": bytes(data),
            "timestamp": t,
        })()

    def _make_cluster_msg(
        self,
        dist_long: float,
        dist_lat: float,
        rcs: float,
        t: float,
        cluster_id: int = 0,
    ):
        data = bytearray(8)

        def wb(val_int, bit_start, bit_len):
            val_int = max(0, min(int(val_int), (1 << bit_len) - 1))
            for i in range(bit_len):
                byte_idx = (bit_start + i) // 8
                bit_idx  = (bit_start + i) % 8
                if byte_idx < len(data):
                    data[byte_idx] |= (((val_int >> i) & 1) << bit_idx)

        wb(int(cluster_id) & 0xFF, 0, 8)
        wb((dist_long + 500.0) / 0.2, 19, 13)
        wb((dist_lat  + 102.3) / 0.2, 24, 10)
        wb((rcs       + 64.0)  / 0.5, 56,  8)
        return type("Msg", (), {
            "arbitration_id": CAN_ID_CLUSTER_GENERAL,
            "data": bytes(data),
            "timestamp": t,
        })()

    def run(self, duration_s=None):
        print(f"[DEMO] 生成 {self.n_frames} 帧虚拟数据...")
        print("-" * 60)
        base_time = time.time()

        for frame_idx in range(self.n_frames):
            t = base_time + frame_idx * 0.071   # 约14Hz
            n_clusters = random.choices([1, 2, 3], weights=[20, 70, 10])[0]

            self.process_message(
                self._make_status_msg(frame_idx % 65536, n_clusters, t)
            )
            for c in range(n_clusters):
                self.process_message(self._make_cluster_msg(
                    dist_long=random.uniform(5.0, 100.0),
                    dist_lat =random.uniform(-5.0, 5.0),
                    rcs      =random.uniform(-15.0, 20.0),
                    t        =t + 0.001 * (c + 1),
                    cluster_id=c,
                ))
            print(f"\r[DEMO] 帧 {frame_idx + 1:4d}/{self.n_frames}  "
                  f"簇数={n_clusters}", end="", flush=True)

        self._flush_frame()
        self.csv_file.close()
        self._print_summary()


def _message_timestamp(msg: Any) -> float:
    ts = getattr(msg, "timestamp", None)
    return float(ts) if ts is not None else time.time()


# ─────────────────────────────────────────────
#  UI / MainController：后台常驻接收 + 按需写 CSV
# ─────────────────────────────────────────────
class ClusterCsvRuntime:
    """
    与 Scout 共用 can0：单独线程 recv；始终解析 0x600/0x701 供界面显示。
    CSV 仅在 begin_recording～end_recording 之间写入；格式与 ClusterLogger / init_csv 一致。
    """

    __slots__ = (
        "iface",
        "bitrate",
        "sensor_id",
        "enable_ins",
        "id_status",
        "id_general",
        "_lock",
        "_stop",
        "bus",
        "_th",
        "_writing",
        "csv_file",
        "csv_writer",
        "csv_path",
        "current_status",
        "frame_clusters",
        "frame_start_ts",
        "total_frames",
        "total_clusters",
        "run_number",
        "_last_clusters",
        "_last_clusters_ts",
        "_last_meas_counter_for_gap",
    )

    def __init__(
        self,
        iface: str,
        bitrate: int,
        sensor_id: int = 0,
        enable_ins: bool = True,
    ) -> None:
        self.iface = str(iface)
        self.bitrate = int(bitrate)
        self.sensor_id = int(sensor_id)
        self.enable_ins = bool(enable_ins)
        off = int(sensor_id) * 0x10
        self.id_status = CAN_ID_CLUSTER_STATUS + off
        self.id_general = CAN_ID_CLUSTER_GENERAL + off
        self._lock = threading.Lock()
        self._stop = False
        self.bus = None  # type: Optional[can.BusABC]
        self._th: Optional[threading.Thread] = None
        self._writing = False
        self.csv_file = None
        self.csv_writer = None
        self.csv_path: Optional[str] = None
        self.current_status: Dict[str, Any] = {}
        self.frame_clusters: List[Dict[str, Any]] = []
        self.frame_start_ts: Optional[float] = None
        self.total_frames = 0
        self.total_clusters = 0
        self.run_number = 1
        self._last_clusters: List[Dict[str, Any]] = []
        self._last_clusters_ts: Optional[float] = None
        self._last_meas_counter_for_gap: Optional[int] = None

    def start(self) -> None:
        if self._th is not None:
            return
        self.bus = can.interface.Bus(
            channel=self.iface, interface="socketcan", bitrate=self.bitrate
        )
        self._th = threading.Thread(target=self._recv_loop, daemon=True)
        self._th.start()

    def _recv_loop(self) -> None:
        assert self.bus is not None
        while not self._stop:
            try:
                msg = self.bus.recv(timeout=0.4)
            except Exception:
                continue
            if msg is None:
                continue
            if getattr(msg, "is_extended_id", False):
                continue
            self._handle_message(msg)

    def _handle_message(self, msg: Any) -> None:
        """始终解析 Cluster 帧以更新界面快照；仅在 _writing 时写入 CSV。"""
        with self._lock:
            aid = int(msg.arbitration_id)
            if aid == self.id_status:
                self._flush_frame_unsafe()
                new_status = parse_cluster_status(bytes(msg.data))
                mc = int(new_status["MeasCounter"])
                if CLUSTER_CSV_MEAS_COUNTER_GAP_FILL and self._last_meas_counter_for_gap is not None:
                    prev = self._last_meas_counter_for_gap
                    if mc != prev:
                        expected_next = (prev + 1) % 65536
                        gap = (mc - expected_next + 65536) % 65536
                        if 0 < gap <= CLUSTER_CSV_MAX_MC_GAP:
                            for _ in range(gap):
                                self._write_nan_csv_row_unsafe()
                self._last_meas_counter_for_gap = mc
                self.current_status = new_status
                self.frame_start_ts = _message_timestamp(msg)
            elif aid == self.id_general:
                if self.frame_start_ts is None:
                    return
                if len(self.frame_clusters) < MAX_CLUSTERS:
                    c = parse_cluster_general(bytes(msg.data))
                    if cluster_in_roi(float(c["DX"]), float(c["DY"])):
                        self.frame_clusters.append(c)

    def _write_nan_csv_row_unsafe(self) -> None:
        if not self._writing or self.csv_writer is None or self.csv_file is None:
            return
        self.csv_writer.writerow(build_nan_cluster_csv_row())
        self.csv_file.flush()
        self.total_frames += 1

    def _flush_frame_unsafe(self) -> None:
        if self.frame_start_ts is None:
            return
        ts = float(self.frame_start_ts)
        self._last_clusters_ts = ts
        self._last_clusters = [{k: c[k] for k in c} for c in self.frame_clusters]

        if self._writing and self.csv_writer is not None and self.csv_file is not None:
            ins_extras = None
            if self.enable_ins:
                try:
                    from imu_gnss_pose import sample_ins_for_radar_csv_row

                    ins_extras = sample_ins_for_radar_csv_row()
                except Exception:
                    ins_extras = None
            row = build_data_row(ts, self.frame_clusters, ins_extras)
            self.csv_writer.writerow(row)
            self.csv_file.flush()
            self.total_frames += 1
            self.total_clusters += len(self.frame_clusters)

        self.frame_clusters = []
        self.frame_start_ts = None

    def get_cluster_display_snapshot(self) -> Tuple[float, List[Dict[str, Any]]]:
        """
        UI 雷达图：优先返回当前未封装的帧内簇（延迟更低），否则返回上一完整帧。
        DX/DY/RCS 与 parse_cluster_general 一致。
        """
        with self._lock:
            if self.frame_clusters and self.frame_start_ts is not None:
                tlf = float(self.frame_start_ts)
                return tlf, [{k: c[k] for k in c} for c in self.frame_clusters]
            ts = 0.0 if self._last_clusters_ts is None else float(self._last_clusters_ts)
            return ts, [{k: c[k] for k in c} for c in self._last_clusters]

    def begin_recording(
        self,
        output_dir: str,
        run_number: int = 1,
        stem: str = "cluster_rcs",
        calibration: Optional[Union[str, float]] = None,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = (stem or "cluster_rcs").replace("/", "_").replace("\\", "_")[:120]
        path = os.path.join(output_dir, f"{safe}_{ts}.csv")
        with self._lock:
            if self._writing:
                self._close_file_unsafe()
            self._flush_soft_reset_unsafe()
            self.csv_path = path
            self.csv_file, self.csv_writer = init_csv(
                path, int(run_number), calibration=calibration
            )
            self.run_number = int(run_number)
            self.total_frames = 0
            self.total_clusters = 0
            self._last_meas_counter_for_gap = None
            self._writing = True
            if self.enable_ins:
                try:
                    from imu_gnss_pose import reset_ins_cluster_csv_row_hold

                    reset_ins_cluster_csv_row_hold()
                except Exception:
                    pass

    def _flush_soft_reset_unsafe(self) -> None:
        self.frame_clusters = []
        self.frame_start_ts = None

    def _close_file_unsafe(self) -> None:
        self._flush_frame_unsafe()
        self._flush_soft_reset_unsafe()
        if self.csv_file is not None:
            try:
                self.csv_file.close()
            except Exception:
                pass
        self.csv_file = None
        self.csv_writer = None
        self.csv_path = None
        self._writing = False

    def end_recording(self) -> Optional[str]:
        with self._lock:
            if not self._writing:
                return None
            self._flush_frame_unsafe()
            self._flush_soft_reset_unsafe()
            out = self.csv_path
            if self.csv_file is not None:
                try:
                    self.csv_file.close()
                except Exception:
                    pass
            self.csv_file = None
            self.csv_writer = None
            self.csv_path = None
            self._writing = False
            return out

    def is_recording(self) -> bool:
        with self._lock:
            return self._writing

    def snapshot_stats(self) -> Tuple[int, int]:
        with self._lock:
            return int(self.total_frames), int(self.total_clusters)

    def stop(self) -> None:
        self._stop = True
        try:
            if self.bus is not None:
                self.bus.shutdown()
        except Exception:
            pass
        self.bus = None


# ─────────────────────────────────────────────
#  命令行入口
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="ARS40X Cluster_1_General (0x701) CAN采集脚本"
    )
    parser.add_argument("--interface", "-i", default="socketcan",
        choices=["socketcan", "pcan", "kvaser", "vector", "ixxat", "usb2can", "slcan"],
        help="CAN接口类型 (默认: socketcan)")
    parser.add_argument("--channel", "-c", default="can0",
        help="CAN通道 (默认: can0，PCAN示例: PCAN_USBBUS1)")
    parser.add_argument("--bitrate", "-b", type=int, default=500000,
        help="CAN波特率 bps (默认: 500000)")
    parser.add_argument("--sensor-id", "-s", type=int, default=0,
        choices=range(8), help="传感器ID 0~7 (默认: 0)")
    parser.add_argument("--run-number", "-r", type=int, default=1,
        help="Run Number 写入CSV元数据 (默认: 1)")
    parser.add_argument(
        "--calibration",
        type=float,
        default=None,
        metavar="VAL",
        help="可选。CSV 元数据 Calibration（三位小数，如与 DRI 对齐用 -6.71）。"
        "不设时读环境变量 CLUSTER_CSV_CALIBRATION（如运行前 export CLUSTER_CSV_CALIBRATION=-6.710）；"
        "均未设则为 0.000。本参数优先于环境变量。",
    )
    parser.add_argument("--duration", "-d", type=float, default=None,
        help="采集时长(秒)，不填则持续到Ctrl+C")
    parser.add_argument("--output-dir", "-o", default=".",
        help="CSV输出目录 (默认: 当前目录)")
    parser.add_argument("--demo", action="store_true",
        help="离线演示模式，无需CAN硬件")
    parser.add_argument("--demo-frames", type=int, default=200,
        help="演示模式生成帧数 (默认: 200)")
    parser.add_argument(
        "--ins",
        action="store_true",
        help="融合北云惯导 INSPVAXA（imu_gnss_pose）：填充 Heading/Speed/TargetHeading；"
        "R/ViewAng 仍可由首簇几何推导",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  ARS40X Cluster_1_General (0x701) 采集脚本")
    print("  Continental ARS 404-21 / ARS 408-21")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.demo:
        print("[DEMO] 离线演示模式（无需CAN硬件）")
        DemoLogger(
            output_dir=args.output_dir,
            run_number=args.run_number,
            n_frames=args.demo_frames,
            calibration=args.calibration,
        ).run()
        return

    if args.ins:
        try:
            from imu_gnss_pose import get_robot_pose

            get_robot_pose()
            print("[INFO] 惯导客户端已启动（UDP 3002 INSPVAXA），用于 CSV 航向/车速")
        except Exception as exc:
            print(f"[WARN] 惯导未就绪，Heading/Speed 将为 NaN: {exc}")

    print(f"[INFO] 接口={args.interface} | 通道={args.channel} | "
          f"波特率={args.bitrate} | 传感器ID={args.sensor_id}")

    try:
        bus = can.interface.Bus(
            interface=args.interface,
            channel=args.channel,
            bitrate=args.bitrate,
        )
    except Exception as e:
        print(f"[ERROR] 无法打开CAN接口: {e}")
        print("[HINT]  Linux SocketCAN: sudo ip link set can0 up type can bitrate 500000")
        return

    logger = ClusterLogger(
        bus=bus,
        output_dir=args.output_dir,
        sensor_id=args.sensor_id,
        run_number=args.run_number,
        enable_ins=args.ins,
        calibration=args.calibration,
    )
    try:
        logger.run(duration_s=args.duration)
    finally:
        bus.shutdown()
        print("[INFO] CAN总线已关闭")
        if args.ins:
            try:
                from imu_gnss_pose import shutdown_imu_client

                shutdown_imu_client()
            except Exception:
                pass


if __name__ == "__main__":
    main()
