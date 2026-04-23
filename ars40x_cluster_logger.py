"""
ARS40X Radar - Cluster_1_General (0x701) CAN采集与解析脚本
=============================================================
支持传感器: Continental ARS 404-21 / ARS 408-21
CAN消息:    0x600 Cluster_0_Status  (帧头，含簇数量)
            0x701 Cluster_1_General (位置/速度/RCS)
            0x702 Cluster_2_Quality (距离/速度 RMS、虚警概率、多普勒模糊态、有效性；需 RadarCfg_SendQuality)
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
  落盘文件名（ClusterCsvRuntime / UI 采集）:
    默认 {stem}_Raw_{YYYYMMDD_HHMMSS}.csv，与常见 DRI *Raw.csv 命名习惯一致；UI 也可传入显式文件名
    （例如星型测量的“30度第1次测量.csv”）。直线与圆周共用同一表结构，仅 stem（及圆周时横向 ROI）不同。
  惯导补充（--ins）:
    Heading/Speed 等与雷达「强制逐行对齐」：仅在每个雷达帧（0x600 触发）写一行时采样一次惯导，
    行频=雷达帧频；INS 短时缺失可在缓存窗口内沿用上一帧（见 imu_gnss_pose.sample_ins_for_radar_csv_row）。
    R：默认由本帧首簇（ROI 内）DX/DY 推导；若小车正在「前进直线段」路径跟踪（UI 传入 segment_kind=line），
    则由底盘线程写入沿路径到终点的剩余距离并优先用作 R（与直线 RCS 距离轴一致）。
    ViewAng 仍由首簇 DX/DY 推导。

  空间门（Cluster 写入 CSV 前筛选）：
    直线/默认：纵向 DX∈[4, 60] m、横向 |DY|≤3 m。
    圆周 RCS（UI 启动「圆周 Cluster」采集时）：同一纵向门，横向放宽为 |DY|≤10 m（与车头朝切向、目标在侧向的几何一致）。

  落盘前处理链（默认与「弱回波剔除 → 702 →（可选）双簇合并 → 距离 KF」一致，均可环境变量关闭/调参）：
    ① CLUSTER_RCS_MIN_DBSM：0x701 RCS（dBsm）下限，默认 -10；设为 none/off 关闭。
    ② 0x702 质量门：见下节 CLUSTER_QUALITY_*（需 SendQuality）。
    ③ CLUSTER_MERGE_TOP2：排序后前两簇非相干功率叠加（dB 域功率和，非算术平均）为单条簇再写槽位；**默认关闭**，落盘保留各簇原始 RCS00/RCS01/…；需旧行为时设 CLUSTER_MERGE_TOP2=1。
    ④ CLUSTER_DX_KF_ENABLE：一阶卡尔曼平滑首簇 DX/DY（默认开启）；CLUSTER_DX_KF_Q / CLUSTER_DX_KF_R 过程/测量方差，默认 Q=0.04、R=0.04。

  质量门（0x702，可选；默认开启，见 CLUSTER_QUALITY_FILTER）：
    解析 Cluster_2_Quality，与当前测量周期内同 Cluster_ID 的 0x701 条目合并后，在写 CSV 前可剔除差质量点（默认已放宽，可用环境变量收紧）：
    - Cluster_Pdh0=0 仍丢弃；最小等级阈 CLUSTER_QUALITY_MIN_PDH（默认 1，旧默认 2）
    - InvalidState：默认不因手册码过滤（CLUSTER_QUALITY_REJECT_INVALID 空）；可设为逗号列表收紧
    - AmbigState：默认不过滤（未设 CLUSTER_QUALITY_AMBIG_MODES 或与 * 等价）；可设为 3,4 等收紧
    - RMS 索引：默认上限 31（5bit 全量），CLUSTER_QUALITY_MAX_*_RMS_IDX 可改小以收紧
    未收到 0x702 或某簇无对应质量帧时：默认仍保留该簇；CLUSTER_QUALITY_STRICT=1 时无质量数据的簇也丢弃。

  簇槽位排序（与 DRI Raw / 雷达 range 序一致）：
    写入 DX00/DY00/RCS00、DX01/… 前，按纵向距离 DX 降序（DX00 为最大 DX，由远及近）。
    需要改为升序时设置环境变量 CLUSTER_CSV_SORT_DX_DESC=0（或 false/off）。

  丢帧与 NaN：
    无 0x600 帧头时收到的 0x701 一律丢弃，不生成行。
    若 MeasCounter 相对上一帧不连续，可选在写入新帧之前插入若干全 NaN 占位行（缺口宽度受 CLUSTER_CSV_MAX_MC_GAP 限制）；
    默认关闭；需启用时设置环境变量 CLUSTER_CSV_MEAS_COUNTER_GAP_FILL=1（或 true/yes/on）。

信号解析（与 ARS40X Technical Documentation V1.8 Table 33 一致，Intel 载荷内从 bit0 起算）:
  Cluster_ID（8）后，DistLong(13)/DistLat(10) 由多段字段拼接，**非**连续 13/10 bit 一次读取：
  raw_DistLong = (Cluster_DistLong1<<5) | Cluster_DistLong2 → DX = raw×0.2 − 500 (m)
  raw_DistLat  = (Cluster_DistLat1<<8) | Cluster_DistLat2  → DY = raw×0.2 − 102.3 (m)
  raw_VrelLong = (Cluster_VrelLong1<<2) | Cluster_VrelLong2 → Vx = raw×0.25 − 128 (m/s)
  raw_VrelLat  = (Cluster_VrelLat1<<3) | Cluster_VrelLat2 → Vy = raw×0.25 − 64 (m/s)
  Cluster_DynProp、Cluster_RCS 见 Table 33 位序。
  Table 30 Cluster_0_Status：MeasCounter 为 byte2 高字节、byte3 低字节组成 16 位计数器。

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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union


# ─────────────────────────────────────────────
#  CAN ID 定义
# ─────────────────────────────────────────────
CAN_ID_CLUSTER_STATUS  = 0x600   # Cluster_0_Status  (列表头)
CAN_ID_CLUSTER_GENERAL = 0x701   # Cluster_1_General (簇数据)
CAN_ID_CLUSTER_QUALITY = 0x702   # Cluster_2_Quality (簇质量，手册 Table 35–37)

# ── 0x702 质量过滤（环境变量可覆盖）──────────────────────────────────
_QF_ENV = os.getenv("CLUSTER_QUALITY_FILTER", "").strip().lower()
CLUSTER_QUALITY_FILTER: bool = _QF_ENV not in ("0", "false", "no", "off")

_STRICT_ENV = os.getenv("CLUSTER_QUALITY_STRICT", "").strip().lower()
CLUSTER_QUALITY_STRICT: bool = _STRICT_ENV in ("1", "true", "yes", "on")

_MINPDH_ENV = os.getenv("CLUSTER_QUALITY_MIN_PDH", "").strip()
try:
    CLUSTER_QUALITY_MIN_PDH: int = int(_MINPDH_ENV) if _MINPDH_ENV else 1
except ValueError:
    CLUSTER_QUALITY_MIN_PDH = 1

def _parse_csv_int_set(env_val: str, default: str) -> frozenset:
    raw = (env_val or default).strip()
    out: Set[int] = set()
    for part in raw.replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            if p.lower().startswith("0x"):
                out.add(int(p, 16))
            else:
                out.add(int(p, 10))
        except ValueError:
            continue
    return frozenset(out)

# InvalidState：默认不拒绝任何编码（空集）；可通过 CLUSTER_QUALITY_REJECT_INVALID=1,2,13,14 等收紧
CLUSTER_QUALITY_REJECT_INVALID: frozenset = _parse_csv_int_set(
    os.getenv("CLUSTER_QUALITY_REJECT_INVALID", ""),
    "",
)
# AmbigState：默认不过滤（与 * 等价）；设为 3,4 等可只保留推荐模态
_AMB_RAW = os.getenv("CLUSTER_QUALITY_AMBIG_MODES", "").strip()
if (not _AMB_RAW) or _AMB_RAW.lower() in ("*", "any", "all", "off"):
    CLUSTER_QUALITY_AMBIG_OK: Optional[frozenset] = None
else:
    CLUSTER_QUALITY_AMBIG_OK = _parse_csv_int_set(_AMB_RAW, "3,4")

def _parse_rms_idx_cap(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    try:
        return int(v) if v else int(default)
    except ValueError:
        return int(default)

# 5bit RMS 索引最大 31；默认放宽为 31，仅排除 Table 37 范围外异常
CLUSTER_QUALITY_MAX_DIST_LONG_RMS_IDX = _parse_rms_idx_cap(
    "CLUSTER_QUALITY_MAX_DIST_LONG_RMS_IDX", 31
)
CLUSTER_QUALITY_MAX_DIST_LAT_RMS_IDX = _parse_rms_idx_cap(
    "CLUSTER_QUALITY_MAX_DIST_LAT_RMS_IDX", 31
)
CLUSTER_QUALITY_MAX_VREL_LONG_RMS_IDX = _parse_rms_idx_cap(
    "CLUSTER_QUALITY_MAX_VREL_LONG_RMS_IDX", 31
)
CLUSTER_QUALITY_MAX_VREL_LAT_RMS_IDX = _parse_rms_idx_cap(
    "CLUSTER_QUALITY_MAX_VREL_LAT_RMS_IDX", 31
)

# Table 37：参数 0x0..0x1A（5bit 信号 0x1B..0x1F 手册未列，按最劣档处理）
_CLUSTER_RMS_UPPER_BOUND_M: Tuple[float, ...] = (
    0.005, 0.006, 0.008, 0.011, 0.014, 0.018, 0.023, 0.029, 0.038, 0.049,
    0.063, 0.081, 0.105, 0.135, 0.174, 0.224, 0.288, 0.371, 0.478, 0.616,
    0.794, 1.023, 1.317, 1.697, 2.187, 2.817, 3.630,
)


def cluster_rms_index_upper_bound(idx: int) -> float:
    """Table 37：返回该 RMS 索引对应的上界（m 或 m/s，视信号而定）。"""
    i = int(idx)
    if i < 0:
        return float(_CLUSTER_RMS_UPPER_BOUND_M[0])
    if i < len(_CLUSTER_RMS_UPPER_BOUND_M):
        return float(_CLUSTER_RMS_UPPER_BOUND_M[i])
    return float(_CLUSTER_RMS_UPPER_BOUND_M[-1]) * 2.0

MAX_CLUSTERS = 20   # 每帧最多记录簇数，与原始CSV保持一致（DX00~DX19）

# MeasCounter 不连续时插入占位的全 NaN 行（视为丢帧）；过大间隔视为计数翻转/异常，不插入以免刷屏。
_MC_GAP_ENV = os.getenv("CLUSTER_CSV_MAX_MC_GAP", "").strip()
try:
    CLUSTER_CSV_MAX_MC_GAP: int = int(_MC_GAP_ENV) if _MC_GAP_ENV else 256
except ValueError:
    CLUSTER_CSV_MAX_MC_GAP = 256

_MFILL_ENV = os.getenv("CLUSTER_CSV_MEAS_COUNTER_GAP_FILL", "").strip().lower()
# 默认关闭：避免雷达 MeasCounter 步进与预期不一致时出现「一行数据、一行全 NaN」；需占位时再显式开启。
CLUSTER_CSV_MEAS_COUNTER_GAP_FILL: bool = _MFILL_ENV in ("1", "true", "yes", "on")

# 落盘空间门：DistLong→DX（纵向，前方为正）、DistLat→DY（横向）；仅采集该扇区内的簇。
CLUSTER_ROI_DX_MIN_M = 4.0
CLUSTER_ROI_DX_MAX_M = 60.0
CLUSTER_ROI_ABS_DY_MAX_M = 3.0
# 圆周段采集（ClusterCsvRuntime begin_recording orbit_roi=True）：仅放宽横向，纵向与直线一致
CLUSTER_ROI_ORBIT_ABS_DY_MAX_M = 10.0


def cluster_in_roi_ex(
    dx: float,
    dy: float,
    *,
    dx_min: float = CLUSTER_ROI_DX_MIN_M,
    dx_max: float = CLUSTER_ROI_DX_MAX_M,
    abs_dy_max: float = CLUSTER_ROI_ABS_DY_MAX_M,
) -> bool:
    """空间门：纵向 [dx_min, dx_max]，横向 |DY|≤abs_dy_max。"""
    if not math.isfinite(dx) or not math.isfinite(dy):
        return False
    if dx < float(dx_min) or dx > float(dx_max):
        return False
    if abs(dy) > float(abs_dy_max):
        return False
    return True


def cluster_in_roi(dx: float, dy: float) -> bool:
    """默认（直线）ROI：前方纵向 [CLUSTER_ROI_DX_MIN_M, CLUSTER_ROI_DX_MAX_M]，|DY|≤CLUSTER_ROI_ABS_DY_MAX_M。"""
    return cluster_in_roi_ex(dx, dy)


# ── 0x701 落盘增强：RCS 门限 / 双簇合并 / 首簇距离 KF（环境变量）────────────────
_RCS_MIN_ENV = os.getenv("CLUSTER_RCS_MIN_DBSM", "-10").strip().lower()
if _RCS_MIN_ENV in ("", "none", "off", "disable"):
    CLUSTER_RCS_MIN_DBSM: Optional[float] = None
else:
    try:
        CLUSTER_RCS_MIN_DBSM = float(_RCS_MIN_ENV)
    except ValueError:
        CLUSTER_RCS_MIN_DBSM = -10.0

_MT2_ENV = os.getenv("CLUSTER_MERGE_TOP2", "").strip().lower()
CLUSTER_MERGE_TOP2: bool = _MT2_ENV in ("1", "true", "yes", "on")

_KF_ON_ENV = os.getenv("CLUSTER_DX_KF_ENABLE", "").strip().lower()
CLUSTER_DX_KF_ENABLE: bool = _KF_ON_ENV not in ("0", "false", "no", "off")

try:
    CLUSTER_DX_KF_Q = float(os.getenv("CLUSTER_DX_KF_Q", "0.04").strip() or "0.04")
except ValueError:
    CLUSTER_DX_KF_Q = 0.04
try:
    CLUSTER_DX_KF_R = float(os.getenv("CLUSTER_DX_KF_R", "0.04").strip() or "0.04")
except ValueError:
    CLUSTER_DX_KF_R = 0.04


def _combine_rcs_db_incoherent_sum(vals: Sequence[float]) -> Optional[float]:
    """dBsm 非相干功率叠加（与 Radar Signal Processing / DRI 一致）。"""
    acc: List[float] = []
    for v in vals:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            acc.append(fv)
    if not acc:
        return None
    if len(acc) == 1:
        return float(acc[0])
    p_lin = 0.0
    for v in acc:
        p_lin += 10.0 ** (v / 10.0)
    if p_lin <= 0.0 or not math.isfinite(p_lin):
        return None
    return float(10.0 * math.log10(p_lin))


class _DistKalman1D:
    """随机游走 + 标量观测；用于抑制 DX/DY 量化台阶。"""

    __slots__ = ("q", "r", "x", "p")

    def __init__(self, q: float, r: float) -> None:
        self.q = max(float(q), 1e-12)
        self.r = max(float(r), 1e-12)
        self.x: Optional[float] = None
        self.p = 1.0

    def reset(self) -> None:
        self.x = None
        self.p = 1.0

    def update(self, z: float) -> float:
        z = float(z)
        if self.x is None:
            self.x = z
            self.p = self.r
            return z
        p_pred = self.p + self.q
        k = p_pred / (p_pred + self.r)
        self.x = float(self.x + k * (z - self.x))
        self.p = (1.0 - k) * p_pred
        return float(self.x)


class _DistKalmanDualXY:
    __slots__ = ("kx", "ky")

    def __init__(self, q: float, r: float) -> None:
        self.kx = _DistKalman1D(q, r)
        self.ky = _DistKalman1D(q, r)

    def reset(self) -> None:
        self.kx.reset()
        self.ky.reset()

    def apply_to_first_cluster(self, c: Dict[str, Any]) -> None:
        dx = float(c["DX"])
        dy = float(c["DY"])
        c["DX"] = round(self.kx.update(dx), 2)
        c["DY"] = round(self.ky.update(dy), 2)


def _create_dist_kf_dual() -> Optional[_DistKalmanDualXY]:
    if not CLUSTER_DX_KF_ENABLE:
        return None
    return _DistKalmanDualXY(CLUSTER_DX_KF_Q, CLUSTER_DX_KF_R)


def _filter_clusters_rcs_minimum(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if CLUSTER_RCS_MIN_DBSM is None:
        return list(clusters)
    thr = float(CLUSTER_RCS_MIN_DBSM)
    out: List[Dict[str, Any]] = []
    for c in clusters:
        try:
            rcs = float(c.get("RCS", float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(rcs) and rcs >= thr:
            out.append(c)
    return out


def _merge_top_two_clusters_if_enabled(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将排序后最远两簇在功率域非相干叠加为一个等效簇（几何取平均）。
    仅当 CLUSTER_MERGE_TOP2=1 时启用；默认关闭以便 CSV 保留原始多槽 RCS。
    """
    if not CLUSTER_MERGE_TOP2 or len(clusters) < 2:
        return list(clusters)
    c0 = dict(clusters[0])
    c1 = dict(clusters[1])
    dx = (float(c0["DX"]) + float(c1["DX"])) * 0.5
    dy = (float(c0["DY"]) + float(c1["DY"])) * 0.5
    r0 = float(c0["RCS"])
    r1 = float(c1["RCS"])
    cr = _combine_rcs_db_incoherent_sum([r0, r1])
    if cr is None:
        cr = r0 if math.isfinite(r0) else r1
    try:
        cid0 = int(c0.get("ClusterID", 0))
        cid1 = int(c1.get("ClusterID", 1))
        cid_m = min(cid0, cid1)
    except (TypeError, ValueError):
        cid_m = int(c0.get("ClusterID", 0))
    merged = dict(c0)
    merged["DX"] = round(dx, 2)
    merged["DY"] = round(dy, 2)
    merged["RCS"] = round(float(cr), 2)
    merged["ClusterID"] = int(cid_m)
    return [merged] + list(clusters[2:])


def finalize_clusters_pipeline(
    frame_clusters: List[Dict[str, Any]],
    frame_quality: Dict[int, Dict[str, Any]],
    dist_kf: Optional[_DistKalmanDualXY],
) -> List[Dict[str, Any]]:
    """
    单帧 0x701→落盘行 的最终簇列表：RCS 门限 → 0x702 合并过滤 → DX 排序 → 可选双簇合并 → 可选 KF。
    frame_quality 由调用方在本函数返回后 clear。
    """
    clusters = _filter_clusters_rcs_minimum(frame_clusters)
    merged = _merge_quality_into_clusters(clusters, frame_quality)
    merged = sort_clusters_for_dri_csv(merged)
    merged = _merge_top_two_clusters_if_enabled(merged)
    if dist_kf is not None and merged:
        dist_kf.apply_to_first_cluster(merged[0])
    return merged


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
    Table 30 Cluster_0_Status（DLC≥5，与 V1.8 一致）:
      Byte0 NofClustersNear, Byte1 NofClustersFar,
      Byte2–3 MeasCounter（高字节在前：MeasCounter = data[2]<<8 | data[3]）,
      Byte4 低 4bit Reserved、高 4bit InterfaceVersion。
    """
    raw = bytes(data or b"")
    d = (raw[:5] + b"\x00" * 5)[:5]
    b4 = d[4]
    meas = ((d[2] & 0xFF) << 8) | (d[3] & 0xFF)
    return {
        "NofClustersNear": int(d[0]),
        "NofClustersFar": int(d[1]),
        "MeasCounter": int(meas),
        "InterfaceVersion": int((b4 >> 4) & 0x0F),
    }


# ─────────────────────────────────────────────
#  0x701 Cluster_1_General 解析
# ─────────────────────────────────────────────
def parse_cluster_general(data: bytes) -> dict:
    """
    Continental Standardized ARS Interface **Table 33 Cluster_1_General**（V1.8，Intel 载荷）:
      按 64bit 小端布局解析；DistLong/DistLat 为分段字段拼接（见手册图示），不得用连续 bit19..31 / 24..33 误读。

    物理量（与手册 Resolution / Offset 一致）:
      DX  = ((DistLong1<<5)|DistLong2)×0.2 − 500
      DY  = ((DistLat1<<8)|DistLat2)×0.2 − 102.3
      Vx  = ((VrelLong1<<2)|VrelLong2)×0.25 − 128（可选，供调试）
      Vy  = ((VrelLat1<<3)|VrelLat2)×0.25 − 64
      RCS = raw×0.5 − 64
    """
    if len(data) < 8:
        data = bytes(data) + bytes(8 - len(data))
    v = int.from_bytes(data[:8], "little")

    cluster_id = v & 0xFF
    dist_long1 = (v >> 8) & 0xFF
    dist_lat1 = (v >> 16) & 0x3
    dist_long2 = (v >> 19) & 0x1F
    dist_lat2 = (v >> 24) & 0xFF
    dist_long_raw = (dist_long1 << 5) | dist_long2
    dist_lat_raw = (dist_lat1 << 8) | dist_lat2
    dist_long = dist_long_raw * 0.2 - 500.0
    dist_lat = dist_lat_raw * 0.2 - 102.3

    vrel_long1 = (v >> 32) & 0xFF
    vrel_lat1 = (v >> 40) & 0x3F
    vrel_long2 = (v >> 46) & 0x3
    dyn_prop = (v >> 48) & 0x7
    vrel_lat2 = (v >> 53) & 0x7
    rcs_raw = (v >> 56) & 0xFF

    vrel_long_raw = (vrel_long1 << 2) | vrel_long2
    vrel_lat_raw = (vrel_lat1 << 3) | vrel_lat2
    vrel_long = vrel_long_raw * 0.25 - 128.0
    vrel_lat = vrel_lat_raw * 0.25 - 64.0
    rcs = rcs_raw * 0.5 - 64.0

    return {
        "ClusterID": int(cluster_id),
        "DX": round(dist_long, 2),
        "DY": round(dist_lat, 2),
        "DynProp": int(dyn_prop),
        "RCS": round(rcs, 2),
        "VrelLong": round(vrel_long, 2),
        "VrelLat": round(vrel_lat, 2),
    }


# ─────────────────────────────────────────────
#  0x702 Cluster_2_Quality 解析（Table 35–37）
# ─────────────────────────────────────────────
def parse_cluster_quality(data: bytes) -> dict:
    """
    Continental **Cluster_2_Quality**（V1.8，Intel 载荷，与 Table 33/图示相同的位序约定）。

    手册 Table 35 中 Cluster_Pdh0 的 Start=24 与 Cluster_DistLat_rms(22,5) 在「连续 LSB 位域」下重叠；
    按 Figure 23 字节打包习惯，本实现将 Pdh0 置于 DistLat_rms 之后（bit 27 起），
    Cluster_VrelLat_rms → bit30、Cluster_AmbigState → bit35、Cluster_InvalidState → bit38，
    与 Table 36 中 Ambig=32/Invalid=35 相差 3bit 的字段对齐争议见环境变量 CLUSTER_702_LAYOUT。

    返回字段均为解析整数下标/枚举；RMS 物理上界用 cluster_rms_index_upper_bound() 查 Table 37。
    """
    d = bytes(data or b"")
    raw = (d + b"\x00" * 8)[:8]

    layout = os.getenv("CLUSTER_702_LAYOUT", "").strip().lower()
    if layout in ("table36", "t36", "manual32"):
        # 尝试严格按 Table 36 的 Ambig=32、Invalid=35 对齐（可能与 Pdh0/DistLat 手册行冲突，供对照 DBC 用）
        cid = extract_bits(raw, 0, 8)
        dlr = extract_bits(raw, 11, 5)
        vlr = extract_bits(raw, 17, 5)
        dlat = extract_bits(raw, 22, 5)
        pdh = extract_bits(raw, 24, 3)
        vlat = extract_bits(raw, 28, 5)
        amb = extract_bits(raw, 32, 3)
        inv = extract_bits(raw, 35, 5)
    else:
        cid = extract_bits(raw, 0, 8)
        dlr = extract_bits(raw, 11, 5)
        vlr = extract_bits(raw, 17, 5)
        dlat = extract_bits(raw, 22, 5)
        pdh = extract_bits(raw, 27, 3)
        vlat = extract_bits(raw, 30, 5)
        amb = extract_bits(raw, 35, 3)
        inv = extract_bits(raw, 38, 5)

    return {
        "ClusterID": int(cid),
        "DistLongRmsIdx": int(dlr),
        "VrelLongRmsIdx": int(vlr),
        "DistLatRmsIdx": int(dlat),
        "Pdh0": int(pdh),
        "VrelLatRmsIdx": int(vlat),
        "AmbigState": int(amb),
        "InvalidState": int(inv),
        "DistLongRmsUpperM": round(cluster_rms_index_upper_bound(dlr), 4),
        "DistLatRmsUpperM": round(cluster_rms_index_upper_bound(dlat), 4),
        "VrelLongRmsUpperMps": round(cluster_rms_index_upper_bound(vlr), 4),
        "VrelLatRmsUpperMps": round(cluster_rms_index_upper_bound(vlat), 4),
    }


def encode_cluster_quality_payload(
    cluster_id: int = 0,
    dist_long_rms_idx: int = 0,
    vrel_long_rms_idx: int = 0,
    dist_lat_rms_idx: int = 0,
    pdh0: int = 7,
    vrel_lat_rms_idx: int = 0,
    ambig_state: int = 3,
    invalid_state: int = 0,
    *,
    layout: str = "default",
) -> bytes:
    """与 parse_cluster_quality 互逆（供测试/仿真）；layout 取值同 CLUSTER_702_LAYOUT。"""
    lay = (layout or "default").strip().lower()
    v = 0
    v |= int(cluster_id) & 0xFF
    if lay in ("table36", "t36", "manual32"):
        v |= (int(dist_long_rms_idx) & 0x1F) << 11
        v |= (int(vrel_long_rms_idx) & 0x1F) << 17
        v |= (int(dist_lat_rms_idx) & 0x1F) << 22
        v |= (int(pdh0) & 0x07) << 24
        v |= (int(vrel_lat_rms_idx) & 0x1F) << 28
        v |= (int(ambig_state) & 0x07) << 32
        v |= (int(invalid_state) & 0x1F) << 35
    else:
        v |= (int(dist_long_rms_idx) & 0x1F) << 11
        v |= (int(vrel_long_rms_idx) & 0x1F) << 17
        v |= (int(dist_lat_rms_idx) & 0x1F) << 22
        v |= (int(pdh0) & 0x07) << 27
        v |= (int(vrel_lat_rms_idx) & 0x1F) << 30
        v |= (int(ambig_state) & 0x07) << 35
        v |= (int(invalid_state) & 0x1F) << 38
    return v.to_bytes(8, "little")


def cluster_quality_accepts(general: Dict[str, Any], quality: Optional[Dict[str, Any]]) -> bool:
    """
    是否保留该簇用于写 CSV / 统计。general 为 parse_cluster_general 结果；quality 为同 Cluster_ID 的
    parse_cluster_quality 结果，若无则 quality=None。
    """
    if not CLUSTER_QUALITY_FILTER:
        return True
    if quality is None:
        return not CLUSTER_QUALITY_STRICT
    try:
        pdh = int(quality.get("Pdh0", 0))
    except (TypeError, ValueError):
        return not CLUSTER_QUALITY_STRICT
    if pdh <= 0:
        return False
    if pdh < int(CLUSTER_QUALITY_MIN_PDH):
        return False

    try:
        inv = int(quality.get("InvalidState", 0))
    except (TypeError, ValueError):
        inv = 0
    if inv in CLUSTER_QUALITY_REJECT_INVALID:
        return False

    try:
        amb = int(quality.get("AmbigState", 0))
    except (TypeError, ValueError):
        amb = 0
    if CLUSTER_QUALITY_AMBIG_OK is not None and amb not in CLUSTER_QUALITY_AMBIG_OK:
        return False

    try:
        dlr = int(quality.get("DistLongRmsIdx", 0))
        dlat = int(quality.get("DistLatRmsIdx", 0))
        vlr = int(quality.get("VrelLongRmsIdx", 0))
        vlat = int(quality.get("VrelLatRmsIdx", 0))
    except (TypeError, ValueError):
        return False
    if dlr > int(CLUSTER_QUALITY_MAX_DIST_LONG_RMS_IDX):
        return False
    if dlat > int(CLUSTER_QUALITY_MAX_DIST_LAT_RMS_IDX):
        return False
    if vlr > int(CLUSTER_QUALITY_MAX_VREL_LONG_RMS_IDX):
        return False
    if vlat > int(CLUSTER_QUALITY_MAX_VREL_LAT_RMS_IDX):
        return False
    return True


def _merge_quality_into_clusters(
    clusters: List[Dict[str, Any]],
    quality_by_id: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """按 Cluster_ID 合并 0x702，并按 cluster_quality_accepts 过滤。"""
    out: List[Dict[str, Any]] = []
    for c in clusters:
        try:
            cid = int(c.get("ClusterID", -1))
        except (TypeError, ValueError):
            cid = -1
        q = quality_by_id.get(cid) if cid >= 0 else None
        if not cluster_quality_accepts(c, q):
            continue
        if q:
            merged = dict(c)
            for k, v in q.items():
                if k != "ClusterID":
                    merged[k] = v
            out.append(merged)
        else:
            out.append(c)
    return out


def encode_cluster_general_payload(
    cluster_id: int = 0,
    dx: float = 0.0,
    dy: float = 0.0,
    dyn_prop: int = 0,
    rcs: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
) -> bytes:
    """
    Table 33 编码（与 parse_cluster_general 互逆，供演示帧 / 单测）。
    """
    dist_long_raw = int(round((float(dx) + 500.0) / 0.2))
    dist_lat_raw = int(round((float(dy) + 102.3) / 0.2))
    dist_long_raw = max(0, min(dist_long_raw, 0x1FFF))
    dist_lat_raw = max(0, min(dist_lat_raw, 0x3FF))
    dist_long1 = (dist_long_raw >> 5) & 0xFF
    dist_long2 = dist_long_raw & 0x1F
    dist_lat1 = (dist_lat_raw >> 8) & 0x3
    dist_lat2 = dist_lat_raw & 0xFF

    vrel_long_raw = int(round((float(vx) + 128.0) / 0.25))
    vrel_lat_raw = int(round((float(vy) + 64.0) / 0.25))
    vrel_long_raw = max(0, min(vrel_long_raw, 0x3FF))
    vrel_lat_raw = max(0, min(vrel_lat_raw, 0x1FF))
    vrel_long1 = (vrel_long_raw >> 2) & 0xFF
    vrel_long2 = vrel_long_raw & 0x3
    vrel_lat1 = (vrel_lat_raw >> 3) & 0x3F
    vrel_lat2 = vrel_lat_raw & 0x7

    rcs_i = int(round((float(rcs) + 64.0) / 0.5))
    rcs_i = max(0, min(rcs_i, 0xFF))

    v = 0
    v |= int(cluster_id) & 0xFF
    v |= dist_long1 << 8
    v |= dist_lat1 << 16
    v |= dist_long2 << 19
    v |= dist_lat2 << 24
    v |= vrel_long1 << 32
    v |= vrel_lat1 << 40
    v |= vrel_long2 << 46
    v |= (int(dyn_prop) & 0x7) << 48
    v |= vrel_lat2 << 53
    v |= rcs_i << 56
    return v.to_bytes(8, "little")


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


_SORT_DX_DESC_ENV = os.getenv("CLUSTER_CSV_SORT_DX_DESC", "").strip().lower()
# 默认降序（DX00 最大）；仅当显式设为 0/false/no/off 时为升序
CLUSTER_CSV_SORT_DX_DESC: bool = _SORT_DX_DESC_ENV not in ("0", "false", "no", "off")


def sort_clusters_for_dri_csv(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    同一帧内簇按 DX（DistLong，纵向距离）排序后再映射到 DX00/DX01/…。
    默认降序（DX00 最大，与雷达 range 列表习惯一致）；CLUSTER_CSV_SORT_DX_DESC=0/false/off 时为升序。
    同等 DX 时按 Cluster_ID 升序，保证稳定、可复现。
    """
    if len(clusters) <= 1:
        return list(clusters)

    def sort_key(c: Dict[str, Any]) -> Tuple[bool, float, int]:
        try:
            dx = float(c.get("DX", float("nan")))
        except (TypeError, ValueError):
            dx = float("nan")
        non_finite = not math.isfinite(dx)
        dx_key = -dx if CLUSTER_CSV_SORT_DX_DESC else dx
        try:
            cid = int(c.get("ClusterID", 0))
        except (TypeError, ValueError):
            cid = 0
        return (non_finite, dx_key, cid)

    return sorted(clusters, key=sort_key)


def build_data_row(
    timestamp: float,
    clusters: List[Dict[str, Any]],
    ins_extras: Optional[Dict[str, float]] = None,
) -> List[Any]:
    """
    [Time, R, ViewAng, Heading, Speed, MeasType, TargetHeading,
     MeasRadOrAng, Aim, DX00..RCS19]

    ins_extras: sample_ins_for_radar_csv_row()（每雷达帧仅调用一次；行频=雷达帧频）。
    R / ViewAng：
      - 若路径跟踪线程提供了「前进直线段沿路径到终点剩余距离」，则 R 优先写该值（与 DRI 第 2 列语义对齐）；
      - 否则若本帧有簇，R 由首簇 DX/DY 推导平面距离，ViewAng=atan2(DY,DX)(°)；
      - 无覆盖且无簇时 R/ViewAng 为 NaN。
    MeasType：来自 Table 33 首簇 DynProp（CAN）；无簇时为 NaN（不再写死 0）。
    """
    r_path_rem: Optional[float] = None
    try:
        from imu_gnss_pose import get_cluster_csv_straight_path_remaining_m

        r_path_rem = get_cluster_csv_straight_path_remaining_m()
    except Exception:
        r_path_rem = None

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

    if r_path_rem is not None and math.isfinite(float(r_path_rem)):
        r_for_csv = round(float(r_path_rem), 4)
    elif clusters:
        r_for_csv = r_geom
    else:
        r_for_csv = "NaN"

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
        r_for_csv,
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
        self.id_quality = CAN_ID_CLUSTER_QUALITY + offset

        # 帧缓存
        self.current_status  = {}
        self.frame_clusters  = []    # 当前帧收集的簇列表
        self.frame_start_ts  = None  # 当前帧时间戳（取0x600到达时刻）
        self.frame_quality: Dict[int, Dict[str, Any]] = {}  # 本测量周期 0x702，按 Cluster_ID

        # 统计
        self.total_frames   = 0
        self.total_clusters = 0

        self._last_meas_counter_for_gap: Optional[int] = None
        self._dist_kf_dual: Optional[_DistKalmanDualXY] = _create_dist_kf_dual()

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
        if CLUSTER_RCS_MIN_DBSM is not None:
            print(f"[INFO] RCS 下限: ≥{CLUSTER_RCS_MIN_DBSM} dBsm（弱回波剔除）")
        if CLUSTER_MERGE_TOP2:
            print("[INFO] 双簇合并: 排序后前两簇 → 单目标（非相干功率叠加，非算术平均）")
        if CLUSTER_DX_KF_ENABLE:
            print(
                f"[INFO] 首簇距离卡尔曼: Q={CLUSTER_DX_KF_Q}, R={CLUSTER_DX_KF_R} "
                f"（CLUSTER_DX_KF_ENABLE=0 可关闭）"
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
        merged = finalize_clusters_pipeline(
            self.frame_clusters,
            self.frame_quality,
            self._dist_kf_dual,
        )
        self.frame_quality.clear()
        ins_extras = None
        if self.enable_ins:
            try:
                from imu_gnss_pose import sample_ins_for_radar_csv_row

                ins_extras = sample_ins_for_radar_csv_row()
            except Exception:
                ins_extras = None
        row = build_data_row(self.frame_start_ts, merged, ins_extras)
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        self.total_frames   += 1
        self.total_clusters += len(merged)
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

        elif msg.arbitration_id == self.id_quality:
            if self.frame_start_ts is None:
                return
            q = parse_cluster_quality(bytes(msg.data))
            self.frame_quality[int(q["ClusterID"])] = q

    def run(self, duration_s=None):
        start_time = time.time()
        print(f"[INFO] 开始采集... (Ctrl+C 停止)")
        print(
            f"[INFO] 监听 0x{self.id_status:03X}(状态) / 0x{self.id_general:03X}(General) "
            f"/ 0x{self.id_quality:03X}(Quality)"
        )
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
        self.id_quality     = CAN_ID_CLUSTER_QUALITY
        self.current_status = {}
        self.frame_clusters = []
        self.frame_start_ts = None
        self.frame_quality = {}
        self.total_frames   = 0
        self.total_clusters = 0
        self.n_frames       = n_frames
        self._last_meas_counter_for_gap = None
        self._dist_kf_dual: Optional[_DistKalmanDualXY] = _create_dist_kf_dual()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"cluster_log_demo_{ts}.csv")
        self.csv_file, self.csv_writer = init_csv(
            self.csv_path, run_number, calibration=calibration
        )
        print(f"[DEMO] 输出文件: {self.csv_path}")

    def _make_status_msg(self, meas_cnt: int, n_near: int, t: float):
        mc = int(meas_cnt) & 0xFFFF
        nn = int(n_near) & 0xFF
        data = bytes(
            [
                nn,
                0,
                (mc >> 8) & 0xFF,
                mc & 0xFF,
                0x10,
            ]
        )
        return type("Msg", (), {
            "arbitration_id": CAN_ID_CLUSTER_STATUS,
            "data": data,
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
        payload = encode_cluster_general_payload(
            cluster_id=int(cluster_id),
            dx=float(dist_long),
            dy=float(dist_lat),
            dyn_prop=0,
            rcs=float(rcs),
            vx=0.0,
            vy=0.0,
        )
        return type("Msg", (), {
            "arbitration_id": CAN_ID_CLUSTER_GENERAL,
            "data": payload,
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
        "id_quality",
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
        "frame_quality",
        "frame_start_ts",
        "total_frames",
        "total_clusters",
        "run_number",
        "_last_clusters",
        "_last_clusters_ts",
        "_last_meas_counter_for_gap",
        "_dist_kf_dual",
        "_roi_abs_dy_max",
        "_orbit_roi_session",
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
        self.id_quality = CAN_ID_CLUSTER_QUALITY + off
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
        self.frame_quality: Dict[int, Dict[str, Any]] = {}
        self.frame_start_ts: Optional[float] = None
        self.total_frames = 0
        self.total_clusters = 0
        self.run_number = 1
        self._last_clusters: List[Dict[str, Any]] = []
        self._last_clusters_ts: Optional[float] = None
        self._last_meas_counter_for_gap: Optional[int] = None
        self._dist_kf_dual: Optional[_DistKalmanDualXY] = None
        self._roi_abs_dy_max: float = float(CLUSTER_ROI_ABS_DY_MAX_M)
        self._orbit_roi_session: bool = False

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
                    if cluster_in_roi_ex(
                        float(c["DX"]),
                        float(c["DY"]),
                        abs_dy_max=self._roi_abs_dy_max,
                    ):
                        self.frame_clusters.append(c)
            elif aid == self.id_quality:
                if self.frame_start_ts is None:
                    return
                q = parse_cluster_quality(bytes(msg.data))
                self.frame_quality[int(q["ClusterID"])] = q

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
        merged = finalize_clusters_pipeline(
            self.frame_clusters,
            self.frame_quality,
            self._dist_kf_dual,
        )
        self.frame_quality.clear()
        self._last_clusters_ts = ts
        self._last_clusters = [{k: c[k] for k in c} for c in merged]

        if self._writing and self.csv_writer is not None and self.csv_file is not None:
            ins_extras = None
            if self.enable_ins:
                try:
                    from imu_gnss_pose import sample_ins_for_radar_csv_row

                    ins_extras = sample_ins_for_radar_csv_row()
                except Exception:
                    ins_extras = None
            row = build_data_row(ts, merged, ins_extras)
            self.csv_writer.writerow(row)
            self.csv_file.flush()
            self.total_frames += 1
            self.total_clusters += len(merged)

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
                snap: List[Dict[str, Any]] = []
                for c in self.frame_clusters:
                    mc = {k: c[k] for k in c}
                    try:
                        cid = int(c.get("ClusterID", -1))
                    except (TypeError, ValueError):
                        cid = -1
                    if cid >= 0:
                        q = self.frame_quality.get(cid)
                        if q:
                            for kk, vv in q.items():
                                if kk != "ClusterID":
                                    mc[kk] = vv
                    snap.append(mc)
                return tlf, snap
            ts = 0.0 if self._last_clusters_ts is None else float(self._last_clusters_ts)
            return ts, [{k: c[k] for k in c} for c in self._last_clusters]

    def begin_recording(
        self,
        output_dir: str,
        run_number: int = 1,
        stem: str = "cluster_rcs",
        calibration: Optional[Union[str, float]] = None,
        *,
        orbit_roi: bool = False,
        filename: Optional[str] = None,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        if filename:
            safe_name = os.path.basename(str(filename)).replace("/", "_").replace("\\", "_").strip()
            safe_name = safe_name[:180] or "cluster_rcs.csv"
            if not safe_name.lower().endswith(".csv"):
                safe_name += ".csv"
            base_path = os.path.join(output_dir, safe_name)
            path = base_path
            root, ext = os.path.splitext(base_path)
            i = 2
            while os.path.exists(path):
                path = f"{root}_{i}{ext or '.csv'}"
                i += 1
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe = (stem or "cluster_rcs").replace("/", "_").replace("\\", "_")[:120]
            # 与实验室 DRI Raw 命名一致（如 1_M1_S1_Raw.csv → 此处为 stem_Raw_时间戳.csv，避免重名）
            path = os.path.join(output_dir, f"{safe}_Raw_{ts}.csv")
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
            self.frame_quality = {}
            self._last_meas_counter_for_gap = None
            self._dist_kf_dual = _create_dist_kf_dual()
            self._writing = True
            self._orbit_roi_session = bool(orbit_roi)
            self._roi_abs_dy_max = (
                float(CLUSTER_ROI_ORBIT_ABS_DY_MAX_M)
                if orbit_roi
                else float(CLUSTER_ROI_ABS_DY_MAX_M)
            )
            if self.enable_ins:
                try:
                    from imu_gnss_pose import reset_ins_cluster_csv_row_hold

                    reset_ins_cluster_csv_row_hold()
                except Exception:
                    pass

    def _flush_soft_reset_unsafe(self) -> None:
        self.frame_clusters = []
        self.frame_quality = {}
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
        self._roi_abs_dy_max = float(CLUSTER_ROI_ABS_DY_MAX_M)
        self._orbit_roi_session = False

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
            self._roi_abs_dy_max = float(CLUSTER_ROI_ABS_DY_MAX_M)
            self._orbit_roi_session = False
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
