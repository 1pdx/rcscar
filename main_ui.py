 # main_ui.py
# -*- coding: utf-8 -*-
import csv
import hashlib
import importlib.util
import os
import platform
import re
import shutil
import sys
import math
import time
import threading
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from ars40x_cluster_logger import (
    MAX_CLUSTERS,
    build_columns,
    format_cluster_csv_calibration,
)

import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "#0D47A1")
pg.setConfigOption("antialias", True)

from imu_gnss_pose import (
    get_robot_pose,
    get_absolute_robot_pose,
    PoseSolution,
    get_status_summary,
    align_enu_y_axis_with_points,
    clear_enu_calibration,
    get_enu_calibration_summary,
    set_position_origin_to_current,
    define_local_frame_from_two_geodetic_points,
    absolute_planar_xy_from_geodetic,
)
from main_controller import MainController, QueuedPathTask
from car_control import FORWARD_STRAIGHT_TRACKING_KWARGS
from star_trajectory_planner import (
    StarMeasurementSpec,
    build_star_measurement_plan,
 )


_RADAR_MODULE = None
_RCS_REF_MODULE = None
_DATA_ANALYSIS_MODULE = None
_CJK_FONT_FAMILY: Optional[str] = None
_CJK_FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "Source Han Serif SC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "AR PL UMing CN",
    "AR PL UKai CN",
    "Droid Sans Fallback",
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
    "SimSun",
    "NSimSun",
    "Microsoft JhengHei",
    "Microsoft JhengHei UI",
    "PingFang SC",
    "Heiti SC",
    "Songti SC",
    "Hiragino Sans GB",
]

_TRACKING_MOTION_TRACE_FIELDS: Tuple[str, ...] = (
    "timestamp",
    "relative_time_s",
    "run_key",
    "run_label",
    "tracking_mode",
    "segment_index",
    "speed_sign",
    "speed_mps",
    "nominal_speed_abs_mps",
    "lookahead_base_m",
    "arrival_dist_m",
    "slow_down_dist_m",
    "stanley_gain",
    "stanley_softening_distance_m",
    "stanley_term_rad",
    "lateral_pid_kp",
    "lateral_pid_ki",
    "lateral_pid_kd",
    "heading_pid_kp",
    "heading_pid_ki",
    "heading_pid_kd",
    "yaw_rate_pid_kp",
    "yaw_rate_pid_ki",
    "yaw_rate_pid_kd",
    "cmd_v_mps",
    "cmd_w_radps",
    "desired_v_mps",
    "desired_w_radps",
    "feedback_v_mps",
    "feedback_w_radps",
    "lateral_error_m",
    "stanley_lateral_pd_output_radps",
    "heading_error_rad",
    "yaw_rate_error_radps",
    "lateral_pid_output_radps",
    "heading_pid_output_radps",
    "yaw_rate_pid_output_radps",
    "path_curvature_inv_m",
    "path_ff_w_radps",
    "profile_speed_mps",
    "path_s_m",
    "dist_to_goal_m",
    "pose_age_s",
    "current_x_m",
    "current_y_m",
    "nearest_x_m",
    "nearest_y_m",
    "lookahead_x_m",
    "lookahead_y_m",
    "segment_kind",
    "segment_trajectory_name",
    "motion_direction",
)

_TRACKING_MOTION_INTERNAL_FIELDS: Tuple[str, ...] = (
    *_TRACKING_MOTION_TRACE_FIELDS,
    "motion_distance_m",
    "yaw_rate_feedback_valid",
)


def _configure_utf8_stdio() -> None:
    # Force UTF-8 logs on Raspberry Pi terminals with incomplete locale setup.
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                continue


def _iter_local_font_files() -> List[Path]:
    fonts_dir = Path(__file__).with_name("fonts")
    if not fonts_dir.is_dir():
        return []

    font_paths: List[Path] = []
    for ext in ("*.ttf", "*.otf", "*.ttc"):
        font_paths.extend(sorted(fonts_dir.glob(ext)))
    return font_paths


def _download_local_cjk_font() -> Optional[Path]:
    fonts_dir = Path(__file__).with_name("fonts")
    fonts_dir.mkdir(parents=True, exist_ok=True)
    target_path = fonts_dir / "NotoSansCJKsc-Regular.otf"
    if target_path.exists():
        try:
            if target_path.stat().st_size > 1024 * 1024:
                return target_path
        except OSError:
            pass

    download_urls = [
        "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
        "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    ]
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    for url in download_urls:
        try:
            with urllib.request.urlopen(url, timeout=20) as resp, tmp_path.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
            if tmp_path.stat().st_size <= 1024 * 1024:
                raise RuntimeError("downloaded font file is unexpectedly small")
            tmp_path.replace(target_path)
            print(f"[main_ui] Downloaded bundled CJK font: {target_path}")
            return target_path
        except Exception as exc:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            print(f"[main_ui] Failed to download CJK font from {url}: {exc}")
    return None


def _resolve_cjk_font_family() -> Optional[str]:
    global _CJK_FONT_FAMILY
    if _CJK_FONT_FAMILY:
        return _CJK_FONT_FAMILY

    from matplotlib import font_manager

    available = set()
    for font_path in _iter_local_font_files():
        try:
            font_manager.fontManager.addfont(str(font_path))
        except Exception:
            continue
        try:
            font_id = QtGui.QFontDatabase.addApplicationFont(str(font_path))
        except Exception:
            font_id = -1
        if font_id >= 0:
            try:
                available.update(QtGui.QFontDatabase.applicationFontFamilies(font_id))
            except Exception:
                pass

    available.update(f.name for f in font_manager.fontManager.ttflist)
    try:
        available.update(str(name) for name in QtGui.QFontDatabase().families())
    except Exception:
        pass

    for name in _CJK_FONT_CANDIDATES:
        if name in available:
            _CJK_FONT_FAMILY = name
            return _CJK_FONT_FAMILY

    try:
        from matplotlib import ft2font

        sample_char = "中"
        font_paths = []
        for ext in ("ttf", "otf", "ttc"):
            font_paths.extend(font_manager.findSystemFonts(fontext=ext))
        for font_path in font_paths:
            try:
                if ft2font.FT2Font(font_path).get_char_index(ord(sample_char)) == 0:
                    continue
                name = font_manager.FontProperties(fname=font_path).get_name()
                if name:
                    _CJK_FONT_FAMILY = name
                    return _CJK_FONT_FAMILY
            except Exception:
                continue
    except Exception:
        pass

    downloaded_font = _download_local_cjk_font()
    if downloaded_font is not None:
        try:
            font_manager.fontManager.addfont(str(downloaded_font))
        except Exception:
            pass
        try:
            font_id = QtGui.QFontDatabase.addApplicationFont(str(downloaded_font))
            if font_id >= 0:
                families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
                if families:
                    _CJK_FONT_FAMILY = str(families[0])
                    return _CJK_FONT_FAMILY
        except Exception:
            pass
        try:
            name = font_manager.FontProperties(fname=str(downloaded_font)).get_name()
            if name:
                _CJK_FONT_FAMILY = name
                return _CJK_FONT_FAMILY
        except Exception:
            pass

    return None


_configure_utf8_stdio()


def _load_radar_processing_module():
    global _RADAR_MODULE
    if _RADAR_MODULE is not None:
        return _RADAR_MODULE

    module_name = "radar_signal_processing"
    if module_name in sys.modules:
        _RADAR_MODULE = sys.modules[module_name]
        return _RADAR_MODULE

    module_path = Path(__file__).with_name("Radar Signal Processing.py")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load radar processing module: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    _RADAR_MODULE = mod
    return _RADAR_MODULE


_radar_mod = _load_radar_processing_module()
RcsRunRecorder = _radar_mod.RcsRunRecorder
AssocLock = _radar_mod.AssocLock
ObjMeas = _radar_mod.ObjMeas
CurvePoint = _radar_mod.CurvePoint
combine_rcs_db_incoherent_sum = getattr(
    _radar_mod, "combine_rcs_db_incoherent_sum", None
)
RCS_POINT_RCS_EMA_ALPHA = float(getattr(_radar_mod, "RCS_POINT_RCS_EMA_ALPHA", 0.35))
RCS_BIN_INCOHERENT_SUM_MAX_TIME_SPAN_S = float(
    getattr(_radar_mod, "RCS_BIN_INCOHERENT_SUM_MAX_TIME_SPAN_S", 0.08)
)
# 与 fit_curve 一致：分箱后沿距离对 RCS 做 SG（无 scipy 时模块内回退中值+滑动平均）
_peak_smooth_rcs_series = getattr(_radar_mod, "_peak_smooth_rcs_series", None)

# Cluster CSV 解析（对齐 DRI Raw）：Time 归一为段内相对秒；剔除首目标平面位置突变（如 DX 跳变）
try:
    CLUSTER_RCS_PARSE_MAX_DIST_STEP_M = float(
        os.getenv("CLUSTER_RCS_PARSE_MAX_DIST_STEP_M", "5.0")
    )
except ValueError:
    CLUSTER_RCS_PARSE_MAX_DIST_STEP_M = 5.0
try:
    CLUSTER_RCS_PARSE_TIME_GAP_RESET_S = float(
        os.getenv("CLUSTER_RCS_PARSE_TIME_GAP_RESET_S", "0.35")
    )
except ValueError:
    CLUSTER_RCS_PARSE_TIME_GAP_RESET_S = 0.35


def _load_rcs_reference_module():
    global _RCS_REF_MODULE
    if _RCS_REF_MODULE is not None:
        return _RCS_REF_MODULE

    module_name = "rcs_reference_data"
    if module_name in sys.modules:
        _RCS_REF_MODULE = sys.modules[module_name]
        return _RCS_REF_MODULE

    module_path = Path(__file__).with_name("rcs_reference_data.py")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load rcs reference module: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    _RCS_REF_MODULE = mod
    return _RCS_REF_MODULE


def _load_data_analysis_module():
    global _DATA_ANALYSIS_MODULE
    if _DATA_ANALYSIS_MODULE is not None:
        return _DATA_ANALYSIS_MODULE

    module_name = "recorded_data_analysis_ui"
    if module_name in sys.modules:
        _DATA_ANALYSIS_MODULE = sys.modules[module_name]
        return _DATA_ANALYSIS_MODULE

    module_path = Path(__file__).resolve().parent / "recorded_data" / "data_analysis_ui.py"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load data analysis module: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    _DATA_ANALYSIS_MODULE = mod
    return _DATA_ANALYSIS_MODULE


try:
    _rcs_ref_mod = _load_rcs_reference_module()
except Exception as _rcs_ref_err:
    _rcs_ref_mod = None
    print(f"[main_ui] Failed to load rcs reference module: {_rcs_ref_err}")

PRESET_PATHS_CSV = Path(__file__).with_name("preset_paths.csv")
SegmentRange = Tuple[int, int, int, bool, float, float, float]
DEFAULT_SEGMENT_SPEED_MPS = 0.5
DEFAULT_RADIAL_MEASUREMENT_SPEED_MPS = 0.7
DEFAULT_LINE_PLAN_DIST_M = 50.0
DEFAULT_LINE_PLAN_SPEED_MPS = 0.7
DEFAULT_ACCEL_DIST_M = 1.5
DEFAULT_DECEL_DIST_M = 1.5
MIN_SEGMENT_SPEED_MPS = 0.08
MIN_PROFILE_SPEED_MPS = 0.12
PATH_DUPLICATE_EPS_M = 1e-4
PATH_RESAMPLE_STEP_FLOOR_M = 0.03
PATH_RESAMPLE_STEP_MIN_M = 0.08
PATH_RESAMPLE_STEP_MAX_M = 0.25
PATH_RESAMPLE_BOUNDARY_GAP_RATIO = 0.45
PATH_SMOOTH_WINDOW_SCALE = 2.4
PATH_SMOOTH_WINDOW_MIN_M = 0.16
PATH_SMOOTH_WINDOW_MAX_M = 0.42
PATH_SMOOTH_BLEND = 0.55
PATH_SMOOTH_CORNER_LIMIT_DEG = 35.0
PATH_SMOOTH_MAX_SHIFT_SCALE = 0.55
PATH_SMOOTH_MAX_SHIFT_MIN_M = 0.015
PATH_SMOOTH_MAX_SHIFT_MAX_M = 0.05
RADAR_TARGET_FRESH_S = 0.35
RADAR_EMERGENCY_STOP_DEFAULT_ENABLED = False
STRAIGHT_SEGMENT_POINT_COUNT = 2
RCS_MAX_DISTANCE_M = 60.0
# 直线距离-RCS：绘图与拟合有效距离门（与 Radar Signal Processing 中 RCS_FILTER_X_* 一致）
RCS_STRAIGHT_X_MIN_M = 4.0
RCS_STRAIGHT_X_MAX_M = 60.0
RCS_STRAIGHT_EMA_ALPHA = 0.5  # 0~1，越小越平滑（建议 0.2~0.35）
RCS_FIT_GRID_STEP_M = 0.1
RCS_FIT_LINE_WIDTH = 1.8
RCS_EXPORT_DPI = 320
RADAR_TARGET_CHECK_TITLE = (
    "雷达 Cluster 检查图（0x701）：直线默认 前方 4–60m、左右 ±3m；"
    "圆周 RCS 采集进行中为 ±10m"
)
ACTION_CLEAR_LOG_TEXT = "清空日志"
ACTION_SAVE_LOG_TEXT = "保存日志"
ACTION_EXPORT_CSV_TEXT = "导出 CSV"
ACTION_SAVE_MOTION_DATA_TEXT = "保存运动数据"
ACTION_EMERGENCY_STOP_TEXT = "紧急停止"
PATH_COORD_MODE_FIXED_ORIGIN = "fixed_origin"
# 历史预设 CSV 中可能出现，导入时按「无固定原点」处理
_LEGACY_PATH_COORD_MODE_CURRENT_POSE = "current_pose"
# 轨迹锚点：用户在校准平面系下手动输入的 (x,y)，与 get_robot_pose() 同系
PATH_ORIGIN_MANUAL_KEY = "__manual_xy_anchor__"
# 旧版预设/会话中可能出现，解析时与手动锚点同等对待（坐标以帧内数值为准）
PATH_ORIGIN_CALIB_PLANE_KEY = "__calib_plane_origin__"
DATA_SAVE_ROOT_DIR_NAME = "recorded_data"
RCS_DATA_DIR_NAME = "rcs_data"
class _ClusterDisplayTarget:
    """Cluster 0x701：散点图按「目标物」合并后的一点（oid 为簇 ClusterID 或分组代表序号）。"""

    __slots__ = ("oid", "x", "y", "rcs_db", "t")

    def __init__(self, idx: int, dx: float, dy: float, rcs: float, ts: float) -> None:
        self.oid = int(idx)
        self.x = float(dx)
        self.y = float(dy)
        self.rcs_db = float(rcs)
        self.t = float(ts)

    def xy_raw(self) -> Tuple[float, float]:
        return self.x, self.y


class _ClusterSafetyProxy:
    """急停逻辑仅需车体前向距离 x（此处 x=DX）。"""

    __slots__ = ("x", "y")

    def __init__(self, c: Dict[str, Any]) -> None:
        self.x = float(c["DX"])
        self.y = float(c["DY"])


# 圆周段 RCS：以前方约 40 m 处目标为采样对象；落盘仅 Cluster Raw CSV，极坐标图为离线/预览用
ORBIT_RCS_NOMINAL_FORWARD_M = 40.0
# 已锁定且已有采样点后：|DX−40|≤FORWARD_GATE_M 才写入/关联
ORBIT_RCS_FORWARD_GATE_M = 8.0
# 圆周采集尚未写入任何点时：用更宽半宽便于首个目标进门（见 _orbit_rcs_effective_forward_gate_m）
ORBIT_RCS_FORWARD_GATE_RELAXED_M = 16.0
# 圆周 RCS 极坐标图：径向网格每格固定 5 dBsm；相邻时间点跳变达到该阈值视为异常点剔除
ORBIT_RCS_RADIAL_TICK_STEP_DB = 5.0
ORBIT_RCS_ADJACENT_JUMP_REJECT_DB = 15.0
# 圆周 RCS：时间平铺到 2π 后按角度分箱，箱内 RCS 取算术平均后连成拟合曲线
ORBIT_RCS_ANGLE_BIN_DEG = 1.0
# 同一帧内相距不大于该值的簇合并为同一目标散点（与相同 Cluster_ID 合并一致采用）
RCS_DISPLAY_TARGET_MERGE_RADIUS_M = 2.5
FORWARD_STRAIGHT_RCS_TARGET_NAME = "前进直线段"
MOTION_DATA_DIR_NAME = "motion_data"
VEHICLE_BODY_SIZE_M = 0.5
VEHICLE_HEADING_GAP_M = 0.06
VEHICLE_HEADING_SHAFT_M = 0.22
VEHICLE_HEADING_HEAD_LEN_M = 0.10
VEHICLE_HEADING_HEAD_HALF_WIDTH_M = 0.07
VIRTUAL_PREVIEW_POSE_SOURCE = "VIRTUAL_PREVIEW"
# 14" 1920×1080 设计基准：可用区与基准一致时缩放系数为 1.0，刚好铺满工作区
UI_LAYOUT_BASE_WIDTH = 1920
UI_LAYOUT_BASE_HEIGHT = 1080
UI_SCALE_MIN = 0.60
# 纵向布局压缩（约缩短 1/3），减轻全屏时总高度溢出
UI_VERTICAL_LAYOUT_FACTOR = 2.0 / 3.0


def _layout_compact_v(ui_scale: float, design_px: int) -> int:
    return max(1, int(round(float(design_px) * UI_VERTICAL_LAYOUT_FACTOR * float(ui_scale))))


@dataclass
class LoadedRcsCurve:
    file_path: str
    display_name: str
    segments: List[List[CurvePoint]]
    fitted: Tuple[np.ndarray, np.ndarray]
    point_count: int
    # 与 segments 一一对应：CSV 含 SegIdx 列时按分段升序编号；否则为单段 [1]
    segment_run_labels: Optional[List[int]] = None


@dataclass
class AggregatedRcsFile:
    trajectory_name: str
    target_name: str
    file_path: str
    segments: List[List[CurvePoint]]


@dataclass
class RadialMeasurementSpec:
    angle_cycles: List[Tuple[int, int]]
    # 距离目标点最近距离（m）
    inner_radius_m: float
    # 固定直线长度（m）
    line_length_m: float
    speed_mps: float
    accel_dist_m: float
    decel_dist_m: float


@dataclass
class RadialMeasurementPlan:
    task_names: List[str]
    local_points: List[Tuple[float, float]]
    ranges: List[SegmentRange]
    range_task_names: List[Optional[str]]
    transition_count: int


@dataclass
class PathReferenceFrame:
    mode: str = PATH_COORD_MODE_FIXED_ORIGIN
    origin_key: str = ""
    origin_label: str = ""
    origin_x_m: float = 0.0
    origin_y_m: float = 0.0
    origin_z_m: float = 0.0

    def is_fixed_origin(self) -> bool:
        return bool(str(self.origin_key or "").strip())


def _apply_matplotlib_font() -> None:
    matplotlib.rcParams["axes.unicode_minus"] = False
    preferred: List[str] = []
    family = _resolve_cjk_font_family()
    if family:
        preferred.append(family)
    for name in _CJK_FONT_CANDIDATES:
        if name not in preferred:
            preferred.append(name)
    # Prefer a concrete CJK-capable font when available; otherwise fallback to a list.
    # This avoids "square boxes/garbled" legend text when the default font lacks CJK glyphs.
    if family:
        matplotlib.rcParams["font.family"] = family
    else:
        matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = preferred
    matplotlib.rcParams["font.serif"] = preferred
    matplotlib.rcParams["font.monospace"] = preferred


class BatteryIndicator(QtWidgets.QWidget):
    def __init__(
        self,
        title: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._percent: Optional[float] = None
        self._detail = "--"
        self._status = "unavailable"
        self._visual_scale = 1.0
        self.setMinimumSize(140, 70)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.setToolTip(f"{self._title}: --")

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(
            max(118, int(round(150 * self._visual_scale))),
            max(56, int(round(68 * self._visual_scale))),
        )

    def set_visual_scale(self, scale: float) -> None:
        self._visual_scale = max(UI_SCALE_MIN, min(1.0, float(scale)))
        self.setMinimumSize(
            max(108, int(round(145 * self._visual_scale))),
            max(52, int(round(66 * self._visual_scale))),
        )
        self.updateGeometry()
        self.update()

    def set_status(
        self,
        percent: Optional[float],
        detail: str,
        status: str = "ok",
        tooltip: Optional[str] = None,
    ) -> None:
        self._percent = None if percent is None else max(0.0, min(100.0, float(percent)))
        self._detail = detail or "--"
        self._status = status or "ok"
        self.setToolTip(tooltip or f"{self._title}: {self._detail}")
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        outer = QtCore.QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(255, 255, 255, 220))
        painter.drawRoundedRect(outer, 11, 11)

        outline_color = QtGui.QColor("#2E7D32")
        background_color = QtGui.QColor("#F1F8E9")
        detail_color = QtGui.QColor("#33691E")
        fill_color = QtGui.QColor("#A5D6A7")
        text_color = QtGui.QColor("#1B5E20")

        if self._percent is not None:
            if self._percent >= 65.0:
                fill_color = QtGui.QColor("#2E7D32")
                text_color = QtGui.QColor("white")
            elif self._percent >= 30.0:
                fill_color = QtGui.QColor("#43A047")
                text_color = QtGui.QColor("white")
            else:
                fill_color = QtGui.QColor("#66BB6A")
        elif self._status in {"no_feedback", "stale", "read_failed", "read_error", "invalid"}:
            outline_color = QtGui.QColor("#558B2F")
            background_color = QtGui.QColor("#F9FBE7")
            fill_color = QtGui.QColor(197, 225, 165, 120)

        title_rect = QtCore.QRectF(outer.left() + 10, outer.top() + 6, outer.width() - 20, 16)
        title_font = QtGui.QFont(painter.font())
        title_font.setBold(True)
        title_font.setPointSize(max(7, int(round(9 * self._visual_scale))))
        painter.setFont(title_font)
        painter.setPen(detail_color)
        painter.drawText(title_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self._title)

        body_rect = QtCore.QRectF(outer.left() + 10, outer.top() + 26, outer.width() - 26, 28)
        tip_rect = QtCore.QRectF(body_rect.right() + 2, body_rect.top() + 7, 7, body_rect.height() - 14)

        painter.setPen(QtGui.QPen(outline_color, 2))
        painter.setBrush(background_color)
        painter.drawRoundedRect(body_rect, 6, 6)
        painter.drawRoundedRect(tip_rect, 2.5, 2.5)

        inner_rect = body_rect.adjusted(4, 4, -4, -4)
        if self._percent is not None and inner_rect.width() > 0 and self._percent > 0.0:
            fill_width = inner_rect.width() * (self._percent / 100.0)
            if fill_width > 0.0:
                fill_width = max(6.0, fill_width)
                fill_width = min(inner_rect.width(), fill_width)
                fill_rect = QtCore.QRectF(
                    inner_rect.left(),
                    inner_rect.top(),
                    fill_width,
                    inner_rect.height(),
                )
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(fill_color)
                painter.drawRoundedRect(fill_rect, 4, 4)

        value_text = "--" if self._percent is None else f"{self._percent:.0f}%"
        value_font = QtGui.QFont(painter.font())
        value_font.setBold(True)
        value_font.setPointSize(max(8, int(round(10 * self._visual_scale))))
        painter.setFont(value_font)
        painter.setPen(text_color)
        painter.drawText(body_rect, QtCore.Qt.AlignCenter, value_text)

        detail_rect = QtCore.QRectF(outer.left() + 10, outer.bottom() - 20, outer.width() - 20, 14)
        detail_font = QtGui.QFont(painter.font())
        detail_font.setPointSize(max(7, int(round(8 * self._visual_scale))))
        painter.setFont(detail_font)
        fm = QtGui.QFontMetrics(detail_font)
        detail_text = fm.elidedText(self._detail, QtCore.Qt.ElideRight, int(detail_rect.width()))
        painter.setPen(detail_color)
        painter.drawText(detail_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, detail_text)


class TrajectoryPlannerDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        default_dist: float = DEFAULT_LINE_PLAN_DIST_M,
        default_radius: float = 40.0,
        default_angle: float = 360.0,
        default_circle_direction: str = "ccw",
        default_line_speed: float = DEFAULT_LINE_PLAN_SPEED_MPS,
        default_circle_speed: float = DEFAULT_SEGMENT_SPEED_MPS,
        default_accel_dist: float = DEFAULT_ACCEL_DIST_M,
        default_decel_dist: float = DEFAULT_DECEL_DIST_M,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("轨迹规划")
        self.setMinimumWidth(700)

        self._segments: List[dict] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        hint = QtWidgets.QLabel(
            "路径在校准平面系中生成：右手系 +X 朝右、+Y 朝前；航向逆时针（CCW）为正，角速度 w>0 为左转。"
            "直线沿全局 +Y（前进为正、倒车为负）。跟踪时车在路径右侧则横向误差为正，控制器应产生正角速度贴回路径。"
            "圆弧圆心由「当前路径切向」与半径、顺/逆时针决定（与底盘 move_circle / Stanley 一致），"
            "沿当前运动方向左/右转弯，不以坐标原点为圆心。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["序号", "类型", "参数", "RCS"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self.table, 3)

        segment_box = QtWidgets.QHBoxLayout()

        line_group = QtWidgets.QGroupBox("直线段")
        line_form = QtWidgets.QFormLayout(line_group)
        self.line_dist = QtWidgets.QDoubleSpinBox()
        self.line_dist.setRange(0.1, 200.0)
        self.line_dist.setDecimals(2)
        self.line_dist.setValue(default_dist)
        self.line_dist.setSuffix(" m")
        self.line_speed = QtWidgets.QDoubleSpinBox()
        self.line_speed.setRange(0.05, 3.0)
        self.line_speed.setDecimals(2)
        self.line_speed.setValue(default_line_speed)
        self.line_speed.setSuffix(" m/s")
        self.line_accel_dist = QtWidgets.QDoubleSpinBox()
        self.line_accel_dist.setRange(0.0, 50.0)
        self.line_accel_dist.setDecimals(2)
        self.line_accel_dist.setValue(default_accel_dist)
        self.line_accel_dist.setSuffix(" m")
        self.line_decel_dist = QtWidgets.QDoubleSpinBox()
        self.line_decel_dist.setRange(0.0, 50.0)
        self.line_decel_dist.setDecimals(2)
        self.line_decel_dist.setValue(default_decel_dist)
        self.line_decel_dist.setSuffix(" m")
        self.line_round = QtWidgets.QCheckBox("往返")
        self.line_dir_forward = QtWidgets.QRadioButton("前进")
        self.line_dir_reverse = QtWidgets.QRadioButton("倒车")
        self.line_dir_forward.setChecked(True)
        self.line_round_count = QtWidgets.QSpinBox()
        self.line_round_count.setRange(1, 100)
        self.line_round_count.setValue(1)
        self.line_round_count.setSuffix(" 次")
        self.line_round_count.setEnabled(False)
        self.line_round.toggled.connect(self.line_round_count.setEnabled)
        line_round_row = QtWidgets.QWidget()
        line_round_row_layout = QtWidgets.QHBoxLayout(line_round_row)
        line_round_row_layout.setContentsMargins(0, 0, 0, 0)
        line_round_row_layout.setSpacing(8)
        line_round_row_layout.addWidget(self.line_round)
        line_round_row_layout.addWidget(self.line_dir_forward)
        line_round_row_layout.addWidget(self.line_dir_reverse)
        line_round_row_layout.addStretch(1)
        self.line_rcs_start = QtWidgets.QCheckBox("本段触发采集RCS（Cluster 0x701）")
        self.btn_add_line = QtWidgets.QPushButton("添加直线段")
        self.btn_add_line.clicked.connect(self._add_line_segment)
        line_form.addRow("距离", self.line_dist)
        line_form.addRow("速度", self.line_speed)
        line_form.addRow("加速段", self.line_accel_dist)
        line_form.addRow("减速段", self.line_decel_dist)
        line_form.addRow("往返", line_round_row)
        line_form.addRow("次数", self.line_round_count)
        line_form.addRow("RCS(Cluster)", self.line_rcs_start)
        line_form.addRow(self.btn_add_line)

        circle_group = QtWidgets.QGroupBox("圆弧段")
        circle_form = QtWidgets.QFormLayout(circle_group)
        self.circle_radius = QtWidgets.QDoubleSpinBox()
        self.circle_radius.setRange(0.1, 200.0)
        self.circle_radius.setDecimals(2)
        self.circle_radius.setValue(default_radius)
        self.circle_radius.setSuffix(" m")
        self.circle_angle = QtWidgets.QDoubleSpinBox()
        self.circle_angle.setRange(1.0, 3600.0)
        self.circle_angle.setDecimals(1)
        self.circle_angle.setValue(default_angle)
        self.circle_angle.setSuffix(" °")
        self.circle_speed = QtWidgets.QDoubleSpinBox()
        self.circle_speed.setRange(0.05, 3.0)
        self.circle_speed.setDecimals(2)
        self.circle_speed.setValue(default_circle_speed)
        self.circle_speed.setSuffix(" m/s")
        self.circle_accel_dist = QtWidgets.QDoubleSpinBox()
        self.circle_accel_dist.setRange(0.0, 50.0)
        self.circle_accel_dist.setDecimals(2)
        self.circle_accel_dist.setValue(default_accel_dist)
        self.circle_accel_dist.setSuffix(" m")
        self.circle_decel_dist = QtWidgets.QDoubleSpinBox()
        self.circle_decel_dist.setRange(0.0, 50.0)
        self.circle_decel_dist.setDecimals(2)
        self.circle_decel_dist.setValue(default_decel_dist)
        self.circle_decel_dist.setSuffix(" m")
        self.circle_dir = QtWidgets.QComboBox()
        self.circle_dir.addItems(["逆时针", "顺时针"])
        self.circle_dir.setCurrentIndex(1 if str(default_circle_direction).strip().lower() == "cw" else 0)
        self.circle_rcs_start = QtWidgets.QCheckBox("本段触发采集RCS（Cluster）")
        self.btn_add_circle = QtWidgets.QPushButton("添加圆弧段")
        self.btn_add_circle.clicked.connect(self._add_circle_segment)
        circle_form.addRow("半径", self.circle_radius)
        circle_form.addRow("角度", self.circle_angle)
        circle_form.addRow("速度", self.circle_speed)
        circle_form.addRow("加速段", self.circle_accel_dist)
        circle_form.addRow("减速段", self.circle_decel_dist)
        circle_form.addRow("方向", self.circle_dir)
        circle_form.addRow("RCS(Cluster)", self.circle_rcs_start)
        circle_form.addRow(self.btn_add_circle)

        segment_box.addWidget(line_group, 1)
        segment_box.addWidget(circle_group, 1)
        layout.addLayout(segment_box)

        ops = QtWidgets.QHBoxLayout()
        self.btn_remove = QtWidgets.QPushButton("移除选中")
        self.btn_clear = QtWidgets.QPushButton("清空")
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(self._clear_segments)
        ops.addWidget(self.btn_remove)
        ops.addWidget(self.btn_clear)
        ops.addStretch(1)
        layout.addLayout(ops)

        action_box = QtWidgets.QHBoxLayout()
        self.btn_apply = QtWidgets.QPushButton("应用并返回")
        self.btn_cancel = QtWidgets.QPushButton("取消")
        self.btn_apply.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        action_box.addStretch(1)
        action_box.addWidget(self.btn_apply)
        action_box.addWidget(self.btn_cancel)
        layout.addLayout(action_box)

    def _add_line_segment(self) -> None:
        dist = float(self.line_dist.value())
        round_trip = self.line_round.isChecked()
        round_trip_count = int(self.line_round_count.value()) if round_trip else 1
        direction_sign = 1 if self.line_dir_forward.isChecked() else -1
        self._segments.append(
            {
                "type": "line",
                "distance": dist,
                "speed_mps": float(self.line_speed.value()),
                "accel_dist": float(self.line_accel_dist.value()),
                "decel_dist": float(self.line_decel_dist.value()),
                "round_trip": round_trip,
                "round_trip_count": round_trip_count,
                "direction_sign": direction_sign,
                "rcs_start": self.line_rcs_start.isChecked(),
            }
        )
        self._refresh_table()

    def _add_circle_segment(self) -> None:
        radius = float(self.circle_radius.value())
        angle = float(self.circle_angle.value())
        direction = "ccw" if self.circle_dir.currentIndex() == 0 else "cw"
        self._segments.append(
            {
                "type": "circle",
                "radius": radius,
                "angle": angle,
                "direction": direction,
                "speed_mps": float(self.circle_speed.value()),
                "accel_dist": float(self.circle_accel_dist.value()),
                "decel_dist": float(self.circle_decel_dist.value()),
                "rcs_start": self.circle_rcs_start.isChecked(),
            }
        )
        self._refresh_table()

    def _remove_selected(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return
        rows = sorted((idx.row() for idx in selected), reverse=True)
        for row in rows:
            if 0 <= row < len(self._segments):
                del self._segments[row]
        self._refresh_table()

    def _clear_segments(self) -> None:
        self._segments.clear()
        self._refresh_table()

    def _refresh_table(self) -> None:
        self.table.setRowCount(len(self._segments))
        for i, seg in enumerate(self._segments):
            type_text = "直线" if seg["type"] == "line" else "圆弧"
            if seg["type"] == "line":
                dir_text = "前进" if int(seg.get("direction_sign", 1)) >= 0 else "倒车"
                params = (
                    f"沿+Y 距离={seg['distance']:.2f}m v={float(seg.get('speed_mps', DEFAULT_SEGMENT_SPEED_MPS)):.2f}m/s "
                    f"加={float(seg.get('accel_dist', DEFAULT_ACCEL_DIST_M)):.2f}m "
                    f"减={float(seg.get('decel_dist', DEFAULT_DECEL_DIST_M)):.2f}m"
                )
                if seg.get("round_trip"):
                    count = max(1, int(seg.get("round_trip_count", 1)))
                    params += f" 往返x{count}(先{dir_text})"
                else:
                    params += f" {dir_text}"
            else:
                direction = "逆时针" if seg.get("direction") == "ccw" else "顺时针"
                params = (
                    f"切向圆弧 R={seg['radius']:.2f}m θ={seg['angle']:.1f}° {direction} "
                    f"v={float(seg.get('speed_mps', DEFAULT_SEGMENT_SPEED_MPS)):.2f}m/s "
                    f"加={float(seg.get('accel_dist', DEFAULT_ACCEL_DIST_M)):.2f}m "
                    f"减={float(seg.get('decel_dist', DEFAULT_DECEL_DIST_M)):.2f}m"
                )
            rcs_text = "触发Cluster RCS" if seg.get("rcs_start") else "--"
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(type_text))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(params))
            self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(rcs_text))

    def get_segments(self) -> List[dict]:
        return list(self._segments)


class RadialMeasurementDialog(QtWidgets.QDialog):
    # 0°, 30°, …, 330°（每隔 30° 一个方向）
    _ANGLE_OPTIONS = list(range(0, 360, 30))

    def __init__(
        self,
        target_point: Tuple[float, float],
        default_angle_cycles: Optional[Dict[int, int]] = None,
        default_inner_radius: float = 4.0,
        default_line_length_m: float = 50.0,
        default_speed: float = DEFAULT_RADIAL_MEASUREMENT_SPEED_MPS,
        default_accel_dist: float = DEFAULT_ACCEL_DIST_M,
        default_decel_dist: float = DEFAULT_DECEL_DIST_M,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("星型测量")
        self.setMinimumWidth(680)

        defaults = {
            int(angle) % 360: max(1, int(count))
            for angle, count in (default_angle_cycles or {}).items()
        }
        if not defaults:
            defaults = {angle: 1 for angle in self._ANGLE_OPTIONS}
        target_x = float(target_point[0])
        target_y = float(target_point[1])
        self._angle_checks: Dict[int, QtWidgets.QCheckBox] = {}
        self._angle_cycle_spins: Dict[int, QtWidgets.QSpinBox] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        intro = QtWidgets.QLabel(
            "围绕已标定目标物生成每隔30°的一组直线测量轨迹。"
            "0° 以轨迹原点指向目标物的方向为基准（右侧为正30°），每个角度都可以单独设置往返测量次数。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        target_label = QtWidgets.QLabel(
            f"当前目标物坐标: x={target_x:.2f} m, y={target_y:.2f} m"
        )
        target_label.setStyleSheet("color: #455A64;")
        layout.addWidget(target_label)

        angle_group = QtWidgets.QGroupBox("测量角度与往返次数")
        angle_grid = QtWidgets.QGridLayout(angle_group)
        angle_grid.setContentsMargins(10, 10, 10, 10)
        angle_grid.setHorizontalSpacing(16)
        angle_grid.setVerticalSpacing(8)
        angle_grid.addWidget(QtWidgets.QLabel("启用"), 0, 0)
        angle_grid.addWidget(QtWidgets.QLabel("角度"), 0, 1)
        angle_grid.addWidget(QtWidgets.QLabel("往返次数"), 0, 2)
        angle_grid.addWidget(QtWidgets.QLabel("启用"), 0, 3)
        angle_grid.addWidget(QtWidgets.QLabel("角度"), 0, 4)
        angle_grid.addWidget(QtWidgets.QLabel("往返次数"), 0, 5)
        for index, angle in enumerate(self._ANGLE_OPTIONS):
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(angle in defaults)
            spin = QtWidgets.QSpinBox()
            spin.setRange(1, 20)
            spin.setValue(int(defaults.get(angle, 1)))
            spin.setSuffix(" 次")
            spin.setEnabled(checkbox.isChecked())
            checkbox.toggled.connect(spin.setEnabled)
            self._angle_checks[angle] = checkbox
            self._angle_cycle_spins[angle] = spin
            row = 1 + (index % 6)
            col_base = 0 if index < 6 else 3
            angle_label = QtWidgets.QLabel(f"{angle}°")
            angle_grid.addWidget(checkbox, row, col_base + 0)
            angle_grid.addWidget(angle_label, row, col_base + 1)
            angle_grid.addWidget(spin, row, col_base + 2)
        layout.addWidget(angle_group)

        angle_ops = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("全选")
        btn_clear = QtWidgets.QPushButton("清空")
        btn_all.clicked.connect(self._select_all_angles)
        btn_clear.clicked.connect(self._clear_all_angles)
        angle_ops.addWidget(btn_all)
        angle_ops.addWidget(btn_clear)
        angle_ops.addStretch(1)
        layout.addLayout(angle_ops)

        form = QtWidgets.QFormLayout()
        self.inner_radius = QtWidgets.QDoubleSpinBox()
        self.inner_radius.setRange(0.2, 200.0)
        self.inner_radius.setDecimals(2)
        self.inner_radius.setValue(default_inner_radius)
        self.inner_radius.setSuffix(" m")

        self.line_length = QtWidgets.QDoubleSpinBox()
        self.line_length.setRange(0.5, 300.0)
        self.line_length.setDecimals(2)
        self.line_length.setValue(float(default_line_length_m))
        self.line_length.setSuffix(" m")

        self.speed_spin = QtWidgets.QDoubleSpinBox()
        self.speed_spin.setRange(MIN_SEGMENT_SPEED_MPS, 3.0)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setValue(default_speed)
        self.speed_spin.setSuffix(" m/s")

        self.accel_dist_spin = QtWidgets.QDoubleSpinBox()
        self.accel_dist_spin.setRange(0.0, 50.0)
        self.accel_dist_spin.setDecimals(2)
        self.accel_dist_spin.setValue(default_accel_dist)
        self.accel_dist_spin.setSuffix(" m")

        self.decel_dist_spin = QtWidgets.QDoubleSpinBox()
        self.decel_dist_spin.setRange(0.0, 50.0)
        self.decel_dist_spin.setDecimals(2)
        self.decel_dist_spin.setValue(default_decel_dist)
        self.decel_dist_spin.setSuffix(" m")

        form.addRow("距离目标最近距离", self.inner_radius)
        form.addRow("固定直线长度", self.line_length)
        form.addRow("巡航速度", self.speed_spin)
        form.addRow("加速段", self.accel_dist_spin)
        form.addRow("减速段", self.decel_dist_spin)
        layout.addLayout(form)

        hint = QtWidgets.QLabel(
            "说明: 0° 为轨迹原点(锚点)到目标物的方向，角度按“向右(顺时针)为正”增加。"
            "系统会自动把不同角度的测量轨迹放在目标外侧进行平滑连接，尽量减少原地大角度掉头。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #546E7A;")
        layout.addWidget(hint)

        btns = QtWidgets.QHBoxLayout()
        btn_ok = QtWidgets.QPushButton("生成预览")
        btn_cancel = QtWidgets.QPushButton("取消")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btns.addStretch(1)
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)

    def _select_all_angles(self) -> None:
        for checkbox in self._angle_checks.values():
            checkbox.setChecked(True)

    def _clear_all_angles(self) -> None:
        for checkbox in self._angle_checks.values():
            checkbox.setChecked(False)

    def values(self) -> RadialMeasurementSpec:
        angle_cycles = [
            (angle, int(self._angle_cycle_spins[angle].value()))
            for angle in self._ANGLE_OPTIONS
            if self._angle_checks[angle].isChecked()
        ]
        inner = float(self.inner_radius.value())
        length = max(0.5, float(self.line_length.value()))
        return RadialMeasurementSpec(
            angle_cycles=angle_cycles,
            inner_radius_m=inner,
            line_length_m=length,
            speed_mps=float(self.speed_spin.value()),
            accel_dist_m=float(self.accel_dist_spin.value()),
            decel_dist_m=float(self.decel_dist_spin.value()),
        )


class PresetSequenceDialog(QtWidgets.QDialog):
    def __init__(
        self,
        preset_names: List[str],
        checked_names: Optional[List[str]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("预设轨迹序列")
        self.setMinimumWidth(460)
        self._import_requested = False
        checked = set(checked_names or [])

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        hint = QtWidgets.QLabel("勾选需要执行的轨迹，按列表顺序自动串联并规划过渡段。")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        for name in preset_names:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(
                item.flags()
                | QtCore.Qt.ItemIsEnabled
                | QtCore.Qt.ItemIsUserCheckable
            )
            item.setCheckState(QtCore.Qt.Checked if name in checked else QtCore.Qt.Unchecked)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget, 1)

        ops = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("全选")
        btn_clear = QtWidgets.QPushButton("清空")
        btn_import = QtWidgets.QPushButton("导入文件")
        btn_all.clicked.connect(self._select_all)
        btn_clear.clicked.connect(self._clear_all)
        btn_import.clicked.connect(self._request_import)
        ops.addWidget(btn_all)
        ops.addWidget(btn_clear)
        ops.addStretch(1)
        ops.addWidget(btn_import)
        layout.addLayout(ops)

        actions = QtWidgets.QHBoxLayout()
        btn_apply = QtWidgets.QPushButton("加载选中")
        btn_cancel = QtWidgets.QPushButton("取消")
        btn_apply.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        actions.addStretch(1)
        actions.addWidget(btn_apply)
        actions.addWidget(btn_cancel)
        layout.addLayout(actions)

    def _select_all(self) -> None:
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item is not None:
                item.setCheckState(QtCore.Qt.Checked)

    def _clear_all(self) -> None:
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item is not None:
                item.setCheckState(QtCore.Qt.Unchecked)

    def _request_import(self) -> None:
        self._import_requested = True
        self.accept()

    def import_requested(self) -> bool:
        return self._import_requested

    def checked_names(self) -> List[str]:
        names: List[str] = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item is not None and item.checkState() == QtCore.Qt.Checked:
                names.append(item.text())
        return names


class RcsReferenceConfigDialog(QtWidgets.QDialog):
    def __init__(
        self,
        class_options: List[str],
        angle_options: List[str],
        class_labels: Optional[Dict[str, str]] = None,
        default_class: Optional[str] = None,
        default_angle: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("选择RCS参考产品")
        self.setMinimumWidth(460)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        intro_label = QtWidgets.QLabel(
            "请先选择参考产品类型和角度。后续导入的 RCS 数据绘图和对比都会使用这组参考上下限。"
        )
        intro_label.setWordWrap(True)
        layout.addWidget(intro_label)

        form = QtWidgets.QFormLayout()
        self.product_combo = QtWidgets.QComboBox()
        self.product_combo.addItem("请选择参考产品", "")
        labels = dict(class_labels or {})
        for name in class_options:
            label = str(labels.get(name, name)).strip()
            display = f"{label} ({name})" if label and label != name else name
            self.product_combo.addItem(display, name)

        self.angle_combo = QtWidgets.QComboBox()
        self.angle_combo.addItem("请选择角度", "")
        for ang in angle_options:
            self.angle_combo.addItem(ang, ang)

        if default_class:
            idx = self.product_combo.findData(default_class)
            if idx >= 0:
                self.product_combo.setCurrentIndex(idx)
        if default_angle:
            idx = self.angle_combo.findData(default_angle)
            if idx >= 0:
                self.angle_combo.setCurrentIndex(idx)

        form.addRow("产品类型", self.product_combo)
        form.addRow("角度", self.angle_combo)
        layout.addLayout(form)

        btns = QtWidgets.QHBoxLayout()
        btn_ok = QtWidgets.QPushButton("确定")
        btn_cancel = QtWidgets.QPushButton("取消")
        btn_ok.clicked.connect(self._accept_selection)
        btn_cancel.clicked.connect(self.reject)
        btns.addStretch(1)
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)

    def _accept_selection(self) -> None:
        if not self.values()[0]:
            QtWidgets.QMessageBox.information(self, "未选择产品", "请先选择参考产品类型。")
            return
        if not self.values()[1]:
            QtWidgets.QMessageBox.information(self, "未选择角度", "请先选择参考角度。")
            return
        self.accept()

    def values(self) -> Tuple[Optional[str], Optional[str]]:
        return (
            str(self.product_combo.currentData() or "").strip() or None,
            str(self.angle_combo.currentData() or "").strip() or None,
        )


class RcsPlotConfigDialog(QtWidgets.QDialog):
    def __init__(
        self,
        file_path: str,
        reference_summary: str,
        default_custom_name: Optional[str] = None,
        default_calibration_db: float = 0.0,
        enable_curve_csv_export: bool = True,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("RCS绘图选项")
        self.setMinimumWidth(460)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        file_label = QtWidgets.QLabel(f"数据文件: {Path(file_path).name}")
        file_label.setWordWrap(True)
        layout.addWidget(file_label)

        ref_label = QtWidgets.QLabel(f"当前参考: {reference_summary}")
        ref_label.setWordWrap(True)
        ref_label.setStyleSheet("color: #546E7A;")
        layout.addWidget(ref_label)

        form = QtWidgets.QFormLayout()
        self.custom_name_edit = QtWidgets.QLineEdit()
        self.custom_name_edit.setPlaceholderText("自定义绘图名称，可留空")
        self.custom_name_edit.setText(str(default_custom_name or "").strip())
        form.addRow("自定义名称", self.custom_name_edit)
        self.calibration_spin = QtWidgets.QDoubleSpinBox()
        self.calibration_spin.setRange(-80.0, 80.0)
        self.calibration_spin.setDecimals(2)
        self.calibration_spin.setSingleStep(0.5)
        self.calibration_spin.setSuffix(" dB")
        self.calibration_spin.setValue(float(default_calibration_db))
        self.calibration_spin.setToolTip(
            "叠加到本次绘图中的测量 RCS（纵轴与圆周色标）；不修改磁盘上的原始数据；参考上下限曲线不偏移。"
        )
        form.addRow("RCS标定值", self.calibration_spin)
        layout.addLayout(form)

        self.export_curve_csv_check = QtWidgets.QCheckBox(
            "同时生成 _Filtered.csv 和 *_combined.csv"
        )
        self.export_curve_csv_check.setToolTip(
            "仅用于距离-RCS Cluster Raw：按每帧 R 最近的两个簇生成 X/RCS 分箱 Filtered，再输出最终曲线 CSV。"
        )
        self.export_curve_csv_check.setChecked(bool(enable_curve_csv_export))
        self.export_curve_csv_check.setVisible(bool(enable_curve_csv_export))
        layout.addWidget(self.export_curve_csv_check)

        btns = QtWidgets.QHBoxLayout()
        btn_ok = QtWidgets.QPushButton("开始绘图")
        btn_cancel = QtWidgets.QPushButton("取消")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btns.addStretch(1)
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)

    def values(self) -> Tuple[Optional[str], float, bool]:
        custom_name = str(self.custom_name_edit.text() or "").strip()
        export_curve_csv = (
            bool(self.export_curve_csv_check.isChecked())
            if self.export_curve_csv_check.isVisible()
            else False
        )
        return (
            custom_name or None,
            float(self.calibration_spin.value()),
            export_curve_csv,
        )


class RcsCompareConfigDialog(QtWidgets.QDialog):
    def __init__(
        self,
        file_paths: List[str],
        reference_summary: str,
        default_display_names: List[str],
        default_calibration_db: float = 0.0,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._file_paths = list(file_paths)
        self.setWindowTitle("RCS对比绘图选项")
        self.setMinimumWidth(620)
        self.resize(720, 420)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        intro_label = QtWidgets.QLabel(
            f"已选择 {len(self._file_paths)} 个RCS数据文件，请确认每条对比曲线的显示名称。"
        )
        intro_label.setWordWrap(True)
        layout.addWidget(intro_label)

        hint_label = QtWidgets.QLabel(f"当前参考: {reference_summary}")
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #546E7A;")
        layout.addWidget(hint_label)

        form_cal = QtWidgets.QFormLayout()
        self.calibration_spin = QtWidgets.QDoubleSpinBox()
        self.calibration_spin.setRange(-80.0, 80.0)
        self.calibration_spin.setDecimals(2)
        self.calibration_spin.setSingleStep(0.5)
        self.calibration_spin.setSuffix(" dB")
        self.calibration_spin.setValue(float(default_calibration_db))
        self.calibration_spin.setToolTip(
            "叠加到对比图中各条测量曲线的 RCS；不修改原始文件；参考上下限不偏移。"
        )
        form_cal.addRow("RCS标定值", self.calibration_spin)
        layout.addLayout(form_cal)

        self.name_table = QtWidgets.QTableWidget(len(self._file_paths), 2)
        self.name_table.setHorizontalHeaderLabels(["数据文件", "对比名称"])
        self.name_table.verticalHeader().setVisible(False)
        self.name_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.name_table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
            | QtWidgets.QAbstractItemView.SelectedClicked
        )
        header = self.name_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)

        default_names = list(default_display_names)
        while len(default_names) < len(self._file_paths):
            default_names.append("")

        for row, file_path in enumerate(self._file_paths):
            file_item = QtWidgets.QTableWidgetItem(Path(file_path).name)
            file_item.setFlags(file_item.flags() & ~QtCore.Qt.ItemIsEditable)
            file_item.setToolTip(file_path)
            self.name_table.setItem(row, 0, file_item)

            display_name = str(default_names[row] or "").strip() or Path(file_path).stem
            name_item = QtWidgets.QTableWidgetItem(display_name)
            self.name_table.setItem(row, 1, name_item)

        layout.addWidget(self.name_table, 1)

        btns = QtWidgets.QHBoxLayout()
        btn_ok = QtWidgets.QPushButton("开始对比绘图")
        btn_cancel = QtWidgets.QPushButton("取消")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btns.addStretch(1)
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)

    def values(self) -> Tuple[List[str], float]:
        display_names: List[str] = []
        for row, file_path in enumerate(self._file_paths):
            item = self.name_table.item(row, 1)
            display_name = str(item.text() if item is not None else "").strip()
            display_names.append(display_name or Path(file_path).stem)

        return display_names, float(self.calibration_spin.value())


class EnuCalibrationDialog(QtWidgets.QDialog):
    _COL_INDEX = 0
    _COL_SYS_X = 1
    _COL_SYS_Y = 2
    _COL_SYS_Z = 3
    _COL_RTK_LAT = 4
    _COL_RTK_LON = 5
    _COL_RTK_H = 6

    def __init__(
        self,
        points: List[Dict[str, Any]],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._points: List[Dict[str, Any]] = [dict(point) for point in points]
        self._calibration_changed = False

        self.setWindowTitle("ENU坐标校准")
        self.setMinimumWidth(860)
        self.resize(940, 480)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        intro_label = QtWidgets.QLabel(
            "1. 把小车开到第一个控制点并点击“记录当前点”，系统会自动记录当前 GNSS 经纬度，高程仅做记录。 "
            "2. 开到第二个控制点后再次记录。 "
            "3. “两点方位校准”：用这两次采样时刻的平面坐标 (x,y) 定 Y 轴，并把当前车位设为原点。 "
            "4. “按表中经纬度定系”：直接用表格里两点的经纬度定系，原点为您选的点，另一点方向为 +Y；"
            "可双击编辑经纬度列（适合地图量测或外业坐标）。主界面轨迹锚点请直接输入校准平面系下的 x,y（米）。"
        )
        intro_label.setWordWrap(True)
        layout.addWidget(intro_label)

        self.label_summary = QtWidgets.QLabel()
        self.label_summary.setWordWrap(True)
        self.label_summary.setStyleSheet("color: #455A64;")
        layout.addWidget(self.label_summary)

        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            [
                "点号",
                "系统X(m)",
                "系统Y(m)",
                "系统Z(m,固定0)",
                "GNSS纬度",
                "GNSS经度",
                "GNSS高程(m,仅记录)",
            ]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(self._COL_INDEX, QtWidgets.QHeaderView.ResizeToContents)
        for column in (
            self._COL_SYS_X,
            self._COL_SYS_Y,
            self._COL_SYS_Z,
            self._COL_RTK_LAT,
            self._COL_RTK_LON,
            self._COL_RTK_H,
        ):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        geodetic_row = QtWidgets.QHBoxLayout()
        geodetic_row.addWidget(QtWidgets.QLabel("经纬度定系 — 原点与 +Y："))
        self.combo_geodetic_origin = QtWidgets.QComboBox()
        self.combo_geodetic_origin.addItem("以前一点为原点，+Y 指向后一点", "first_origin")
        self.combo_geodetic_origin.addItem("以后一点为原点，+Y 指向前一点", "second_origin")
        geodetic_row.addWidget(self.combo_geodetic_origin, 1)
        self.btn_apply_geodetic = QtWidgets.QPushButton("按表中经纬度定系")
        self.btn_apply_geodetic.clicked.connect(self._apply_geodetic_frame)
        geodetic_row.addWidget(self.btn_apply_geodetic)
        layout.addLayout(geodetic_row)

        button_row = QtWidgets.QHBoxLayout()
        self.btn_record = QtWidgets.QPushButton("记录当前点")
        self.btn_delete = QtWidgets.QPushButton("删除选中")
        self.btn_apply = QtWidgets.QPushButton("两点方位校准")
        self.btn_clear = QtWidgets.QPushButton("清除ENU校准")
        self.btn_close = QtWidgets.QPushButton("关闭")
        self.btn_record.clicked.connect(self._record_current_point)
        self.btn_delete.clicked.connect(self._delete_selected_point)
        self.btn_apply.clicked.connect(self._apply_heading_alignment)
        self.btn_clear.clicked.connect(self._clear_calibration)
        self.btn_close.clicked.connect(self.accept)
        button_row.addWidget(self.btn_record)
        button_row.addWidget(self.btn_delete)
        button_row.addStretch(1)
        button_row.addWidget(self.btn_apply)
        button_row.addWidget(self.btn_clear)
        button_row.addWidget(self.btn_close)
        layout.addLayout(button_row)

        hint_label = QtWidgets.QLabel(
            "说明: 系统按二维局部水平面处理。 “两点方位校准”以采样时的系统坐标定轴并以当前车位为原点；"
            "“按表中经纬度定系”以表格中最后两行的经纬度定轴，并以所选点为原点（与当前车位无关）。"
            "两种做法都会把连线方向对齐到平面坐标 +Y，并对 INS 航向叠加同一平面旋转。"
            "轨迹规划/跟踪在同一平面系下使用 (x,y)；航向由路径几何与惯导闭环得到。"
            "CSV 为局部轨迹；主界面「轨迹锚点 x,y」把局部原点映射到校准平面系。"
        )
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #607D8B;")
        layout.addWidget(hint_label)

        self._refresh_table()
        self._update_summary_label()

    @staticmethod
    def _read_only_item(text: str) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(text)
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        return item

    @staticmethod
    def _editable_item(text: str) -> QtWidgets.QTableWidgetItem:
        return QtWidgets.QTableWidgetItem(text)

    @staticmethod
    def _format_float(value: Any, digits: int = 3) -> str:
        return f"{float(value):.{digits}f}"

    @staticmethod
    def _wrap_angle_rad(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _log_to_parent(self, message: str) -> None:
        parent = self.parent()
        logger = getattr(parent, "_log", None)
        if callable(logger):
            logger(message)

    def _refresh_table(self) -> None:
        self.table.setRowCount(len(self._points))
        for row, point in enumerate(self._points):
            self.table.setItem(row, self._COL_INDEX, self._read_only_item(str(row + 1)))
            self.table.setItem(
                row,
                self._COL_SYS_X,
                self._read_only_item(self._format_float(point.get("sys_x", 0.0))),
            )
            self.table.setItem(
                row,
                self._COL_SYS_Y,
                self._read_only_item(self._format_float(point.get("sys_y", 0.0))),
            )
            self.table.setItem(
                row,
                self._COL_SYS_Z,
                self._read_only_item(self._format_float(point.get("sys_z", 0.0))),
            )
            self.table.setItem(
                row,
                self._COL_RTK_LAT,
                self._editable_item(str(point.get("rtk_lat", "") or "")),
            )
            self.table.setItem(
                row,
                self._COL_RTK_LON,
                self._editable_item(str(point.get("rtk_lon", "") or "")),
            )
            self.table.setItem(
                row,
                self._COL_RTK_H,
                self._editable_item(str(point.get("rtk_h", "") or "")),
            )

        if self._points:
            last_row = len(self._points) - 1
            self.table.selectRow(last_row)
            self.table.scrollToItem(self.table.item(last_row, self._COL_INDEX))

    def _sync_points_from_table(self) -> None:
        synced_points: List[Dict[str, Any]] = []
        for row in range(self.table.rowCount()):
            synced_points.append(
                {
                    "sys_x": float(self.table.item(row, self._COL_SYS_X).text()),
                    "sys_y": float(self.table.item(row, self._COL_SYS_Y).text()),
                    "sys_z": float(self.table.item(row, self._COL_SYS_Z).text()),
                    "rtk_lat": str(self.table.item(row, self._COL_RTK_LAT).text()).strip(),
                    "rtk_lon": str(self.table.item(row, self._COL_RTK_LON).text()).strip(),
                    "rtk_h": str(self.table.item(row, self._COL_RTK_H).text()).strip(),
                }
            )
        self._points = synced_points

    def _update_summary_label(self) -> None:
        summary = get_enu_calibration_summary()
        if summary.enabled:
            rms_text = "--" if summary.rms_error_m is None else f"{summary.rms_error_m:.3f} m"
            text = (
                f"当前 ENU 校准: 已启用 | 点数={summary.point_count} | "
                f"rot={summary.rotation_deg:.4f}° | tx={summary.translation_x_m:.3f} m | "
                f"ty={summary.translation_y_m:.3f} m | rms={rms_text}"
            )
        else:
            text = "当前 ENU 校准: 未启用"
        self.label_summary.setText(text)

    def _record_current_point(self) -> None:
        pose = get_absolute_robot_pose()
        if pose is None:
            QtWidgets.QMessageBox.warning(self, "位姿无效", "当前无法获取实时 GNSS/INS 位姿")
            return

        point_index = len(self._points) + 1
        self._points.append(
            {
                "sys_x": float(pose.x),
                "sys_y": float(pose.y),
                "sys_z": float(pose.z),
                "rtk_lat": f"{float(pose.lat):.8f}",
                "rtk_lon": f"{float(pose.lon):.8f}",
                "rtk_h": f"{float(pose.height):.3f}",
            }
        )
        self._refresh_table()
        self._log_to_parent(
            f"ENU控制点{point_index}已记录: "
            f"x={pose.x:.3f} m, y={pose.y:.3f} m (二维平面，z固定0) | "
            f"lat={pose.lat:.8f}, lon={pose.lon:.8f}, GNSS h={pose.height:.3f} m"
        )

    def _build_heading_alignment_input(self) -> Dict[str, Any]:
        self._sync_points_from_table()

        if len(self._points) < 2:
            raise ValueError("至少需要记录两个控制点")

        point_a = dict(self._points[-2])
        point_b = dict(self._points[-1])
        point_a["row_index"] = len(self._points) - 1
        point_b["row_index"] = len(self._points)
        dx = point_b["sys_x"] - point_a["sys_x"]
        dy = point_b["sys_y"] - point_a["sys_y"]
        baseline_m = math.hypot(dx, dy)
        if baseline_m < 1.0:
            raise ValueError("两点距离过近，至少需要约 1 米以上基线才能稳定计算方位")

        line_yaw_rad = math.atan2(dy, dx)
        delta_rot_rad = self._wrap_angle_rad((math.pi * 0.5) - line_yaw_rad)
        return {
            "point_a": point_a,
            "point_b": point_b,
            "line_yaw_rad": line_yaw_rad,
            "delta_rot_rad": delta_rot_rad,
            "baseline_m": baseline_m,
        }

    def _apply_heading_alignment(self) -> None:
        try:
            heading_input = self._build_heading_alignment_input()
            point_a = heading_input["point_a"]
            point_b = heading_input["point_b"]
            summary = align_enu_y_axis_with_points(
                [
                    (point_a["sys_x"], point_a["sys_y"]),
                    (point_b["sys_x"], point_b["sys_y"]),
                ]
            )
            origin_reset_ok = set_position_origin_to_current()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "两点方位校准失败", str(exc))
            return

        self._calibration_changed = True
        cos_delta = math.cos(float(heading_input["delta_rot_rad"]))
        sin_delta = math.sin(float(heading_input["delta_rot_rad"]))
        for point in self._points:
            old_x = float(point.get("sys_x", 0.0))
            old_y = float(point.get("sys_y", 0.0))
            point["sys_x"] = old_x * cos_delta - old_y * sin_delta
            point["sys_y"] = old_x * sin_delta + old_y * cos_delta
        self._refresh_table()
        self._update_summary_label()
        line_heading_deg = (90.0 - math.degrees(float(heading_input["line_yaw_rad"]))) % 360.0
        delta_deg = math.degrees(float(heading_input["delta_rot_rad"]))
        baseline_m = float(heading_input["baseline_m"])
        point_a_idx = int(point_a["row_index"])
        point_b_idx = int(point_b["row_index"])
        self._log_to_parent(
            "ENU两点方位校准已更新: "
            f"点{point_a_idx}->点{point_b_idx} | "
            f"基线={baseline_m:.3f} m | "
            f"点位方位={line_heading_deg:.2f}° | "
            f"Y轴对齐旋转={delta_deg:+.3f}° | "
            f"tx={summary.translation_x_m:.3f} m | ty={summary.translation_y_m:.3f} m | "
            f"当前位置设原点={'成功' if origin_reset_ok else '失败'}"
        )
        QtWidgets.QMessageBox.information(
            self,
            "两点方位校准完成",
            f"使用控制点 {point_a_idx} -> {point_b_idx}\n"
            f"基线长度: {baseline_m:.3f} m\n"
            f"点位方位: {line_heading_deg:.2f}°\n"
            f"Y轴对齐旋转: {delta_deg:+.3f}°\n"
            f"当前位置设原点: {'成功' if origin_reset_ok else '失败'}",
        )

    @staticmethod
    def _parse_geodetic_fields(point: Dict[str, Any]) -> Tuple[float, float, Optional[float]]:
        lat = float(str(point.get("rtk_lat", "")).strip())
        lon = float(str(point.get("rtk_lon", "")).strip())
        h_raw = str(point.get("rtk_h", "")).strip()
        h_opt: Optional[float] = float(h_raw) if h_raw else None
        return lat, lon, h_opt

    def _apply_geodetic_frame(self) -> None:
        self._sync_points_from_table()
        if len(self._points) < 2:
            QtWidgets.QMessageBox.warning(
                self,
                "点数不足",
                "经纬度定系至少需要两个控制点（使用表格中最后两行）。",
            )
            return

        p_a = dict(self._points[-2])
        p_b = dict(self._points[-1])
        try:
            lat_a, lon_a, h_a = self._parse_geodetic_fields(p_a)
            lat_b, lon_b, h_b = self._parse_geodetic_fields(p_b)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "经纬度无效",
                f"请检查最后两点的纬度/经度是否为有效数字：{exc}",
            )
            return

        # 用 currentIndex 判定，避免部分环境下 currentData() 与字符串比较失效
        use_second_row_as_origin = self.combo_geodetic_origin.currentIndex() == 1
        try:
            if use_second_row_as_origin:
                summary = define_local_frame_from_two_geodetic_points(
                    lat_b,
                    lon_b,
                    lat_a,
                    lon_a,
                    origin_height_m=h_b,
                    axis_height_m=h_a,
                )
                o_lat, o_lon = lat_b, lon_b
            else:
                summary = define_local_frame_from_two_geodetic_points(
                    lat_a,
                    lon_a,
                    lat_b,
                    lon_b,
                    origin_height_m=h_a,
                    axis_height_m=h_b,
                )
                o_lat, o_lon = lat_a, lon_a
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "经纬度定系失败", str(exc))
            return

        self._calibration_changed = True
        for point in self._points:
            try:
                lat, lon, h_opt = self._parse_geodetic_fields(point)
                x, y, z = absolute_planar_xy_from_geodetic(lat, lon, h_opt)
                point["sys_x"] = x
                point["sys_y"] = y
                point["sys_z"] = z
            except (ValueError, RuntimeError):
                continue

        self._refresh_table()
        self._update_summary_label()
        self._log_to_parent(
            "已按经纬度建立平面系: "
            f"原点≈({o_lat:.8f},{o_lon:.8f}) | rot={summary.rotation_deg:.4f}° | "
            f"tx={summary.translation_x_m:.3f} m | ty={summary.translation_y_m:.3f} m | "
            "已根据经纬度刷新表中系统坐标（解析失败的行未修改）。"
        )
        QtWidgets.QMessageBox.information(
            self,
            "经纬度定系完成",
            "已用最后两点的经纬度设定平面系，并以所选点为平面原点。\n"
            "表中系统坐标已按当前 ENU/校准从经纬度重算（若某行经纬度无效则跳过该行）。",
        )

    def _delete_selected_point(self) -> None:
        self._sync_points_from_table()
        selected_ranges = self.table.selectedRanges()
        if not selected_ranges:
            QtWidgets.QMessageBox.information(self, "未选择点位", "请先选择要删除的控制点")
            return

        row = selected_ranges[0].topRow()
        if row < 0 or row >= len(self._points):
            return

        removed_index = row + 1
        self._points.pop(row)
        self._refresh_table()
        self._log_to_parent(f"ENU控制点{removed_index}已删除")

    def _clear_calibration(self) -> None:
        clear_enu_calibration()
        self._calibration_changed = True
        self._update_summary_label()
        self._log_to_parent("ENU校准已清除")
        QtWidgets.QMessageBox.information(self, "已清除", "ENU坐标校准参数已清除")

    def calibration_changed(self) -> bool:
        return bool(self._calibration_changed)

    def points(self) -> List[Dict[str, Any]]:
        self._sync_points_from_table()
        return [dict(point) for point in self._points]


class MainWindow(QtWidgets.QMainWindow):
    segment_rcs_start_requested = QtCore.pyqtSignal(int, str)
    segment_rcs_finish_requested = QtCore.pyqtSignal(int, str)
    tracking_metrics_reported = QtCore.pyqtSignal(object)
    motion_run_finished = QtCore.pyqtSignal(str)
    """
    RCS 雷达小车一体化调试 UI：

      - 主窗口：左侧状态信息，中间轨迹规划图，右侧控制按钮
      - 运行日志、雷达目标检查图、RCS 绘图在独立弹窗中（通过对应按钮打开）
    """

    def __init__(self) -> None:
        super().__init__()
        _apply_matplotlib_font()
        self.setWindowTitle("RCS 雷达小车调试 UI")
        self._screen_available_geometry = self._get_available_screen_geometry()
        self._ui_scale = self._compute_ui_scale(self._screen_available_geometry)
        self._apply_window_scale()
        self._apply_initial_geometry()

        self._is_linux = platform.system().lower() == "linux"

        # 初始化控制器
        self.controller = MainController(self._is_linux)

        self.rcs_recorder = RcsRunRecorder()
        self.rcs_lock = AssocLock()
        self._rcs_recording = False
        self._rcs_fitted: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._rcs_show_only_fitted = False
        self._rcs_target_name: Optional[str] = None
        self._loaded_rcs_curves: List[LoadedRcsCurve] = []
        self._rcs_ref_class_options: List[str] = []
        self._rcs_ref_angle_options: List[str] = []
        self._rcs_ref_labels: Dict[str, str] = {}
        self._rcs_ref_limits: Optional[Dict[str, np.ndarray]] = None
        self._rcs_ref_class: Optional[str] = None
        self._rcs_ref_angle: Optional[str] = None
        self._rcs_plot_mode = "distance"
        self._rcs_orbit_samples: List[Dict[str, float]] = []
        self._rcs_ax_projection = "cartesian"
        self._data_save_root = Path(__file__).resolve().parent / DATA_SAVE_ROOT_DIR_NAME
        self._motion_save_dir = self._data_save_root / MOTION_DATA_DIR_NAME
        self._rcs_save_dir = self._data_save_root / RCS_DATA_DIR_NAME
        self._current_path_name = "轨迹"
        self._loaded_path_frame = PathReferenceFrame()
        self._rcs_active_segment_index: Optional[int] = None
        self._rcs_active_path_name: Optional[str] = None
        self._rcs_active_target_name: Optional[str] = None
        self._rcs_saved_trajectory_files: Dict[str, AggregatedRcsFile] = {}
        self._rcs_relock_events: List[str] = []
        # 一次 RCS 录制开始时固定，用于聚合文件名；避免断联重锁后「目标ID」变化写入另一组缓存文件
        self._rcs_snapshot_file_target_name: Optional[str] = None
        # 圆弧轨迹段 RCS：距离门控 + 极坐标保存（径向 dBsm）
        self._orbit_rcs_active: bool = False
        self._orbit_rcs_rows: List[Dict[str, float]] = []
        self._orbit_polar_cached_series: Optional[Dict[str, Any]] = None
        # 前进直线段：合并记录（不按 ID 分开），将所有目标点写入同一 CSV 并统一拟合直线
        self._straight_rcs_collect_all: bool = False
        self._straight_rcs_last_by_oid: Dict[int, float] = {}
        self._straight_rcs_rcs_ema_by_oid: Dict[int, float] = {}
        # 前进直线段：UI 叠加显示“第N次”测量数据（不依赖保存文件解析）
        self._straight_rcs_runs: List[List[CurvePoint]] = []
        self._straight_rcs_max_runs: int = 30
        # 与 rcs_recorder.segments 对齐：第 i 段对应的“次数”显示（通常由 CSV 的 SegIdx 列解析得到）
        self._rcs_segment_run_labels: Optional[List[int]] = None
        # RCS plotting workflow state
        self._rcs_base_file_path: Optional[str] = None
        self._rcs_curve_csv_source_paths: List[str] = []
        # When True, do not overlay reference limit bands/lines on the plot.
        # Used for "选择数据绘RCS图" as requested.
        self._rcs_hide_reference_limits: bool = False
        # 绘图时叠加到测量 RCS（dB），不改动落盘原始数据；参考上下限不偏移。
        self._rcs_plot_calibration_db: float = 0.0
        self._last_car_control_mode: Optional[int] = None
        self._ensure_recorded_data_dirs()
        self._migrate_existing_saved_data()
        self._load_rcs_reference_options()
        self._radar_emergency_active = False
        self._radar_emergency_stop_enabled = RADAR_EMERGENCY_STOP_DEFAULT_ENABLED
        self._radar_auto_relock_enabled = True
        self._radar_target_fresh_s = RADAR_TARGET_FRESH_S
        self._motion_active = False
        # 运动异常急停监测（防止非设计原地自转 / 角速度抽搐）
        self._motion_guard_enabled = True
        self._motion_guard_low_v_mps = 0.08
        self._motion_guard_spin_w_radps = 1.10
        self._motion_guard_spin_hold_s = 0.35
        self._motion_guard_dw_dt_thresh = 7.0
        self._motion_guard_jerk_hold_s = 0.25
        self._motion_guard_jerk_w_floor = 0.70
        self._motion_guard_popup_cooldown_s = 1.5
        self._motion_guard_last_popup_ts = 0.0
        self._motion_guard_design = "unknown"
        self._motion_guard_nominal_speed = 0.0
        self._motion_guard_w_limit = 1.5
        self._motion_guard_allow_spin_until = 0.0
        self._motion_guard_spin_since: Optional[float] = None
        self._motion_guard_jerk_since: Optional[float] = None
        self._motion_guard_prev_ts: Optional[float] = None
        self._motion_guard_prev_w: Optional[float] = None
        self._last_motion_interrupt_reason = ""
        self._last_motion_interrupt_ts = 0.0
        self._motion_stop_popup_cooldown_s = 1.5
        self._motion_stop_last_popup_ts = 0.0
        self._motion_stop_last_popup_key = ""
        self._radar_stop_threshold = 1.0
        self._radar_plot_wide_lateral_active = False
        self._path_speed = DEFAULT_SEGMENT_SPEED_MPS
        self._path_tracking_mode = "stanley"
        self._tracking_tuning: Dict[str, float] = {
            "lateral_kp": 2.58,
            "lateral_ki": 0.30,
            "lateral_kd": 0.52,
            "heading_kp": 2.65,
            "heading_ki": 0.32,
            "heading_kd": 0.68,
            "stanley_lateral_kp": 1.05,
            "stanley_lateral_kd": 0.46,
        }
        self._traj_default_dist = DEFAULT_LINE_PLAN_DIST_M
        self._traj_default_radius = 40.0
        self._traj_default_angle = 360.0
        self._traj_default_circle_direction = "ccw"
        self._traj_default_line_speed = DEFAULT_LINE_PLAN_SPEED_MPS
        self._traj_default_circle_speed = DEFAULT_SEGMENT_SPEED_MPS
        self._traj_default_accel_dist = DEFAULT_ACCEL_DIST_M
        self._traj_default_decel_dist = DEFAULT_DECEL_DIST_M
        self._radial_measurement_spec: Optional[RadialMeasurementSpec] = None
        self._radial_default_angle_cycles = {angle: 5 for angle in range(0, 360, 30)}
        # 星型测量：UI 改为「固定直线长度」+「距离目标最近距离」
        self._radial_default_inner_radius = 4.0
        self._radial_default_outer_radius = 54.0
        self._radial_default_line_length_m = 46.0
        self._radial_default_speed_mps = DEFAULT_RADIAL_MEASUREMENT_SPEED_MPS
        self._radial_rcs_session_dir: Optional[Path] = None
        self._radial_rcs_expected_segments: List[str] = []
        self._radial_rcs_started_segments: Set[str] = set()
        self._radial_rcs_finished_segments: Set[str] = set()
        self._radial_rcs_failed_segments: Set[str] = set()
        self._preset_paths: Dict[str, List[Tuple[float, float]]] = {}
        self._preset_ranges: Dict[str, List[SegmentRange]] = {}
        self._preset_segment_kinds: Dict[str, List[str]] = {}
        self._preset_path_frames: Dict[str, PathReferenceFrame] = {}
        self._planned_ranges: Optional[List[SegmentRange]] = None
        # 与轨迹规划器一致的逐段类型：\"line\" / \"circle\"；缺省则回退到几何判直
        self._planned_segment_kinds: Optional[List[str]] = None
        self._planned_range_task_names: List[Optional[str]] = []
        self._loaded_task_sequence_names: List[str] = []
        self._loaded_task_transition_count = 0
        self.segment_rcs_start_requested.connect(self._on_segment_rcs_start_requested)
        self.segment_rcs_finish_requested.connect(self._on_segment_rcs_finish_requested)
        self.tracking_metrics_reported.connect(self._on_tracking_metrics_reported)
        self.motion_run_finished.connect(self._on_motion_run_finished)
        self._tracking_run_records: List[Dict[str, Any]] = []
        self._tracking_sample_buffers: Dict[str, List[Dict[str, Any]]] = {}
        self._tracking_sample_lock = threading.Lock()
        self._motion_full_session_active = False
        self._motion_full_session_accumulator: List[Dict[str, Any]] = []
        self._data_analysis_window = None
        self._runtime_log_lines: List[str] = []
        self._trim_next_motion_record = False
        self._trim_next_motion_record_ts = 0.0
        self._trim_next_motion_record_reason = ""

        # ================== UI 结构 ==================
        # 主界面不再包 QScrollArea，避免中部出现纵向滚动条；由布局按比例分配高度。
        central = QtWidgets.QWidget()
        central.setObjectName("MainContent")
        central.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(
            self._compact_v(16),
            self._compact_v(16),
            self._compact_v(16),
            self._compact_v(16),
        )
        vbox.setSpacing(self._compact_v(14))

        self._build_control_panel(vbox)
        self._build_plots(vbox)
        self._apply_theme()
        self._update_rcs_save_dir_button_tooltip()
        self._load_preset_paths()

        # 轨迹缓存（历史 + 规划）
        self.path_x: List[float] = []
        self.path_y: List[float] = []
        self.planned_x: List[float] = []
        self.planned_y: List[float] = []
        self._planned_path_curve_arrows: List[Any] = []
        self.loaded_path_points: List[Tuple[float, float]] = []
        self.loaded_path_local_points: List[Tuple[float, float]] = []
        self._path_preview_uses_virtual_pose = False
        self._enu_calibration_points: List[Dict[str, Any]] = []
        self._target_marking_mode = False
        self.target_point: Optional[Tuple[float, float]] = None

        self.traj_enabled = True
        self.tracked_target_id: Optional[int] = None

        # 默认禁用运动按钮，等待 IMU 状态良好后自动解锁
        self._update_run_path_button_state()

        # ================== 定时更新 ==================
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        # 主界面周期刷新（位姿、小车模型、轨迹、IMU 状态栏、雷达等）：默认 50ms≈20Hz。若 INSPVAX 为 50Hz 且希望显示更跟手，可改为 20ms（≈50Hz，CPU 更忙）。
        self.timer.start(50)

        # 启动时全屏显示
        self.setWindowState(self.windowState() | QtCore.Qt.WindowMaximized)

        self._log("UI已启动")
        self._log(
            f"系统: {platform.system()} | RCS采集: Cluster CSV(0x701)"
            f"{' +接收线程' if getattr(self.controller, 'cluster_csv_runtime', None) else ''}"
        )
        self._log(f"CAN状态: {self.controller.get_can_status()}")
        self._log(
            f"界面缩放: scale={self._ui_scale:.2f} | screen={self._screen_available_geometry.width()}x{self._screen_available_geometry.height()}"
        )

    def _get_active_screen(self) -> Optional[QtGui.QScreen]:
        screen = None
        if hasattr(QtGui.QGuiApplication, "screenAt"):
            screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())
        if screen is None:
            screen = QtWidgets.QApplication.primaryScreen()
        return screen

    def _get_available_screen_geometry(self) -> QtCore.QRect:
        screen = self._get_active_screen()
        if screen is None:
            return QtCore.QRect(0, 0, UI_LAYOUT_BASE_WIDTH, UI_LAYOUT_BASE_HEIGHT)
        available = screen.availableGeometry()
        if available.width() <= 0 or available.height() <= 0:
            return QtCore.QRect(0, 0, UI_LAYOUT_BASE_WIDTH, UI_LAYOUT_BASE_HEIGHT)
        return available

    def _compute_ui_scale(self, available: QtCore.QRect) -> float:
        scale_w = available.width() / float(UI_LAYOUT_BASE_WIDTH)
        scale_h = available.height() / float(UI_LAYOUT_BASE_HEIGHT)
        return max(UI_SCALE_MIN, min(1.0, scale_w, scale_h))

    def _scaled_px(self, value: int) -> int:
        return max(1, int(round(float(value) * self._ui_scale)))

    def _compact_v(self, design_px: int) -> int:
        return _layout_compact_v(self._ui_scale, design_px)

    def _apply_window_scale(self) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        font = QtGui.QFont(app.font())
        family = _resolve_cjk_font_family()
        if family:
            font.setFamily(family)
        point_size = font.pointSizeF()
        if point_size <= 0:
            point_size = 10.0
        font.setPointSizeF(max(7.5, point_size * self._ui_scale * 0.90))
        app.setFont(font)
        self.setFont(font)

    def _apply_initial_geometry(self) -> None:
        available = self._screen_available_geometry
        if available.width() <= 0 or available.height() <= 0:
            self.resize(UI_LAYOUT_BASE_WIDTH, UI_LAYOUT_BASE_HEIGHT)
            return

        target_w = min(int(UI_LAYOUT_BASE_WIDTH * self._ui_scale), available.width())
        target_h = min(int(UI_LAYOUT_BASE_HEIGHT * self._ui_scale), available.height())
        self.resize(target_w, target_h)

        x = available.x() + max(0, (available.width() - target_w) // 2)
        y = available.y() + max(0, (available.height() - target_h) // 2)
        self.move(x, y)

    def _build_control_panel(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """构建控制面板"""
        panel = QtWidgets.QFrame()
        panel.setObjectName("ControlPanel")
        panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        hbox = QtWidgets.QHBoxLayout(panel)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(self._scaled_px(14))

        side_column = QtWidgets.QWidget()
        side_column.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        side_vbox = QtWidgets.QVBoxLayout(side_column)
        side_vbox.setContentsMargins(0, 0, 0, 0)
        side_vbox.setSpacing(self._compact_v(16))

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(max(1, self._scaled_px(6)))
        main_splitter.addWidget(side_column)

        # 左侧：状态信息
        info_frame = QtWidgets.QFrame()
        info_frame.setObjectName("InfoCard")
        info_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        vinfo = QtWidgets.QVBoxLayout(info_frame)
        vinfo.setContentsMargins(
            self._compact_v(16),
            self._compact_v(16),
            self._compact_v(16),
            self._compact_v(16),
        )
        vinfo.setSpacing(self._compact_v(8))

        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setColor(QtGui.QColor(0, 0, 0, 140))
        shadow.setOffset(0, 6)
        info_frame.setGraphicsEffect(shadow)

        self.label_status_header = QtWidgets.QLabel("系统状态")
        self.label_status_header.setObjectName("InfoHeader")

        self.pi_battery_indicator = BatteryIndicator("树莓派")
        self.car_battery_indicator = BatteryIndicator("小车底盘")
        self.pi_battery_indicator.set_visual_scale(self._ui_scale)
        self.car_battery_indicator.set_visual_scale(self._ui_scale)
        power_strip = QtWidgets.QHBoxLayout()
        power_strip.setContentsMargins(0, 0, 0, 0)
        power_strip.setSpacing(self._scaled_px(10))
        power_strip.addWidget(self.pi_battery_indicator, 0, QtCore.Qt.AlignLeft)
        power_strip.addWidget(self.car_battery_indicator, 0, QtCore.Qt.AlignLeft)
        power_strip.addStretch(1)

        status_label_min_width = self._scaled_px(280)
        self.label_pose = QtWidgets.QLabel("等待 INSPVAXA / INS 数据...")
        self.label_pose.setWordWrap(True)
        self.label_pose.setMinimumWidth(status_label_min_width)

        self.label_enu_calibration = QtWidgets.QLabel("ENU校准: 未启用")
        self.label_enu_calibration.setWordWrap(True)
        self.label_enu_calibration.setMinimumWidth(status_label_min_width)

        self.label_imu_status = QtWidgets.QLabel("IMU: 初始化中...")
        self.label_imu_status.setWordWrap(True)
        self.label_imu_status.setMinimumWidth(status_label_min_width)

        self.label_power = QtWidgets.QLabel("树莓派电量: -- | 小车电量: --")
        self.label_power.setWordWrap(True)
        self.label_power.setMinimumWidth(status_label_min_width)

        self.label_can = QtWidgets.QLabel("CAN 状态: 未知")
        self.label_can.setMinimumWidth(status_label_min_width)

        self.label_tracked = QtWidgets.QLabel("锁定目标: --")
        self.label_tracked.setMinimumWidth(status_label_min_width)

        self.label_heading = QtWidgets.QLabel("方位角: --")
        self.label_heading.setMinimumWidth(status_label_min_width)

        self.btn_open_log_window = QtWidgets.QPushButton("运行日志窗口")
        self.btn_open_log_window.setToolTip("在独立窗口中查看与操作运行日志")
        self.btn_open_log_window.clicked.connect(self._on_open_log_window_clicked)

        status_actions = QtWidgets.QGridLayout()
        status_actions.setContentsMargins(0, 0, 0, 0)
        status_actions.setHorizontalSpacing(self._scaled_px(8))
        status_actions.setVerticalSpacing(self._compact_v(8))
        btn_clear_log = QtWidgets.QPushButton(ACTION_CLEAR_LOG_TEXT)
        btn_save_log = QtWidgets.QPushButton(ACTION_SAVE_LOG_TEXT)
        btn_export_log_csv = QtWidgets.QPushButton(ACTION_EXPORT_CSV_TEXT)
        btn_save_motion_data = QtWidgets.QPushButton(ACTION_SAVE_MOTION_DATA_TEXT)
        btn_emergency_stop = QtWidgets.QPushButton(ACTION_EMERGENCY_STOP_TEXT)
        btn_emergency_stop.setToolTip("立即停止当前轨迹/分段控制循环，并下发底盘零速")
        btn_emergency_stop.setStyleSheet(
            "background-color: #C62828; color: #FFFFFF; font-weight: bold; padding: 6px 10px;"
        )
        btn_emergency_stop.clicked.connect(self._on_emergency_stop_clicked)
        btn_clear_log.clicked.connect(self._on_clear_log)
        btn_save_log.clicked.connect(self._on_save_log)
        btn_export_log_csv.clicked.connect(self._on_export_metrics_csv)
        btn_save_motion_data.clicked.connect(self._on_save_motion_data)
        status_actions.addWidget(btn_clear_log, 0, 0)
        status_actions.addWidget(btn_save_log, 0, 1)
        status_actions.addWidget(btn_export_log_csv, 1, 0)
        status_actions.addWidget(btn_save_motion_data, 1, 1)
        status_actions.addWidget(btn_emergency_stop, 2, 0, 1, 2)

        vinfo.addLayout(power_strip)
        vinfo.addWidget(self.label_status_header)
        vinfo.addWidget(self.label_pose)
        vinfo.addWidget(self.label_enu_calibration)
        vinfo.addWidget(self.label_imu_status)
        vinfo.addWidget(self.label_power)
        vinfo.addWidget(self.label_can)
        vinfo.addWidget(self.label_tracked)
        vinfo.addWidget(self.label_heading)
        vinfo.addWidget(self.btn_open_log_window)
        vinfo.addLayout(status_actions)
        vinfo.addStretch(1)

        side_vbox.addWidget(info_frame, 3)

        # 中间：轨迹规划图
        traj_frame = QtWidgets.QFrame()
        traj_frame.setObjectName("PlotFrame")
        traj_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        traj_vbox = QtWidgets.QVBoxLayout(traj_frame)
        traj_vbox.setContentsMargins(
            self._compact_v(8),
            self._compact_v(8),
            self._compact_v(8),
            self._compact_v(8),
        )

        traj_label = QtWidgets.QLabel("轨迹规划图 (蓝色:实际轨迹, 橙色实线+方向箭头:规划轨迹, 正方形:小车模型, 红色:航向)")
        traj_label.setObjectName("PlotLabel")
        traj_vbox.addWidget(traj_label)

        self.traj_plot = pg.PlotWidget()
        self.traj_plot.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.traj_plot.setLabel('left', '北向坐标 Y', 'm')
        self.traj_plot.setLabel('bottom', '东向坐标 X', 'm')
        self.traj_plot.setAspectLocked(True)
        self.traj_plot.showGrid(x=True, y=True, alpha=0.3)
        self._update_traj_axis_labels()

        # 轨迹曲线
        self.traj_curve = self.traj_plot.plot([], [], pen=pg.mkPen(color='#0D47A1', width=2), name="实际轨迹")
        self.traj_planned_curve = self.traj_plot.plot(
            [],
            [],
            pen=pg.mkPen(color="#FF9800", width=2, style=QtCore.Qt.SolidLine),
            name="规划轨迹",
        )

        self.vehicle_body_curve = self.traj_plot.plot(
            [],
            [],
            pen=pg.mkPen(color="#5D4037", width=2),
            name="小车模型",
        )
        self.vehicle_heading_curve = self.traj_plot.plot(
            [],
            [],
            pen=pg.mkPen(color="#D32F2F", width=2),
            name="航向",
        )
        self.vehicle_center_marker = pg.ScatterPlotItem(
            size=6,
            pen=pg.mkPen(color="#5D4037", width=1),
            brush=pg.mkBrush(93, 64, 55, 220),
        )
        self.traj_plot.addItem(self.vehicle_center_marker)
        self.vehicle_center_marker.setData([], [])
        self.vehicle_center_marker.setZValue(9)

        self.target_marker = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen(color="r", width=1),
            brush=pg.mkBrush(255, 0, 0, 200),
            symbol="s",
        )
        self.traj_plot.addItem(self.target_marker)
        self.target_marker.setData([], [])
        self.target_marker.setZValue(10)
        self.traj_plot.scene().sigMouseClicked.connect(self._on_traj_plot_clicked)

        traj_vbox.addWidget(self.traj_plot, 1)

        # 左侧下方：按钮控制
        btn_frame = QtWidgets.QFrame()
        btn_frame.setObjectName("ButtonCard")
        btn_frame.setMinimumWidth(self._scaled_px(320))
        btn_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        btn_card_shadow = QtWidgets.QGraphicsDropShadowEffect(btn_frame)
        btn_card_shadow.setBlurRadius(24)
        btn_card_shadow.setColor(QtGui.QColor(0, 0, 0, 118))
        btn_card_shadow.setOffset(0, 5)
        btn_frame.setGraphicsEffect(btn_card_shadow)
        vbtn = QtWidgets.QVBoxLayout(btn_frame)
        vbtn.setContentsMargins(
            self._compact_v(12),
            self._compact_v(12),
            self._compact_v(12),
            self._compact_v(12),
        )
        vbtn.setSpacing(self._compact_v(10))

        self.btn_load_path = QtWidgets.QPushButton("轨迹规划")
        self.btn_radial_measure = QtWidgets.QPushButton("星型测量")
        self.btn_run_path = QtWidgets.QPushButton("执行轨迹")
        self.btn_preset_path = QtWidgets.QPushButton("预设轨迹 / 导入")
        self.btn_delete_preset = QtWidgets.QPushButton("删除预设轨迹")
        self.btn_calib = QtWidgets.QPushButton("标定目标物位置")
        self.btn_enu_calib = QtWidgets.QPushButton("ENU坐标校准")
        self.btn_data_analysis = QtWidgets.QPushButton("误差分析")
        self.btn_save_traj = QtWidgets.QPushButton("锁定雷达目标/检查锁定")
        self.btn_select_rcs_save_dir = QtWidgets.QPushButton("选择数据绘RCS图")
        self.btn_select_rcs_reference = QtWidgets.QPushButton("选择RCS参考产品")
        self.btn_plot_saved_rcs = QtWidgets.QPushButton("导入数据对比")
        self.btn_heading_calib = QtWidgets.QPushButton("清除历史轨迹")

        self.btn_load_path.clicked.connect(self._on_load_path)
        self.btn_radial_measure.clicked.connect(self._on_radial_measure_clicked)
        self.btn_run_path.clicked.connect(self._on_run_path)
        self.btn_preset_path.clicked.connect(self._on_preset_path_clicked)
        delete_handler = getattr(self, "_on_delete_preset_clicked", None)
        if callable(delete_handler):
            self.btn_delete_preset.clicked.connect(delete_handler)
        else:
            self.btn_delete_preset.setEnabled(False)
        self.btn_calib.clicked.connect(self._on_calib_clicked)
        self.btn_enu_calib.clicked.connect(self._on_enu_calibration_clicked)
        self.btn_data_analysis.clicked.connect(self._on_data_analysis_clicked)
        self.btn_save_traj.clicked.connect(self._on_radar_lock_check_clicked)
        self.btn_select_rcs_save_dir.clicked.connect(self._on_select_rcs_save_dir)
        self.btn_select_rcs_reference.clicked.connect(self._on_select_rcs_reference)
        self.btn_plot_saved_rcs.clicked.connect(self._on_plot_saved_rcs_clicked)
        self.btn_heading_calib.clicked.connect(self._on_heading_calib_clicked)
        self.label_path_tracking_mode = QtWidgets.QLabel("轨迹跟踪算法")
        self.combo_path_tracking_mode = QtWidgets.QComboBox()
        self.combo_path_tracking_mode.addItem("Stanley", "stanley")
        self.combo_path_tracking_mode.addItem("Stanley + PID", "stanley_pid")
        self.combo_path_tracking_mode.currentIndexChanged.connect(self._on_path_tracking_mode_changed)
        tracking_mode_index = self.combo_path_tracking_mode.findData(self._path_tracking_mode)
        if tracking_mode_index < 0:
            tracking_mode_index = 0
        self.combo_path_tracking_mode.setCurrentIndex(tracking_mode_index)
        self.label_path_coord_hint = QtWidgets.QLabel(
            "轨迹与锚点均在「校准后平面系」中（与 get_robot_pose 的 x,y 同系）。"
            "ENU 用「前一点/后一点为原点、另一点方向为 +Y」定系后，点「置(0,0)」即把局部轨迹原点绑在该定系原点上；"
            "规划/导入的局部轨迹只随锚点平移，不会自动改到小车当前位置。"
        )
        self.label_path_coord_hint.setWordWrap(True)
        self.label_path_origin_point = QtWidgets.QLabel("轨迹锚点 x,y（校准平面 m）")
        anchor_row = QtWidgets.QWidget()
        anchor_h = QtWidgets.QHBoxLayout(anchor_row)
        anchor_h.setContentsMargins(0, 0, 0, 0)
        anchor_h.setSpacing(6)
        self.spin_path_anchor_x = QtWidgets.QDoubleSpinBox()
        self.spin_path_anchor_y = QtWidgets.QDoubleSpinBox()
        for sp in (self.spin_path_anchor_x, self.spin_path_anchor_y):
            sp.setRange(-100000.0, 100000.0)
            sp.setDecimals(3)
            sp.setSuffix(" m")
        self.spin_path_anchor_x.setPrefix("x ")
        self.spin_path_anchor_y.setPrefix("y ")
        self.btn_path_anchor_zero = QtWidgets.QPushButton("置(0,0)")
        self.btn_path_anchor_zero.setToolTip("与 ENU 校准定义的平面原点一致")
        anchor_h.addWidget(self.spin_path_anchor_x, 1)
        anchor_h.addWidget(self.spin_path_anchor_y, 1)
        anchor_h.addWidget(self.btn_path_anchor_zero)
        self.spin_path_anchor_x.valueChanged.connect(self._on_path_anchor_xy_changed)
        self.spin_path_anchor_y.valueChanged.connect(self._on_path_anchor_xy_changed)
        self.btn_path_anchor_zero.clicked.connect(self._on_path_anchor_zero_clicked)
        self.label_path_origin_summary = QtWidgets.QLabel()
        self.label_path_origin_summary.setWordWrap(True)
        self.label_path_origin_summary.setStyleSheet("color: #546E7A;")
        self.label_rcs_reference_summary = QtWidgets.QLabel()
        self.label_rcs_reference_summary.setWordWrap(True)
        self.label_rcs_reference_summary.setStyleSheet("color: #546E7A;")
        self.btn_select_rcs_save_dir.setToolTip(
            f"从 {self._rcs_save_dir} 选择历史RCS数据文件并绘图（不叠加参考上下限）。"
        )
        self.btn_select_rcs_reference.setToolTip("选择产品型号与角度，并导入RCS上下限（用于对比/验收）")
        self.btn_plot_saved_rcs.setToolTip(
            "在当前已绘制的数据基础上，再选择一组数据进行拟合曲线对比（可自定义名称）。"
        )

        vbtn.addWidget(self.label_path_tracking_mode)
        vbtn.addWidget(self.combo_path_tracking_mode)
        # 隐藏轨迹跟踪算法选择，固定使用初始化时的默认（见 _path_tracking_mode）
        self.label_path_tracking_mode.setVisible(False)
        self.combo_path_tracking_mode.setVisible(False)
        vbtn.addWidget(self.label_path_coord_hint)
        vbtn.addWidget(self.label_path_origin_point)
        vbtn.addWidget(anchor_row)
        vbtn.addWidget(self.label_path_origin_summary)
        path_btn_grid = QtWidgets.QGridLayout()
        path_btn_grid.setContentsMargins(0, 0, 0, 0)
        path_btn_grid.setHorizontalSpacing(self._scaled_px(8))
        path_btn_grid.setVerticalSpacing(self._compact_v(8))
        path_btn_grid.setColumnStretch(0, 1)
        path_btn_grid.setColumnStretch(1, 1)
        path_btn_grid.addWidget(self.btn_load_path, 0, 0)
        path_btn_grid.addWidget(self.btn_radial_measure, 0, 1)
        self.chk_rcs_all_forward_straight = QtWidgets.QCheckBox(
            "前进直线段均触发 Cluster RCS 采集（往返时每段前进各一段）"
        )
        self.chk_rcs_all_forward_straight.setChecked(True)
        self.chk_rcs_all_forward_straight.setToolTip(
            "勾选后：凡前进近似直线分段都会自动触发 Cluster(0x701) RCS CSV，"
            "与 CSV 里是否逐段勾选无关；倒车段仍以文件中的 rcs_start 为准。"
        )
        path_btn_grid.addWidget(self.chk_rcs_all_forward_straight, 1, 0, 1, 2)
        path_action_pairs = [
            (self.btn_run_path, self.btn_preset_path),
            (self.btn_delete_preset, self.btn_calib),
            (self.btn_enu_calib, self.btn_data_analysis),
            (self.btn_heading_calib, self.btn_save_traj),
            (self.btn_select_rcs_save_dir, self.btn_select_rcs_reference),
        ]
        for i, (left_btn, right_btn) in enumerate(path_action_pairs):
            path_btn_grid.addWidget(left_btn, i + 2, 0)
            path_btn_grid.addWidget(right_btn, i + 2, 1)
        vbtn.addLayout(path_btn_grid)
        vbtn.addWidget(self.label_rcs_reference_summary)
        vbtn.addWidget(self.btn_plot_saved_rcs)
        vbtn.addStretch(1)
        self._use_calibration_plane_path_origin()
        self._update_rcs_reference_summary_label()

        side_vbox.addWidget(btn_frame, 4)
        main_splitter.addWidget(traj_frame)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 5)
        sw = max(800, self._screen_available_geometry.width())
        main_splitter.setSizes(
            [
                max(self._scaled_px(380), int(sw * 0.26)),
                max(self._scaled_px(900), int(sw * 0.62)),
            ]
        )
        hbox.addWidget(main_splitter, 1)

        # 主窗口仅保留上方轨迹与控制区；运行日志 / 雷达检查 / RCS 图仅在弹窗中构建
        parent_layout.addWidget(panel, 1)

    def _build_plots(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """运行日志、雷达目标检查、RCS 绘图均在独立弹窗中，不占用主窗口底部。"""
        del parent_layout
        self._build_auxiliary_plot_dialogs()

    def _create_rcs_frame(self) -> QtWidgets.QFrame:
        """构建雷达跟踪图 + RCS 控件"""
        rcs_frame = QtWidgets.QFrame()
        rcs_frame.setObjectName("PlotFrame")
        rcs_vbox = QtWidgets.QVBoxLayout(rcs_frame)
        rcs_vbox.setContentsMargins(
            self._compact_v(8),
            self._compact_v(8),
            self._compact_v(8),
            self._compact_v(8),
        )

        rcs_label = QtWidgets.QLabel("雷达跟踪图 (点击雷达点锁定)")
        rcs_label.setObjectName("PlotLabel")
        rcs_vbox.addWidget(rcs_label)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.addWidget(QtWidgets.QLabel("绘图模式"))
        self.rcs_plot_mode_combo = QtWidgets.QComboBox()
        self.rcs_plot_mode_combo.addItem("距离-RCS", "distance")
        self.rcs_plot_mode_combo.addItem("圆周-RCS", "orbit")
        self.rcs_plot_mode_combo.setCurrentIndex(0)
        self.rcs_plot_mode_combo.currentIndexChanged.connect(self._on_rcs_plot_mode_changed)
        mode_row.addWidget(self.rcs_plot_mode_combo, 1)
        rcs_vbox.addLayout(mode_row)

        self.rcs_canvas = FigureCanvas(Figure(figsize=(4.6, 3.8), dpi=100))
        self.rcs_canvas.setMinimumSize(self._scaled_px(360), self._compact_v(220))
        self.rcs_canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.rcs_ax = self.rcs_canvas.figure.add_subplot(111)
        self.rcs_canvas.figure.subplots_adjust(bottom=0.18)
        rcs_vbox.addWidget(self.rcs_canvas, 1)

        self.rcs_status_label = QtWidgets.QLabel("Cluster RCS 采集: 未开始")
        rcs_vbox.addWidget(self.rcs_status_label)
        rcs_btns = QtWidgets.QHBoxLayout()
        self.btn_save_rcs_image = QtWidgets.QPushButton("保存RCS图片")
        self.btn_save_rcs_image.clicked.connect(self._on_rcs_save_image)
        rcs_btns.addWidget(self.btn_save_rcs_image)
        rcs_btns.addStretch(1)
        rcs_vbox.addLayout(rcs_btns)

        self._draw_rcs()
        return rcs_frame

    def _on_rcs_plot_mode_changed(self, _index: int) -> None:
        self._rcs_plot_mode = self._get_rcs_plot_mode()
        self._draw_rcs()

    def _get_rcs_plot_mode(self) -> str:
        if hasattr(self, "rcs_plot_mode_combo") and self.rcs_plot_mode_combo is not None:
            mode = self.rcs_plot_mode_combo.currentData()
            if isinstance(mode, str) and mode:
                return mode
        return str(getattr(self, "_rcs_plot_mode", "distance") or "distance")

    def _select_rcs_plot_mode(self, mode: str) -> None:
        """切换 RCS 绘图模式（距离-RCS / 圆周-RCS），并同步内部状态。"""
        m = str(mode or "").strip()
        if m not in {"distance", "orbit"}:
            return
        self._rcs_plot_mode = m
        combo = getattr(self, "rcs_plot_mode_combo", None)
        if combo is None:
            return
        for i in range(combo.count()):
            if combo.itemData(i) == m:
                combo.blockSignals(True)
                combo.setCurrentIndex(i)
                combo.blockSignals(False)
                return

    def _ensure_rcs_axes(self, plot_mode: str) -> None:
        projection = "polar" if plot_mode == "orbit" else "cartesian"
        if projection == getattr(self, "_rcs_ax_projection", "") and getattr(self, "rcs_ax", None) is not None:
            return
        figure = self.rcs_canvas.figure
        figure.clf()
        if projection == "polar":
            self.rcs_ax = figure.add_subplot(111, projection="polar")
            figure.subplots_adjust(bottom=0.10, top=0.90)
        else:
            self.rcs_ax = figure.add_subplot(111)
            figure.subplots_adjust(bottom=0.18)
        self._rcs_ax_projection = projection

    def _append_rcs_orbit_sample(
        self,
        point: Optional[CurvePoint],
        pose: Optional[PoseSolution] = None,
    ) -> None:
        if point is None or self.target_point is None:
            return
        resolved_pose = pose if pose is not None else get_robot_pose()
        if resolved_pose is None:
            return

        dx = float(resolved_pose.x) - float(self.target_point[0])
        dy = float(resolved_pose.y) - float(self.target_point[1])
        orbit_radius_m = math.hypot(dx, dy)
        if orbit_radius_m <= 1e-3:
            return

        self._rcs_orbit_samples.append(
            {
                "timestamp": float(point.t),
                "angle_rad": float(math.atan2(dy, dx)),
                "orbit_radius_m": float(orbit_radius_m),
                "rcs_raw": float(point.rcs_raw),
                "rcs_filt": float(point.rcs_filt),
                "car_x_m": float(resolved_pose.x),
                "car_y_m": float(resolved_pose.y),
            }
        )
        if len(self._rcs_orbit_samples) > 12000:
            self._rcs_orbit_samples = self._rcs_orbit_samples[-12000:]

    def _build_orbit_polar_series_from_rows(
        self, rows: List[Dict[str, float]]
    ) -> Optional[Dict[str, Any]]:
        """极径为 RCS(dBsm)，极角按采样时间顺序线性映射到 [0, 2π]。

        Matplotlib 极坐标径向 r 需非负，内部用 floor 平移，刻度仍标真实 dBsm。
        相邻时间点 RCS 跳变达到 ORBIT_RCS_ADJACENT_JUMP_REJECT_DB 时剔除当前点。
        若数据来自 Cluster Raw 重放，rcs_filt 应与直线段相同：
        先行内 RCS00+RCS01 线性功率合并（见 _parse_cluster_rcs_csv / _cluster_rcs_rowwise_rcs00_rcs01_linear_power）。
        不使用 theta_rad 作极角；与几何方位角无关。
        """
        if len(rows) < 2:
            return None

        def _row_ok(r: Dict[str, float]) -> bool:
            try:
                t = float(r["t"])
                v = float(r["rcs_filt"])
                return bool(np.isfinite(t) and np.isfinite(v))
            except (KeyError, TypeError, ValueError):
                return False

        rows_sorted = sorted((r for r in rows if _row_ok(r)), key=lambda r: float(r["t"]))
        if len(rows_sorted) < 2:
            return None

        filtered_rows: List[Dict[str, float]] = []
        last_v: Optional[float] = None
        jump_thr = float(ORBIT_RCS_ADJACENT_JUMP_REJECT_DB)
        for r in rows_sorted:
            v = float(r["rcs_filt"])
            if last_v is not None and abs(v - last_v) >= jump_thr:
                continue
            filtered_rows.append(r)
            last_v = v
        if len(filtered_rows) < 2:
            return None

        t_arr = np.asarray([float(r["t"]) for r in filtered_rows], dtype=float)
        values = np.asarray([float(r["rcs_filt"]) for r in filtered_rows], dtype=float)
        span_t = float(t_arr[-1] - t_arr[0])
        if span_t <= 1e-12:
            angles = np.linspace(0.0, 2.0 * math.pi, t_arr.size, dtype=float, endpoint=True)
        else:
            angles = (t_arr - t_arr[0]) / span_t * (2.0 * math.pi)
        fit_angles, fit_values = self._bin_orbit_rcs_by_angle(
            angles,
            values,
            ORBIT_RCS_ANGLE_BIN_DEG,
        )

        span_deg = 360.0
        rmin = float(np.min(values))
        rmax = float(np.max(values))
        tick_step = max(float(ORBIT_RCS_RADIAL_TICK_STEP_DB), 1e-6)
        tick_start = math.floor(rmin / tick_step) * tick_step
        tick_end = math.ceil(rmax / tick_step) * tick_step
        if tick_end <= tick_start:
            tick_end = tick_start + tick_step
        tick_values = np.arange(tick_start, tick_end + tick_step * 0.5, tick_step, dtype=float)
        floor = tick_start - max(0.5, tick_step * 0.1)
        radii = values - floor
        tick_positions = tick_values - floor
        r_top = float(np.max(tick_positions)) + tick_step * 0.2
        return {
            "mode": "abs_dbsm",
            "angles": angles,
            "radii": radii,
            "values": values,
            "tick_values": tick_values,
            "tick_positions": tick_positions,
            "r_top": r_top,
            "span_deg": span_deg,
            "value_min": rmin,
            "value_max": rmax,
            "r_floor": floor,
            "radial_tick_step_db": tick_step,
            "raw_point_count": len(rows_sorted),
            "filtered_point_count": len(filtered_rows),
            "jump_reject_db": jump_thr,
            "orbit_angle_by_time": True,
            "fit_angles": fit_angles,
            "fit_values": fit_values,
            "angle_bin_deg": float(ORBIT_RCS_ANGLE_BIN_DEG),
            "fit_point_count": int(fit_angles.size),
        }

    @staticmethod
    def _bin_orbit_rcs_by_angle(
        angles: np.ndarray,
        values: np.ndarray,
        bin_deg: float = ORBIT_RCS_ANGLE_BIN_DEG,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """按平铺后的角度分箱，箱内 RCS 使用 dBsm 算术平均。"""
        a = np.asarray(angles, dtype=float)
        v = np.asarray(values, dtype=float)
        if a.size == 0 or v.size == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        n = min(int(a.size), int(v.size))
        a = a[:n]
        v = v[:n]
        mask = np.isfinite(a) & np.isfinite(v)
        if int(np.count_nonzero(mask)) < 2:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        a = a[mask]
        v = v[mask]

        two_pi = 2.0 * math.pi
        try:
            bin_deg_eff = float(bin_deg)
        except (TypeError, ValueError):
            bin_deg_eff = 1.0
        if not math.isfinite(bin_deg_eff):
            bin_deg_eff = 1.0
        bin_deg_eff = max(bin_deg_eff, 1e-3)
        n_bins = max(1, int(math.ceil(360.0 / bin_deg_eff)))
        bin_w = two_pi / float(n_bins)
        # 这里保留 [0, 2π] 的铺展语义；终点 2π 归入最后一个角度箱，不回绕到 0。
        a = np.clip(a, 0.0, np.nextafter(two_pi, 0.0))
        bin_ids = np.floor(a / bin_w).astype(np.int64)
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)

        by_bin: Dict[int, List[float]] = defaultdict(list)
        for bid, val in zip(bin_ids.tolist(), v.tolist()):
            fv = float(val)
            if math.isfinite(fv):
                by_bin[int(bid)].append(fv)
        if not by_bin:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        out_angles: List[float] = []
        out_values: List[float] = []
        for bid in sorted(by_bin.keys()):
            vals = by_bin[int(bid)]
            if not vals:
                continue
            out_angles.append((float(bid) + 0.5) * bin_w)
            out_values.append(float(np.mean(np.asarray(vals, dtype=float))))
        return np.asarray(out_angles, dtype=float), np.asarray(out_values, dtype=float)

    def _planned_segment_kind_for_index(self, seg_idx: int) -> Optional[str]:
        """返回规划器逐段类型：circle / line；无元数据时返回 None（由几何判定）。"""
        if seg_idx < 1:
            return None
        kinds = getattr(self, "_planned_segment_kinds", None) or []
        if seg_idx > len(kinds):
            return None
        raw = str(kinds[seg_idx - 1] or "").strip().lower()
        return raw if raw in ("circle", "line") else None

    def _planned_range_task_name_for_index(self, seg_idx: int) -> str:
        if seg_idx < 1:
            return ""
        names = list(getattr(self, "_planned_range_task_names", []) or [])
        if seg_idx > len(names):
            return ""
        return str(names[seg_idx - 1] or "").strip()

    def _segment_index_is_radial_named_line(self, seg_idx: int) -> bool:
        """星型测量中带任务名的测量往返直线段，不能被稳定化后的轻微曲率误判为曲线。"""
        if getattr(self, "_radial_measurement_spec", None) is None:
            return False
        name = self._planned_range_task_name_for_index(seg_idx)
        return self._parse_radial_rcs_task_name(name) is not None

    def _segment_geometry_is_straight(self, seg_idx: int, points: List[Tuple[float, float]]) -> bool:
        """分段是否按直线处理：优先规划器标注，其次折线曲率判据。"""
        kind = self._planned_segment_kind_for_index(seg_idx)
        if kind == "circle":
            return False
        if kind == "line":
            return len(points) >= 2
        if self._segment_index_is_radial_named_line(seg_idx):
            return len(points) >= 2
        return self._is_nearly_straight_segment(points)

    def _segment_index_is_arc_segment(self, seg_idx: int) -> bool:
        # 星型/径向测量轨迹：按「直线-距离RCS」工作流记录与绘图，
        # 不启用圆周RCS(40±gate)门控，否则不同朝向直线段会被误判为圆弧段导致 orbit_rcs 落盘。
        if getattr(self, "_radial_measurement_spec", None) is not None:
            return False
        ranges = getattr(self, "_planned_ranges", None) or []
        if seg_idx < 1 or seg_idx > len(ranges):
            return False
        if self._planned_segment_kind_for_index(seg_idx) == "circle":
            return True
        sr = self._normalize_segment_range(ranges[seg_idx - 1])
        start_idx, end_idx = int(sr[0]), int(sr[1])
        if end_idx <= start_idx:
            return False
        pts = self.loaded_path_points[start_idx : end_idx + 1]
        if len(pts) < 2:
            return False
        return not self._is_nearly_straight_segment(pts)

    def _segment_index_is_forward_straight_segment(self, seg_idx: Optional[int]) -> bool:
        if seg_idx is None:
            return False
        ranges = getattr(self, "_planned_ranges", None) or []
        if seg_idx < 1 or seg_idx > len(ranges):
            return False
        if self._planned_segment_kind_for_index(seg_idx) == "circle":
            return False
        sr = self._normalize_segment_range(ranges[seg_idx - 1])
        start_idx, end_idx = int(sr[0]), int(sr[1])
        speed_sign = float(sr[2])
        if end_idx <= start_idx:
            return False
        if speed_sign <= 0:
            return False
        pts = self.loaded_path_points[start_idx : end_idx + 1]
        if len(pts) < 2:
            return False
        return self._segment_geometry_is_straight(seg_idx, pts)

    def _segment_index_is_forward_curved_segment(self, seg_idx: Optional[int]) -> bool:
        """前进且为圆弧段（或几何非直线），用于分段轨迹中启动圆周 RCS（极坐标采样 CSV + 宽横向 Cluster CSV）。"""
        if seg_idx is None:
            return False
        ranges = getattr(self, "_planned_ranges", None) or []
        if seg_idx < 1 or seg_idx > len(ranges):
            return False
        sr = self._normalize_segment_range(ranges[seg_idx - 1])
        start_idx, end_idx = int(sr[0]), int(sr[1])
        speed_sign = float(sr[2])
        if end_idx <= start_idx:
            return False
        if speed_sign <= 0:
            return False
        if getattr(self, "_radial_measurement_spec", None) is not None:
            return False
        if self._planned_segment_kind_for_index(seg_idx) == "circle":
            return True
        if self._planned_segment_kind_for_index(seg_idx) == "line":
            return False
        pts = self.loaded_path_points[start_idx : end_idx + 1]
        if len(pts) < 2:
            return False
        return not self._is_nearly_straight_segment(pts)

    @staticmethod
    def _is_orbit_rcs_polar_csv(path: Path) -> bool:
        """是否为圆周极坐标采样 CSV（文件名含 __orbit_rcs__，或表头含极坐标列）。"""
        if path.suffix.lower() != ".csv":
            return False
        if "__orbit_rcs__" in path.name:
            return True
        try:
            with path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
            if not header:
                return False
            h = {str(x).strip().lower() for x in header if str(x).strip()}
            need = {"t", "theta_rad", "rcs_raw", "rcs_filt", "oid"}
            x_ok = ("x_m" in h or "x" in h) and ("y_m" in h or "y" in h)
            return need.issubset(h) and x_ok
        except OSError:
            return False

    @staticmethod
    def _cluster_csv_has_dri_cluster_header(path: Path) -> bool:
        """是否为 Cluster Raw（Time + DX00…RCS19）表头。"""
        if path.suffix.lower() != ".csv":
            return False
        try:
            with path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    if row[0].strip() == "Time" and any((c or "").strip() == "DX00" for c in row):
                        return True
        except OSError:
            return False
        return False

    @staticmethod
    def _cluster_csv_data_row_count(path: Path) -> int:
        """统计 Cluster Raw 表头后的数据行数，用于判断分段 CSV 是否真正写入。"""
        if path.suffix.lower() != ".csv":
            return 0
        count = 0
        header_seen = False
        try:
            with path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    if not header_seen:
                        if row[0].strip() == "Time" and any((c or "").strip() == "DX00" for c in row):
                            header_seen = True
                        continue
                    if any(str(c or "").strip() for c in row):
                        count += 1
        except OSError:
            return 0
        return count

    @staticmethod
    def _is_orbit_cluster_raw_csv(path: Path) -> bool:
        """是否为圆周段 Cluster Raw CSV（如 *_orbit_cluster_segXX_Raw_*.csv）。"""
        if path.suffix.lower() != ".csv":
            return False
        marker_tokens = ("orbit_cluster", "_orbit_", "__orbit_rcs__", "圆周")
        name_has_marker = any(token in path.name.casefold() for token in marker_tokens)
        metadata_has_marker = False
        try:
            with path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    row_text = " ".join(str(c or "").strip() for c in row).casefold()
                    metadata_has_marker = metadata_has_marker or any(
                        token in row_text for token in marker_tokens
                    )
                    if row[0].strip() == "Time" and any((c or "").strip() == "DX00" for c in row):
                        return bool(name_has_marker or metadata_has_marker)
        except OSError:
            return False
        return False

    def _parse_orbit_rcs_csv(self, file_path: str) -> List[Dict[str, float]]:
        path = Path(file_path)
        rows: List[Dict[str, float]] = []
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("圆周RCS CSV 无表头或为空")
            lower = {str(k).strip().lower(): k for k in reader.fieldnames if k is not None}

            def _col(*names: str) -> Optional[str]:
                for n in names:
                    if n in lower:
                        return lower[n]
                return None

            c_t = _col("t", "time_s", "time")
            c_th = _col("theta_rad")
            c_rr = _col("rcs_raw")
            c_rf = _col("rcs_filt", "rcs_filtered")
            c_x = _col("x_m", "x")
            c_y = _col("y_m", "y")
            c_oid = _col("oid", "cluster_id")
            if not all([c_t, c_th, c_rr, c_rf, c_x, c_y, c_oid]):
                raise ValueError("圆周RCS CSV 表头缺少必需列 (t,theta_rad,rcs_raw,rcs_filt,x_m,y_m,oid)")
            for raw in reader:
                if not raw:
                    continue
                try:
                    def _get(key: str) -> str:
                        v = raw.get(key)
                        if v is None:
                            raise ValueError
                        return str(v).strip()

                    rows.append(
                        {
                            "t": float(_get(c_t)),
                            "theta_rad": float(_get(c_th)),
                            "rcs_raw": float(_get(c_rr)),
                            "rcs_filt": float(_get(c_rf)),
                            "x": float(_get(c_x)),
                            "y": float(_get(c_y)),
                            "oid": int(float(_get(c_oid))),
                        }
                    )
                except (ValueError, TypeError, KeyError):
                    continue
        if len(rows) < 2:
            raise ValueError("圆周RCS CSV 有效点不足")
        return rows

    def _get_rcs_orbit_plot_series(self) -> Optional[Dict[str, Any]]:
        if self._orbit_polar_cached_series is not None:
            return self._orbit_polar_cached_series
        if not self._rcs_orbit_samples:
            return None

        rows_like: List[Dict[str, float]] = []
        for i, sample in enumerate(self._rcs_orbit_samples):
            try:
                v = float(sample.get("rcs_filt", float("nan")))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v):
                continue
            try:
                t = float(sample.get("timestamp", float("nan")))
            except (TypeError, ValueError):
                t = float("nan")
            if not np.isfinite(t):
                t = float(i)
            rows_like.append({"t": t, "rcs_filt": v})
        if len(rows_like) >= 2:
            ser = self._build_orbit_polar_series_from_rows(rows_like)
            if ser is not None:
                return ser

        return None

    def _create_log_panel(self) -> QtWidgets.QFrame:
        log_frame = QtWidgets.QFrame()
        log_frame.setObjectName("LogFrame")
        log_vbox = QtWidgets.QVBoxLayout(log_frame)
        log_vbox.setContentsMargins(
            self._compact_v(8),
            self._compact_v(8),
            self._compact_v(8),
            self._compact_v(8),
        )

        log_label = QtWidgets.QLabel("运行日志")
        log_label.setObjectName("PlotLabel")
        log_vbox.addWidget(log_label)

        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.log_edit.setMaximumBlockCount(800)
        self.log_edit.setMinimumHeight(self._compact_v(120))
        log_vbox.addWidget(self.log_edit, 1)

        log_btns = QtWidgets.QHBoxLayout()
        btn_clear_log = QtWidgets.QPushButton(ACTION_CLEAR_LOG_TEXT)
        btn_save_log = QtWidgets.QPushButton(ACTION_SAVE_LOG_TEXT)
        btn_export_log_csv = QtWidgets.QPushButton(ACTION_EXPORT_CSV_TEXT)
        btn_save_motion_data = QtWidgets.QPushButton(ACTION_SAVE_MOTION_DATA_TEXT)
        btn_emergency_stop = QtWidgets.QPushButton(ACTION_EMERGENCY_STOP_TEXT)
        btn_emergency_stop.setToolTip("立即停止当前轨迹/分段控制循环，并下发底盘零速")
        btn_emergency_stop.setStyleSheet(
            "background-color: #C62828; color: #FFFFFF; font-weight: bold; padding: 6px 10px;"
        )
        btn_emergency_stop.clicked.connect(self._on_emergency_stop_clicked)
        btn_clear_log.clicked.connect(self._on_clear_log)
        btn_save_log.clicked.connect(self._on_save_log)
        btn_export_log_csv.clicked.connect(self._on_export_metrics_csv)
        btn_save_motion_data.clicked.connect(self._on_save_motion_data)
        log_btns.addWidget(btn_clear_log)
        log_btns.addWidget(btn_save_log)
        log_btns.addWidget(btn_export_log_csv)
        log_btns.addWidget(btn_save_motion_data)
        log_btns.addWidget(btn_emergency_stop)
        log_btns.addStretch(1)
        log_vbox.addLayout(log_btns)
        return log_frame

    def _create_radar_target_frame(self) -> QtWidgets.QFrame:
        radar_frame = QtWidgets.QFrame()
        radar_frame.setObjectName("PlotFrame")
        radar_vbox = QtWidgets.QVBoxLayout(radar_frame)
        radar_vbox.setContentsMargins(
            self._compact_v(8),
            self._compact_v(8),
            self._compact_v(8),
            self._compact_v(8),
        )

        radar_label = QtWidgets.QLabel(RADAR_TARGET_CHECK_TITLE)
        radar_label.setObjectName("PlotLabel")
        radar_vbox.addWidget(radar_label)

        hint_label = QtWidgets.QLabel("点击雷达点可锁定目标，锁定结果会同步到主界面。")
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #546E7A;")
        radar_vbox.addWidget(hint_label)

        self.radar_plot = pg.PlotWidget()
        self.radar_plot.setLabel("left", "前方 DistLong(DX)", "m")
        self.radar_plot.setLabel("bottom", "横向 DistLat(DY)", "m")
        self.radar_plot.setXRange(-2.5, 2.5)
        self.radar_plot.setYRange(0, 62)
        self.radar_plot.setAspectLocked(True)
        self.radar_plot.showGrid(x=True, y=True, alpha=0.3)
        self.radar_plot.setMinimumSize(self._scaled_px(520), self._scaled_px(420))

        self.radar_scatter = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 0, 0, 120),
        )
        self.radar_scatter.sigClicked.connect(self._on_radar_point_clicked)
        self.radar_plot.addItem(self.radar_scatter)
        radar_vbox.addWidget(self.radar_plot, 1)
        return radar_frame

    def _build_auxiliary_plot_dialogs(self) -> None:
        # 仅在此处各创建一份控件，避免主窗口嵌套与弹窗重复构建导致 self.radar_plot 等被覆盖
        self._log_viewer_dialog = QtWidgets.QDialog(self)
        self._log_viewer_dialog.setWindowTitle("运行日志")
        self._log_viewer_dialog.resize(self._scaled_px(720), self._scaled_px(520))
        log_layout = QtWidgets.QVBoxLayout(self._log_viewer_dialog)
        log_layout.setContentsMargins(
            self._compact_v(12),
            self._compact_v(12),
            self._compact_v(12),
            self._compact_v(12),
        )
        log_layout.setSpacing(self._compact_v(10))
        log_layout.addWidget(self._create_log_panel(), 1)

        self._radar_target_dialog = QtWidgets.QDialog(self)
        self._radar_target_dialog.setWindowTitle("锁定雷达目标检查")
        self._radar_target_dialog.resize(self._scaled_px(760), self._scaled_px(640))
        radar_layout = QtWidgets.QVBoxLayout(self._radar_target_dialog)
        radar_layout.setContentsMargins(
            self._compact_v(12),
            self._compact_v(12),
            self._compact_v(12),
            self._compact_v(12),
        )
        radar_layout.setSpacing(self._compact_v(10))
        radar_layout.addWidget(self._create_radar_target_frame(), 1)

        self._rcs_viewer_dialog = QtWidgets.QDialog(self)
        self._rcs_viewer_dialog.setWindowTitle("RCS绘图")
        self._rcs_viewer_dialog.resize(self._scaled_px(860), self._scaled_px(720))
        rcs_layout = QtWidgets.QVBoxLayout(self._rcs_viewer_dialog)
        rcs_layout.setContentsMargins(
            self._compact_v(12),
            self._compact_v(12),
            self._compact_v(12),
            self._compact_v(12),
        )
        rcs_layout.setSpacing(self._compact_v(10))
        rcs_layout.addWidget(self._create_rcs_frame(), 1)

    def _place_radar_target_dialog_at_trajectory_top_right(self) -> None:
        """将「锁定雷达目标检查」弹窗置于轨迹规划图区域的右上角（外框对齐）。"""
        dialog = self._radar_target_dialog
        anchor = self.traj_plot
        top_right = anchor.mapToGlobal(anchor.rect().topRight())
        fr = dialog.frameGeometry()
        fr.moveTopRight(top_right)
        screen = QtWidgets.QApplication.screenAt(fr.center())
        if screen is None:
            screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            ag = screen.availableGeometry()
            dx = dy = 0
            if fr.left() < ag.left():
                dx = ag.left() - fr.left()
            elif fr.right() > ag.right():
                dx = ag.right() - fr.right()
            if fr.top() < ag.top():
                dy = ag.top() - fr.top()
            elif fr.bottom() > ag.bottom():
                dy = ag.bottom() - fr.bottom()
            if dx or dy:
                fr.translate(dx, dy)
        dialog.move(fr.topLeft())

    @staticmethod
    def _show_tool_dialog(dialog: QtWidgets.QDialog) -> None:
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _apply_theme(self) -> None:
        """应用主题样式"""
        card_radius = self._scaled_px(8)
        card_padding = self._scaled_px(8)
        info_header_font = self._scaled_px(13)
        info_header_padding = self._compact_v(8)
        plot_label_font = self._scaled_px(11)
        plot_label_padding = self._compact_v(4)
        button_padding_v = self._compact_v(4)
        button_padding_h = self._scaled_px(9)
        button_radius = self._scaled_px(4)
        button_min_height = self._compact_v(14)
        button_font = self._scaled_px(9)
        side_button_padding_v = self._compact_v(6)
        side_button_padding_h = self._scaled_px(8)
        side_button_min_height = self._compact_v(30)
        side_button_font = self._scaled_px(9)
        side_card_label_font = self._scaled_px(9)
        side_combo_font = self._scaled_px(9)
        side_combo_pad_v = self._compact_v(3)
        side_combo_pad_h = self._scaled_px(8)
        side_combo_min_h = self._compact_v(22)
        line_edit_padding_v = self._compact_v(5)
        line_edit_padding_h = self._scaled_px(8)
        group_radius = self._scaled_px(4)
        group_margin_top = self._compact_v(10)
        group_padding_top = self._compact_v(10)
        group_title_left = self._scaled_px(10)
        group_title_padding = self._scaled_px(5)
        radio_spacing = self._scaled_px(5)
        radio_indicator = self._scaled_px(13)
        radio_indicator_radius = self._scaled_px(7)
        stylesheet = """
            QMainWindow {{
                background-color: #f5f5f5;
            }}
            #MainContent {{
                background-color: #f5f5f5;
            }}
            #ControlPanel {{
                background-color: transparent;
            }}
            #InfoCard {{
                background-color: white;
                border-radius: {card_radius}px;
                padding: {card_padding}px;
            }}
            #ButtonCard {{
                background-color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(13, 71, 161, 0.14);
                border-radius: {card_radius}px;
                padding: {card_padding}px;
            }}
            #PlotFrame, #LogFrame {{
                background-color: white;
                border-radius: {card_radius}px;
            }}
            #InfoHeader {{
                font-size: {info_header_font}px;
                font-weight: bold;
                color: #0D47A1;
                padding-bottom: {info_header_padding}px;
            }}
            #PlotLabel {{
                font-size: {plot_label_font}px;
                font-weight: bold;
                color: #0D47A1;
                padding: {plot_label_padding}px;
            }}
            QPushButton {{
                background-color: #0D47A1;
                color: white;
                border: none;
                padding: {button_padding_v}px {button_padding_h}px;
                border-radius: {button_radius}px;
                font-weight: bold;
                min-height: {button_min_height}px;
                font-size: {button_font}px;
            }}
            #ButtonCard QPushButton {{
                padding: {side_button_padding_v}px {side_button_padding_h}px;
                min-height: {side_button_min_height}px;
                font-size: {side_button_font}px;
                background-color: rgba(13, 71, 161, 0.92);
                border: 1px solid rgba(255, 255, 255, 0.22);
            }}
            #ButtonCard QLabel {{
                font-size: {side_card_label_font}px;
            }}
            #ButtonCard QComboBox {{
                font-size: {side_combo_font}px;
                padding: {side_combo_pad_v}px {side_combo_pad_h}px;
                min-height: {side_combo_min_h}px;
            }}
            QPushButton:hover {{
                background-color: #1565C0;
            }}
            QPushButton:pressed {{
                background-color: #003C8F;
            }}
            QPushButton:disabled {{
                background-color: #90CAF9;
                color: #E3F2FD;
            }}
            #ButtonCard QPushButton:hover {{
                background-color: rgba(21, 101, 192, 0.96);
            }}
            #ButtonCard QPushButton:pressed {{
                background-color: rgba(0, 60, 143, 0.98);
            }}
            #ButtonCard QPushButton:disabled {{
                background-color: rgba(144, 202, 249, 0.75);
                color: rgba(255, 255, 255, 0.85);
                border: 1px solid rgba(255, 255, 255, 0.12);
            }}
            QLineEdit {{
                padding: {line_edit_padding_v}px {line_edit_padding_h}px;
                border: 1px solid #BBDEFB;
                border-radius: {button_radius}px;
                background-color: white;
            }}
            QLineEdit:focus {{
                border-color: #0D47A1;
            }}
            QGroupBox {{
                font-weight: bold;
                border: 1px solid #BBDEFB;
                border-radius: {group_radius}px;
                margin-top: {group_margin_top}px;
                padding-top: {group_padding_top}px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: {group_title_left}px;
                padding: 0 {group_title_padding}px 0 {group_title_padding}px;
            }}
            QRadioButton {{
                spacing: {radio_spacing}px;
            }}
            QRadioButton::indicator {{
                width: {radio_indicator}px;
                height: {radio_indicator}px;
            }}
            QRadioButton::indicator:unchecked {{
                border: 1px solid #0D47A1;
                border-radius: {radio_indicator_radius}px;
                background-color: white;
            }}
            QRadioButton::indicator:checked {{
                border: 1px solid #0D47A1;
                border-radius: {radio_indicator_radius}px;
                background-color: #0D47A1;
            }}
        """.format(
            card_radius=card_radius,
            card_padding=card_padding,
            info_header_font=info_header_font,
            info_header_padding=info_header_padding,
            plot_label_font=plot_label_font,
            plot_label_padding=plot_label_padding,
            button_padding_v=button_padding_v,
            button_padding_h=button_padding_h,
            button_radius=button_radius,
            button_min_height=button_min_height,
            button_font=button_font,
            side_button_padding_v=side_button_padding_v,
            side_button_padding_h=side_button_padding_h,
            side_button_min_height=side_button_min_height,
            side_button_font=side_button_font,
            side_card_label_font=side_card_label_font,
            side_combo_font=side_combo_font,
            side_combo_pad_v=side_combo_pad_v,
            side_combo_pad_h=side_combo_pad_h,
            side_combo_min_h=side_combo_min_h,
            line_edit_padding_v=line_edit_padding_v,
            line_edit_padding_h=line_edit_padding_h,
            group_radius=group_radius,
            group_margin_top=group_margin_top,
            group_padding_top=group_padding_top,
            group_title_left=group_title_left,
            group_title_padding=group_title_padding,
            radio_spacing=radio_spacing,
            radio_indicator=radio_indicator,
            radio_indicator_radius=radio_indicator_radius,
        )
        self.setStyleSheet(stylesheet)

    @staticmethod
    def _battery_indicator_detail(power: Dict[str, Any]) -> str:
        voltage = power.get("voltage")
        mode_text = str(power.get("mode_text") or "").strip()
        status = str(power.get("status") or "").strip()

        if voltage is not None:
            detail = f"{float(voltage):.1f}V"
            if mode_text and mode_text != "--":
                detail += f" {mode_text}"
            return detail

        return {
            "unavailable": "不可用",
            "read_failed": "读取失败",
            "read_error": "读取异常",
            "no_feedback": "无 0x211",
            "stale": "反馈过期",
            "invalid": "电压无效",
        }.get(status, "--")

    def _update_power_indicators(self, power_snapshot: Dict[str, Dict[str, Any]]) -> None:
        pi_info = power_snapshot.get("pi", {})
        car_info = power_snapshot.get("car", {})

        self.pi_battery_indicator.set_status(
            pi_info.get("percent"),
            self._battery_indicator_detail(pi_info),
            str(pi_info.get("status") or "unavailable"),
            str(pi_info.get("text") or "树莓派电量: --"),
        )
        self.car_battery_indicator.set_status(
            car_info.get("percent"),
            self._battery_indicator_detail(car_info),
            str(car_info.get("status") or "unavailable"),
            str(car_info.get("text") or "小车电量: --"),
        )

    def _handle_car_control_mode_transition(
        self,
        power_snapshot: Dict[str, Dict[str, Any]],
    ) -> None:
        car_info = power_snapshot.get("car", {})
        raw_mode = car_info.get("mode")
        try:
            mode = int(raw_mode) if raw_mode is not None else None
        except (TypeError, ValueError):
            mode = None

        previous_mode = self._last_car_control_mode
        self._last_car_control_mode = mode
        status_text = str(car_info.get("status") or "").strip().lower()
        feedback_age_s = car_info.get("feedback_age_s")
        try:
            feedback_age_s = (
                None if feedback_age_s is None else max(0.0, float(feedback_age_s))
            )
        except (TypeError, ValueError):
            feedback_age_s = None

        # 首次收到模式反馈时只做初始化，避免上电时非 CAN 状态误触发。
        if previous_mode is None:
            return
        if mode == previous_mode:
            return
        if feedback_age_s is not None and feedback_age_s > 1.5:
            return
        if status_text in {"stale", "no_feedback"}:
            return

        switched_out_of_can = previous_mode == 1 and mode != 1
        remote_priority = (
            mode == 3
            or status_text == "remote"
        )
        non_can_control = (
            mode is not None
            and mode != 1
            and status_text in {"remote", "standby", "mode_unknown"}
        )
        if not (switched_out_of_can and (remote_priority or non_can_control)):
            return

        mode_text = str(car_info.get("mode_text") or "").strip()
        if not mode_text and mode is not None:
            mode_text = f"模式{mode}"
        handover_to_remote = bool(remote_priority)
        reason = (
            "检测到底盘切换到手柄控制"
            if handover_to_remote
            else f"检测到底盘切换到{mode_text or '非CAN控制'}"
        )
        self._handle_motion_session_interrupt(
            reason,
            request_stop=True,
            trim_motion_record_to_now=handover_to_remote,
            save_rcs_raw=not handover_to_remote,
            popup_title="\u63a7\u5236\u6a21\u5f0f\u5207\u6362",
            popup_text=(
                f"{reason}\n已自动停止当前底盘运动指令下发。"
                + (
                    "\n已自动保存切换前的运动数据，切换后的手柄运动不记录。"
                    if handover_to_remote
                    else "\n运动数据将自动保存。"
                )
            ),
        )

    def _log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        self._runtime_log_lines.append(line)
        if len(self._runtime_log_lines) > 2000:
            self._runtime_log_lines = self._runtime_log_lines[-2000:]
        if hasattr(self, "log_edit") and self.log_edit is not None:
            self.log_edit.appendPlainText(line)
            sb = self.log_edit.verticalScrollBar()
            sb.setValue(sb.maximum())
        else:
            print(line)

    def _emit_tracking_metrics(self, record: Dict[str, Any]) -> None:
        self.tracking_metrics_reported.emit(dict(record))

    @staticmethod
    def _csv_float_text(value: Any, digits: int = 6) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        if not math.isfinite(number):
            return ""
        return f"{number:.{digits}f}"

    @staticmethod
    def _csv_int_text(value: Any) -> str:
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return ""

    def _compact_tracking_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        compact: Dict[str, Any] = {}
        for key in _TRACKING_MOTION_INTERNAL_FIELDS:
            if key in sample:
                compact[key] = sample.get(key)

        compact["run_key"] = str(sample.get("run_key") or "")
        compact["run_label"] = str(sample.get("run_label") or "轨迹")
        compact["tracking_mode"] = str(sample.get("tracking_mode") or "stanley")
        compact["segment_kind"] = str(sample.get("segment_kind") or "")
        compact["segment_trajectory_name"] = str(sample.get("segment_trajectory_name") or "")

        _, motion_direction, _ = self._motion_direction_fields(sample)
        compact["motion_direction"] = motion_direction
        return compact

    @staticmethod
    def _sample_has_valid_yaw_feedback(sample: Dict[str, Any]) -> bool:
        raw_value = sample.get("yaw_rate_feedback_valid")
        if raw_value is None:
            return False
        try:
            return int(raw_value) != 0
        except (TypeError, ValueError):
            return False

    def _append_tracking_sample(self, sample: Dict[str, Any]) -> None:
        run_key = str(sample.get("run_key") or "")
        if not run_key:
            return
        sample_copy = self._compact_tracking_sample(sample)
        with self._tracking_sample_lock:
            self._tracking_sample_buffers.setdefault(run_key, []).append(sample_copy)
            if getattr(self, "_motion_full_session_active", False):
                self._motion_full_session_accumulator.append(dict(sample_copy))

    def _arm_trim_next_motion_record(self, cutoff_ts: float, reason: str) -> None:
        self._trim_next_motion_record = True
        self._trim_next_motion_record_ts = max(0.0, float(cutoff_ts))
        self._trim_next_motion_record_reason = str(reason or "").strip()

    def _clear_trim_next_motion_record(self) -> None:
        self._trim_next_motion_record = False
        self._trim_next_motion_record_ts = 0.0
        self._trim_next_motion_record_reason = ""

    def _maybe_trim_motion_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if not bool(getattr(self, "_trim_next_motion_record", False)):
            return record

        exit_reason = str(record.get("exit_reason") or "").strip().lower()
        if exit_reason != "stop_flag":
            return record

        cutoff_ts = float(getattr(self, "_trim_next_motion_record_ts", 0.0) or 0.0)
        if cutoff_ts <= 0.0:
            return record

        samples = list(record.get("motion_samples") or [])
        if not samples:
            return record

        trimmed_samples = []
        for sample in samples:
            raw_ts = sample.get("timestamp")
            try:
                sample_ts = float(raw_ts)
            except (TypeError, ValueError):
                sample_ts = 0.0
            if sample_ts <= cutoff_ts + 1e-3:
                trimmed_samples.append(dict(sample))

        if len(trimmed_samples) >= len(samples):
            return record

        trimmed = dict(record)
        trimmed["motion_samples"] = trimmed_samples
        trimmed["record_trimmed_at_handover"] = 1
        if self._trim_next_motion_record_reason:
            trimmed["stop_reason"] = str(trimmed.get("stop_reason") or self._trim_next_motion_record_reason)
        if not trimmed_samples:
            trimmed["samples"] = 0
            trimmed["feedback_samples"] = 0
            trimmed["duration_s"] = 0.0
            trimmed["peak_abs_lateral_error_m"] = 0.0
            trimmed["peak_abs_heading_error_deg"] = 0.0
            trimmed["peak_abs_yaw_rate_error_radps"] = 0.0
            trimmed["peak_abs_cmd_w_radps"] = 0.0
            trimmed["peak_abs_feedback_w_radps"] = 0.0
            trimmed["motion_distance_total_m"] = 0.0
            trimmed["motion_distance_total_signed_m"] = 0.0
            return trimmed

        last_sample = dict(trimmed_samples[-1])
        distance_m, signed_distance_m, distance_kind, distance_label = self._motion_distance_fields(last_sample)
        trimmed["samples"] = len(trimmed_samples)
        trimmed["feedback_samples"] = sum(
            1
            for sample in trimmed_samples
            if self._sample_has_valid_yaw_feedback(sample)
        )
        try:
            trimmed["duration_s"] = max(0.0, float(last_sample.get("relative_time_s", 0.0)))
        except (TypeError, ValueError):
            trimmed["duration_s"] = float(record.get("duration_s", 0.0) or 0.0)
        trimmed["peak_abs_lateral_error_m"] = max(
            abs(float(sample.get("lateral_error_m", 0.0) or 0.0))
            for sample in trimmed_samples
        )
        trimmed["peak_abs_heading_error_deg"] = max(
            abs(math.degrees(float(sample.get("heading_error_rad", 0.0) or 0.0)))
            for sample in trimmed_samples
        )
        trimmed["peak_abs_yaw_rate_error_radps"] = max(
            abs(float(sample.get("yaw_rate_error_radps", 0.0) or 0.0))
            for sample in trimmed_samples
        )
        trimmed["peak_abs_cmd_w_radps"] = max(
            abs(float(sample.get("cmd_w_radps", 0.0) or 0.0))
            for sample in trimmed_samples
        )
        trimmed["peak_abs_feedback_w_radps"] = max(
            abs(float(sample.get("feedback_w_radps", 0.0) or 0.0))
            for sample in trimmed_samples
        )
        trimmed["motion_distance_total_m"] = distance_m
        trimmed["motion_distance_total_signed_m"] = signed_distance_m
        trimmed["motion_distance_kind"] = distance_kind
        trimmed["motion_distance_label"] = distance_label
        return trimmed

    def _pop_tracking_samples(self, run_key: str) -> List[Dict[str, Any]]:
        if not run_key:
            return []
        with self._tracking_sample_lock:
            samples = self._tracking_sample_buffers.pop(run_key, [])
        return [dict(sample) for sample in samples]

    def _on_tracking_metrics_reported(self, record: Dict[str, Any]) -> None:
        rec = dict(record)
        if not bool(rec.get("completed")):
            exit_reason = str(rec.get("exit_reason") or "").strip().lower()
            if exit_reason == "stop_flag":
                stop_reason = self._get_recent_motion_interrupt_reason()
                if stop_reason:
                    rec["stop_reason"] = stop_reason
        rec["exit_reason_text"] = self._describe_tracking_exit_reason(rec)
        rec["motion_samples"] = self._pop_tracking_samples(str(rec.get("run_key") or ""))
        rec = self._maybe_trim_motion_record(rec)
        if getattr(self, "_motion_full_session_active", False):
            self._log(self._format_tracking_metrics_summary(rec))
            self._clear_trim_next_motion_record()
            er = str(rec.get("exit_reason") or "").strip().lower()
            if er in ("pose_lost", "pose_stale", "stop_flag"):
                self._maybe_popup_tracking_stop_reason(rec)
            return
        rec["record_index"] = len(self._tracking_run_records) + 1
        self._tracking_run_records.append(rec)
        self._clear_trim_next_motion_record()
        if int(rec.get("record_trimmed_at_handover", 0) or 0) != 0:
            self._log("检测到手柄接管，运动数据已按切换时刻截断后保存。")
        self._log(self._format_tracking_metrics_summary(rec))
        self._autosave_motion_record(rec)
        self._maybe_popup_tracking_stop_reason(rec)
        run_label = str(rec.get("run_label") or "").strip()
        if self._motion_active and run_label and not run_label.startswith("分段"):
            self._on_motion_run_finished(run_label)

    def _finalize_full_motion_session_record(self) -> None:
        """将一次规划执行中多段采样合并为一条运动记录（CSV 一条），各行仍保留 segment_index 供分析 UI 二次截取。"""
        samples = [dict(s) for s in self._motion_full_session_accumulator]
        if not samples:
            return
        def _seg_sort_key(samp: Dict[str, Any]) -> Tuple[float, int]:
            try:
                ts = float(samp.get("timestamp") or 0.0)
            except (TypeError, ValueError):
                ts = 0.0
            try:
                si = int(samp.get("segment_index") or 0)
            except (TypeError, ValueError):
                si = 0
            return ts, si

        samples.sort(key=_seg_sort_key)
        try:
            t0 = float(samples[0].get("timestamp") or 0.0)
        except (TypeError, ValueError):
            t0 = 0.0
        session_wall = time.time()
        unified_key = f"{session_wall:.6f}|完整运动"
        for s in samples:
            try:
                ts = float(s.get("timestamp") or t0)
            except (TypeError, ValueError):
                ts = t0
            s["relative_time_s"] = max(0.0, ts - t0)
            s["run_key"] = unified_key
            s["run_label"] = "完整运动"
        last = dict(samples[-1])
        rec: Dict[str, Any] = {
            "timestamp": float(last.get("timestamp") or time.time()),
            "run_key": unified_key,
            "run_label": "完整运动",
            "tracking_mode": "mixed",
            "motion_session_unified": 1,
            "completed": True,
            "exit_reason": "goal_arrived",
            "exit_reason_text": "完整规划运动（多段合并）",
            "motion_samples": samples,
            "samples": len(samples),
            "speed_mps": 0.0,
        }
        for key in ("profile_speed_mps", "desired_v_mps", "cmd_v_mps"):
            try:
                v = abs(float(last.get(key) or 0.0))
                if v > 1e-6:
                    rec["speed_mps"] = v
                    break
            except (TypeError, ValueError):
                continue
        try:
            rec["duration_s"] = max(0.0, float(last.get("relative_time_s") or 0.0))
        except (TypeError, ValueError):
            rec["duration_s"] = 0.0
        rec["feedback_samples"] = sum(
            1 for s in samples if self._sample_has_valid_yaw_feedback(s)
        )
        rec["peak_abs_lateral_error_m"] = max(
            abs(float(s.get("lateral_error_m") or 0.0)) for s in samples
        )
        rec["peak_abs_heading_error_deg"] = max(
            abs(math.degrees(float(s.get("heading_error_rad") or 0.0))) for s in samples
        )
        rec["peak_abs_yaw_rate_error_radps"] = max(
            abs(float(s.get("yaw_rate_error_radps") or 0.0)) for s in samples
        )
        rec["peak_abs_cmd_w_radps"] = max(
            abs(float(s.get("cmd_w_radps") or 0.0)) for s in samples
        )
        rec["peak_abs_feedback_w_radps"] = max(
            abs(float(s.get("feedback_w_radps") or 0.0)) for s in samples
        )
        dist_m, signed_m, kind, dist_label = self._motion_distance_fields(last)
        rec["motion_distance_total_m"] = dist_m
        rec["motion_distance_total_signed_m"] = signed_m
        rec["motion_distance_kind"] = kind
        rec["motion_distance_label"] = dist_label
        rec["exit_reason_text"] = self._describe_tracking_exit_reason(rec)
        rec["record_index"] = len(self._tracking_run_records) + 1
        self._tracking_run_records.append(rec)
        self._log(self._format_tracking_metrics_summary(rec))
        self._autosave_motion_record(rec)

    @staticmethod
    def _tracking_mode_label(mode: Optional[str]) -> str:
        value = str(mode or "stanley").strip().lower()
        if value == "stanley":
            return "Stanley"
        if value == "stanley_pid":
            return "Stanley + PID"
        if value == "circle_orbit":
            return "Circle Orbit"
        if value == "mixed":
            return "多段合并"
        # 兼容历史记录：旧数据可能仍写入 pid
        return "PID"

    @staticmethod
    def _motion_direction_fields(record: Dict[str, Any]) -> Tuple[int, str, str]:
        sign = 1
        raw_sign = record.get("speed_sign")
        if raw_sign is not None:
            try:
                sign = -1 if float(raw_sign) < 0.0 else 1
            except (TypeError, ValueError):
                sign = 1
        else:
            for key in ("speed_mps", "profile_speed_mps", "desired_v_mps", "cmd_v_mps"):
                raw_value = record.get(key)
                if raw_value is None:
                    continue
                try:
                    sign = -1 if float(raw_value) < 0.0 else 1
                    break
                except (TypeError, ValueError):
                    continue
        direction = str(record.get("motion_direction") or "").strip().lower()
        if direction not in ("forward", "reverse"):
            direction = "reverse" if sign < 0 else "forward"
        label = str(record.get("motion_direction_label") or "").strip()
        if not label:
            label = "倒回" if direction == "reverse" else "前进"
        return sign, direction, label

    def _motion_distance_fields(
        self,
        record: Dict[str, Any],
        *,
        total: bool = False,
    ) -> Tuple[float, float, str, str]:
        speed_sign, _, _ = self._motion_direction_fields(record)
        segment_kind = str(record.get("segment_kind") or "").strip().lower()
        tracking_mode = str(record.get("tracking_mode") or "").strip().lower()

        if total:
            raw_distance = record.get("motion_distance_total_m", record.get("motion_distance_m"))
            raw_signed = record.get(
                "motion_distance_total_signed_m",
                record.get("motion_distance_signed_m"),
            )
        else:
            raw_distance = record.get("motion_distance_m", record.get("path_s_m"))
            raw_signed = record.get("motion_distance_signed_m")

        try:
            distance = max(0.0, float(raw_distance))
        except (TypeError, ValueError):
            distance = 0.0

        kind = str(record.get("motion_distance_kind") or "").strip()
        label = str(record.get("motion_distance_label") or "").strip()
        if not kind:
            if tracking_mode == "circle_orbit" or segment_kind == "circle":
                kind = "circle_segment"
            elif segment_kind == "line":
                kind = "line_reverse" if speed_sign < 0 else "line_forward"
            else:
                kind = "reverse_segment" if speed_sign < 0 else "forward_segment"
        if not label:
            if kind == "circle_segment":
                label = "圆周段"
            elif kind == "line_reverse":
                label = "直线段后退向"
            elif kind == "line_forward":
                label = "直线段前进向"
            elif kind == "reverse_segment":
                label = "后退向"
            else:
                label = "前进向"

        try:
            signed_distance = float(raw_signed)
        except (TypeError, ValueError):
            signed_distance = distance if kind == "circle_segment" else float(speed_sign * distance)
        return distance, signed_distance, kind, label

    def _reset_motion_stop_context(self) -> None:
        self._last_motion_interrupt_reason = ""
        self._last_motion_interrupt_ts = 0.0

    def _remember_motion_interrupt_reason(self, reason: str) -> None:
        text = str(reason or "").strip()
        if not text:
            return
        self._last_motion_interrupt_reason = text
        self._last_motion_interrupt_ts = time.time()

    def _get_recent_motion_interrupt_reason(self, max_age_s: float = 15.0) -> str:
        text = str(getattr(self, "_last_motion_interrupt_reason", "") or "").strip()
        if not text:
            return ""
        ts = float(getattr(self, "_last_motion_interrupt_ts", 0.0) or 0.0)
        if ts <= 0.0:
            return ""
        if time.time() - ts > max(0.5, float(max_age_s)):
            return ""
        return text

    def _show_motion_stop_popup(self, title: str, text: str) -> None:
        popup_title = str(title or "\u8fd0\u52a8\u505c\u8f66").strip() or "\u8fd0\u52a8\u505c\u8f66"
        popup_text = str(text or "").strip()
        if not popup_text:
            return
        now = time.time()
        popup_key = f"{popup_title}|{popup_text}"
        if (
            popup_key == self._motion_stop_last_popup_key
            and (now - self._motion_stop_last_popup_ts) < float(self._motion_stop_popup_cooldown_s)
        ):
            return
        self._motion_stop_last_popup_key = popup_key
        self._motion_stop_last_popup_ts = now
        QtWidgets.QMessageBox.warning(self, popup_title, popup_text)

    def _describe_tracking_exit_reason(self, record: Dict[str, Any]) -> str:
        exit_reason = str(record.get("exit_reason") or "").strip().lower()
        stop_reason = str(record.get("stop_reason") or "").strip()
        completed = bool(record.get("completed"))
        if completed or exit_reason == "goal_arrived":
            return "\u6b63\u5e38\u5b8c\u6210\uff1a\u5df2\u5230\u8fbe\u76ee\u6807"
        if exit_reason == "pose_stale":
            return "\u5f02\u5e38\u505c\u8f66\uff1a\u5b9a\u4f4d\u6570\u636e\u8d85\u65f6"
        if exit_reason == "pose_lost":
            return "\u5f02\u5e38\u505c\u8f66\uff1a\u65e0\u6cd5\u83b7\u53d6\u5f53\u524d\u4f4d\u59ff"
        if exit_reason == "stop_flag":
            if stop_reason:
                return f"\u5f02\u5e38\u505c\u8f66\uff1a{stop_reason}"
            return "\u5f02\u5e38\u505c\u8f66\uff1a\u6536\u5230\u505c\u6b62\u6216\u6025\u505c\u6307\u4ee4"
        if exit_reason == "loop_exit":
            return "\u5f02\u5e38\u505c\u8f66\uff1a\u63a7\u5236\u5faa\u73af\u63d0\u524d\u9000\u51fa"
        if exit_reason:
            return f"\u5f02\u5e38\u505c\u8f66\uff1a{exit_reason}"
        return "\u5f02\u5e38\u505c\u8f66\uff1a\u672a\u77e5\u539f\u56e0"

    def _maybe_popup_tracking_stop_reason(self, record: Dict[str, Any]) -> None:
        if bool(record.get("completed")):
            return
        exit_reason = str(record.get("exit_reason") or "").strip().lower()
        if exit_reason == "goal_arrived":
            return
        stop_reason = str(record.get("stop_reason") or "").strip()
        if exit_reason == "stop_flag" and stop_reason:
            return
        label = str(record.get("run_label") or "\u8f68\u8ff9").strip() or "\u8f68\u8ff9"
        mode_label = self._tracking_mode_label(str(record.get("tracking_mode") or "stanley"))
        detail = str(record.get("exit_reason_text") or self._describe_tracking_exit_reason(record))
        popup_text = (
            f"\u8fd0\u52a8\u4efb\u52a1: {label}\n"
            f"\u63a7\u5236\u6a21\u5f0f: {mode_label}\n"
            f"\u505c\u8f66\u539f\u56e0: {detail}"
        )
        self._show_motion_stop_popup("\u8fd0\u52a8\u505c\u8f66\u539f\u56e0", popup_text)

    def _format_tracking_metrics_summary(self, record: Dict[str, Any]) -> str:
        record_index = int(record.get("record_index", len(self._tracking_run_records)))
        label = str(record.get("run_label") or "轨迹")
        mode = str(record.get("tracking_mode") or "stanley")
        mode_label = self._tracking_mode_label(mode)
        _, _, direction_label = self._motion_direction_fields(record)
        started_at = time.strftime(
            "%H:%M:%S", time.localtime(float(record.get("timestamp", time.time())))
        )
        speed_mps = float(record.get("speed_mps", 0.0))
        peak_lat = float(record.get("peak_abs_lateral_error_m", 0.0))
        peak_head_deg = float(record.get("peak_abs_heading_error_deg", 0.0))
        peak_yaw_rate_err = float(record.get("peak_abs_yaw_rate_error_radps", 0.0))
        peak_w = float(record.get("peak_abs_cmd_w_radps", 0.0))
        peak_fb_w = float(record.get("peak_abs_feedback_w_radps", 0.0))
        feedback_samples = int(record.get("feedback_samples", 0))
        duration_s = float(record.get("duration_s", 0.0))
        status = str(
            record.get("exit_reason_text")
            or self._describe_tracking_exit_reason(record)
        )
        feedback_text = (
            f"{peak_fb_w:.3f}rad/s" if feedback_samples > 0 else "N/A"
        )
        return (
            f"实验记录#{record_index} | 开始={started_at} | {label} | 模式={mode_label} | 方向={direction_label} | 速度={speed_mps:.2f}m/s | "
            f"横向误差峰值={peak_lat:.3f}m | 航向误差峰值={peak_head_deg:.2f}deg | "
            f"角速度跟踪误差峰值={peak_yaw_rate_err:.3f}rad/s | "
            f"命令角速度峰值(|w_cmd|)={peak_w:.3f}rad/s | "
            f"反馈角速度峰值(|w_fb|)={feedback_text} | 时长={duration_s:.2f}s | 结束={status}"
        )

    def _compose_runtime_log_text(self) -> str:
        text = "\n".join(self._runtime_log_lines)
        lines: List[str] = []
        if text:
            lines.append(text.rstrip())
        if self._tracking_run_records:
            if lines:
                lines.append("")
            lines.append("===== 实验指标汇总 =====")
            for idx, record in enumerate(self._tracking_run_records, start=1):
                rec = dict(record)
                rec["record_index"] = idx
                lines.append(self._format_tracking_metrics_summary(rec))
        result = "\n".join(lines).rstrip()
        return result + ("\n" if result else "")

    def _on_clear_log(self) -> None:
        self._runtime_log_lines.clear()
        if hasattr(self, "log_edit") and self.log_edit is not None:
            self.log_edit.clear()
        self._tracking_run_records.clear()
        with self._tracking_sample_lock:
            self._tracking_sample_buffers.clear()
        self._motion_full_session_accumulator.clear()
        self._motion_full_session_active = False
        self._clear_trim_next_motion_record()
        self._reset_motion_stop_context()

    def _on_save_log(self) -> None:
        default_name = time.strftime("runtime_log_%Y%m%d_%H%M%S.txt")
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存运行日志",
            str(Path(self._motion_save_dir) / default_name),
            "文本文件 (*.txt);;日志文件 (*.log);;所有文件 (*)",
        )
        if not filename:
            return
        suffixes = (".txt", ".log")
        if not filename.lower().endswith(suffixes):
            filename += ".txt"
        content = self._compose_runtime_log_text()
        try:
            Path(filename).write_text(content, encoding="utf-8")
            self._log(f"运行日志已保存: {filename}")
            QtWidgets.QMessageBox.information(self, "保存成功", f"运行日志已保存至:\n{filename}")
        except Exception as exc:
            self._log(f"运行日志保存失败: {exc}")
            QtWidgets.QMessageBox.warning(self, "保存失败", f"运行日志保存失败:\n{exc}")

    def _tracking_metrics_csv_headers(self) -> List[str]:
        return [
            "record_index",
            "timestamp",
            "started_at",
            "run_label",
            "tracking_mode",
            "tracking_mode_label",
            "speed_mps",
            "speed_sign",
            "motion_direction",
            "motion_direction_label",
            "motion_distance_total_m",
            "motion_distance_total_signed_m",
            "motion_distance_kind",
            "motion_distance_label",
            "nominal_speed_abs_mps",
            "lookahead_base_m",
            "arrival_dist_m",
            "slow_down_dist_m",
            "stanley_gain",
            "stanley_softening_distance_m",
            "stanley_max_bias_deg",
            "lateral_pid_kp",
            "lateral_pid_ki",
            "lateral_pid_kd",
            "heading_pid_kp",
            "heading_pid_ki",
            "heading_pid_kd",
            "yaw_rate_pid_kp",
            "yaw_rate_pid_ki",
            "yaw_rate_pid_kd",
            "segment_index",
            "segment_kind",
            "segment_trajectory_name",
            "segment_start_idx",
            "segment_end_idx",
            "segment_cruise_speed_mps",
            "segment_accel_dist_m",
            "segment_decel_dist_m",
            "segment_start_speed_mps",
            "segment_end_speed_mps",
            "waypoints_count",
            "duration_s",
            "samples",
            "feedback_samples",
            "peak_abs_lateral_error_m",
            "peak_abs_heading_error_deg",
            "peak_abs_yaw_rate_error_radps",
            "peak_abs_cmd_w_radps",
            "peak_abs_feedback_w_radps",
            "completed",
            "exit_reason",
            "exit_reason_text",
            "stop_reason",
        ]

    def _tracking_metrics_csv_row(
        self,
        record_index: int,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        timestamp = float(record.get("timestamp", time.time()))
        mode = str(record.get("tracking_mode") or "stanley")
        speed_sign, motion_direction, motion_direction_label = self._motion_direction_fields(record)
        motion_distance_total_m, motion_distance_total_signed_m, motion_distance_kind, motion_distance_label = (
            self._motion_distance_fields(record, total=True)
        )
        return {
            "record_index": record_index,
            "timestamp": f"{timestamp:.3f}",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
            "run_label": str(record.get("run_label") or "轨迹"),
            "tracking_mode": mode,
            "tracking_mode_label": self._tracking_mode_label(mode),
            "speed_mps": f"{float(record.get('speed_mps', 0.0)):.6f}",
            "speed_sign": int(speed_sign),
            "motion_direction": motion_direction,
            "motion_direction_label": motion_direction_label,
            "motion_distance_total_m": f"{motion_distance_total_m:.6f}",
            "motion_distance_total_signed_m": f"{motion_distance_total_signed_m:.6f}",
            "motion_distance_kind": motion_distance_kind,
            "motion_distance_label": motion_distance_label,
            "nominal_speed_abs_mps": f"{float(record.get('nominal_speed_abs_mps', abs(float(record.get('speed_mps', 0.0))))):.6f}",
            "lookahead_base_m": f"{float(record.get('lookahead_base_m', 0.0)):.6f}",
            "arrival_dist_m": f"{float(record.get('arrival_dist_m', 0.0)):.6f}",
            "slow_down_dist_m": f"{float(record.get('slow_down_dist_m', 0.0)):.6f}",
            "stanley_gain": f"{float(record.get('stanley_gain', 0.0)):.6f}",
            "stanley_softening_distance_m": f"{float(record.get('stanley_softening_distance_m', 0.0)):.6f}",
            "stanley_max_bias_deg": f"{float(record.get('stanley_max_bias_deg', 0.0)):.6f}",
            "lateral_pid_kp": f"{float(record.get('lateral_pid_kp', 0.0)):.6f}",
            "lateral_pid_ki": f"{float(record.get('lateral_pid_ki', 0.0)):.6f}",
            "lateral_pid_kd": f"{float(record.get('lateral_pid_kd', 0.0)):.6f}",
            "heading_pid_kp": f"{float(record.get('heading_pid_kp', 0.0)):.6f}",
            "heading_pid_ki": f"{float(record.get('heading_pid_ki', 0.0)):.6f}",
            "heading_pid_kd": f"{float(record.get('heading_pid_kd', 0.0)):.6f}",
            "yaw_rate_pid_kp": f"{float(record.get('yaw_rate_pid_kp', 0.0)):.6f}",
            "yaw_rate_pid_ki": f"{float(record.get('yaw_rate_pid_ki', 0.0)):.6f}",
            "yaw_rate_pid_kd": f"{float(record.get('yaw_rate_pid_kd', 0.0)):.6f}",
            "segment_index": int(record.get("segment_index", 0)),
            "segment_kind": str(record.get("segment_kind") or ""),
            "segment_trajectory_name": str(record.get("segment_trajectory_name") or ""),
            "segment_start_idx": int(record.get("segment_start_idx", 0)),
            "segment_end_idx": int(record.get("segment_end_idx", 0)),
            "segment_cruise_speed_mps": f"{float(record.get('segment_cruise_speed_mps', 0.0)):.6f}",
            "segment_accel_dist_m": f"{float(record.get('segment_accel_dist_m', 0.0)):.6f}",
            "segment_decel_dist_m": f"{float(record.get('segment_decel_dist_m', 0.0)):.6f}",
            "segment_start_speed_mps": f"{float(record.get('segment_start_speed_mps', 0.0)):.6f}",
            "segment_end_speed_mps": f"{float(record.get('segment_end_speed_mps', 0.0)):.6f}",
            "waypoints_count": int(record.get("waypoints_count", 0)),
            "duration_s": f"{float(record.get('duration_s', 0.0)):.6f}",
            "samples": int(record.get("samples", 0)),
            "feedback_samples": int(record.get("feedback_samples", 0)),
            "peak_abs_lateral_error_m": f"{float(record.get('peak_abs_lateral_error_m', 0.0)):.6f}",
            "peak_abs_heading_error_deg": f"{float(record.get('peak_abs_heading_error_deg', 0.0)):.6f}",
            "peak_abs_yaw_rate_error_radps": f"{float(record.get('peak_abs_yaw_rate_error_radps', 0.0)):.6f}",
            "peak_abs_cmd_w_radps": f"{float(record.get('peak_abs_cmd_w_radps', 0.0)):.6f}",
            "peak_abs_feedback_w_radps": f"{float(record.get('peak_abs_feedback_w_radps', 0.0)):.6f}",
            "completed": int(bool(record.get("completed"))),
            "exit_reason": str(record.get("exit_reason") or ""),
            "exit_reason_text": str(
                record.get("exit_reason_text") or self._describe_tracking_exit_reason(record)
            ),
            "stop_reason": str(record.get("stop_reason") or ""),
        }

    def _on_export_metrics_csv(self) -> None:
        if not self._tracking_run_records:
            QtWidgets.QMessageBox.information(self, "无可导出数据", "当前没有实验记录可导出。")
            return

        default_name = time.strftime("tracking_metrics_%Y%m%d_%H%M%S.csv")
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出实验记录 CSV",
            str(Path(self._motion_save_dir) / default_name),
            "CSV 文件 (*.csv);;所有文件 (*)",
        )
        if not filename:
            return
        if not filename.lower().endswith(".csv"):
            filename += ".csv"

        headers = self._tracking_metrics_csv_headers()
        try:
            with open(filename, "w", encoding="utf-8-sig", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=headers)
                writer.writeheader()
                for idx, record in enumerate(self._tracking_run_records, start=1):
                    writer.writerow(self._tracking_metrics_csv_row(idx, record))
            self._refresh_data_analysis_window(focus_path=Path(filename))
            self._log(f"实验记录 CSV 已导出: {filename}")
            QtWidgets.QMessageBox.information(self, "导出成功", f"实验记录 CSV 已保存至:\n{filename}")
        except Exception as exc:
            self._log(f"实验记录 CSV 导出失败: {exc}")
            QtWidgets.QMessageBox.warning(self, "导出失败", f"实验记录 CSV 导出失败:\n{exc}")

    def _tracking_motion_sample_headers(self) -> List[str]:
        return list(_TRACKING_MOTION_TRACE_FIELDS)

    def _tracking_motion_sample_row(
        self,
        record_index: int,
        sample_index: int,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        del record_index, sample_index
        _, motion_direction, _ = self._motion_direction_fields(sample)
        return {
            "timestamp": self._csv_float_text(sample.get("timestamp")),
            "relative_time_s": self._csv_float_text(sample.get("relative_time_s")),
            "run_key": str(sample.get("run_key") or ""),
            "run_label": str(sample.get("run_label") or "轨迹"),
            "tracking_mode": str(sample.get("tracking_mode") or "stanley"),
            "segment_index": self._csv_int_text(sample.get("segment_index")),
            "speed_sign": self._csv_float_text(sample.get("speed_sign")),
            "speed_mps": self._csv_float_text(sample.get("speed_mps")),
            "nominal_speed_abs_mps": self._csv_float_text(sample.get("nominal_speed_abs_mps")),
            "lookahead_base_m": self._csv_float_text(sample.get("lookahead_base_m")),
            "arrival_dist_m": self._csv_float_text(sample.get("arrival_dist_m")),
            "slow_down_dist_m": self._csv_float_text(sample.get("slow_down_dist_m")),
            "stanley_gain": self._csv_float_text(sample.get("stanley_gain")),
            "stanley_softening_distance_m": self._csv_float_text(
                sample.get("stanley_softening_distance_m")
            ),
            "stanley_term_rad": self._csv_float_text(sample.get("stanley_term_rad")),
            "lateral_pid_kp": self._csv_float_text(sample.get("lateral_pid_kp")),
            "lateral_pid_ki": self._csv_float_text(sample.get("lateral_pid_ki")),
            "lateral_pid_kd": self._csv_float_text(sample.get("lateral_pid_kd")),
            "heading_pid_kp": self._csv_float_text(sample.get("heading_pid_kp")),
            "heading_pid_ki": self._csv_float_text(sample.get("heading_pid_ki")),
            "heading_pid_kd": self._csv_float_text(sample.get("heading_pid_kd")),
            "yaw_rate_pid_kp": self._csv_float_text(sample.get("yaw_rate_pid_kp")),
            "yaw_rate_pid_ki": self._csv_float_text(sample.get("yaw_rate_pid_ki")),
            "yaw_rate_pid_kd": self._csv_float_text(sample.get("yaw_rate_pid_kd")),
            "cmd_v_mps": self._csv_float_text(sample.get("cmd_v_mps")),
            "cmd_w_radps": self._csv_float_text(sample.get("cmd_w_radps")),
            "desired_v_mps": self._csv_float_text(sample.get("desired_v_mps")),
            "desired_w_radps": self._csv_float_text(sample.get("desired_w_radps")),
            "feedback_v_mps": self._csv_float_text(sample.get("feedback_v_mps")),
            "feedback_w_radps": self._csv_float_text(sample.get("feedback_w_radps")),
            "lateral_error_m": self._csv_float_text(sample.get("lateral_error_m")),
            "stanley_lateral_pd_output_radps": self._csv_float_text(
                sample.get("stanley_lateral_pd_output_radps")
            ),
            "heading_error_rad": self._csv_float_text(sample.get("heading_error_rad")),
            "yaw_rate_error_radps": self._csv_float_text(sample.get("yaw_rate_error_radps")),
            "lateral_pid_output_radps": self._csv_float_text(
                sample.get("lateral_pid_output_radps")
            ),
            "heading_pid_output_radps": self._csv_float_text(
                sample.get("heading_pid_output_radps")
            ),
            "yaw_rate_pid_output_radps": self._csv_float_text(
                sample.get("yaw_rate_pid_output_radps")
            ),
            "path_curvature_inv_m": self._csv_float_text(
                sample.get("path_curvature_inv_m")
            ),
            "path_ff_w_radps": self._csv_float_text(sample.get("path_ff_w_radps")),
            "profile_speed_mps": self._csv_float_text(sample.get("profile_speed_mps")),
            "path_s_m": self._csv_float_text(sample.get("path_s_m")),
            "dist_to_goal_m": self._csv_float_text(sample.get("dist_to_goal_m")),
            "pose_age_s": self._csv_float_text(sample.get("pose_age_s")),
            "current_x_m": self._csv_float_text(sample.get("current_x_m")),
            "current_y_m": self._csv_float_text(sample.get("current_y_m")),
            "nearest_x_m": self._csv_float_text(sample.get("nearest_x_m")),
            "nearest_y_m": self._csv_float_text(sample.get("nearest_y_m")),
            "lookahead_x_m": self._csv_float_text(sample.get("lookahead_x_m")),
            "lookahead_y_m": self._csv_float_text(sample.get("lookahead_y_m")),
            "segment_kind": str(sample.get("segment_kind") or ""),
            "segment_trajectory_name": str(sample.get("segment_trajectory_name") or ""),
            "motion_direction": motion_direction,
        }

    @staticmethod
    def _safe_filename_token(text: str) -> str:
        cleaned = re.sub(r'[\\/:*?"<>|]+', "_", str(text).strip())
        cleaned = cleaned.replace(" ", "_")
        return cleaned or "run"

    def _ensure_recorded_data_dirs(self) -> None:
        self._data_save_root.mkdir(parents=True, exist_ok=True)
        self._motion_save_dir.mkdir(parents=True, exist_ok=True)
        self._rcs_save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _unique_output_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        counter = 2
        while True:
            candidate = path.with_name(f"{stem}_{counter:02d}{suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    def _move_existing_output_to_dir(
        self,
        src_path: Path,
        target_dir: Path,
    ) -> Optional[Path]:
        if not src_path.exists() or not src_path.is_file():
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            if src_path.parent.resolve() == target_dir.resolve():
                return src_path
        except Exception:
            pass

        target_path = target_dir / src_path.name
        if target_path.exists():
            target_path = self._unique_output_path(target_path)
        shutil.move(str(src_path), str(target_path))
        return target_path

    def _migrate_matching_outputs(
        self,
        patterns: List[str],
        target_dir: Path,
    ) -> List[str]:
        moved: List[str] = []
        for pattern in patterns:
            for path in sorted(Path.cwd().glob(pattern)):
                try:
                    migrated = self._move_existing_output_to_dir(path, target_dir)
                except Exception as exc:
                    self._log(f"历史数据迁移失败: {path} -> {target_dir} | {exc}")
                    continue
                if migrated is not None:
                    moved.append(str(migrated))
        return moved

    def _migrate_existing_saved_data(self) -> None:
        moved: List[str] = []
        legacy_rcs_dir = Path.cwd() / "rcs_raw_data"
        try:
            if legacy_rcs_dir.exists() and legacy_rcs_dir.is_dir():
                if legacy_rcs_dir.resolve() != self._rcs_save_dir.resolve():
                    for path in sorted(legacy_rcs_dir.iterdir()):
                        if not path.is_file():
                            continue
                        migrated = self._move_existing_output_to_dir(path, self._rcs_save_dir)
                        if migrated is not None:
                            moved.append(str(migrated))
                    try:
                        legacy_rcs_dir.rmdir()
                    except OSError:
                        pass
        except Exception as exc:
            self._log(f"历史RCS目录迁移失败: {legacy_rcs_dir} | {exc}")

        moved.extend(
            self._migrate_matching_outputs(
                [
                    "*__*__*.txt",
                    "*__orbit_rcs__*.csv",
                    "rcs_fit_id*.png",
                    "fitted_id*.pdf",
                    "rcs_plot_*.png",
                    "rcs_compare_*.png",
                ],
                self._rcs_save_dir,
            )
        )
        moved.extend(
            self._migrate_matching_outputs(
                [
                    "motion_trace_*.csv",
                    "tracking_metrics_*.csv",
                ],
                self._motion_save_dir,
            )
        )
        if moved:
            self._log(
                f"历史运动/RCS数据已归档到 {self._data_save_root} | 数量={len(moved)}"
            )

    def _default_motion_trace_filename(self, record: Dict[str, Any]) -> str:
        timestamp = float(record.get("timestamp", time.time()))
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
        traj_name = str(
            record.get("segment_trajectory_name")
            or record.get("run_label")
            or "轨迹"
        ).strip() or "轨迹"
        traj_token = self._safe_filename_token(traj_name)
        mode_token = self._safe_filename_token(str(record.get("tracking_mode") or "pid"))
        return f"motion_trace_{traj_token}_{stamp}_{mode_token}.csv"

    def _save_motion_record_csv(
        self,
        record: Dict[str, Any],
        target_dir: Path,
        *,
        skip_if_saved: bool = False,
        mark_saved: bool = True,
    ) -> Optional[str]:
        samples = list(record.get("motion_samples") or [])
        if not samples:
            return None

        existing_path_text = str(record.get("motion_saved_path") or "").strip()
        if skip_if_saved and existing_path_text:
            return existing_path_text

        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / self._default_motion_trace_filename(record)
        if existing_path_text:
            existing_path = Path(existing_path_text)
            try:
                if existing_path.exists() and existing_path.parent.resolve() == target_dir.resolve():
                    file_path = existing_path
                elif file_path.exists():
                    file_path = self._unique_output_path(file_path)
            except Exception:
                if file_path.exists():
                    file_path = self._unique_output_path(file_path)
        elif file_path.exists():
            file_path = self._unique_output_path(file_path)

        headers = self._tracking_motion_sample_headers()
        record_index = int(record.get("record_index", 0))
        with file_path.open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=headers)
            writer.writeheader()
            for sample_index, sample in enumerate(samples, start=1):
                writer.writerow(
                    self._tracking_motion_sample_row(record_index, sample_index, sample)
                )

        if mark_saved:
            record["motion_saved_path"] = str(file_path)
            record["motion_saved_at"] = time.time()
        return str(file_path)

    def _save_motion_records(
        self,
        records: List[Dict[str, Any]],
        target_dir: Path,
        *,
        skip_if_saved: bool = False,
        mark_saved: bool = True,
    ) -> List[str]:
        saved_files: List[str] = []
        for record in records:
            file_path = self._save_motion_record_csv(
                record,
                target_dir,
                skip_if_saved=skip_if_saved,
                mark_saved=mark_saved,
            )
            if file_path:
                saved_files.append(file_path)
        return saved_files

    def _autosave_motion_record(self, record: Dict[str, Any]) -> Optional[str]:
        try:
            file_path = self._save_motion_record_csv(
                record,
                Path(self._motion_save_dir),
                skip_if_saved=True,
                mark_saved=True,
            )
        except Exception as exc:
            self._log(f"运动数据自动保存失败: {exc}")
            return None
        if file_path:
            self._refresh_data_analysis_window(focus_path=Path(file_path))
            self._log(f"运动数据已自动保存: {file_path}")
        return file_path

    def _on_emergency_stop_clicked(self) -> None:
        if self.controller.car is not None:
            try:
                self.controller.car.stop()
            except Exception as exc:
                self._log(f"紧急停止：下发零速失败: {exc}")
        else:
            self._log("紧急停止：CAN 未连接，无法下发底盘停止指令")
        if not self._motion_active and not self._rcs_recording:
            self._log("紧急停止：已请求停止（当前无进行中的轨迹跟踪或 RCS 记录）")
            return
        self._handle_motion_session_interrupt(
            "用户按下紧急停止",
            request_stop=False,
            trim_motion_record_to_now=True,
            popup_title="紧急停止",
            popup_text="已停止当前运动指令下发。",
        )

    def _handle_motion_session_interrupt(
        self,
        reason: str,
        *,
        request_stop: bool = False,
        trim_motion_record_to_now: bool = False,
        save_rcs_raw: bool = True,
        popup_title: Optional[str] = None,
        popup_text: Optional[str] = None,
    ) -> None:
        had_motion = bool(self._motion_active)
        had_rcs = bool(self._rcs_recording)
        if not had_motion and not had_rcs:
            return
        self._remember_motion_interrupt_reason(reason)
        if trim_motion_record_to_now and had_motion:
            self._arm_trim_next_motion_record(time.time(), reason)

        if request_stop and self.controller.car is not None:
            try:
                self.controller.car.stop()
            except Exception as exc:
                self._log(f"{reason} | 停止底盘控制失败: {exc}")

        self._motion_active = False
        self._radar_emergency_active = False
        self._reset_motion_guard_runtime(clear_profile=False)

        raw_path = None
        if self._rcs_recording:
            _, raw_path, _ = self._finalize_rcs_recording(save_raw=save_rcs_raw)

        parts = [reason]
        if had_motion:
            if trim_motion_record_to_now:
                parts.append("切换前的运动数据将自动保存，切换后的手柄运动不记录")
            else:
                parts.append(f"运动数据将自动保存到 {self._motion_save_dir}")
        if raw_path:
            parts.append(f"RCS已保存 {raw_path}")
        elif had_rcs:
            parts.append("RCS数据未保存" if not save_rcs_raw else f"RCS数据目录={self._rcs_save_dir}")
        self._log(" | ".join(parts))
        if popup_title:
            self._show_motion_stop_popup(popup_title, popup_text or reason)

    def _on_motion_run_finished(self, source: str) -> None:
        if not self._motion_active and not self._rcs_recording:
            self._radar_emergency_active = False
            self._reset_motion_guard_runtime(clear_profile=False)
            return

        self._motion_active = False
        self._radar_emergency_active = False
        self._reset_motion_guard_runtime(clear_profile=False)

        raw_path = None
        if self._rcs_recording:
            _, raw_path, _ = self._finalize_rcs_recording(save_raw=True)

        summary = f"运动结束: {str(source or '控制线程退出').strip()}"
        if raw_path:
            summary += f" | RCS已自动保存 {raw_path}"
        self._log(summary)

    def _set_current_path_name(self, name: Optional[str]) -> None:
        text = str(name or "").strip()
        self._current_path_name = text or "轨迹"

    def _get_current_path_name(self) -> str:
        return str(getattr(self, "_current_path_name", "") or "轨迹")

    @classmethod
    def _normalize_path_reference_frame(
        cls,
        frame: Optional[PathReferenceFrame],
    ) -> PathReferenceFrame:
        if frame is None:
            return PathReferenceFrame()
        raw_mode = str(frame.mode or "").strip()
        if raw_mode in (PATH_COORD_MODE_FIXED_ORIGIN, ""):
            pass
        elif raw_mode == _LEGACY_PATH_COORD_MODE_CURRENT_POSE:
            # 旧版「当前车位姿」轨迹与固定原点不可混用，导入后需重新选原点并规划
            return PathReferenceFrame(mode=PATH_COORD_MODE_FIXED_ORIGIN)
        origin_key = str(frame.origin_key or "").strip()
        origin_label = str(frame.origin_label or "").strip()
        if not origin_key and origin_label:
            origin_key = "__saved_origin__"
        return PathReferenceFrame(
            mode=PATH_COORD_MODE_FIXED_ORIGIN,
            origin_key=origin_key,
            origin_label=origin_label,
            origin_x_m=float(frame.origin_x_m),
            origin_y_m=float(frame.origin_y_m),
            origin_z_m=float(frame.origin_z_m),
        )

    @classmethod
    def _path_reference_frames_match(
        cls,
        frame_a: Optional[PathReferenceFrame],
        frame_b: Optional[PathReferenceFrame],
        tol_m: float = 1e-4,
    ) -> bool:
        a = cls._normalize_path_reference_frame(frame_a)
        b = cls._normalize_path_reference_frame(frame_b)
        if a.is_fixed_origin() != b.is_fixed_origin():
            return False
        if not a.is_fixed_origin():
            return True
        return (
            abs(a.origin_x_m - b.origin_x_m) <= tol_m
            and abs(a.origin_y_m - b.origin_y_m) <= tol_m
            and abs(a.origin_z_m - b.origin_z_m) <= max(tol_m, 1e-3)
        )

    def _resolve_path_reference_frame(
        self,
        frame: Optional[PathReferenceFrame],
    ) -> PathReferenceFrame:
        normalized = self._normalize_path_reference_frame(frame)
        if not normalized.is_fixed_origin():
            return normalized
        # 锚点平面坐标以帧内 origin_x/y/z 为准（手动输入或预设保存值），
        # 不再根据 origin_key 去反查 ENU 表里「记录控制点」（与当前系 x,y 无直接对应）。
        key = str(normalized.origin_key or "").strip()
        label = str(normalized.origin_label or "").strip() or "手动锚点"
        if key in (PATH_ORIGIN_CALIB_PLANE_KEY, PATH_ORIGIN_MANUAL_KEY, "__saved_origin__"):
            nk = PATH_ORIGIN_MANUAL_KEY
        elif key.startswith("enu_point_"):
            nk = PATH_ORIGIN_MANUAL_KEY
        else:
            nk = key or PATH_ORIGIN_MANUAL_KEY
        return PathReferenceFrame(
            mode=PATH_COORD_MODE_FIXED_ORIGIN,
            origin_key=nk,
            origin_label=label,
            origin_x_m=float(normalized.origin_x_m),
            origin_y_m=float(normalized.origin_y_m),
            origin_z_m=float(normalized.origin_z_m),
        )

    @classmethod
    def _format_path_reference_frame(cls, frame: Optional[PathReferenceFrame]) -> str:
        normalized = cls._normalize_path_reference_frame(frame)
        if not normalized.is_fixed_origin():
            return "轨迹锚点未就绪（请检查界面初始化）"
        label = normalized.origin_label or "手动锚点"
        return (
            f"{label} | x₀={normalized.origin_x_m:.3f} m, "
            f"y₀={normalized.origin_y_m:.3f} m"
        )

    def _path_anchor_frame_from_ui(self) -> PathReferenceFrame:
        sx = getattr(self, "spin_path_anchor_x", None)
        sy = getattr(self, "spin_path_anchor_y", None)
        xv = float(sx.value()) if sx is not None else 0.0
        yv = float(sy.value()) if sy is not None else 0.0
        return PathReferenceFrame(
            mode=PATH_COORD_MODE_FIXED_ORIGIN,
            origin_key=PATH_ORIGIN_MANUAL_KEY,
            origin_label="手动锚点",
            origin_x_m=xv,
            origin_y_m=yv,
            origin_z_m=0.0,
        )

    def _set_path_anchor_spinboxes(self, x_m: float, y_m: float) -> None:
        sx = getattr(self, "spin_path_anchor_x", None)
        sy = getattr(self, "spin_path_anchor_y", None)
        if sx is None or sy is None:
            return
        sx.blockSignals(True)
        sy.blockSignals(True)
        sx.setValue(float(x_m))
        sy.setValue(float(y_m))
        sx.blockSignals(False)
        sy.blockSignals(False)

    def _get_path_anchor_reference_frame(self) -> PathReferenceFrame:
        return self._path_anchor_frame_from_ui()

    def _use_calibration_plane_path_origin(self) -> None:
        """锚点置为校准平面原点 (0,0)。"""
        self._set_path_anchor_spinboxes(0.0, 0.0)
        self._loaded_path_frame = self._path_anchor_frame_from_ui()
        self._update_path_coordinate_widgets(frame_override=self._loaded_path_frame)

    def _apply_path_reference_frame_to_controls(
        self,
        frame: Optional[PathReferenceFrame],
    ) -> None:
        normalized = self._resolve_path_reference_frame(frame)
        self._set_path_anchor_spinboxes(normalized.origin_x_m, normalized.origin_y_m)
        self._loaded_path_frame = self._path_anchor_frame_from_ui()
        self._update_path_coordinate_widgets(frame_override=self._loaded_path_frame)

    def _update_path_coordinate_widgets(
        self,
        frame_override: Optional[PathReferenceFrame] = None,
    ) -> None:
        summary_widget = getattr(self, "label_path_origin_summary", None)
        if summary_widget is None:
            return
        if frame_override is not None:
            nf = self._normalize_path_reference_frame(frame_override)
            if nf.is_fixed_origin():
                self._loaded_path_frame = nf
        else:
            self._loaded_path_frame = self._path_anchor_frame_from_ui()
        f = self._loaded_path_frame
        if not f.is_fixed_origin():
            summary_widget.setText("请在上方输入轨迹锚点平面坐标 x, y（米）。")
            return
        summary_widget.setText(
            f"锚点（校准平面系）: x₀={float(f.origin_x_m):.3f} m, "
            f"y₀={float(f.origin_y_m):.3f} m — 局部规划原点 (0,0) 落在该处；"
            f"与小车位置无关，除非您把锚点改成车位姿。"
        )

    def _on_path_anchor_xy_changed(self, *_args: Any) -> None:
        self._loaded_path_frame = self._path_anchor_frame_from_ui()
        self._update_path_coordinate_widgets(frame_override=self._loaded_path_frame)
        if self._radial_measurement_spec is None and len(self.loaded_path_local_points) >= 2:
            self._apply_planned_local_points(
                list(self.loaded_path_local_points),
                frame=self._path_anchor_frame_from_ui(),
            )

    def _on_path_anchor_zero_clicked(self) -> None:
        self._set_path_anchor_spinboxes(0.0, 0.0)
        self._on_path_anchor_xy_changed()

    def _get_selected_rcs_target_name(self) -> str:
        name = str(self._rcs_target_name or "").strip()
        if name:
            return name
        if self.tracked_target_id is not None:
            return f"目标ID{int(self.tracked_target_id)}"
        return "未命名目标"

    def _get_live_rcs_target_name(self) -> str:
        if self.tracked_target_id is not None:
            return f"目标ID{int(self.tracked_target_id)}"
        return "未命名目标"

    @staticmethod
    def _extract_radar_target_oid(target: Any) -> Optional[int]:
        try:
            return int(getattr(target, "oid", getattr(target, "id", getattr(target, "cid", 0))))
        except Exception:
            return None

    @staticmethod
    def _get_radar_target_age_s(target: Any, now_ts: Optional[float] = None) -> Optional[float]:
        raw_t = getattr(target, "t", None)
        if raw_t is None:
            return None
        try:
            age_s = float(now_ts if now_ts is not None else time.time()) - float(raw_t)
        except (TypeError, ValueError):
            return None
        return max(0.0, age_s)

    def _is_radar_target_fresh(self, target: Any, now_ts: Optional[float] = None) -> bool:
        age_s = self._get_radar_target_age_s(target, now_ts=now_ts)
        if age_s is None:
            return True
        return age_s <= float(self._radar_target_fresh_s)

    def _get_best_radar_relock_target(
        self,
        exclude_oid: Optional[int] = None,
    ) -> Optional[Any]:
        del exclude_oid
        return None

    @staticmethod
    def _is_live_target_name(text: Optional[str]) -> bool:
        value = str(text or "").strip()
        return not value or bool(re.fullmatch(r"目标ID\d+", value))

    def _set_tracked_radar_target(
        self,
        target: Any,
        *,
        auto: bool = False,
        reason: Optional[str] = None,
    ) -> bool:
        oid = self._extract_radar_target_oid(target)
        if oid is None:
            return False

        prev_oid = self.tracked_target_id
        changed = prev_oid != oid
        self.tracked_target_id = oid
        if hasattr(self.controller, "set_selected_radar_id"):
            self.controller.set_selected_radar_id(oid)

        if self._rcs_recording and changed and self._is_live_target_name(self._rcs_active_target_name):
            self._rcs_active_target_name = self._get_live_rcs_target_name()

        try:
            detail = (
                f"id={oid} x={float(getattr(target, 'x', 0.0)):.2f}m "
                f"y={float(getattr(target, 'y', 0.0)):.2f}m "
                f"rcs={float(getattr(target, 'rcs_db', 0.0)):.1f}dBsm"
            )
        except Exception:
            detail = f"id={oid}"

        if auto:
            if not changed:
                return False
            if prev_oid is not None:
                message = f"雷达目标自动重锁: {prev_oid} -> {detail}"
            else:
                message = f"雷达目标自动锁定: {detail}"
            if reason:
                message += f" | 原因={reason}"
            self._log(message)
            if self._rcs_recording:
                self._rcs_relock_events.append(message)
        else:
            self._log(f"雷达目标锁定: {detail}")
        return changed

    def _set_target_point_from_global(
        self,
        x: float,
        y: float,
        *,
        log_prefix: str,
        detail: Optional[str] = None,
    ) -> None:
        gx = float(x)
        gy = float(y)
        self.target_point = (gx, gy)
        if self.target_marker is not None:
            self.target_marker.setData([gx], [gy])
        self._target_marking_mode = False

        message = f"{log_prefix}: x={gx:.2f}m, y={gy:.2f}m"
        if detail:
            message += f" | {detail}"
        self._log(message)

        if self._radial_measurement_spec is not None:
            anchor_pose = get_robot_pose()
            if anchor_pose is None:
                anchor_pose = self._build_virtual_preview_pose()
            if self._apply_radial_measurement_spec(
                self._radial_measurement_spec,
                anchor_pose=anchor_pose,
                announce_virtual_preview=False,
            ):
                self._log("目标物位置已更新，星型测量轨迹已自动重建。")

    def _project_radar_target_to_global(
        self,
        target: Any,
        pose: Optional[PoseSolution] = None,
    ) -> Optional[Tuple[Tuple[float, float], float, float, bool]]:
        try:
            x_rel = float(getattr(target, "x", 0.0))
            y_rel = float(getattr(target, "y", 0.0))
        except (TypeError, ValueError):
            return None

        resolved_pose = pose if pose is not None else get_robot_pose()
        using_virtual_pose = resolved_pose is None
        if resolved_pose is None:
            resolved_pose = self._build_virtual_preview_pose()

        global_point = self._local_points_to_global(
            [(x_rel, y_rel)],
            resolved_pose,
        )[0]
        return (
            (float(global_point[0]), float(global_point[1])),
            x_rel,
            y_rel,
            using_virtual_pose,
        )

    def _maybe_auto_relock_selected_target(
        self,
        targets: List[Any],
        now_ts: Optional[float] = None,
    ) -> None:
        if (
            not self._radar_auto_relock_enabled
            or self._rcs_recording
            or self.tracked_target_id is None
            or getattr(self.controller, "cluster_csv_runtime", None) is not None
        ):
            return

        tracked_oid = int(self.tracked_target_id)
        for target in targets:
            if self._extract_radar_target_oid(target) != tracked_oid:
                continue
            if self._is_radar_target_fresh(target, now_ts=now_ts):
                return

        replacement = self._get_best_radar_relock_target(exclude_oid=tracked_oid)
        if replacement is None:
            return
        self._set_tracked_radar_target(
            replacement,
            auto=True,
            reason="当前锁定目标消失",
        )

    @staticmethod
    def _normalized_name_key(text: Optional[str]) -> str:
        return re.sub(r"[\s_]+", "", str(text or "").strip().lower())

    def _update_rcs_save_dir_button_tooltip(self) -> None:
        if hasattr(self, "btn_select_rcs_save_dir") and self.btn_select_rcs_save_dir is not None:
            self.btn_select_rcs_save_dir.setToolTip(
                f"从 {self._rcs_save_dir} 选择 Cluster Raw CSV；圆周-RCS 图由 Raw 经行内功率合并后按时间平铺。"
            )

    def _on_select_rcs_save_dir(self) -> None:
        # Repurposed: choose a saved RCS data file and plot it.
        # Saved data is auto-written to recorded_data/rcs_data, so we do not change the save directory here.
        self._show_tool_dialog(self._rcs_viewer_dialog)
        if self._rcs_recording:
            QtWidgets.QMessageBox.information(
                self,
                "Cluster RCS 采集中",
                "请先等待当前 Cluster RCS 采集结束后再绘制历史数据。",
            )
            return

        file_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "选择已保存的 Cluster Raw CSV（可多选合并拟合，仅距离-RCS）",
            str(self._rcs_save_dir),
            "CSV 表格 (*.csv);;所有文件(*)",
        )
        if not file_paths:
            return
        # 多选合并：仅支持距离-RCS 文件，合并后只绘制一条拟合曲线
        if len(file_paths) >= 2:
            try:
                curve = self._build_merged_loaded_rcs_curve(file_paths, display_name=None)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "载入失败", f"合并读取RCS数据失败: {exc}")
                self._log(f"RCS合并绘图载入失败: {exc}")
                return
            ref_summary = self._format_rcs_reference_summary()
            plot_dlg = RcsPlotConfigDialog(
                str(file_paths[0]),
                ref_summary,
                default_custom_name=curve.display_name,
                default_calibration_db=self._rcs_plot_calibration_db,
                enable_curve_csv_export=True,
                parent=self,
            )
            if plot_dlg.exec_() != QtWidgets.QDialog.Accepted:
                return
            custom_name, cal_db, export_curve_csv = plot_dlg.values()
            self._rcs_plot_calibration_db = cal_db
            if custom_name:
                curve = replace(curve, display_name=custom_name)
            show_ref = bool(self._rcs_ref_class and self._rcs_ref_angle and self._rcs_ref_limits is not None)
            self._rcs_hide_reference_limits = not show_ref
            self._rcs_base_file_path = None
            self._rcs_curve_csv_source_paths = [str(p) for p in file_paths]
            self._loaded_rcs_curves = []
            self._orbit_polar_cached_series = None
            self.rcs_recorder.reset()
            self.rcs_recorder.segments = [list(seg) for seg in curve.segments]
            self.rcs_recorder._cur = []
            self.rcs_recorder._ended = True
            self.rcs_recorder.oid = None
            self._rcs_recording = False
            self._rcs_show_only_fitted = False
            self._rcs_orbit_samples = []
            self._rcs_active_segment_index = None
            self._rcs_active_path_name = None
            self._rcs_active_target_name = None
            self._rcs_fitted = curve.fitted
            self._rcs_target_name = curve.display_name
            self._rcs_segment_run_labels = curve.segment_run_labels
            for i in range(self.rcs_plot_mode_combo.count()):
                if self.rcs_plot_mode_combo.itemData(i) == "distance":
                    self.rcs_plot_mode_combo.blockSignals(True)
                    self.rcs_plot_mode_combo.setCurrentIndex(i)
                    self.rcs_plot_mode_combo.blockSignals(False)
                    break
            self._rcs_plot_mode = "distance"
            self._ensure_rcs_axes("distance")
            cal_note = f" | 标定{cal_db:+.2f}dB" if abs(float(cal_db)) > 1e-9 else ""
            csv_note = ""
            if export_curve_csv:
                _filtered_paths, combined_path = self._try_export_rcs_curve_csvs(
                    list(file_paths),
                    cal_db,
                )
                if combined_path is not None:
                    csv_note = f" | CSV={combined_path.name}"
            self.rcs_status_label.setText(
                f"RCS绘图: 已合并 {len(file_paths)} 个文件 | 总点数={curve.point_count}{cal_note}{csv_note}"
            )
            self._log(
                f"RCS合并绘图数据已载入: 文件数={len(file_paths)} | 总点数={curve.point_count}"
                f"{cal_note}{csv_note}"
            )
            self._draw_rcs()
            return

        file_path = str(file_paths[0])

        is_orbit_file = MainWindow._is_orbit_rcs_polar_csv(Path(file_path))
        if is_orbit_file:
            try:
                orows = self._parse_orbit_rcs_csv(file_path)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "载入失败", f"读取圆周RCS CSV 失败: {exc}")
                self._log(f"圆周RCS载入失败: {exc}")
                return
            ref_summary = self._format_rcs_reference_summary()
            stem = Path(file_path).stem
            plot_dlg = RcsPlotConfigDialog(
                file_path,
                ref_summary,
                default_custom_name=stem,
                default_calibration_db=self._rcs_plot_calibration_db,
                enable_curve_csv_export=False,
                parent=self,
            )
            if plot_dlg.exec_() != QtWidgets.QDialog.Accepted:
                return
            custom_name, cal_db, _export_curve_csv = plot_dlg.values()
            self._rcs_plot_calibration_db = cal_db
            self._orbit_polar_cached_series = self._build_orbit_polar_series_from_rows(orows)
            # 圆周图不叠加距离-RCS参考上下限
            self._rcs_hide_reference_limits = True
            self._rcs_base_file_path = str(file_path)
            self._rcs_curve_csv_source_paths = []
            self._loaded_rcs_curves = []
            self.rcs_recorder.reset()
            self._rcs_segment_run_labels = None
            self.rcs_recorder._cur = []
            self.rcs_recorder._ended = True
            self.rcs_recorder.oid = None
            self._rcs_recording = False
            self._rcs_show_only_fitted = False
            self._rcs_orbit_samples = []
            self._rcs_fitted = None
            self._rcs_active_segment_index = None
            self._rcs_active_path_name = None
            self._rcs_active_target_name = None
            self._rcs_target_name = custom_name or stem
            for i in range(self.rcs_plot_mode_combo.count()):
                if self.rcs_plot_mode_combo.itemData(i) == "orbit":
                    self.rcs_plot_mode_combo.blockSignals(True)
                    self.rcs_plot_mode_combo.setCurrentIndex(i)
                    self.rcs_plot_mode_combo.blockSignals(False)
                    break
            self._rcs_plot_mode = "orbit"
            self._ensure_rcs_axes("orbit")
            cal_note = f" | 标定{cal_db:+.2f}dB" if abs(float(cal_db)) > 1e-9 else ""
            self.rcs_status_label.setText(
                f"圆周RCS: 已载入 {Path(file_path).name} | 点数={len(orows)} | "
                f"角度分箱均值曲线 | 径向=RCS(dBsm){cal_note}"
            )
            self._log(
                f"圆周RCS CSV 已载入: {file_path} | 点数={len(orows)} | "
                f"平铺到2π后按{ORBIT_RCS_ANGLE_BIN_DEG:g}°分箱算术平均{cal_note}"
            )
            self._draw_rcs()
            return

        if MainWindow._is_orbit_cluster_raw_csv(Path(file_path)):
            try:
                orows = self._orbit_plot_rows_from_cluster_raw_path(file_path)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "载入失败", f"读取圆周 Cluster Raw 失败: {exc}")
                self._log(f"圆周 Cluster Raw 载入失败: {exc}")
                return
            if len(orows) < 2:
                QtWidgets.QMessageBox.warning(self, "载入失败", "圆周 Cluster Raw 有效 RCS 点不足")
                self._log(f"圆周 Cluster Raw 载入失败: 有效点不足 | 文件={file_path}")
                return
            ref_summary = self._format_rcs_reference_summary()
            stem = Path(file_path).stem
            plot_dlg = RcsPlotConfigDialog(
                file_path,
                ref_summary,
                default_custom_name=stem,
                default_calibration_db=self._rcs_plot_calibration_db,
                enable_curve_csv_export=False,
                parent=self,
            )
            if plot_dlg.exec_() != QtWidgets.QDialog.Accepted:
                return
            custom_name, cal_db, _export_curve_csv = plot_dlg.values()
            self._rcs_plot_calibration_db = cal_db
            self._orbit_polar_cached_series = self._build_orbit_polar_series_from_rows(orows)
            self._rcs_hide_reference_limits = True
            self._rcs_base_file_path = str(file_path)
            self._rcs_curve_csv_source_paths = []
            self._loaded_rcs_curves = []
            self.rcs_recorder.reset()
            self._rcs_segment_run_labels = None
            self.rcs_recorder._cur = []
            self.rcs_recorder._ended = True
            self.rcs_recorder.oid = None
            self._rcs_recording = False
            self._rcs_show_only_fitted = False
            self._rcs_orbit_samples = []
            self._rcs_fitted = None
            self._rcs_active_segment_index = None
            self._rcs_active_path_name = None
            self._rcs_active_target_name = None
            self._rcs_target_name = custom_name or stem
            self._select_rcs_plot_mode("orbit")
            self._ensure_rcs_axes("orbit")
            cal_note = f" | 标定{cal_db:+.2f}dB" if abs(float(cal_db)) > 1e-9 else ""
            self.rcs_status_label.setText(
                f"圆周RCS: 已载入 {Path(file_path).name} | 点数={len(orows)} | "
                f"RCS00/RCS01功率合并 | 角度分箱均值曲线 | 径向=RCS(dBsm){cal_note}"
            )
            self._log(
                f"圆周 Cluster Raw 已载入: {file_path} | 点数={len(orows)} | "
                f"RCS00/RCS01功率合并 | 时间平铺到2π后按{ORBIT_RCS_ANGLE_BIN_DEG:g}°分箱算术平均{cal_note}"
            )
            self._draw_rcs()
            return

        try:
            curve = self._build_loaded_rcs_curve(file_path, display_name=None)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "载入失败", f"读取RCS数据失败: {exc}")
            self._log(f"RCS绘图载入失败: {exc}")
            return

        ref_summary = self._format_rcs_reference_summary()
        plot_dlg = RcsPlotConfigDialog(
            file_path,
            ref_summary,
            default_custom_name=curve.display_name,
            default_calibration_db=self._rcs_plot_calibration_db,
            enable_curve_csv_export=True,
            parent=self,
        )
        if plot_dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        custom_name, cal_db, export_curve_csv = plot_dlg.values()
        self._rcs_plot_calibration_db = cal_db
        if custom_name:
            curve = replace(curve, display_name=custom_name)

        # Base plot: show reference limits if user has selected a reference product.
        show_ref = bool(self._rcs_ref_class and self._rcs_ref_angle and self._rcs_ref_limits is not None)
        self._rcs_hide_reference_limits = not show_ref
        self._rcs_base_file_path = str(curve.file_path)
        self._rcs_curve_csv_source_paths = [str(curve.file_path)]
        self._loaded_rcs_curves = []
        self._orbit_polar_cached_series = None
        if MainWindow._cluster_csv_has_dri_cluster_header(Path(curve.file_path)):
            try:
                orows = self._orbit_plot_rows_from_cluster_raw_path(str(curve.file_path))
                if len(orows) >= 2:
                    self._orbit_polar_cached_series = self._build_orbit_polar_series_from_rows(orows)
            except Exception:
                self._orbit_polar_cached_series = None
        self.rcs_recorder.reset()
        self.rcs_recorder.segments = [list(seg) for seg in curve.segments]
        self.rcs_recorder._cur = []
        self.rcs_recorder._ended = True
        self.rcs_recorder.oid = None
        self._rcs_recording = False
        self._rcs_show_only_fitted = False
        self._rcs_orbit_samples = []
        self._rcs_active_segment_index = None
        self._rcs_active_path_name = None
        self._rcs_active_target_name = None
        self._rcs_fitted = curve.fitted
        self._rcs_target_name = curve.display_name or Path(curve.file_path).stem
        self._rcs_segment_run_labels = curve.segment_run_labels

        for i in range(self.rcs_plot_mode_combo.count()):
            if self.rcs_plot_mode_combo.itemData(i) == "distance":
                self.rcs_plot_mode_combo.blockSignals(True)
                self.rcs_plot_mode_combo.setCurrentIndex(i)
                self.rcs_plot_mode_combo.blockSignals(False)
                break
        self._rcs_plot_mode = "distance"
        self._ensure_rcs_axes("distance")

        cal_note = f" | 标定{cal_db:+.2f}dB" if abs(float(cal_db)) > 1e-9 else ""
        csv_note = ""
        if export_curve_csv:
            _filtered_paths, combined_path = self._try_export_rcs_curve_csvs(
                [str(curve.file_path)],
                cal_db,
            )
            if combined_path is not None:
                csv_note = f" | CSV={combined_path.name}"
        self.rcs_status_label.setText(
            f"RCS绘图: 已载入 {Path(curve.file_path).name} | 点数={curve.point_count}{cal_note}{csv_note}"
        )
        self._log(
            f"RCS绘图数据已载入(无参考上下限): 文件={curve.file_path} | 点数={curve.point_count}"
            f"{cal_note}{csv_note}"
        )
        self._draw_rcs()

    def _compose_rcs_raw_filename(
        self,
        trajectory_name: Optional[str] = None,
        target_name: Optional[str] = None,
        segment_index: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> str:
        if getattr(self, "_radial_measurement_spec", None) is not None:
            tn = str(trajectory_name or self._get_current_path_name() or "").strip()
            if self._parse_radial_rcs_task_name(tn) is not None:
                return f"{self._radial_rcs_segment_file_stem(tn)}.csv"
        traj_token = self._safe_filename_token(trajectory_name or self._get_current_path_name())
        target_token = self._safe_filename_token(target_name or self._get_selected_rcs_target_name())
        seg_token = f"_seg{int(segment_index):02d}" if segment_index is not None else ""
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(float(timestamp or time.time())))
        return f"{traj_token}__{target_token}{seg_token}__{stamp}.csv"

    def _resolve_rcs_save_dir(self, trajectory_name: Optional[str]) -> Path:
        """RCS 落盘目录。

        星型/径向测量：一次执行共用一个会话文件夹；每段 CSV 文件名为「X度第N次测量」样式（见 _format_radial_rcs_segment_label）。
        """
        base = Path(self._rcs_save_dir)
        if getattr(self, "_radial_measurement_spec", None) is not None:
            session_dir = self._ensure_radial_rcs_session_dir()
            if session_dir is not None:
                return session_dir
        return base

    @staticmethod
    def _write_curvepoints_cluster_rcs_csv(
        file_path: Path,
        segments: List[List[CurvePoint]],
        *,
        data_file_display: str,
        include_seg_idx: bool,
        run_number: int = 1,
        calibration=None,
    ) -> None:
        """与 ars40x_cluster_logger / DRI Raw 元数据一致；可选 SegIdx 列用于多段合并。"""
        base_cols = build_columns()
        headers = list(base_cols) + (["SegIdx"] if include_seg_idx else [])
        n_slots = (len(base_cols) - 9) // 3
        cal_s = format_cluster_csv_calibration(calibration)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Data Type", "Raw"])
            w.writerow(["Data File", str(data_file_display)])
            w.writerow(["Run Number", int(run_number)])
            w.writerow(["Calibration", cal_s])
            w.writerow([])
            w.writerow(headers)
            for seg_idx, seg in enumerate(segments):
                for p in seg:
                    dx = float(p.x_raw)
                    dy = float(p.y_raw)
                    r_geom = round(math.hypot(dx, dy), 4)
                    view_deg = round(math.degrees(math.atan2(dy, dx)), 4)
                    row: List[Any] = [
                        round(float(p.t), 4),
                        r_geom,
                        view_deg,
                        "NaN",
                        "NaN",
                        "NaN",
                        "NaN",
                        "NaN",
                        "NaN",
                    ]
                    for i in range(n_slots):
                        if i == 0:
                            row += [round(dx, 2), round(dy, 2), round(float(p.rcs_raw), 2)]
                        else:
                            row += ["NaN", "NaN", "NaN"]
                    if include_seg_idx:
                        row.append(int(seg_idx))
                    w.writerow(row)

    def _save_rcs_raw_snapshot(
        self,
        trajectory_name: Optional[str] = None,
        target_name: Optional[str] = None,
        segment_index: Optional[int] = None,
    ) -> Optional[str]:
        if self.rcs_recorder.point_count() <= 0:
            return None
        save_dir = self._resolve_rcs_save_dir(trajectory_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / self._compose_rcs_raw_filename(
            trajectory_name=trajectory_name,
            target_name=target_name,
            segment_index=segment_index,
        )
        segs = [list(s) for s in self.rcs_recorder.segments if s]
        if not segs and self.rcs_recorder._cur:
            segs = [list(self.rcs_recorder._cur)]
        if not segs:
            return None
        self._write_curvepoints_cluster_rcs_csv(
            file_path,
            segs,
            data_file_display=str(file_path),
            include_seg_idx=len(segs) > 1,
        )
        self._write_rcs_fitted_companion_file(file_path, self.rcs_recorder)
        return str(file_path)

    @staticmethod
    def _write_rcs_fitted_companion_file(raw_file_path: Path, recorder: RcsRunRecorder) -> Optional[str]:
        """在原始落盘旁写入拟合曲线采样 `*_fitted.csv`。"""
        try:
            fit_path = raw_file_path.with_name(f"{raw_file_path.stem}_fitted.csv")
            fit = recorder.fit_curve()
            with fit_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["x_m", "rcs_fit_dBsm"])
                if fit is not None:
                    xs, ys = fit
                    for a, b in zip(np.asarray(xs, dtype=float).ravel(), np.asarray(ys, dtype=float).ravel()):
                        if np.isfinite(a) and np.isfinite(b):
                            w.writerow([f"{float(a):.4f}", f"{float(b):.4f}"])
            return str(fit_path)
        except Exception:
            return None

    def _reset_rcs_trajectory_file_cache(self) -> None:
        self._rcs_saved_trajectory_files.clear()

    def _save_rcs_trajectory_snapshot(
        self,
        trajectory_name: Optional[str],
        target_name: Optional[str],
        segment_points: List[CurvePoint],
    ) -> Optional[str]:
        if not segment_points:
            return None

        resolved_trajectory_name = str(trajectory_name or self._get_current_path_name()).strip() or "轨迹"
        resolved_target_name = str(target_name or self._get_selected_rcs_target_name()).strip() or "未命名目标"
        cache_key = f"{resolved_trajectory_name}\n{resolved_target_name}"
        record = self._rcs_saved_trajectory_files.get(cache_key)

        save_dir = self._resolve_rcs_save_dir(resolved_trajectory_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        if record is None:
            file_path = save_dir / self._compose_rcs_raw_filename(
                trajectory_name=resolved_trajectory_name,
                target_name=resolved_target_name,
                segment_index=None,
            )
            if file_path.exists():
                stem = file_path.stem
                suffix = file_path.suffix or ".csv"
                counter = 2
                while True:
                    candidate = file_path.with_name(f"{stem}_{counter:02d}{suffix}")
                    if not candidate.exists():
                        file_path = candidate
                        break
                    counter += 1
            record = AggregatedRcsFile(
                trajectory_name=resolved_trajectory_name,
                target_name=resolved_target_name,
                file_path=str(file_path),
                segments=[],
            )
            self._rcs_saved_trajectory_files[cache_key] = record

        record.segments.append(list(segment_points))
        fp = Path(record.file_path)
        self._write_curvepoints_cluster_rcs_csv(
            fp,
            record.segments,
            data_file_display=str(fp),
            include_seg_idx=True,
        )
        tmp_rec = RcsRunRecorder()
        tmp_rec.segments = [list(s) for s in record.segments]
        tmp_rec._ended = True
        self._write_rcs_fitted_companion_file(fp, tmp_rec)
        return str(record.file_path)

    def _guess_rcs_plot_defaults(
        self,
        file_path: str,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        stem = Path(file_path).stem
        parts = stem.split("__")
        guessed_name = ""
        if len(parts) >= 2:
            guessed_name = re.sub(r"_seg\d+$", "", parts[1], flags=re.IGNORECASE)
        elif parts:
            guessed_name = parts[0]
        guessed_name = guessed_name.replace("_", " ").strip()

        guessed_class = None
        guessed_key = self._normalized_name_key(guessed_name)
        if guessed_key:
            for name in self._rcs_ref_class_options:
                if self._normalized_name_key(name) == guessed_key:
                    guessed_class = name
                    break

        guessed_angle = (
            self._rcs_ref_angle
            if self._rcs_ref_angle in self._rcs_ref_angle_options
            else None
        )
        custom_name = guessed_name or guessed_class or self._rcs_target_name or ""
        return guessed_class, guessed_angle, custom_name or None

    @staticmethod
    def _common_nonempty_value(values: List[Optional[str]]) -> Optional[str]:
        filtered = [str(value).strip() for value in values if str(value or "").strip()]
        if not filtered:
            return None
        first = filtered[0]
        return first if all(value == first for value in filtered[1:]) else None

    @staticmethod
    def _ensure_unique_rcs_display_names(
        file_paths: List[str],
        display_names: List[Optional[str]],
    ) -> List[str]:
        result: List[str] = []
        counts: Dict[str, int] = {}
        for index, file_path in enumerate(file_paths):
            raw_name = display_names[index] if index < len(display_names) else None
            base_name = str(raw_name or "").strip() or Path(file_path).stem
            key = base_name.casefold()
            counts[key] = counts.get(key, 0) + 1
            if counts[key] > 1:
                base_name = f"{base_name} ({counts[key]})"
            result.append(base_name)
        return result

    @staticmethod
    def _cluster_rcs_csv_slot_dx_rcs(
        row: List[str],
        col: Dict[str, int],
        slot_index: int,
    ) -> Optional[Tuple[float, float]]:
        """读取 DX/RCS 槽位；RCS 或 DX 任一无效则跳过该槽。"""
        dxk = f"DX{slot_index:02d}"
        rck = f"RCS{slot_index:02d}"
        if dxk not in col or rck not in col:
            return None
        try:
            dx = MainWindow._cluster_csv_cell_float(row[col[dxk]])
            rcs = MainWindow._cluster_csv_cell_float(row[col[rck]])
        except (ValueError, KeyError, IndexError):
            return None
        if not math.isfinite(dx) or not math.isfinite(rcs):
            return None
        return float(dx), float(rcs)

    @staticmethod
    def _filtered_rcs_csv_path(raw_path: Path) -> Path:
        return raw_path.with_name(f"{raw_path.stem}_Filtered.csv")

    @staticmethod
    def _build_filtered_rcs_rows_from_cluster_raw_path(
        raw_path: Path,
        calibration_db: float,
    ) -> Tuple[List[List[str]], List[Tuple[float, float]], int]:
        """
        Filtered CSV 数据：
        每帧按 abs(DX[i]-R) 取最近两个有效槽，记录 (DX, RCS+calibration)，再按 0.1m X 分箱算术平均。
        """
        metadata_rows: List[List[str]] = []
        bins: Dict[int, List[float]] = defaultdict(list)
        selected_count = 0
        with raw_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header_cells: Optional[List[str]] = None
            col: Dict[str, int] = {}
            for row in reader:
                if row and row[0].strip() == "Time" and any(
                    (c or "").strip() == "DX00" for c in row
                ):
                    header_cells = [(c or "").strip() for c in row]
                    col = {name: idx for idx, name in enumerate(header_cells)}
                    break
                metadata_rows.append(list(row))
            if not header_cells:
                raise ValueError("Cluster RCS CSV：未找到表头行（含 Time、DX00）")
            for k in ("R", "DX00", "RCS00"):
                if k not in col:
                    raise ValueError(f"Cluster RCS CSV 缺少列「{k}」")
            max_ix = max(col.values())

            for raw_row in reader:
                if not raw_row or not any(str(c).strip() for c in raw_row):
                    continue
                row = list(raw_row)
                while len(row) <= max_ix:
                    row.append("")
                try:
                    range_val = MainWindow._cluster_csv_cell_float(row[col["R"]])
                except (ValueError, KeyError, IndexError):
                    continue
                if not math.isfinite(range_val):
                    continue

                candidates: List[Tuple[float, int, float, float]] = []
                for slot in range(int(MAX_CLUSTERS)):
                    parsed = MainWindow._cluster_rcs_csv_slot_dx_rcs(row, col, slot)
                    if parsed is None:
                        continue
                    dx, rcs = parsed
                    candidates.append(
                        (abs(float(dx) - float(range_val)), int(slot), float(dx), float(rcs))
                    )
                if not candidates:
                    continue

                candidates.sort(key=lambda item: (float(item[0]), int(item[1])))
                for _dist_err, _slot, dx, rcs in candidates[:2]:
                    bin_id = int(round(float(dx) * 10.0))
                    bins[bin_id].append(float(rcs) + float(calibration_db))
                    selected_count += 1

        if not bins:
            raise ValueError("Cluster RCS CSV 中没有可生成 Filtered 的有效帧")

        rows: List[Tuple[float, float]] = []
        for bin_id in range(min(bins.keys()), max(bins.keys()) + 1):
            values = bins.get(int(bin_id), [])
            r_val = float(bin_id) / 10.0
            if values:
                rows.append((r_val, float(sum(values)) / float(len(values))))
            else:
                rows.append((r_val, float("nan")))
        return metadata_rows, rows, selected_count

    @staticmethod
    def _prepare_rcs_two_column_metadata(
        metadata_rows: List[List[str]],
        output_path: Path,
        calibration_db: float,
    ) -> List[List[str]]:
        rows = [list(r) for r in metadata_rows]
        saw_data_file = False
        saw_calibration = False
        for row in rows:
            if not row:
                continue
            key = str(row[0] or "").strip().casefold()
            if key == "data file":
                while len(row) < 2:
                    row.append("")
                row[1] = str(output_path)
                saw_data_file = True
            elif key == "calibration":
                while len(row) < 2:
                    row.append("")
                row[1] = format_cluster_csv_calibration(calibration_db)
                saw_calibration = True

        insert_at = next(
            (
                i
                for i, row in enumerate(rows)
                if not row or not any(str(c).strip() for c in row)
            ),
            len(rows),
        )
        if not saw_data_file:
            rows.insert(insert_at, ["Data File", str(output_path)])
            insert_at += 1
        if not saw_calibration:
            rows.insert(insert_at, ["Calibration", format_cluster_csv_calibration(calibration_db)])

        if rows and any(str(c).strip() for c in rows[-1]):
            rows.append([])
        elif not rows:
            rows.append([])
        return rows

    @staticmethod
    def _write_rcs_two_column_csv(
        output_path: Path,
        metadata_rows: List[List[str]],
        columns: Tuple[str, str],
        rows: List[Tuple[float, float]],
        *,
        calibration_db: float,
        include_nan_rcs: bool,
        x_decimals: int,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        meta = MainWindow._prepare_rcs_two_column_metadata(
            metadata_rows, output_path, calibration_db
        )
        with output_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for row in meta:
                w.writerow(row)
            w.writerow(list(columns))
            for x_val, rcs_val in rows:
                if not math.isfinite(float(x_val)):
                    continue
                if not math.isfinite(float(rcs_val)):
                    if not include_nan_rcs:
                        continue
                    rcs_text = "NaN"
                else:
                    rcs_text = f"{float(rcs_val):.4f}"
                w.writerow([f"{float(x_val):.{int(x_decimals)}f}", rcs_text])

    @staticmethod
    def _build_combined_rcs_rows_from_filtered_sets(
        filtered_sets: List[List[Tuple[float, float]]],
    ) -> List[Tuple[float, float]]:
        by_bin: Dict[int, List[float]] = defaultdict(list)
        for rows in filtered_sets:
            for r_val, rcs_val in rows:
                if not math.isfinite(float(r_val)) or not math.isfinite(float(rcs_val)):
                    continue
                by_bin[int(round(float(r_val) * 10.0))].append(float(rcs_val))
        if not by_bin:
            return []

        xs: List[float] = []
        ys: List[float] = []
        for bin_id in sorted(by_bin.keys()):
            vals = by_bin[int(bin_id)]
            if not vals:
                continue
            xs.append(float(bin_id) / 10.0)
            ys.append(float(sum(vals)) / float(len(vals)))

        x_arr = np.asarray(xs, dtype=float)
        y_arr = np.asarray(ys, dtype=float)
        if y_arr.size and _peak_smooth_rcs_series is not None:
            try:
                y_arr = np.asarray(_peak_smooth_rcs_series(y_arr), dtype=float)
            except Exception:
                pass
        out: List[Tuple[float, float]] = []
        for x_val, y_val in zip(x_arr.tolist(), y_arr.tolist()):
            if math.isfinite(float(x_val)) and math.isfinite(float(y_val)):
                out.append((float(x_val), float(y_val)))
        return out

    @staticmethod
    def _combined_rcs_csv_path(raw_paths: List[Path]) -> Optional[Path]:
        paths = [Path(p) for p in raw_paths if str(p).strip()]
        if not paths:
            return None
        first = paths[0]
        if len(paths) == 1:
            return first.with_name(f"{first.stem}_combined.csv")
        source_key = "\n".join(
            str(p.resolve()) if p.exists() else str(p) for p in paths
        )
        digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:8]
        return first.with_name(f"{first.stem}_combined_{len(paths)}files_{digest}.csv")

    def _export_rcs_filtered_and_combined_csvs(
        self,
        file_paths: List[str],
        calibration_db: float,
        combined_rows_override: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[List[Path], Optional[Path]]:
        filtered_paths: List[Path] = []
        filtered_sets: List[List[Tuple[float, float]]] = []
        combined_metadata: Optional[List[List[str]]] = None
        raw_paths: List[Path] = []

        for file_path in file_paths:
            raw_path = Path(file_path)
            raw_paths.append(raw_path)
            metadata_rows, filtered_rows, _selected_count = (
                MainWindow._build_filtered_rcs_rows_from_cluster_raw_path(
                    raw_path, calibration_db
                )
            )
            filtered_path = MainWindow._filtered_rcs_csv_path(raw_path)
            MainWindow._write_rcs_two_column_csv(
                filtered_path,
                metadata_rows,
                ("X", "RCS"),
                filtered_rows,
                calibration_db=calibration_db,
                include_nan_rcs=True,
                x_decimals=1,
            )
            filtered_paths.append(filtered_path)
            filtered_sets.append(filtered_rows)
            if combined_metadata is None:
                combined_metadata = metadata_rows

        combined_rows = (
            list(combined_rows_override)
            if combined_rows_override is not None
            else MainWindow._build_combined_rcs_rows_from_filtered_sets(filtered_sets)
        )
        combined_path: Optional[Path] = None
        if combined_metadata is not None:
            combined_path = MainWindow._combined_rcs_csv_path(raw_paths)
        if combined_path is not None:
            MainWindow._write_rcs_two_column_csv(
                combined_path,
                combined_metadata,
                ("X", "RCS"),
                combined_rows,
                calibration_db=calibration_db,
                include_nan_rcs=False,
                x_decimals=1,
            )
        return filtered_paths, combined_path

    def _try_export_rcs_curve_csvs(
        self,
        file_paths: List[str],
        calibration_db: float,
    ) -> Tuple[List[Path], Optional[Path]]:
        try:
            green_rows = self._current_distance_rcs_green_curve_rows()
            filtered_paths, combined_path = self._export_rcs_filtered_and_combined_csvs(
                file_paths,
                calibration_db,
                combined_rows_override=green_rows if green_rows else None,
            )
        except Exception as exc:
            self._log(f"RCS曲线CSV生成失败: {exc}")
            QtWidgets.QMessageBox.warning(self, "RCS曲线CSV生成失败", str(exc))
            return [], None
        filtered_preview = ", ".join(p.name for p in filtered_paths[:4])
        if len(filtered_paths) > 4:
            filtered_preview += ", ..."
        self._log(
            "RCS曲线CSV已生成: "
            f"Filtered={filtered_preview or '无'}"
            + (f" | combined={combined_path}" if combined_path else "")
        )
        return filtered_paths, combined_path

    def _build_loaded_rcs_curve(
        self,
        file_path: str,
        display_name: Optional[str],
    ) -> LoadedRcsCurve:
        segments, segment_run_labels = self._parse_saved_rcs_raw_file(file_path)
        recorder = RcsRunRecorder()
        recorder.segments = [list(seg) for seg in segments]
        recorder._cur = []
        recorder._ended = True
        recorder.oid = None

        grid = np.arange(0.0, RCS_MAX_DISTANCE_M + RCS_FIT_GRID_STEP_M * 0.5, RCS_FIT_GRID_STEP_M)
        fitted = recorder.fitted_curve(grid)
        fitted = self._clip_curve_by_distance(*fitted)
        label = str(display_name or "").strip() or Path(file_path).stem
        point_count = sum(len(seg) for seg in segments)
        return LoadedRcsCurve(
            file_path=str(file_path),
            display_name=label,
            segments=segments,
            fitted=fitted,
            point_count=point_count,
            segment_run_labels=segment_run_labels,
        )

    def _build_merged_loaded_rcs_curve(
        self,
        file_paths: List[str],
        display_name: Optional[str],
    ) -> LoadedRcsCurve:
        paths = [str(p) for p in file_paths if str(p).strip()]
        if not paths:
            raise ValueError("未选择可合并的RCS数据文件")
        merged_segments: List[List[CurvePoint]] = []
        merged_run_labels: List[int] = []
        run_base = 1
        for p in paths:
            if MainWindow._is_orbit_rcs_polar_csv(Path(p)) or MainWindow._is_orbit_cluster_raw_csv(Path(p)):
                raise ValueError("合并拟合仅支持距离-RCS，不支持圆周-RCS 数据")
            segs, _ = self._parse_saved_rcs_raw_file(p)
            merged_segments.extend(segs)
            merged_run_labels.extend(list(range(run_base, run_base + len(segs))))
            run_base += len(segs)
        recorder = RcsRunRecorder()
        recorder.segments = [list(seg) for seg in merged_segments]
        recorder._cur = []
        recorder._ended = True
        recorder.oid = None
        grid = np.arange(0.0, RCS_MAX_DISTANCE_M + RCS_FIT_GRID_STEP_M * 0.5, RCS_FIT_GRID_STEP_M)
        fitted = recorder.fitted_curve(grid)
        fitted = self._clip_curve_by_distance(*fitted)
        label = str(display_name or "").strip() or f"合并({len(paths)}个文件)"
        point_count = sum(len(seg) for seg in merged_segments)
        # 用第一个文件名作为展示基准（真实来源为多文件合并）
        pseudo_path = str(paths[0])
        return LoadedRcsCurve(
            file_path=pseudo_path,
            display_name=label,
            segments=merged_segments,
            fitted=fitted,
            point_count=point_count,
            segment_run_labels=merged_run_labels,
        )

    def _load_saved_rcs_for_compare(
        self,
        file_paths: List[str],
        ref_class: Optional[str],
        ref_angle: Optional[str],
        display_names: List[Optional[str]],
    ) -> None:
        resolved_paths = [str(path) for path in file_paths if str(path).strip()]
        if not resolved_paths:
            raise ValueError("未选择可载入的RCS数据文件")

        resolved_names = self._ensure_unique_rcs_display_names(resolved_paths, display_names)
        loaded_curves = [
            self._build_loaded_rcs_curve(path, resolved_names[index])
            for index, path in enumerate(resolved_paths)
        ]

        self.rcs_recorder.reset()
        self.rcs_recorder._cur = []
        self.rcs_recorder._ended = True
        self.rcs_recorder.oid = None
        self._rcs_recording = False
        self._rcs_show_only_fitted = False
        self._rcs_orbit_samples = []
        self._orbit_polar_cached_series = None
        self._rcs_active_segment_index = None
        self._rcs_active_path_name = None
        self._rcs_active_target_name = None
        self._loaded_rcs_curves = loaded_curves if len(loaded_curves) > 1 else []
        self._rcs_curve_csv_source_paths = list(resolved_paths)

        if len(loaded_curves) == 1:
            curve = loaded_curves[0]
            self.rcs_recorder.segments = [list(seg) for seg in curve.segments]
            self._rcs_fitted = curve.fitted
            self._rcs_target_name = curve.display_name or Path(curve.file_path).stem
            self._rcs_segment_run_labels = curve.segment_run_labels
        else:
            self._rcs_fitted = None
            compare_title = self._common_nonempty_value([curve.display_name for curve in loaded_curves])
            if compare_title:
                compare_title = f"{compare_title} 对比"
            else:
                compare_title = f"RCS对比({len(loaded_curves)}组)"
            self._rcs_target_name = compare_title
            self._rcs_segment_run_labels = None

        self._apply_rcs_reference_selection(ref_class, ref_angle, log_change=False)

        total_points = sum(curve.point_count for curve in loaded_curves)
        if len(loaded_curves) == 1:
            self.rcs_status_label.setText(
                f"RCS绘图: 已载入 {Path(loaded_curves[0].file_path).name} | 点数={loaded_curves[0].point_count}"
            )
            self._log(
                f"RCS绘图数据已载入: 文件={loaded_curves[0].file_path} | 点数={loaded_curves[0].point_count}"
                f" | 产品={self._rcs_ref_class or '未选择'}"
                f" | 角度={self._rcs_ref_angle or '未选择'}"
                f" | 名称={self._rcs_target_name or 'RCS'}"
            )
        else:
            joined_names = " / ".join(curve.display_name for curve in loaded_curves[:6])
            if len(loaded_curves) > 6:
                joined_names += " / ..."
            self.rcs_status_label.setText(
                f"RCS绘图: 已载入 {len(loaded_curves)} 组数据对比 | 总点数={total_points}"
            )
            self._log(
                f"RCS对比数据已载入: 数量={len(loaded_curves)} | 总点数={total_points}"
                f" | 产品={self._rcs_ref_class or '未选择'}"
                f" | 角度={self._rcs_ref_angle or '未选择'}"
                f" | 曲线={joined_names}"
            )

        self._draw_rcs()

    @staticmethod
    def _cluster_csv_cell_float(cell: str) -> float:
        s = str(cell).strip()
        if not s or s.upper() == "NAN":
            return float("nan")
        return float(s)

    @staticmethod
    def _group_cluster_dicts_for_individual_targets(
        clusters: List[Any],
        merge_radius_m: float,
    ) -> List[Tuple[int, float, float, float]]:
        """
        将同一帧内多个簇合并为「目标物」散点：
        - CAN 声明相同 Cluster_ID 的簇必合并；
        - 否则若平面距离 ≤ merge_radius_m（DX/DY，m）则视为同一物体合并。
        几何：DX/DY 取平均；RCS：非相干功率叠加（与 combine_rcs_db_incoherent_sum 一致）。
        返回 [(oid, dx, dy, rcs_db), ...]，oid 优先取组内最小 Cluster_ID，否则取簇下标最小值。
        """
        n = len(clusters)
        if n <= 0:
            return []

        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        def getxy(i: int) -> Tuple[float, float]:
            c = clusters[i]
            return float(c["DX"]), float(c["DY"])

        def get_cid(i: int) -> Optional[int]:
            c = clusters[i]
            if isinstance(c, dict) and "ClusterID" in c:
                try:
                    return int(c["ClusterID"])
                except (TypeError, ValueError):
                    return None
            return None

        r_m = max(float(merge_radius_m), 1e-6)
        r2 = r_m * r_m
        for i in range(n):
            xi, yi = getxy(i)
            ci = get_cid(i)
            for j in range(i + 1, n):
                cj = get_cid(j)
                same_id = ci is not None and cj is not None and ci == cj
                xj, yj = getxy(j)
                close = (xi - xj) ** 2 + (yi - yj) ** 2 <= r2
                if same_id or close:
                    union(i, j)

        by_root: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            by_root[find(i)].append(i)

        out: List[Tuple[int, float, float, float]] = []
        for _root, idxs in by_root.items():
            idxs_s = sorted(int(k) for k in idxs)
            dxs = [float(clusters[k]["DX"]) for k in idxs_s]
            dys = [float(clusters[k]["DY"]) for k in idxs_s]
            rcs_list = [float(clusters[k]["RCS"]) for k in idxs_s]
            dx_m = float(sum(dxs)) / float(len(dxs))
            dy_m = float(sum(dys)) / float(len(dys))
            if combine_rcs_db_incoherent_sum is not None:
                rcs_m = combine_rcs_db_incoherent_sum(rcs_list)
            else:
                rcs_m = float(
                    10.0
                    * math.log10(sum(10.0 ** (float(v) / 10.0) for v in rcs_list))
                )
            if rcs_m is None:
                rcs_m = rcs_list[0]
            cids_ok = [get_cid(k) for k in idxs_s]
            cids_f = [x for x in cids_ok if x is not None]
            oid = int(min(cids_f)) if cids_f else int(min(idxs_s))
            out.append((oid, dx_m, dy_m, float(rcs_m)))

        out.sort(key=lambda row: (row[0], row[1], row[2]))
        return out

    @staticmethod
    def _parse_cluster_rcs_csv_row_slots(
        row: List[str],
        col: Dict[str, int],
        slot_index: int,
    ) -> Optional[Tuple[float, float, float]]:
        """读取 DX/DY/RCS 槽位；三者均有限值时返回 (dx, dy, rcs_db)，否则 None。"""
        dxk = f"DX{slot_index:02d}"
        dyk = f"DY{slot_index:02d}"
        rck = f"RCS{slot_index:02d}"
        for k in (dxk, dyk, rck):
            if k not in col:
                return None
        try:
            dx = MainWindow._cluster_csv_cell_float(row[col[dxk]])
            dy = MainWindow._cluster_csv_cell_float(row[col[dyk]])
            rcs = MainWindow._cluster_csv_cell_float(row[col[rck]])
        except (ValueError, KeyError, IndexError):
            return None
        if not all(math.isfinite(v) for v in (dx, dy, rcs)):
            return None
        return (float(dx), float(dy), float(rcs))

    @staticmethod
    def _cluster_rcs_curve_point_from_slot(
        t_val: float,
        seg_key: int,
        x_raw: float,
        y_raw: float,
        rcs_val: float,
    ) -> Tuple[int, CurvePoint]:
        r_slant = float(math.hypot(x_raw, y_raw))
        return (
            int(seg_key),
            CurvePoint(
                t=float(t_val),
                x_raw=float(x_raw),
                y_raw=float(y_raw),
                r_raw=float(r_slant),
                x=float(x_raw),
                y=float(y_raw),
                rcs_raw=float(rcs_val),
                rcs_filt=float(rcs_val),
            ),
        )

    @staticmethod
    def _cluster_rcs_rowwise_rcs00_rcs01_linear_power(
        row: List[str],
        col: Dict[str, int],
        t_val: float,
        seg_key: int,
    ) -> List[Tuple[int, CurvePoint]]:
        """
        与 DRI `dri_pipeline_gui._rowwise_linear_power_rcs_db(merge_rcs01=True)` 一致：
        同采样行内 RCS_eff = 10*log10(10^(RCS00/10)+10^(RCS01/10))；几何用主簇 DX00/DY00。
        若仅单槽有效则退化为该槽一点。
        """
        s0 = MainWindow._parse_cluster_rcs_csv_row_slots(row, col, 0)
        s1 = MainWindow._parse_cluster_rcs_csv_row_slots(row, col, 1)
        out: List[Tuple[int, CurvePoint]] = []
        if s0 is not None and s1 is not None:
            x0, y0, r0 = float(s0[0]), float(s0[1]), float(s0[2])
            _x1, _y1, r1 = float(s1[0]), float(s1[1]), float(s1[2])
            if all(math.isfinite(v) for v in (x0, y0, r0, r1)):
                if combine_rcs_db_incoherent_sum is not None:
                    rcs_m = combine_rcs_db_incoherent_sum([r0, r1])
                else:
                    psum = 10.0 ** (r0 / 10.0) + 10.0 ** (r1 / 10.0)
                    rcs_m = float(10.0 * math.log10(max(psum, 1e-300)))
                if rcs_m is None or not math.isfinite(rcs_m):
                    rcs_m = r0
                out.append(
                    MainWindow._cluster_rcs_curve_point_from_slot(
                        t_val, seg_key, x0, y0, float(rcs_m)
                    )
                )
                return out
        if s0 is not None:
            x0, y0, r0 = float(s0[0]), float(s0[1]), float(s0[2])
            if all(math.isfinite(v) for v in (x0, y0, r0)):
                out.append(
                    MainWindow._cluster_rcs_curve_point_from_slot(t_val, seg_key, x0, y0, r0)
                )
                return out
        if s1 is not None:
            x1, y1, r1 = float(s1[0]), float(s1[1]), float(s1[2])
            if all(math.isfinite(v) for v in (x1, y1, r1)):
                out.append(
                    MainWindow._cluster_rcs_curve_point_from_slot(t_val, seg_key, x1, y1, r1)
                )
        return out

    @staticmethod
    def _cluster_rcs_point_with_t(p: CurvePoint, t_new: float) -> CurvePoint:
        return CurvePoint(
            t=float(t_new),
            x_raw=float(p.x_raw),
            y_raw=float(p.y_raw),
            r_raw=float(p.r_raw),
            x=float(p.x),
            y=float(p.y),
            rcs_raw=float(p.rcs_raw),
            rcs_filt=float(p.rcs_filt),
        )

    @staticmethod
    def _cluster_rcs_points_time_relative(points: List[CurvePoint]) -> List[CurvePoint]:
        """与 DRI Raw 一致：段内 Time 从 0 秒起（相对首点时间）。"""
        if not points:
            return points
        t0 = float(points[0].t)
        return [
            MainWindow._cluster_rcs_point_with_t(p, float(p.t) - t0) for p in points
        ]

    @staticmethod
    def _cluster_rcs_points_drop_spatial_spikes(
        points: List[CurvePoint],
        max_dist_step_m: float,
        time_gap_reset_s: float,
    ) -> List[CurvePoint]:
        """
        剔除相邻帧间首目标几何突变点（如 DX/DY 对应 x 方向巨跳）。
        时间间隔 ≥ time_gap_reset_s 时视为新一段，不与前一点比幅值。
        """
        if len(points) <= 1:
            return list(points)
        out: List[CurvePoint] = [points[0]]
        last = points[0]
        for p in points[1:]:
            dt = float(p.t) - float(last.t)
            if not math.isfinite(dt):
                continue
            # 同一时间戳的多槽位（RCS00/RCS01/…）几何可相距较远，不得按「帧间突变」剔除
            if abs(dt) < 1e-9:
                out.append(p)
                last = p
                continue
            if dt >= float(time_gap_reset_s):
                out.append(p)
                last = p
                continue
            dist = math.hypot(
                float(p.x_raw) - float(last.x_raw),
                float(p.y_raw) - float(last.y_raw),
            )
            if dist <= float(max_dist_step_m):
                out.append(p)
                last = p
        return out

    @staticmethod
    def _finalize_cluster_rcs_curve_points(
        points: List[CurvePoint],
    ) -> List[CurvePoint]:
        """排序 → 去空间突变 → 相对时间（DRI 风格）。"""
        if not points:
            return points
        pts = sorted(points, key=lambda p: (float(p.t), float(p.x_raw)))
        pts = MainWindow._cluster_rcs_points_drop_spatial_spikes(
            pts,
            CLUSTER_RCS_PARSE_MAX_DIST_STEP_M,
            CLUSTER_RCS_PARSE_TIME_GAP_RESET_S,
        )
        return MainWindow._cluster_rcs_points_time_relative(pts)

    @staticmethod
    def _parse_cluster_rcs_csv(
        path: Path,
        *,
        merge_rcs00_rcs01_rowwise: bool = True,
        orbit_row_primary_only: bool = False,
    ) -> Tuple[List[List[CurvePoint]], List[int]]:
        """Cluster 0x701 落盘 CSV：每帧一行多槽位 DXii/DYii/RCSii。
        - merge_rcs00_rcs01_rowwise=True（默认）：与 DRI `dri_pipeline_gui._rowwise_linear_power_rcs_db`
          一致，同采样行 RCS00+RCS01 线性功率合成后只生成一点（几何取 DX00/DY00）；槽位 02 起仍逐点展开。
        - orbit_row_primary_only=True：仅保留每帧上述行内合并点（忽略 02+ 槽），供圆周-RCS 按时间平铺绘图。
        - merge_rcs00_rcs01_rowwise=False：每槽位各一点（旧行为）。
        整理：剔除首目标平面位置突变帧，Time 归一为段内相对秒。"""
        tagged: List[Tuple[int, CurvePoint]] = []
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header_cells: Optional[List[str]] = None
            col: Dict[str, int] = {}
            for row in reader:
                if not row:
                    continue
                if row[0].strip() == "Time" and any((c or "").strip() == "DX00" for c in row):
                    header_cells = [(c or "").strip() for c in row]
                    col = {name: idx for idx, name in enumerate(header_cells)}
                    break
            if not header_cells:
                raise ValueError("Cluster RCS CSV：未找到表头行（含 Time、DX00）")
            req = ("Time", "DX00", "DY00", "RCS00")
            for k in req:
                if k not in col:
                    raise ValueError(f"Cluster RCS CSV 缺少列「{k}」")
            max_ix = max(col.values())
            use_seg = "SegIdx" in col

            for raw_row in reader:
                if not raw_row or not any(str(c).strip() for c in raw_row):
                    continue
                row = list(raw_row)
                while len(row) <= max_ix:
                    row.append("")
                try:
                    t_val = MainWindow._cluster_csv_cell_float(row[col["Time"]])
                except (ValueError, KeyError, IndexError):
                    continue
                if not math.isfinite(t_val):
                    continue

                seg_key = 0
                if use_seg:
                    try:
                        seg_key = int(round(float(MainWindow._cluster_csv_cell_float(row[col["SegIdx"]]))))
                    except (ValueError, OverflowError):
                        seg_key = 0

                row_any = False
                if merge_rcs00_rcs01_rowwise:
                    merged01 = MainWindow._cluster_rcs_rowwise_rcs00_rcs01_linear_power(
                        row, col, t_val, seg_key
                    )
                    for item in merged01:
                        tagged.append(item)
                        row_any = True
                    if not orbit_row_primary_only:
                        for slot in range(2, int(MAX_CLUSTERS)):
                            s = MainWindow._parse_cluster_rcs_csv_row_slots(row, col, slot)
                            if s is None:
                                continue
                            x_raw, y_raw, rcs_val = float(s[0]), float(s[1]), float(s[2])
                            if not all(math.isfinite(v) for v in (x_raw, y_raw, rcs_val)):
                                continue
                            row_any = True
                            tagged.append(
                                MainWindow._cluster_rcs_curve_point_from_slot(
                                    t_val, seg_key, x_raw, y_raw, rcs_val
                                )
                            )
                else:
                    for slot in range(int(MAX_CLUSTERS)):
                        s = MainWindow._parse_cluster_rcs_csv_row_slots(row, col, slot)
                        if s is None:
                            continue
                        x_raw, y_raw, rcs_val = float(s[0]), float(s[1]), float(s[2])
                        if not all(math.isfinite(v) for v in (x_raw, y_raw, rcs_val)):
                            continue
                        row_any = True
                        tagged.append(
                            MainWindow._cluster_rcs_curve_point_from_slot(
                                t_val, seg_key, x_raw, y_raw, rcs_val
                            )
                        )
                if not row_any:
                    continue

        if not tagged:
            raise ValueError(
                "Cluster RCS CSV 中没有可用的簇点（至少需要某一槽位 DXii/DYii/RCSii 均为有限值）"
            )

        if use_seg:
            by_seg: Dict[int, List[CurvePoint]] = defaultdict(list)
            for sk, pt in tagged:
                by_seg[int(sk)].append(pt)
            sorted_keys = sorted(by_seg.keys())
            segments: List[List[CurvePoint]] = []
            segment_run_labels: List[int] = []
            for run_i, sk in enumerate(sorted_keys, start=1):
                pts = sorted(by_seg[sk], key=lambda p: (float(p.t), float(p.x_raw)))
                if pts:
                    segments.append(MainWindow._finalize_cluster_rcs_curve_points(pts))
                    segment_run_labels.append(run_i)
            if not segments:
                raise ValueError("Cluster RCS CSV 中没有有效的分段数据")
            return segments, segment_run_labels

        points = [pt for _, pt in tagged]
        points.sort(key=lambda p: (float(p.t), float(p.x_raw)))
        return [MainWindow._finalize_cluster_rcs_curve_points(points)], [1]

    def _parse_saved_rcs_raw_file(self, file_path: str) -> Tuple[List[List[CurvePoint]], List[int]]:
        path = Path(file_path)
        if path.suffix.lower() != ".csv":
            raise ValueError("仅支持载入 .csv 格式的 Cluster RCS 数据")
        return MainWindow._parse_cluster_rcs_csv(path)

    def _orbit_plot_rows_from_cluster_raw_path(self, file_path: str) -> List[Dict[str, float]]:
        """圆周 Raw 专用：每行 RCS00+RCS01 线性功率合并后按时间平铺。

        圆周图只使用时间序列与合并后的 RCS，不使用直线距离图的空间突变剔除。
        """
        path = Path(file_path)
        rows: List[Dict[str, float]] = []
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header_cells: Optional[List[str]] = None
            col: Dict[str, int] = {}
            for row in reader:
                if not row:
                    continue
                if row[0].strip() == "Time" and any((c or "").strip() == "DX00" for c in row):
                    header_cells = [(c or "").strip() for c in row]
                    col = {name: idx for idx, name in enumerate(header_cells)}
                    break
            if not header_cells:
                raise ValueError("Cluster RCS CSV：未找到表头行（含 Time、DX00）")
            for k in ("Time", "DX00", "DY00", "RCS00"):
                if k not in col:
                    raise ValueError(f"Cluster RCS CSV 缺少列「{k}」")
            max_ix = max(col.values())

            for raw_row in reader:
                if not raw_row or not any(str(c).strip() for c in raw_row):
                    continue
                row = list(raw_row)
                while len(row) <= max_ix:
                    row.append("")
                try:
                    t_val = MainWindow._cluster_csv_cell_float(row[col["Time"]])
                except (ValueError, KeyError, IndexError):
                    continue
                if not math.isfinite(t_val):
                    continue
                merged = MainWindow._cluster_rcs_rowwise_rcs00_rcs01_linear_power(
                    row, col, float(t_val), 0
                )
                if not merged:
                    continue
                pt = merged[0][1]
                rcs_f = float(pt.rcs_filt)
                if not math.isfinite(rcs_f):
                    continue
                rows.append({"t": float(t_val), "rcs_filt": rcs_f})
        if not rows:
            raise ValueError("Cluster RCS CSV 中没有可用于圆周绘图的 RCS00/RCS01 数据")
        rows.sort(key=lambda r: float(r["t"]))
        return rows

    def _on_plot_saved_rcs_clicked(self) -> None:
        # Repurposed per request: "导入数据对比"
        # Based on the currently plotted base file, pick another file and overlay fitted curve(s).
        self._show_tool_dialog(self._rcs_viewer_dialog)
        if self._rcs_recording:
            QtWidgets.QMessageBox.information(
                self,
                "Cluster RCS 采集中",
                "请先等待当前 Cluster RCS 采集结束后再进行数据对比。",
            )
            return
        if not self._rcs_base_file_path:
            QtWidgets.QMessageBox.information(
                self,
                "尚未选择基础数据",
                "请先点击“选择数据绘RCS图”，载入一组基础数据后再导入对比数据。",
            )
            self._log("RCS对比已阻止: 尚未选择基础绘图数据文件")
            return

        new_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择对比RCS数据（将与当前曲线叠加对比）",
            str(self._rcs_save_dir),
            "CSV (*.csv);;所有文件(*)",
        )
        if not new_path:
            return

        base_path = str(self._rcs_base_file_path)
        file_paths = [base_path, str(new_path)]
        default_names = [
            str(self._rcs_target_name or Path(base_path).stem).strip() or Path(base_path).stem,
            Path(new_path).stem,
        ]
        reference_summary = self._format_rcs_reference_summary()
        dialog = RcsCompareConfigDialog(
            file_paths=file_paths,
            reference_summary=reference_summary,
            default_display_names=default_names,
            default_calibration_db=self._rcs_plot_calibration_db,
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            self._log("RCS对比已取消")
            return
        display_names, cal_db = dialog.values()
        self._rcs_plot_calibration_db = cal_db

        # Compare view: show reference limits only when a reference is selected and limits are available.
        show_ref = bool(self._rcs_ref_class and self._rcs_ref_angle and self._rcs_ref_limits is not None)
        self._rcs_hide_reference_limits = not show_ref
        try:
            self._load_saved_rcs_for_compare(
                file_paths,
                self._rcs_ref_class if show_ref else None,
                self._rcs_ref_angle if show_ref else None,
                display_names,
            )
        except Exception as exc:
            self._log(f"RCS对比失败: {exc}")
            QtWidgets.QMessageBox.warning(self, "RCS对比失败", f"载入RCS数据失败:\n{exc}")

    def _on_save_motion_data(self) -> None:
        records_with_samples = [
            record for record in self._tracking_run_records
            if record.get("motion_samples")
        ]
        live_snapshot_records = self._build_live_motion_snapshot_records()
        if not records_with_samples and not live_snapshot_records:
            QtWidgets.QMessageBox.information(self, "无运动数据", "当前没有可保存的运动采样数据。")
            return

        target_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "选择运动数据保存目录",
            str(self._motion_save_dir),
        )
        if not target_dir:
            return

        try:
            all_records = records_with_samples + live_snapshot_records
            saved_files = self._save_motion_records(
                all_records,
                Path(target_dir),
                skip_if_saved=False,
                mark_saved=False,
            )
            finished_count = len(records_with_samples)
            live_count = len(live_snapshot_records)
            if live_count > 0:
                self._log(
                    f"已包含运行中快照保存: 已结束={finished_count} 运行中={live_count}"
                )
            if saved_files:
                self._refresh_data_analysis_window(focus_path=Path(saved_files[-1]))
            self._log(f"运动数据已保存: {len(saved_files)}个文件 -> {target_dir}")
            QtWidgets.QMessageBox.information(
                self,
                "保存成功",
                (
                    f"已保存 {len(saved_files)} 个运动数据文件到:\n{target_dir}\n"
                    f"(已结束={finished_count}, 运行中快照={live_count})"
                ),
            )
        except Exception as exc:
            self._log(f"运动数据保存失败: {exc}")
            QtWidgets.QMessageBox.warning(self, "保存失败", f"运动数据保存失败:\n{exc}")

    def _build_live_motion_snapshot_records(self) -> List[Dict[str, Any]]:
        with self._tracking_sample_lock:
            buffer_items = [
                (str(run_key), [dict(sample) for sample in list(samples)])
                for run_key, samples in self._tracking_sample_buffers.items()
                if samples
            ]
        if not buffer_items:
            return []

        base_index = len(self._tracking_run_records) + 1
        now_ts = time.time()
        snapshot_records: List[Dict[str, Any]] = []
        for offset, (run_key, samples) in enumerate(buffer_items):
            if not samples:
                continue
            first_sample = dict(samples[0])
            last_sample = dict(samples[-1])
            run_label = str(
                last_sample.get("run_label")
                or first_sample.get("run_label")
                or "轨迹"
            ).strip() or "轨迹"
            try:
                timestamp = float(last_sample.get("timestamp", now_ts))
            except (TypeError, ValueError):
                timestamp = float(now_ts)
            try:
                duration_s = max(0.0, float(last_sample.get("relative_time_s", 0.0)))
            except (TypeError, ValueError):
                duration_s = 0.0
            try:
                speed_mps = float(
                    last_sample.get(
                        "profile_speed_mps",
                        last_sample.get("desired_v_mps", last_sample.get("cmd_v_mps", 0.0)),
                    )
                )
            except (TypeError, ValueError):
                speed_mps = 0.0

            distance_m, signed_distance_m, distance_kind, distance_label = (
                self._motion_distance_fields(last_sample)
            )
            snapshot_records.append(
                {
                    "record_index": int(base_index + offset),
                    "timestamp": float(timestamp),
                    "run_key": run_key,
                    "run_label": run_label,
                    "tracking_mode": str(last_sample.get("tracking_mode") or "stanley"),
                    "speed_mps": float(speed_mps),
                    "duration_s": float(duration_s),
                    "samples": int(len(samples)),
                    "motion_distance_total_m": float(distance_m),
                    "motion_distance_total_signed_m": float(signed_distance_m),
                    "motion_distance_kind": distance_kind,
                    "motion_distance_label": distance_label,
                    "motion_samples": samples,
                    "record_snapshot_live": 1,
                }
            )
        return snapshot_records

    def _load_rcs_reference_options(self) -> None:
        self._rcs_ref_class_options = []
        self._rcs_ref_angle_options = []
        self._rcs_ref_labels = {}
        if _rcs_ref_mod is None:
            return
        try:
            if hasattr(_rcs_ref_mod, "get_rcs_reference_options"):
                classes, angles = _rcs_ref_mod.get_rcs_reference_options()
                self._rcs_ref_class_options = list(classes) if classes else []
                self._rcs_ref_angle_options = list(angles) if angles else []
            if hasattr(_rcs_ref_mod, "get_rcs_reference_labels"):
                labels = _rcs_ref_mod.get_rcs_reference_labels()
                if isinstance(labels, dict):
                    self._rcs_ref_labels = labels
        except Exception as exc:
            print(f"[main_ui] Failed to read rcs reference options: {exc}")

    def _has_rcs_reference_options(self) -> bool:
        return bool(self._rcs_ref_class_options) and bool(self._rcs_ref_angle_options)

    def _format_rcs_reference_summary(
        self,
        ref_class: Optional[str] = None,
        ref_angle: Optional[str] = None,
    ) -> str:
        if not self._has_rcs_reference_options():
            return "参考库未加载"
        resolved_class = str(ref_class or self._rcs_ref_class or "").strip()
        resolved_angle = str(ref_angle or self._rcs_ref_angle or "").strip()
        if not resolved_class or not resolved_angle:
            return "未选择"
        display_class = str(self._rcs_ref_labels.get(resolved_class, resolved_class)).strip()
        if display_class and display_class != resolved_class:
            return f"{display_class} ({resolved_class}) / {resolved_angle}"
        return f"{resolved_class} / {resolved_angle}"

    def _update_rcs_reference_summary_label(self) -> None:
        summary = self._format_rcs_reference_summary()
        if hasattr(self, "label_rcs_reference_summary") and self.label_rcs_reference_summary is not None:
            self.label_rcs_reference_summary.setText(f"RCS参考: {summary}")
        if hasattr(self, "btn_select_rcs_reference") and self.btn_select_rcs_reference is not None:
            self.btn_select_rcs_reference.setEnabled(self._has_rcs_reference_options())
        if hasattr(self, "btn_plot_saved_rcs") and self.btn_plot_saved_rcs is not None:
            self.btn_plot_saved_rcs.setEnabled(True)
            if self._has_rcs_reference_options():
                tooltip = (
                    f"导入数据对比 | 当前参考: {summary}"
                    if (self._rcs_ref_class and self._rcs_ref_angle)
                    else "导入数据对比：未选择参考产品（将不叠加参考上下限）"
                )
                self.btn_plot_saved_rcs.setToolTip(tooltip)
            else:
                self.btn_plot_saved_rcs.setToolTip("导入数据对比：参考库未加载（将不叠加参考上下限）")

    def _apply_rcs_reference_selection(
        self,
        ref_class: Optional[str],
        ref_angle: Optional[str],
        log_change: bool = True,
    ) -> None:
        resolved_class = str(ref_class or "").strip() or None
        resolved_angle = str(ref_angle or "").strip() or None
        if resolved_class is None:
            resolved_angle = None
        self._rcs_ref_class = resolved_class
        self._rcs_ref_angle = resolved_angle
        self._update_rcs_reference_summary_label()
        self._update_rcs_reference_limits()
        # 一旦用户选择了参考产品，就应立即显示上下限（除非确实没有可用 limits）。
        if self._rcs_ref_class is not None and self._rcs_ref_angle is not None and self._rcs_ref_limits is not None:
            self._rcs_hide_reference_limits = False
            self._draw_rcs()
        if not log_change:
            return
        if self._rcs_ref_class is not None and self._rcs_ref_angle is not None:
            self._log(f"RCS参考已选择: {self._format_rcs_reference_summary()}")
        else:
            self._log("RCS参考已清除")

    def _on_select_rcs_reference(self) -> None:
        if not self._has_rcs_reference_options():
            QtWidgets.QMessageBox.information(
                self,
                "参考库未加载",
                "当前没有可用的 RCS 参考产品和角度配置。",
            )
            self._log("选择RCS参考产品失败: 参考库未加载")
            return
        dialog = RcsReferenceConfigDialog(
            class_options=self._rcs_ref_class_options,
            angle_options=self._rcs_ref_angle_options,
            class_labels=self._rcs_ref_labels,
            default_class=self._rcs_ref_class,
            default_angle=self._rcs_ref_angle,
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        ref_class, ref_angle = dialog.values()
        self._apply_rcs_reference_selection(ref_class, ref_angle)

    def _update_rcs_reference_limits(self) -> None:
        if self._rcs_ref_class is None or self._rcs_ref_angle is None:
            self._rcs_ref_limits = None
            self._draw_rcs()
            return
        if _rcs_ref_mod is None or not hasattr(_rcs_ref_mod, "get_rcs_reference_limits"):
            self._rcs_ref_limits = None
            self._draw_rcs()
            return
        try:
            limits = _rcs_ref_mod.get_rcs_reference_limits(self._rcs_ref_class, self._rcs_ref_angle)
        except Exception as exc:
            self._log(f"参考边界读取失败: {exc}")
            self._rcs_ref_limits = None
            self._draw_rcs()
            return
        self._rcs_ref_limits = limits
        if limits is None:
            self._log(f"未找到参考边界: {self._rcs_ref_class} {self._rcs_ref_angle}")
        self._draw_rcs()

    def _apply_rcs_plot_calibration_to_y(self, ys: np.ndarray) -> np.ndarray:
        c = float(getattr(self, "_rcs_plot_calibration_db", 0.0) or 0.0)
        y = np.asarray(ys, dtype=float)
        if c == 0.0:
            return y
        return y + c

    @staticmethod
    def _clip_curve_by_distance(
        xs: np.ndarray,
        ys: np.ndarray,
        x_min: float = 0.0,
        x_max: float = RCS_MAX_DISTANCE_M,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        if x.size == 0 or y.size == 0:
            return x, y
        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]
        mask = np.isfinite(x) & (x >= float(x_min)) & (x <= float(x_max))
        return x[mask], y[mask]

    def _plot_rcs_reference_limits(
        self,
        ax,
        *,
        show_labels: bool = True,
        linewidth: float = 2.0,
        alpha: float = 1.0,
    ) -> None:
        if bool(getattr(self, "_rcs_hide_reference_limits", False)):
            return
        limits = self._rcs_ref_limits
        if limits is None:
            return

        xs = limits.get("x")
        if xs is None:
            return
        xs = np.asarray(xs, dtype=float)
        mask = np.isfinite(xs) & (xs >= 0.0) & (xs <= RCS_MAX_DISTANCE_M)
        if not np.any(mask):
            return

        lower = limits.get("lower")
        upper = limits.get("upper")
        ref = limits.get("ref")
        ref_color = "#000000"
        if lower is not None:
            lower = np.asarray(lower, dtype=float)
            if lower.shape == xs.shape:
                ax.plot(
                    xs[mask],
                    lower[mask],
                    color=ref_color,
                    linewidth=linewidth,
                    alpha=alpha,
                    label="下界" if show_labels else None,
                )
        if upper is not None:
            upper = np.asarray(upper, dtype=float)
            if upper.shape == xs.shape:
                ax.plot(
                    xs[mask],
                    upper[mask],
                    color=ref_color,
                    linewidth=linewidth,
                    alpha=alpha,
                    label="上界" if show_labels else None,
                )
        if ref is not None:
            ref = np.asarray(ref, dtype=float)
            if ref.shape == xs.shape:
                ax.plot(
                    xs[mask],
                    ref[mask],
                    color=ref_color,
                    linewidth=linewidth,
                    alpha=alpha,
                    label="参考值" if show_labels else None,
                )
        return

        if self._rcs_ref_limits is None:
            return

        xs = self._rcs_ref_limits.get("x")
        if xs is None:
            return
        xs = np.asarray(xs, dtype=float)
        mask = np.isfinite(xs) & (xs >= 0.0) & (xs <= RCS_MAX_DISTANCE_M)
        if not np.any(mask):
            return

        lower = self._rcs_ref_limits.get("lower")
        upper = self._rcs_ref_limits.get("upper")
        ref = self._rcs_ref_limits.get("ref")
        if lower is not None:
            lower = np.asarray(lower, dtype=float)
            if lower.shape == xs.shape:
                ax.plot(xs[mask], lower[mask], color="k", linewidth=2, label="下界")
        if upper is not None:
            upper = np.asarray(upper, dtype=float)
            if upper.shape == xs.shape:
                ax.plot(xs[mask], upper[mask], color="k", linewidth=2, label="上界")
        if ref is not None:
            ref = np.asarray(ref, dtype=float)
            if ref.shape == xs.shape:
                ax.plot(xs[mask], ref[mask], color="b", linewidth=2, label="参考值")

    @staticmethod
    def _thin_rcs_tick_labels(
        positions: np.ndarray,
        values: np.ndarray,
        max_ticks: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if positions.size <= max_ticks or max_ticks <= 0:
            return positions, values
        indices = np.linspace(0, positions.size - 1, max_ticks, dtype=int)
        indices = np.unique(indices)
        return positions[indices], values[indices]

    def _get_rcs_plot_display_title(self, plot_mode: str, compact: bool = False) -> str:
        if compact:
            if plot_mode == "orbit":
                return "Orbit RCS"
            return "RCS Compare" if self._loaded_rcs_curves else "RCS"
        if plot_mode == "orbit":
            return f"{self._get_rcs_title()} | Orbit RCS"
        return self._get_rcs_title()

    def _create_rcs_export_figure(self) -> Figure:
        plot_mode = self._get_rcs_plot_mode()
        # 导出图片固定为 1:1（正方形），避免保存时长宽比变化
        figsize = (7.8, 7.8)
        fig = Figure(figsize=figsize, dpi=140, facecolor="white")
        if plot_mode == "orbit":
            ax = fig.add_subplot(111, projection="polar")
        else:
            ax = fig.add_subplot(111)
        self._render_rcs_plot_on_axis(
            ax,
            plot_mode,
            live=self._rcs_recording,
            compact=False,
            export=True,
        )
        fig.tight_layout(pad=1.1)
        return fig

    @staticmethod
    def _ema_smooth_rcs_array(y: np.ndarray, alpha: float) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.size <= 1:
            return y.copy()
        a = float(alpha)
        if not np.isfinite(a) or a <= 0.0 or a >= 1.0:
            return y.copy()
        out = np.empty_like(y, dtype=float)
        out[0] = float(y[0])
        for i in range(1, int(y.size)):
            out[i] = a * float(y[i]) + (1.0 - a) * float(out[i - 1])
        return out

    @staticmethod
    def _bin_rcs_curve_points_for_plot(points: List[CurvePoint]) -> Tuple[np.ndarray, np.ndarray]:
        """与距离-RCS图中每次测量曲线一致：0.1m 分箱、同周期功率合并、RCS跳变剔除、平滑。"""
        if not points:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        xs = np.asarray([float(p.x) for p in points], dtype=float)
        ys = np.asarray([float(p.rcs_filt) for p in points], dtype=float)
        ts = np.asarray([float(p.t) for p in points], dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ts)
        mask &= (xs >= float(RCS_STRAIGHT_X_MIN_M)) & (xs <= float(RCS_STRAIGHT_X_MAX_M))
        xs = xs[mask]
        ys = ys[mask]
        ts = ts[mask]
        if xs.size == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        order0 = np.argsort(xs)
        xs = xs[order0]
        ys = ys[order0]
        ts = ts[order0]
        if xs.size >= 2:
            keep = np.ones(xs.size, dtype=bool)
            last_idx = 0
            for j in range(1, xs.size):
                if not keep[last_idx]:
                    last_idx = j
                    continue
                if abs(float(ys[j]) - float(ys[last_idx])) > 10.0:
                    keep[j] = False
                else:
                    last_idx = j
            xs = xs[keep]
            ys = ys[keep]
            ts = ts[keep]
            if xs.size == 0:
                return np.asarray([], dtype=float), np.asarray([], dtype=float)

        bin_m = 0.1
        bin_ids = np.floor(xs / float(bin_m)).astype(np.int64)
        uniq = np.unique(bin_ids)
        xb: List[float] = []
        yb: List[float] = []
        t_gate = float(RCS_BIN_INCOHERENT_SUM_MAX_TIME_SPAN_S)
        for bid in uniq:
            m = bin_ids == bid
            if not np.any(m):
                continue
            xb.append(float(np.mean(xs[m])))
            ys_b = ys[m]
            ts_b = ts[m]
            if ys_b.size <= 1:
                yb.append(float(ys_b[0]))
            elif float(np.max(ts_b) - np.min(ts_b)) <= t_gate:
                if combine_rcs_db_incoherent_sum is not None:
                    cr = combine_rcs_db_incoherent_sum(ys_b.tolist())
                    yb.append(float(cr) if cr is not None else float(np.mean(ys_b)))
                else:
                    yb.append(
                        float(
                            10.0
                            * math.log10(float(np.sum(10.0 ** (ys_b.astype(float) / 10.0))))
                        )
                    )
            else:
                yb.append(float(np.mean(ys_b)))
        x_out = np.asarray(xb, dtype=float)
        y_out = np.asarray(yb, dtype=float)
        order = np.argsort(x_out)
        x_out = x_out[order]
        y_out = y_out[order]
        if _peak_smooth_rcs_series is not None:
            y_out = np.asarray(_peak_smooth_rcs_series(y_out), dtype=float)
        else:
            y_out = MainWindow._ema_smooth_rcs_array(y_out, float(RCS_STRAIGHT_EMA_ALPHA))
        return x_out, y_out

    def _current_distance_rcs_green_curve_rows(self) -> List[Tuple[float, float]]:
        """返回当前距离-RCS图绿色融合曲线的 X/RCS 点；无绿色曲线时返回空列表。"""
        if self._get_rcs_plot_mode() != "distance" or getattr(self, "_loaded_rcs_curves", []):
            return []
        recorder = getattr(self, "rcs_recorder", None)
        segments = list(getattr(recorder, "segments", []) or [])
        if not segments:
            return []

        per_run_binned: List[Tuple[np.ndarray, np.ndarray]] = []
        for seg in segments[:30]:
            xb, yb = MainWindow._bin_rcs_curve_points_for_plot(list(seg))
            yb = self._apply_rcs_plot_calibration_to_y(yb)
            if xb.size == 0 or yb.size == 0:
                continue
            per_run_binned.append((np.asarray(xb, dtype=float), np.asarray(yb, dtype=float)))
        if not per_run_binned:
            return []

        bin_m = 0.1
        thr_db = 10.0
        bin_to_xs: Dict[int, List[float]] = {}
        bin_to_ys: Dict[int, List[float]] = {}
        for xb, yb in per_run_binned:
            n = min(int(xb.size), int(yb.size))
            xb2 = np.asarray(xb[:n], dtype=float)
            yb2 = np.asarray(yb[:n], dtype=float)
            m2 = np.isfinite(xb2) & np.isfinite(yb2)
            xb2 = xb2[m2]
            yb2 = yb2[m2]
            if xb2.size == 0:
                continue
            bids = np.floor(xb2 / float(bin_m)).astype(np.int64)
            for k, xk, yk in zip(bids.tolist(), xb2.tolist(), yb2.tolist()):
                bin_to_xs.setdefault(int(k), []).append(float(xk))
                bin_to_ys.setdefault(int(k), []).append(float(yk))

        xb_all: List[float] = []
        yb_all: List[float] = []
        for bid in sorted(bin_to_ys.keys()):
            ys_list = bin_to_ys.get(int(bid), [])
            xs_list = bin_to_xs.get(int(bid), [])
            if not ys_list or not xs_list:
                continue
            ys_arr = np.asarray(ys_list, dtype=float)
            xs_arr = np.asarray(xs_list, dtype=float)
            med = float(np.median(ys_arr))
            keep = np.abs(ys_arr - med) <= float(thr_db)
            if not np.any(keep):
                continue
            xb_all.append(float(np.mean(xs_arr[keep])))
            yb_all.append(float(np.mean(ys_arr[keep])))
        if not xb_all:
            return []
        x_arr = np.asarray(xb_all, dtype=float)
        y_arr = np.asarray(yb_all, dtype=float)
        order_all = np.argsort(x_arr)
        x_arr = x_arr[order_all]
        y_arr = y_arr[order_all]
        if _peak_smooth_rcs_series is not None:
            y_arr = np.asarray(_peak_smooth_rcs_series(y_arr), dtype=float)
        else:
            y_arr = MainWindow._ema_smooth_rcs_array(y_arr, float(RCS_STRAIGHT_EMA_ALPHA))
        out: List[Tuple[float, float]] = []
        for x_val, y_val in zip(x_arr.tolist(), y_arr.tolist()):
            if math.isfinite(float(x_val)) and math.isfinite(float(y_val)):
                out.append((float(x_val), float(y_val)))
        return out

    def _render_rcs_plot_on_axis(
        self,
        ax,
        plot_mode: str,
        *,
        live: bool = False,
        compact: bool = False,
        export: bool = False,
    ) -> None:
        ax.clear()
        ax.set_facecolor("#FCFCFD" if export else "white")

        if plot_mode == "orbit":
            series = self._get_rcs_orbit_plot_series()
            ax.set_title(
                self._get_rcs_plot_display_title(plot_mode, compact=compact),
                fontsize=14 if export else 10,
                fontweight="semibold" if export else "normal",
                pad=14 if export else 8,
            )
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_thetagrids(range(0, 360, 45 if export else 90))
            ax.grid(
                True,
                color="#CFD8DC" if export else "#E5EAEE",
                linestyle="-" if export else "--",
                linewidth=0.85 if export else 0.6,
            )
            ax.set_axisbelow(True)
            ax.set_rlabel_position(135)
            if series is None:
                ax.text(
                    0.5,
                    0.5,
                    "暂无圆周RCS数据\n圆弧段勾选RCS并跑完轨迹后，仅用 Cluster Raw 落盘；\n"
                    "载入同目录 Raw 或切换「圆周-RCS」查看（按时间平铺后角度分箱均值）。\n"
                    "历史 __orbit_rcs__*.csv 仍可载入。",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11 if export else 9,
                    color="#546E7A",
                )
                return

            angles = np.asarray(series["angles"], dtype=float)
            values = self._apply_rcs_plot_calibration_to_y(np.asarray(series["values"], dtype=float))
            fit_angles = np.asarray(series.get("fit_angles", []), dtype=float)
            fit_values = self._apply_rcs_plot_calibration_to_y(
                np.asarray(series.get("fit_values", []), dtype=float)
            )
            orbit_mode = str(series.get("mode", "abs_dbsm"))
            abs_dbsm = orbit_mode == "abs_dbsm"

            if abs_dbsm:
                # 标定后按当前 RCS 重算径向平移与刻度，保持半径刻度显示真实 dBsm。
                rmin = float(np.min(values)) if values.size else 0.0
                rmax = float(np.max(values)) if values.size else 0.0
                tick_step = max(float(series.get("radial_tick_step_db", ORBIT_RCS_RADIAL_TICK_STEP_DB)), 1e-6)
                tick_start = math.floor(rmin / tick_step) * tick_step
                tick_end = math.ceil(rmax / tick_step) * tick_step
                if tick_end <= tick_start:
                    tick_end = tick_start + tick_step
                tick_values = np.arange(
                    tick_start,
                    tick_end + tick_step * 0.5,
                    tick_step,
                    dtype=float,
                )
                floor = tick_start - max(0.5, tick_step * 0.1)
                radii = values - floor if values.size else np.asarray([], dtype=float)
                fit_radii = fit_values - floor if fit_values.size else np.asarray([], dtype=float)
                tick_positions = tick_values - floor
            else:
                radii = np.asarray(series["radii"], dtype=float)
                fit_radii = np.asarray(series.get("fit_radii", []), dtype=float)
                tick_positions = np.asarray(series["tick_positions"], dtype=float)
                tick_values = self._apply_rcs_plot_calibration_to_y(
                    np.asarray(series["tick_values"], dtype=float)
                )

            if compact and not abs_dbsm:
                tick_positions, tick_values = self._thin_rcs_tick_labels(
                    tick_positions,
                    tick_values,
                    3,
                )

            plotted_mean_curve = False
            if fit_angles.size >= 2 and fit_radii.size == fit_angles.size:
                curve_angles = np.asarray(fit_angles, dtype=float)
                curve_radii = np.asarray(fit_radii, dtype=float)
                finite_curve = np.isfinite(curve_angles) & np.isfinite(curve_radii)
                curve_angles = curve_angles[finite_curve]
                curve_radii = curve_radii[finite_curve]
                if curve_angles.size >= 2:
                    if curve_angles.size >= 3:
                        curve_angles = np.concatenate(
                            [curve_angles, np.asarray([2.0 * math.pi], dtype=float)]
                        )
                        curve_radii = np.concatenate(
                            [curve_radii, np.asarray([float(curve_radii[0])], dtype=float)]
                        )
                    ax.plot(
                        curve_angles,
                        curve_radii,
                        "-",
                        color="#1565C0",
                        linewidth=2.25 if export else 1.65,
                        alpha=0.98,
                        zorder=4,
                        label="角度分箱均值",
                    )
                    if export and curve_angles.size >= 3:
                        ax.fill(
                            curve_angles,
                            curve_radii,
                            color="#90CAF9",
                            alpha=0.12,
                            zorder=1,
                        )
                    plotted_mean_curve = True

            if not plotted_mean_curve and angles.size >= 2:
                if abs_dbsm:
                    ax.plot(
                        angles,
                        radii,
                        "-",
                        color="#1565C0",
                        linewidth=1.55 if export else 1.05,
                        alpha=0.93,
                        zorder=2,
                    )
                else:
                    pts = np.column_stack([angles, radii]).astype(float)
                    segs = np.stack([pts[:-1], pts[1:]], axis=1)
                    seg_values = (
                        0.5 * (values[:-1] + values[1:])
                        if values.size == angles.size
                        else values[:-1]
                    )
                    lc = LineCollection(
                        segs,
                        cmap="turbo",
                        linewidths=1.25 if export else 0.85,
                        alpha=0.98 if export else 0.95,
                        transform=ax.transData,
                    )
                    lc.set_array(np.asarray(seg_values, dtype=float))
                    ax.add_collection(lc)
                    ax.fill(
                        angles,
                        radii,
                        color="#90CAF9",
                        alpha=0.18 if export else 0.10,
                    )

            scatter = ax.scatter(
                angles,
                radii,
                c=values,
                cmap="viridis",
                s=34 if export else 14,
                alpha=0.95 if export else 0.9,
                edgecolors="none",
                zorder=3,
            )
            if abs_dbsm and tick_positions.size:
                r_cap = float(np.max(tick_positions)) * 1.08
            elif abs_dbsm and radii.size:
                r_cap = float(np.max(radii)) * 1.08
            else:
                _rt = series.get("r_top")
                if _rt is not None:
                    r_cap = float(_rt)
                else:
                    r_cap = (
                        float(np.max(tick_positions)) * 1.08
                        if tick_positions.size
                        else (float(np.max(radii)) * 1.08 if radii.size else 1.0)
                    )
            if tick_positions.size:
                ax.set_yticks(tick_positions)
                tick_fmt = "{:.1f}" if export else "{:.0f}"
                ax.set_yticklabels([tick_fmt.format(val) for val in tick_values])
            ax.set_ylim(0.0, max(r_cap, float(np.max(radii)) * 1.05) if radii.size else r_cap)
            if export:
                cbar = ax.figure.colorbar(scatter, ax=ax, pad=0.10, fraction=0.05)
                cbar.set_label("RCS (dBsm)")
                cbar.ax.tick_params(labelsize=9)
            return

        ax.set_title(
            self._get_rcs_plot_display_title(plot_mode, compact=compact),
            fontsize=14 if export else 10,
            fontweight="semibold" if export else "normal",
            pad=12 if export else 6,
        )
        ax.set_xlabel("Front Distance (m)" if export else "Distance (m)")
        ax.set_ylabel("RCS (dBsm)" if export else "RCS")
        ax.set_axisbelow(True)
        ax.grid(False)
        ax.yaxis.grid(
            True,
            color="#DCE3E8" if export else "#E7ECF0",
            linestyle="-",
            linewidth=0.8 if export else 0.6,
        )
        if export:
            ax.xaxis.grid(True, color="#EEF2F5", linestyle="--", linewidth=0.65)
        ax.set_xlim(0, RCS_MAX_DISTANCE_M)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        show_legend = export or len(self._loaded_rcs_curves) > 1
        show_ref_labels = export

        if self._rcs_show_only_fitted and self._rcs_fitted is not None:
            xg, yg = self._clip_curve_by_distance(*self._rcs_fitted)
            yg = self._apply_rcs_plot_calibration_to_y(yg)
            if xg.size:
                ax.plot(
                    xg,
                    yg,
                    linewidth=2.2 if export else RCS_FIT_LINE_WIDTH,
                    color="#1565C0",
                    label=None,
                )
            self._plot_rcs_reference_limits(
                ax,
                show_labels=show_ref_labels,
                linewidth=1.8 if export else 1.3,
                alpha=0.95 if export else 0.72,
            )
            handles, labels = ax.get_legend_handles_labels()
            if show_legend and labels:
                ax.legend(loc="best", frameon=False)
            return

        if self._loaded_rcs_curves:
            compare_colors = [
                "#1565C0",
                "#C62828",
                "#2E7D32",
                "#6A1B9A",
                "#EF6C00",
                "#00838F",
                "#5D4037",
                "#283593",
            ]
            for index, curve in enumerate(self._loaded_rcs_curves):
                color = compare_colors[index % len(compare_colors)]
                xg, yg = self._clip_curve_by_distance(*curve.fitted)
                yg = self._apply_rcs_plot_calibration_to_y(yg)
                if xg.size and np.any(np.isfinite(yg)):
                    ax.plot(
                        xg,
                        yg,
                        color=color,
                        linewidth=2.2 if export else RCS_FIT_LINE_WIDTH,
                        label=curve.display_name if show_legend else None,
                    )

            self._plot_rcs_reference_limits(
                ax,
                show_labels=show_ref_labels,
                linewidth=1.9 if export else 1.3,
                alpha=0.95 if export else 0.72,
            )
            # 参考上下限曲线可能覆盖全量 x，绘制后再次强制直线测量的距离门
            ax.set_xlim(0.0, float(RCS_STRAIGHT_X_MAX_M))
            handles, labels = ax.get_legend_handles_labels()
            if show_legend and labels:
                ax.legend(loc="best", frameon=False)
            return

        segments = list(self.rcs_recorder.segments or [])
        run_lbl = list(self._rcs_segment_run_labels or [])
        if segments:
            # 直线测量：显示从 0 开始，但仍只使用 [RCS_STRAIGHT_X_MIN_M, RCS_STRAIGHT_X_MAX_M] 数据做绘制/融合/拟合
            ax.set_xlim(0.0, float(RCS_STRAIGHT_X_MAX_M))
            run_colors = [
                "#1565C0",
                "#F9A825",
                "#C62828",
                "#D81B60",
                "#6A1B9A",
                "#00838F",
                "#5D4037",
                "#283593",
            ]
            def _ema_smooth(y: np.ndarray, alpha: float) -> np.ndarray:
                y = np.asarray(y, dtype=float)
                if y.size <= 1:
                    return y.copy()
                a = float(alpha)
                if not np.isfinite(a) or a <= 0.0:
                    return y.copy()
                if a >= 1.0:
                    return y.copy()
                out = np.empty_like(y, dtype=float)
                out[0] = float(y[0])
                for i in range(1, int(y.size)):
                    out[i] = a * float(y[i]) + (1.0 - a) * float(out[i - 1])
                return out

            def _bin_points_by_x(points: List[CurvePoint]) -> Tuple[np.ndarray, np.ndarray]:
                # 0.1m 分箱；同一雷达周期内（|Δt| 小）多反射点：RCS 按功率线性叠加
                #   P_total = Σ 10^(RCS/10)，RCS_total = 10×log10(P_total)（与 RCS00+RCS01 一致）；
                # 跨时间样本仍用 dB 算术均值以平滑轨迹。
                if not points:
                    return np.asarray([], dtype=float), np.asarray([], dtype=float)
                xs = np.asarray([float(p.x) for p in points], dtype=float)
                ys = np.asarray([float(p.rcs_filt) for p in points], dtype=float)
                ts = np.asarray([float(p.t) for p in points], dtype=float)
                mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ts)
                mask &= (xs >= float(RCS_STRAIGHT_X_MIN_M)) & (xs <= float(RCS_STRAIGHT_X_MAX_M))
                xs = xs[mask]
                ys = ys[mask]
                ts = ts[mask]
                if xs.size == 0:
                    return np.asarray([], dtype=float), np.asarray([], dtype=float)

                # 剔除 RCS 异常跳变点：按 x 排序后，相邻点 |ΔRCS|>10 dBsm 的点直接丢弃
                order0 = np.argsort(xs)
                xs = xs[order0]
                ys = ys[order0]
                ts = ts[order0]
                if xs.size >= 2:
                    keep = np.ones(xs.size, dtype=bool)
                    last_idx = 0
                    for j in range(1, xs.size):
                        if not keep[last_idx]:
                            last_idx = j
                            continue
                        if abs(float(ys[j]) - float(ys[last_idx])) > 10.0:
                            keep[j] = False
                        else:
                            last_idx = j
                    xs = xs[keep]
                    ys = ys[keep]
                    ts = ts[keep]
                    if xs.size == 0:
                        return np.asarray([], dtype=float), np.asarray([], dtype=float)

                bin_m = 0.1
                bin_ids = np.floor(xs / float(bin_m)).astype(np.int64)
                uniq = np.unique(bin_ids)
                xb: List[float] = []
                yb: List[float] = []
                t_gate = float(RCS_BIN_INCOHERENT_SUM_MAX_TIME_SPAN_S)
                for bid in uniq:
                    m = bin_ids == bid
                    if not np.any(m):
                        continue
                    xb.append(float(np.mean(xs[m])))
                    ys_b = ys[m]
                    ts_b = ts[m]
                    if ys_b.size <= 1:
                        yb.append(float(ys_b[0]))
                    elif float(np.max(ts_b) - np.min(ts_b)) <= t_gate:
                        if combine_rcs_db_incoherent_sum is not None:
                            cr = combine_rcs_db_incoherent_sum(ys_b.tolist())
                            yb.append(float(cr) if cr is not None else float(np.mean(ys_b)))
                        else:
                            yb.append(
                                float(
                                    10.0
                                    * math.log10(float(np.sum(10.0 ** (ys_b.astype(float) / 10.0))))
                                )
                            )
                    else:
                        yb.append(float(np.mean(ys_b)))
                x_out = np.asarray(xb, dtype=float)
                y_out = np.asarray(yb, dtype=float)
                order = np.argsort(x_out)
                x_out = x_out[order]
                y_out = y_out[order]
                if _peak_smooth_rcs_series is not None:
                    y_out = np.asarray(_peak_smooth_rcs_series(y_out), dtype=float)
                else:
                    y_out = _ema_smooth(y_out, float(RCS_STRAIGHT_EMA_ALPHA))
                return x_out, y_out

            max_show_segments = 30
            segments_to_show = segments[:max_show_segments]
            segment_sizes: List[int] = []
            # 记录每次分箱后的序列，用于“跨次数异常值剔除”后再融合
            per_run_binned: List[Tuple[np.ndarray, np.ndarray]] = []
            for i, seg in enumerate(segments_to_show):
                xb, yb = _bin_points_by_x(seg)
                yb = self._apply_rcs_plot_calibration_to_y(yb)
                if xb.size == 0 or yb.size == 0:
                    segment_sizes.append(0)
                    per_run_binned.append((np.asarray([], dtype=float), np.asarray([], dtype=float)))
                    continue
                color = run_colors[i % len(run_colors)]
                segment_sizes.append(int(min(int(xb.size), int(yb.size))))
                per_run_binned.append((np.asarray(xb, dtype=float), np.asarray(yb, dtype=float)))
                run_no = (
                    int(run_lbl[i])
                    if i < len(run_lbl) and run_lbl[i] is not None
                    else (i + 1)
                )
                ax.plot(
                    xb,
                    yb,
                    color=color,
                    linewidth=1.6 if export else 1.2,
                    alpha=0.95 if export else 0.88,
                    label=(f"第{run_no}次测量" if (export or len(segments) > 1) else None),
                    zorder=2,
                )
            if live and getattr(self.rcs_recorder, "_cur", None):
                xb, yb = _bin_points_by_x(list(self.rcs_recorder._cur or []))
                yb = self._apply_rcs_plot_calibration_to_y(yb)
                if xb.size and yb.size:
                    color = run_colors[len(segments) % len(run_colors)]
                    next_no = (int(run_lbl[-1]) + 1) if run_lbl else (len(segments) + 1)
                    ax.plot(
                        xb,
                        yb,
                        color=color,
                        linewidth=1.7 if export else 1.25,
                        linestyle="--",
                        alpha=0.95 if export else 0.85,
                        label=f"第{next_no}次测量(进行中)",
                        zorder=3,
                    )
            # 融合曲线：按 0.1m 分箱后，再做“跨次数异常值剔除”
            # 规则：同一分箱上，若某次的 RCS 与该分箱的中位数差值 >10 dBsm，则不参与融合
            bin_m = 0.1
            thr_db = 10.0
            bin_to_xs: Dict[int, List[float]] = {}
            bin_to_ys: Dict[int, List[float]] = {}
            for xb, yb in per_run_binned:
                if xb.size == 0 or yb.size == 0:
                    continue
                n = min(int(xb.size), int(yb.size))
                xb2 = np.asarray(xb[:n], dtype=float)
                yb2 = np.asarray(yb[:n], dtype=float)
                m2 = np.isfinite(xb2) & np.isfinite(yb2)
                xb2 = xb2[m2]
                yb2 = yb2[m2]
                if xb2.size == 0:
                    continue
                bids = np.floor(xb2 / float(bin_m)).astype(np.int64)
                for k, xk, yk in zip(bids.tolist(), xb2.tolist(), yb2.tolist()):
                    bin_to_xs.setdefault(int(k), []).append(float(xk))
                    bin_to_ys.setdefault(int(k), []).append(float(yk))

            xb_all: List[float] = []
            yb_all: List[float] = []
            for bid in sorted(bin_to_ys.keys()):
                ys_list = bin_to_ys.get(int(bid), [])
                xs_list = bin_to_xs.get(int(bid), [])
                if not ys_list or not xs_list:
                    continue
                ys_arr = np.asarray(ys_list, dtype=float)
                xs_arr = np.asarray(xs_list, dtype=float)
                med = float(np.median(ys_arr))
                keep = np.abs(ys_arr - med) <= float(thr_db)
                if not np.any(keep):
                    continue
                xb_all.append(float(np.mean(xs_arr[keep])))
                yb_all.append(float(np.mean(ys_arr[keep])))
            xb_all = np.asarray(xb_all, dtype=float)
            yb_all = np.asarray(yb_all, dtype=float)
            if xb_all.size and np.any(np.isfinite(yb_all)):
                order_all = np.argsort(xb_all)
                xb_all = xb_all[order_all]
                yb_all = yb_all[order_all]
                if _peak_smooth_rcs_series is not None:
                    yb_all = np.asarray(_peak_smooth_rcs_series(yb_all), dtype=float)
                else:
                    yb_all = _ema_smooth(yb_all, float(RCS_STRAIGHT_EMA_ALPHA))
                ax.plot(
                    xb_all,
                    yb_all,
                    linewidth=2.8 if export else 2.2,
                    color="#2E7D32",
                    alpha=0.98 if export else 0.94,
                    label="融合曲线" if len(segments) > 1 else None,
                    zorder=4,
                )
            # 按需求：直线图不显示“次数/分箱点数”旁注，避免图片过于拥挤

            show_legend = bool(
                export or len(self._loaded_rcs_curves) > 1 or len(segments) > 1 or live
            )
            self._plot_rcs_reference_limits(
                ax,
                show_labels=show_ref_labels,
                linewidth=1.9 if export else 1.3,
                alpha=0.95 if export else 0.72,
            )
            handles, labels = ax.get_legend_handles_labels()
            if show_legend and labels:
                ax.legend(loc="best", frameon=False)
            return

        fitted_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if self._rcs_fitted is not None:
            fitted_xy = self._rcs_fitted
        elif self.rcs_recorder.point_count() > 0:
            grid = np.arange(
                0.0, RCS_MAX_DISTANCE_M + RCS_FIT_GRID_STEP_M * 0.5, RCS_FIT_GRID_STEP_M
            )
            fitted_xy = self.rcs_recorder.fitted_curve(grid)
        if fitted_xy is not None:
            xg, yg = self._clip_curve_by_distance(*fitted_xy)
            yg = self._apply_rcs_plot_calibration_to_y(yg)
            if xg.size and np.any(np.isfinite(yg)):
                ax.plot(
                    xg,
                    yg,
                    linewidth=2.2 if export else RCS_FIT_LINE_WIDTH,
                    color="#1565C0",
                    label=None,
                )

        self._plot_rcs_reference_limits(
            ax,
            show_labels=show_ref_labels,
            linewidth=1.9 if export else 1.3,
            alpha=0.95 if export else 0.72,
        )
        handles, labels = ax.get_legend_handles_labels()
        if show_legend and labels:
            ax.legend(loc="best", frameon=False)

    def _draw_rcs(self, live: bool = False) -> None:
        """刷新 RCS 曲线图"""
        plot_mode = self._get_rcs_plot_mode()
        self._ensure_rcs_axes(plot_mode)
        self._render_rcs_plot_on_axis(
            self.rcs_ax,
            plot_mode,
            live=live,
            compact=True,
            export=False,
        )
        self.rcs_canvas.draw_idle()
        return

        ax = self.rcs_ax
        ax.clear()

        if plot_mode == "orbit":
            series = self._get_rcs_orbit_plot_series()
            ax.set_title(f"{self._get_rcs_title()} | 圆周-RCS")
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_thetagrids(range(0, 360, 45))
            ax.grid(True, color="#d0d0d0", linestyle="-", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.set_rlabel_position(135)
            if series is None:
                ax.text(
                    0.5,
                    0.5,
                    "暂无可用圆周RCS数据\n需要锁定/标定目标并完成圆周采样",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                )
                self.rcs_canvas.draw_idle()
                return

            angles = np.asarray(series["angles"], dtype=float)
            radii = np.asarray(series["radii"], dtype=float)
            values = np.asarray(series["values"], dtype=float)
            tick_positions = np.asarray(series["tick_positions"], dtype=float)
            tick_values = np.asarray(series["tick_values"], dtype=float)

            if angles.size >= 2:
                ax.plot(angles, radii, color="#1565C0", linewidth=2.0, label="RCS环向曲线")
                ax.fill(angles, radii, color="#90CAF9", alpha=0.18)
            ax.scatter(
                angles,
                radii,
                c=values,
                cmap="viridis",
                s=20,
                alpha=0.9,
                edgecolors="none",
            )
            if tick_positions.size:
                ax.set_yticks(tick_positions)
                ax.set_yticklabels([f"{val:.1f}" for val in tick_values])
                ax.set_ylim(0.0, float(np.max(tick_positions)) * 1.08)
            span_deg = float(series.get("span_deg", 0.0))
            ax.text(
                0.02,
                1.04,
                f"角度覆盖={span_deg:.0f}°",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                color="#455A64",
            )
            self.rcs_canvas.draw_idle()
            return

        ax.set_title(self._get_rcs_title())
        ax.set_xlabel("Front X (m)")
        ax.set_ylabel("RCS (dBsm)")
        ax.set_axisbelow(True)
        ax.grid(False)
        ax.yaxis.grid(True, color="#e0e0e0", linestyle="-", linewidth=0.6)
        ax.set_xlim(0, RCS_MAX_DISTANCE_M)

        if self._rcs_show_only_fitted and self._rcs_fitted is not None:
            xg, yg = self._clip_curve_by_distance(*self._rcs_fitted)
            if xg.size:
                ax.plot(xg, yg, linewidth=RCS_FIT_LINE_WIDTH, label="拟合")
            self._plot_rcs_reference_limits(ax)
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(loc="best")
            self.rcs_canvas.draw_idle()
            return

        if self._loaded_rcs_curves:
            compare_colors = [
                "#1565C0",
                "#C62828",
                "#2E7D32",
                "#6A1B9A",
                "#EF6C00",
                "#00838F",
                "#5D4037",
                "#283593",
            ]
            for index, curve in enumerate(self._loaded_rcs_curves):
                color = compare_colors[index % len(compare_colors)]
                for seg in curve.segments[:10]:
                    xs = [p.x for p in seg]
                    ys = [p.rcs_filt for p in seg]
                    if xs and ys:
                        ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.16)

                xg, yg = self._clip_curve_by_distance(*curve.fitted)
                if xg.size and np.any(np.isfinite(yg)):
                    ax.plot(
                        xg,
                        yg,
                        color=color,
                        linewidth=RCS_FIT_LINE_WIDTH,
                        label=curve.display_name,
                    )
                    continue

                all_points = [point for seg in curve.segments for point in seg]
                xs = [point.x for point in all_points]
                ys = [point.rcs_filt for point in all_points]
                if xs and ys:
                    ax.plot(xs, ys, color=color, linewidth=1.8, label=curve.display_name)

            self._plot_rcs_reference_limits(ax)
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(loc="best")
            self.rcs_canvas.draw_idle()
            return

        # 多次测量：每次曲线不同颜色，最终拟合曲线单独标注
        run_colors = [
            "#1565C0",  # 蓝
            "#F9A825",  # 黄
            "#C62828",  # 红
            "#2E7D32",  # 绿
            "#6A1B9A",  # 紫
            "#00838F",  # 青
            "#5D4037",  # 棕
            "#283593",  # 靛
        ]
        # 尽量叠加展示所有“第N次”数据；过多时也要保持可读性
        segments = list(self.rcs_recorder.segments or [])
        max_show_segments = 30
        segments_to_show = segments[:max_show_segments]
        segment_sizes: List[int] = []
        for i, seg in enumerate(segments_to_show):
            xs = [p.x for p in seg]
            ys = [p.rcs_filt for p in seg]
            if not xs or not ys:
                segment_sizes.append(0)
                continue
            color = run_colors[i % len(run_colors)]
            segment_sizes.append(min(len(xs), len(ys)))
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=1.15 if export else 0.95,
                alpha=0.78 if export else 0.62,
                label=f"第{i+1}次测量",
            )
        if live and getattr(self.rcs_recorder, "_cur", None):
            xs = [p.x for p in self.rcs_recorder._cur]
            ys = [p.rcs_filt for p in self.rcs_recorder._cur]
            if xs and ys:
                color = run_colors[len(self.rcs_recorder.segments) % len(run_colors)]
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linewidth=1.25 if export else 1.0,
                    linestyle="--",
                    alpha=0.90 if export else 0.75,
                    label=f"第{len(self.rcs_recorder.segments)+1}次测量(进行中)",
                )
        if self._rcs_fitted is not None:
            xg, yg = self._clip_curve_by_distance(*self._rcs_fitted)
            if xg.size:
                ax.plot(
                    xg,
                    yg,
                    linewidth=2.4 if export else 2.0,
                    color="#111827",
                    alpha=0.96 if export else 0.92,
                    label="融合后拟合直线",
                )

        # 旁注：第N次 + 融合后（点数与拟合方程）
        try:
            total_segments = len(segments)
            fused_points = sum(len(s) for s in segments if s)
            lines: List[str] = []
            show_count = len(segments_to_show)
            if total_segments > 0:
                lines.append(f"直线测量次数: {total_segments} 次")
                for i in range(show_count):
                    npt = int(segment_sizes[i]) if i < len(segment_sizes) else 0
                    lines.append(f"第{i+1}次: {npt} 点")
                if total_segments > show_count:
                    lines.append(f"... 其余 {total_segments - show_count} 次未逐条列出")
            if fused_points > 0:
                lines.append(f"融合后: {int(fused_points)} 点")
            if self._rcs_fitted is not None:
                xg, yg = self._clip_curve_by_distance(*self._rcs_fitted)
                if xg.size >= 2:
                    dx = float(xg[-1] - xg[0])
                    if abs(dx) > 1e-9:
                        k = float((yg[-1] - yg[0]) / dx)
                        b = float(yg[0] - k * xg[0])
                        lines.append(f"拟合: y = {k:.3f}x + {b:.3f}")
            if lines:
                ax.text(
                    0.02,
                    0.98,
                    "\n".join(lines),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9 if export else 8.5,
                    color="#37474F",
                    bbox=dict(
                        boxstyle="round,pad=0.35",
                        facecolor="white",
                        edgecolor="#CFD8DC",
                        linewidth=0.8,
                        alpha=0.78 if export else 0.70,
                    ),
                    zorder=10,
                )
        except Exception:
            # 旁注失败不影响主图
            pass

        self._plot_rcs_reference_limits(ax)

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="best", frameon=False)
        self.rcs_canvas.draw_idle()

    def _get_rcs_title(self) -> str:
        if self._loaded_rcs_curves:
            base = (self._rcs_target_name or f"RCS对比({len(self._loaded_rcs_curves)}组)").strip()
            if self._rcs_ref_angle:
                base = f"{base} {self._rcs_ref_angle}"
            return base or "RCS对比"

        base = (self._rcs_target_name or "RCS").strip() or "RCS"
        if self._rcs_ref_angle:
            base = f"{base} {self._rcs_ref_angle}"
        title = base
        if self.rcs_recorder.oid is not None:
            title += f" (ID={self.rcs_recorder.oid})"
        return title

    def _on_rcs_start(
        self,
        segment_index: Optional[int] = None,
        trajectory_name: Optional[str] = None,
    ) -> bool:
        """开始 Cluster(0x701) RCS CSV 录制。"""
        self._orbit_rcs_active = False
        self._orbit_rcs_rows.clear()
        self._orbit_polar_cached_series = None
        save_dir = str(self._resolve_rcs_save_dir(trajectory_name or self._get_current_path_name()))
        if self._radial_measurement_spec is not None:
            stem = self._radial_rcs_segment_file_stem(trajectory_name)
            explicit_filename = f"{stem}.csv"
        else:
            traj_tok = self._safe_filename_token(
                str(trajectory_name or self._get_current_path_name()).strip() or "轨迹"
            )
            seg_part = (
                f"_seg{int(segment_index):02d}"
                if segment_index is not None
                else "_segXX"
            )
            stem = f"{traj_tok}_cluster{seg_part}"
            explicit_filename = None
        if not self.controller.begin_cluster_rcs_capture(
            save_dir,
            stem,
            filename=explicit_filename,
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Cluster RCS",
                "无法启动 CSV：Cluster 接收未就绪或非 Linux CAN（需 can0 Cluster 模式）。",
            )
            self._log("Cluster RCS CSV 启动失败（cluster_csv_runtime 不可用）")
            return False
        self._straight_rcs_collect_all = self._segment_index_is_forward_straight_segment(segment_index)
        self._straight_rcs_last_by_oid.clear()
        self._straight_rcs_rcs_ema_by_oid.clear()
        if not self._straight_rcs_collect_all:
            self._straight_rcs_runs = []
            self._rcs_segment_run_labels = None
        self.rcs_recorder.reset()
        self.rcs_lock.disarm()
        self._rcs_recording = True
        self._rcs_fitted = None
        self._rcs_show_only_fitted = False
        self._rcs_orbit_samples = []
        self._loaded_rcs_curves = []
        self._rcs_curve_csv_source_paths = []
        self._rcs_active_segment_index = segment_index
        self._rcs_active_path_name = str(trajectory_name or self._get_current_path_name()).strip() or "轨迹"
        self._rcs_active_target_name = self._get_live_rcs_target_name()
        self._rcs_snapshot_file_target_name = (
            str(self._rcs_active_target_name or self._get_selected_rcs_target_name()).strip()
            or "未命名目标"
        )
        self._rcs_relock_events = []
        segment_text = f" | 分段={segment_index}" if segment_index is not None else ""
        self.rcs_status_label.setText(f"Cluster RCS CSV: 进行中 | 0x701{segment_text}")
        self._log(
            f"【Cluster 0x701】RCS CSV 采集已开始 | "
            f"轨迹={self._rcs_active_path_name}{segment_text} | 目录={save_dir}"
        )
        self._draw_rcs()
        return True

    def _on_orbit_rcs_start(
        self,
        segment_index: Optional[int] = None,
        trajectory_name: Optional[str] = None,
    ) -> None:
        """
        圆周段：落盘主文件与直线段相同，均为 DRI Raw 风格 Cluster CSV
        （Data Type/Run Number/Calibration + Time,R,ViewAng,…,DX00..RCS19）；
        仅横向 ROI 放宽（orbit_roi）；圆周段不再单独落盘极坐标 CSV，仅 Cluster Raw；极坐标由 Raw 重放或实时缓存绘制。
        """
        save_dir = str(self._resolve_rcs_save_dir(trajectory_name or self._get_current_path_name()))
        traj_tok = self._safe_filename_token(
            str(trajectory_name or self._get_current_path_name()).strip() or "轨迹"
        )
        seg_part = (
            f"_seg{int(segment_index):02d}"
            if segment_index is not None
            else "_segXX"
        )
        stem = f"{traj_tok}_orbit_cluster{seg_part}"
        if not self.controller.begin_cluster_rcs_capture(
            save_dir, stem, orbit_roi=True
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Cluster RCS",
                "无法启动圆周段 CSV：Cluster 接收未就绪或非 Linux CAN。",
            )
            self._log("圆周 Cluster RCS CSV 启动失败")
            return
        self.rcs_recorder.reset()
        self.rcs_recorder._ended = False
        self._rcs_segment_run_labels = None
        self.rcs_lock.disarm()
        self._orbit_rcs_active = True
        self._orbit_rcs_rows.clear()
        self._orbit_polar_cached_series = None
        self._rcs_recording = True
        self._rcs_fitted = None
        self._rcs_show_only_fitted = False
        self._rcs_orbit_samples = []
        self._loaded_rcs_curves = []
        self._rcs_curve_csv_source_paths = []
        self._rcs_active_segment_index = segment_index
        self._rcs_active_path_name = str(trajectory_name or self._get_current_path_name()).strip() or "轨迹"
        self._rcs_active_target_name = self._get_live_rcs_target_name()
        self._rcs_snapshot_file_target_name = (
            str(self._rcs_active_target_name or self._get_selected_rcs_target_name()).strip()
            or "未命名目标"
        )
        self._rcs_relock_events = []
        segment_text = f" | 分段={segment_index}" if segment_index is not None else ""
        self.rcs_status_label.setText(f"圆周 Cluster RCS CSV: 进行中{segment_text}")
        self._log(
            f"【圆周 Cluster 0x701】RCS CSV 已开始 | 轨迹={self._rcs_active_path_name}{segment_text}"
        )
        self._select_rcs_plot_mode("orbit")
        self._draw_rcs()

    def _finalize_orbit_rcs_recording(
        self,
        *,
        save_raw: bool,
        save_fit_image: bool,
        trajectory_name: Optional[str],
        segment_index: Optional[int],
        cluster_raw_csv_path: Optional[str] = None,
    ) -> Tuple[int, Optional[str], Optional[str]]:
        # save_raw / save_fit_image：圆周段不再另存极坐标 CSV；Cluster Raw 已在 _finalize_cluster_rcs_csv_only 落盘。
        self._orbit_rcs_active = False
        self._rcs_recording = False
        rows = list(self._orbit_rcs_rows)
        self._orbit_rcs_rows.clear()
        self.rcs_lock.disarm()
        self.rcs_recorder.reset()
        self.rcs_recorder._cur = []
        self.rcs_recorder._ended = True
        self.rcs_recorder.oid = None

        resolved_segment_index = (
            segment_index if segment_index is not None else self._rcs_active_segment_index
        )
        resolved_trajectory_name = (
            str(trajectory_name or self._rcs_active_path_name or self._get_current_path_name()).strip()
            or "轨迹"
        )
        resolved_target_name = (
            str(
                self._rcs_snapshot_file_target_name
                or self._rcs_active_target_name
                or self._get_selected_rcs_target_name()
            ).strip()
            or "未命名目标"
        )

        if len(rows) < 2:
            self._orbit_polar_cached_series = None
            self._rcs_active_segment_index = None
            self._rcs_active_path_name = None
            self._rcs_active_target_name = None
            self._rcs_snapshot_file_target_name = None
            self._rcs_relock_events = []
            extra = f" | Cluster Raw: {cluster_raw_csv_path}" if cluster_raw_csv_path else ""
            self.rcs_status_label.setText("圆周RCS: 已结束 | 预览点数不足" + extra)
            self._log(
                f"圆周RCS结束: 轨迹={resolved_trajectory_name} | 目标={resolved_target_name} | 点数不足，无极坐标预览{extra}"
            )
            self._draw_rcs()
            return 0, cluster_raw_csv_path, None

        self._orbit_polar_cached_series = self._build_orbit_polar_series_from_rows(rows)
        self._select_rcs_plot_mode("orbit")
        extra = f" | Cluster Raw: {cluster_raw_csv_path}" if cluster_raw_csv_path else ""
        self.rcs_status_label.setText(
            f"圆周RCS: 已结束 | 预览点数={len(rows)} | 角度分箱均值曲线（仅 Raw 落盘）{extra}"
        )
        log_parts = [
            f"圆周RCS结束: 轨迹={resolved_trajectory_name}",
            f"目标={resolved_target_name}",
            f"预览点数={len(rows)}",
            f"极坐标=时间平铺到2π后按{ORBIT_RCS_ANGLE_BIN_DEG:g}°分箱算术平均",
        ]
        if resolved_segment_index is not None:
            log_parts.append(f"分段={resolved_segment_index}")
        if cluster_raw_csv_path:
            log_parts.append(f"ClusterRaw={cluster_raw_csv_path}")
        if self._rcs_relock_events:
            log_parts.append(f"自动重锁={len(self._rcs_relock_events)}次")
        self._log(" | ".join(log_parts))

        self._rcs_active_segment_index = None
        self._rcs_active_path_name = None
        self._rcs_active_target_name = None
        self._rcs_snapshot_file_target_name = None
        self._rcs_relock_events = []
        self._draw_rcs()
        return len(rows), cluster_raw_csv_path, None

    def _finalize_cluster_rcs_csv_only(
        self,
        *,
        save_raw: bool,
        trajectory_name: Optional[str],
        segment_index: Optional[int],
        is_orbit: bool,
    ) -> Tuple[int, Optional[str], Optional[str]]:
        """结束 Cluster(0x701) CSV 采集。"""
        rt = getattr(self.controller, "cluster_csv_runtime", None)
        if rt is None or not rt.is_recording():
            self._rcs_recording = False
            self._orbit_rcs_active = False
            return 0, None, None
        n_frames, n_cl = rt.snapshot_stats()
        csv_path = self.controller.end_cluster_rcs_capture()
        self._rcs_recording = False
        self._orbit_rcs_active = False
        # 圆周段极坐标预览来自内存/Cluster Raw 重放；下次启动直线/圆周采集时在各自入口清空
        self.rcs_recorder.reset()
        self.rcs_recorder._cur = []
        self.rcs_recorder._ended = True
        self.rcs_recorder.oid = None
        self.rcs_lock.disarm()
        self._rcs_fitted = None
        self._loaded_rcs_curves = []
        traj = (
            str(trajectory_name or self._rcs_active_path_name or self._get_current_path_name()).strip()
            or "轨迹"
        )
        seg_txt = f" | 分段={segment_index}" if segment_index is not None else ""
        self.rcs_status_label.setText(
            f"Cluster RCS CSV: 已结束 | 帧≈{n_frames} 簇≈{n_cl}"
            + (f" | {csv_path}" if csv_path else "")
        )
        self._log(
            " | ".join(
                [
                    f"Cluster(0x701) RCS CSV 采集结束: 轨迹={traj}{seg_txt}",
                    f"帧≈{n_frames}",
                    f"簇计数≈{n_cl}",
                    (f"文件={csv_path}" if csv_path else "未生成文件"),
                ]
            )
        )
        self._rcs_active_segment_index = None
        self._rcs_active_path_name = None
        self._rcs_active_target_name = None
        self._rcs_snapshot_file_target_name = None
        self._rcs_relock_events = []
        self._draw_rcs()
        return int(n_frames), csv_path if save_raw else None, None

    def _finalize_rcs_recording(
        self,
        *,
        save_raw: bool = False,
        save_fit_image: bool = False,
        trajectory_name: Optional[str] = None,
        segment_index: Optional[int] = None,
    ) -> Tuple[int, Optional[str], Optional[str]]:
        rt_fin = getattr(self.controller, "cluster_csv_runtime", None)
        if rt_fin is not None and rt_fin.is_recording():
            was_orbit = bool(self._orbit_rcs_active)
            n_frames, csv_path, fit_csv = self._finalize_cluster_rcs_csv_only(
                save_raw=save_raw,
                trajectory_name=trajectory_name,
                segment_index=segment_index,
                is_orbit=was_orbit,
            )
            if was_orbit:
                return self._finalize_orbit_rcs_recording(
                    save_raw=save_raw,
                    save_fit_image=save_fit_image,
                    trajectory_name=trajectory_name,
                    segment_index=segment_index,
                    cluster_raw_csv_path=csv_path,
                )
            return n_frames, csv_path, fit_csv
        if self._orbit_rcs_active:
            return self._finalize_orbit_rcs_recording(
                save_raw=save_raw,
                save_fit_image=save_fit_image,
                trajectory_name=trajectory_name,
                segment_index=segment_index,
                cluster_raw_csv_path=None,
            )
        if self.rcs_recorder.oid is None and self.rcs_recorder.point_count() <= 0:
            self._rcs_recording = False
            self._rcs_active_segment_index = None
            self._rcs_active_path_name = None
            self._rcs_active_target_name = None
            self._rcs_snapshot_file_target_name = None
            return 0, None, None

        self.rcs_recorder.finalize()
        self._rcs_recording = False

        point_count = self.rcs_recorder.point_count()
        finalized_segments = [list(seg) for seg in self.rcs_recorder.segments if seg]
        grid = np.arange(0.0, RCS_MAX_DISTANCE_M + RCS_FIT_GRID_STEP_M * 0.5, RCS_FIT_GRID_STEP_M)
        self._rcs_fitted = self.rcs_recorder.fitted_curve(grid)

        resolved_segment_index = (
            segment_index if segment_index is not None else self._rcs_active_segment_index
        )
        resolved_trajectory_name = (
            str(trajectory_name or self._rcs_active_path_name or self._get_current_path_name()).strip()
            or "轨迹"
        )
        resolved_target_name = (
            str(
                self._rcs_snapshot_file_target_name
                or self._rcs_active_target_name
                or self._get_selected_rcs_target_name()
            ).strip()
            or "未命名目标"
        )
        clip_x0 = (
            float(RCS_STRAIGHT_X_MIN_M)
            if self._segment_index_is_forward_straight_segment(resolved_segment_index)
            else 0.0
        )
        clip_x1 = (
            float(RCS_STRAIGHT_X_MAX_M)
            if self._segment_index_is_forward_straight_segment(resolved_segment_index)
            else float(RCS_MAX_DISTANCE_M)
        )
        self._rcs_fitted = self._clip_curve_by_distance(*self._rcs_fitted, x_min=clip_x0, x_max=clip_x1)

        # 多次往返：前进直线段（speed_sign>0 且近似直线）统一写到同一聚合文件里，不按目标ID拆分
        if self._segment_index_is_forward_straight_segment(resolved_segment_index):
            resolved_target_name = FORWARD_STRAIGHT_RCS_TARGET_NAME
            # 直线测量：把每一次 finalize 出来的段作为“第N次”叠加缓存
            if finalized_segments:
                self._straight_rcs_runs.append(list(finalized_segments[0]))
                if len(self._straight_rcs_runs) > int(self._straight_rcs_max_runs):
                    self._straight_rcs_runs = self._straight_rcs_runs[-int(self._straight_rcs_max_runs) :]
            # 用 UI 缓存的多次数据回填 recorder，用于绘图“第N次”
            if self._straight_rcs_runs:
                self.rcs_recorder.segments = [list(seg) for seg in self._straight_rcs_runs if seg]
                self.rcs_recorder._cur = []
                self.rcs_recorder._ended = True
                self._rcs_segment_run_labels = list(range(1, len(self.rcs_recorder.segments) + 1))
            # 同一目标/同一文件混合多个 ID：拟合时也一起参与，且输出一条直线（融合后 1 次拟合）
            all_points: List[CurvePoint] = []
            for seg in (self._straight_rcs_runs or []):
                all_points.extend(seg)
            lf = self._linear_fit_from_points(all_points)
            if lf is not None:
                self._rcs_fitted = self._clip_curve_by_distance(
                    *lf,
                    x_min=float(RCS_STRAIGHT_X_MIN_M),
                    x_max=float(RCS_STRAIGHT_X_MAX_M),
                )

        raw_path = None
        fit_csv_path: Optional[str] = None
        if save_raw and point_count > 0:
            if resolved_segment_index is not None and finalized_segments:
                raw_path = self._save_rcs_trajectory_snapshot(
                    trajectory_name=resolved_trajectory_name,
                    target_name=resolved_target_name,
                    segment_points=finalized_segments[0],
                )
            else:
                raw_path = self._save_rcs_raw_snapshot(
                    trajectory_name=resolved_trajectory_name,
                    target_name=resolved_target_name,
                    segment_index=resolved_segment_index,
                )
            if raw_path:
                cand = Path(raw_path).with_name(f"{Path(raw_path).stem}_fitted.csv")
                if cand.is_file():
                    fit_csv_path = str(cand)

        point_count_report = int(point_count)
        if raw_path:
            rp = Path(raw_path)
            if rp.is_file():
                try:
                    merged_segs, merged_run_labels = self._parse_saved_rcs_raw_file(str(rp))
                    if merged_segs:
                        merged_rec = RcsRunRecorder()
                        merged_rec.segments = [list(s) for s in merged_segs]
                        merged_rec._cur = []
                        merged_rec._ended = True
                        self._rcs_fitted = self._clip_curve_by_distance(
                            *merged_rec.fitted_curve(grid),
                            x_min=clip_x0,
                            x_max=clip_x1,
                        )
                        # 注意：前进直线段希望保留“第N次”叠加显示，不用保存文件解析结果覆盖 UI 缓存
                        if not self._segment_index_is_forward_straight_segment(resolved_segment_index):
                            self.rcs_recorder.reset()
                            self.rcs_recorder.segments = [list(s) for s in merged_segs]
                            self.rcs_recorder._cur = []
                            self.rcs_recorder._ended = True
                            self._rcs_segment_run_labels = merged_run_labels
                        elif len(merged_run_labels) == len(self.rcs_recorder.segments):
                            # 与落盘 CSV 的 SegIdx 分段对齐（按升序映射为 1..N）
                            self._rcs_segment_run_labels = merged_run_labels
                        point_count_report = sum(len(s) for s in merged_segs)
                except Exception:
                    pass

        fit_path = self._save_rcs_fit_image() if save_fit_image else None

        status_parts = [f"RCS录制: 已结束 | 点数={point_count_report}"]
        if raw_path:
            status_parts.append(f"原始数据已保存 {raw_path}")
        if fit_csv_path:
            status_parts.append(f"拟合曲线数据已保存 {fit_csv_path}")
        if fit_path:
            status_parts.append(f"拟合曲线已保存 {fit_path}")
        self.rcs_status_label.setText(" | ".join(status_parts))

        log_parts = [
            f"RCS录制结束: 轨迹={resolved_trajectory_name}",
            f"目标={resolved_target_name}",
            f"点数={point_count_report}",
        ]
        if resolved_segment_index is not None:
            log_parts.append(f"分段={resolved_segment_index}")
        if raw_path:
            log_parts.append(f"原始数据={raw_path}")
        if fit_csv_path:
            log_parts.append(f"拟合数据={fit_csv_path}")
        if fit_path:
            log_parts.append(f"拟合曲线={fit_path}")
        if self._rcs_relock_events:
            log_parts.append(f"自动重锁={len(self._rcs_relock_events)}次")
        self._log(" | ".join(log_parts))

        self._rcs_active_segment_index = None
        self._rcs_active_path_name = None
        self._rcs_active_target_name = None
        self._rcs_snapshot_file_target_name = None
        self._rcs_relock_events = []
        self._draw_rcs()
        return point_count_report, raw_path, fit_path

    def _has_rcs_plot_content(self) -> bool:
        if self._get_rcs_plot_mode() == "orbit":
            return self._get_rcs_orbit_plot_series() is not None
        if self._loaded_rcs_curves:
            return True
        if self._rcs_fitted is not None:
            xg, yg = self._clip_curve_by_distance(*self._rcs_fitted)
            if xg.size and np.any(np.isfinite(yg)):
                return True
        return self.rcs_recorder.point_count() > 0

    def _default_rcs_image_filename(self) -> str:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if self._get_rcs_plot_mode() == "orbit":
            label = self._safe_filename_token(self._get_selected_rcs_target_name())
            return f"rcs_orbit_{label}_{stamp}.png"
        if self._loaded_rcs_curves:
            if len(self._loaded_rcs_curves) == 1:
                label = self._safe_filename_token(self._loaded_rcs_curves[0].display_name)
                return f"rcs_plot_{label}_{stamp}.png"
            return f"rcs_compare_{len(self._loaded_rcs_curves)}curves_{stamp}.png"
        if self._rcs_fitted is not None and self.rcs_recorder.oid is not None:
            return f"rcs_fit_id{self.rcs_recorder.oid}_{stamp}.png"
        label = self._safe_filename_token(self._get_selected_rcs_target_name())
        return f"rcs_plot_{label}_{stamp}.png"

    def _current_rcs_curve_csv_source_paths(self) -> List[str]:
        """当前距离-RCS图对应的 Raw CSV；保存图片时同步生成 Filtered/combined。"""
        if self._get_rcs_plot_mode() != "distance":
            return []
        candidates = list(getattr(self, "_rcs_curve_csv_source_paths", []) or [])
        if not candidates:
            if self._loaded_rcs_curves:
                candidates = [str(curve.file_path) for curve in self._loaded_rcs_curves]
            elif self._rcs_base_file_path:
                candidates = [str(self._rcs_base_file_path)]

        out: List[str] = []
        seen: Set[str] = set()
        for raw in candidates:
            path = Path(str(raw))
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            if not path.is_file() or path.suffix.lower() != ".csv":
                continue
            if MainWindow._is_orbit_cluster_raw_csv(path):
                continue
            if not MainWindow._cluster_csv_has_dri_cluster_header(path):
                continue
            out.append(str(path))
        return out

    def _on_rcs_save_image(self) -> None:
        if not self._has_rcs_plot_content():
            QtWidgets.QMessageBox.information(self, "无图像", "当前没有可保存的RCS图像。")
            self._log("保存RCS图片失败: 当前没有可导出的图像内容")
            return

        default_path = str(Path(self._rcs_save_dir) / self._default_rcs_image_filename())
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存RCS图片",
            default_path,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;所有文件(*)",
        )
        if not path:
            return

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        fmt = "jpeg" if suffix in {".jpg", ".jpeg"} else "png"
        if not suffix:
            output_path = output_path.with_suffix(".png")
            fmt = "png"

        export_fig = self._create_rcs_export_figure()
        export_fig.savefig(
            str(output_path),
            format=fmt,
            dpi=RCS_EXPORT_DPI,
            bbox_inches="tight",
            pad_inches=0.20,
            facecolor="white",
        )
        export_fig.clf()
        csv_note = ""
        csv_sources = self._current_rcs_curve_csv_source_paths()
        if csv_sources:
            _filtered_paths, combined_path = self._try_export_rcs_curve_csvs(
                csv_sources,
                self._rcs_plot_calibration_db,
            )
            if combined_path is not None:
                csv_note = f" | CSV={combined_path}"
        self.rcs_status_label.setText(f"RCS图片已保存 {output_path}{csv_note}")
        self._log(f"RCS图片已保存: {output_path}{csv_note}")

    def _save_rcs_fit_image(self) -> Optional[str]:
        if self._rcs_fitted is None or self.rcs_recorder.oid is None:
            return None
        xg, yg = self._clip_curve_by_distance(*self._rcs_fitted)
        yg = self._apply_rcs_plot_calibration_to_y(yg)
        valid = np.isfinite(yg)
        if xg.size == 0 or not np.any(valid):
            return None
        stamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"rcs_fit_id{self.rcs_recorder.oid}_{stamp}.png"
        path = Path(self._rcs_save_dir) / filename

        # 拟合图同样导出为 1:1（正方形）
        fig = Figure(figsize=(7.8, 7.8), dpi=140, facecolor="white")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#FCFCFD")
        ax.set_title(f"RCS Fit (ID={self.rcs_recorder.oid})", fontsize=14, fontweight="semibold")
        ax.set_xlabel("Front Distance (m)")
        ax.set_ylabel("RCS (dBsm)")
        ax.grid(False)
        ax.yaxis.grid(True, linestyle="-", linewidth=0.8, color="#DCE3E8")
        ax.xaxis.grid(True, linestyle="--", linewidth=0.65, color="#EEF2F5")
        ax.set_xlim(0, RCS_MAX_DISTANCE_M)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.plot(xg, yg, linewidth=2.2, color="#1565C0", label=None)
        self._plot_rcs_reference_limits(ax, linewidth=1.9, alpha=0.95)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(
            str(path),
            format="png",
            dpi=RCS_EXPORT_DPI,
            bbox_inches="tight",
            pad_inches=0.20,
            facecolor="white",
        )
        fig.clf()
        return str(path)

    def _convert_target_to_objmeas(self, t, now_ts: float) -> Optional[ObjMeas]:
        """把雷达目标对象转换为 ObjMeas，缺失字段使用默认值。"""
        try:
            oid = int(getattr(t, "cid", getattr(t, "id", getattr(t, "oid", 0))))
            x = float(getattr(t, "x", 0.0))
            y = float(getattr(t, "y", 0.0))
            vx = float(getattr(t, "vx", getattr(t, "vx_rel", 0.0)))
            vy = float(getattr(t, "vy", getattr(t, "vy_rel", 0.0)))
            dyn = int(getattr(t, "dyn", 0))
            rcs = getattr(t, "rcs_db", None)
            if rcs is None:
                rcs = getattr(t, "rcs_kalman", getattr(t, "rcs", 0.0))
            meas_t = getattr(t, "t", now_ts)
            try:
                meas_t = float(meas_t)
            except (TypeError, ValueError):
                meas_t = float(now_ts)
            xr = getattr(t, "x_raw", float("nan"))
            yr = getattr(t, "y_raw", float("nan"))
            try:
                xr = float(xr)
                yr = float(yr)
            except (TypeError, ValueError):
                xr = float("nan")
                yr = float("nan")
            if not (math.isfinite(xr) and math.isfinite(yr)):
                xr = float(x)
                yr = float(y)
            return ObjMeas(
                oid=oid,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                dyn=dyn,
                rcs_db=float(rcs),
                t=float(meas_t),
                x_raw=float(xr),
                y_raw=float(yr),
                rcs_kf_db=float(getattr(t, "rcs_kf_db", float("nan"))),
            )
        except Exception:
            return None

    def _find_radar_target_object_by_oid(self, targets: List[Any], oid: int) -> Optional[Any]:
        for t in targets:
            if self._extract_radar_target_oid(t) == int(oid):
                return t
        return None

    def _orbit_rcs_effective_forward_gate_m(self) -> float:
        """圆周 RCS 尚无落盘点时放宽前向距离半宽，便于首次锁定；有采样后恢复窄门。"""
        base = float(ORBIT_RCS_FORWARD_GATE_M)
        if not self._orbit_rcs_active:
            return base
        if len(self._orbit_rcs_rows) > 0:
            return base
        return float(max(base, float(ORBIT_RCS_FORWARD_GATE_RELAXED_M)))

    def _pick_meas_in_forward_gate(
        self,
        candidates: List[ObjMeas],
        prefer_oid: Optional[int],
        gate_m: float,
    ) -> Optional[ObjMeas]:
        nom = float(ORBIT_RCS_NOMINAL_FORWARD_M)
        gate = float(gate_m)
        in_gate = [m for m in candidates if abs(float(m.x) - nom) <= gate]
        if not in_gate:
            return None
        if prefer_oid is not None:
            for m in in_gate:
                if int(m.oid) == int(prefer_oid):
                    return m
        return min(in_gate, key=lambda mm: abs(float(mm.x) - nom))

    def _linear_fit_from_points(self, points: List[CurvePoint]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not points or len(points) < 2:
            return None
        xs = np.asarray([float(p.x) for p in points], dtype=float)
        ys = np.asarray([float(p.rcs_filt) for p in points], dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        mask &= (xs >= float(RCS_STRAIGHT_X_MIN_M)) & (xs <= float(RCS_STRAIGHT_X_MAX_M))
        xs = xs[mask]
        ys = ys[mask]
        if xs.size < 2:
            return None
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        if float(np.max(xs) - np.min(xs)) <= 1e-6:
            return np.asarray([float(xs[0])], dtype=float), np.asarray([float(np.mean(ys))], dtype=float)
        coef = np.polyfit(xs, ys, 1)
        x_fit = np.linspace(float(np.min(xs)), float(np.max(xs)), max(2, int(np.ceil((float(np.max(xs))-float(np.min(xs))) / RCS_FIT_GRID_STEP_M)) + 1))
        y_fit = np.polyval(coef, x_fit).astype(float)
        return x_fit.astype(float), y_fit

    def _merge_cluster_pair_for_orbit_rcs(
        self,
        m_use: ObjMeas,
        gate_meas: List[ObjMeas],
        raw_clusters: Optional[List[Any]],
        nom: float,
        gate: float,
    ) -> ObjMeas:
        """
        圆周采样写入：优先对 CAN 帧内前两簇（对应 RCS00/RCS01 顺序）做功率叠加；
        若原始簇不可用则退化为 gate_meas 中 oid 0 与 1。
        """
        if raw_clusters is not None and len(raw_clusters) >= 2:
            c0 = raw_clusters[0]
            c1 = raw_clusters[1]
            try:
                dx0 = float(c0["DX"])
                dx1 = float(c1["DX"])
                dy0 = float(c0["DY"])
                dy1 = float(c1["DY"])
                r0 = float(c0["RCS"])
                r1 = float(c1["RCS"])
            except (KeyError, TypeError, ValueError):
                r0 = float("nan")
            else:
                if (
                    math.isfinite(r0)
                    and math.isfinite(r1)
                    and abs(dx0 - nom) <= gate
                    and abs(dx1 - nom) <= gate
                ):
                    if combine_rcs_db_incoherent_sum is None:
                        cr = float(
                            10.0
                            * math.log10(
                                10.0 ** (r0 / 10.0) + 10.0 ** (r1 / 10.0)
                            )
                        )
                    else:
                        merged = combine_rcs_db_incoherent_sum([r0, r1])
                        if merged is None:
                            return self._merge_cluster_0_1_for_orbit_rcs_legacy(
                                m_use, gate_meas
                            )
                        cr = float(merged)
                    x_avg = (dx0 + dx1) * 0.5
                    y_avg = (dy0 + dy1) * 0.5
                    return replace(
                        m_use,
                        rcs_db=cr,
                        x=x_avg,
                        y=y_avg,
                        x_raw=x_avg,
                        y_raw=y_avg,
                        rcs_kf_db=float("nan"),
                    )

        return self._merge_cluster_0_1_for_orbit_rcs_legacy(m_use, gate_meas)

    def _merge_cluster_0_1_for_orbit_rcs_legacy(
        self, m_use: ObjMeas, gate_meas: List[ObjMeas]
    ) -> ObjMeas:
        """兼容：关联列表中 oid 为 0、1 的两点（旧帧内序号）。"""
        by_oid = {int(m.oid): m for m in gate_meas}
        if 0 not in by_oid or 1 not in by_oid:
            return m_use
        m0, m1 = by_oid[0], by_oid[1]
        if combine_rcs_db_incoherent_sum is None:
            cr = float(
                10.0
                * math.log10(
                    10.0 ** (float(m0.rcs_db) / 10.0)
                    + 10.0 ** (float(m1.rcs_db) / 10.0)
                )
            )
        else:
            merged = combine_rcs_db_incoherent_sum([m0.rcs_db, m1.rcs_db])
            if merged is None:
                return m_use
            cr = float(merged)
        x_avg = (float(m0.x) + float(m1.x)) * 0.5
        y_avg = (float(m0.y) + float(m1.y)) * 0.5
        xr0, yr0 = m0.xy_raw()
        xr1, yr1 = m1.xy_raw()
        xr = (xr0 + xr1) * 0.5
        yr = (yr0 + yr1) * 0.5
        return replace(
            m_use,
            rcs_db=cr,
            x=x_avg,
            y=y_avg,
            x_raw=xr,
            y_raw=yr,
            rcs_kf_db=float("nan"),
        )

    def _update_orbit_rcs_recording(
        self,
        targets: List[Any],
        raw_clusters: Optional[List[Any]] = None,
    ) -> None:
        """圆周段：只关联前方约 40 m 距离门内的目标并采样 (θ=atan2(y,x), RCS)。"""
        now_ts = time.time()
        candidates: List[ObjMeas] = []
        for t in targets:
            m = self._convert_target_to_objmeas(t, now_ts)
            if m is not None:
                candidates.append(m)
        fresh_candidates = [
            m
            for m in candidates
            if now_ts - float(m.t) <= float(self._radar_target_fresh_s)
        ]

        if not fresh_candidates:
            self.rcs_status_label.setText("圆周RCS: 无新鲜雷达目标")
            self._draw_rcs(live=True)
            return

        nom = float(ORBIT_RCS_NOMINAL_FORWARD_M)
        gate = self._orbit_rcs_effective_forward_gate_m()
        in_gate = [m for m in fresh_candidates if abs(float(m.x) - nom) <= gate]

        if not self.rcs_lock.armed:
            m_use = self._pick_meas_in_forward_gate(
                fresh_candidates, self.tracked_target_id, gate
            )
            if m_use is None:
                self.rcs_status_label.setText(
                    f"圆周RCS: 等待前向≈{ORBIT_RCS_NOMINAL_FORWARD_M:.0f}m目标 (±{gate:.0f}m)"
                )
                self._draw_rcs(live=True)
                return
            self.rcs_lock.arm_from(m_use)
            rt = self._find_radar_target_object_by_oid(targets, int(m_use.oid))
            if rt is not None and self.tracked_target_id != int(m_use.oid):
                self._set_tracked_radar_target(rt, auto=True, reason="圆周RCS距离门内锁定")
        else:
            if not in_gate:
                self.rcs_status_label.setText("圆周RCS: 门内暂无可关联目标")
                self._draw_rcs(live=True)
                return
            prev_oid = self.rcs_lock.last_oid
            m_use = self.rcs_lock.associate(in_gate, now_ts)
            if m_use is None:
                self.rcs_status_label.setText("圆周RCS: 关联保持/搜索门内目标…")
                self._draw_rcs(live=True)
                return
            if prev_oid is not None and int(m_use.oid) != int(prev_oid):
                self._rcs_relock_events.append(f"圆周门内关联切换 {prev_oid}->{m_use.oid} t={now_ts:.2f}")
            if self.tracked_target_id != int(m_use.oid):
                rt = self._find_radar_target_object_by_oid(targets, int(m_use.oid))
                if rt is not None:
                    self._set_tracked_radar_target(rt, auto=True, reason="圆周RCS关联")

        m_use = self._merge_cluster_pair_for_orbit_rcs(
            m_use, in_gate, raw_clusters, nom, gate
        )

        r_raw = float(m_use.rcs_db)
        r_f = float(getattr(m_use, "rcs_kf_db", float("nan")))
        if not math.isfinite(r_f):
            r_f = r_raw
        self.rcs_recorder.oid = int(m_use.oid)
        # 仅缓存 t + 合并后 RCS 供极坐标预览；与 Cluster Raw 行内功率合并一致，不落盘单独极坐标表。
        self._orbit_rcs_rows.append({"t": float(m_use.t), "rcs_filt": float(r_f)})
        self._orbit_polar_cached_series = self._build_orbit_polar_series_from_rows(self._orbit_rcs_rows)
        self.rcs_status_label.setText(
            f"圆周RCS: 进行中 | id={m_use.oid} x={m_use.x:.1f}m RCS={r_raw:.1f}dBsm 点={len(self._orbit_rcs_rows)}"
        )
        self._draw_rcs(live=True)

    def _update_rcs_recording(
        self,
        targets: List,
        raw_clusters: Optional[List[Any]] = None,
    ) -> None:
        """Cluster CSV 写入进度；圆周段仍用簇目标更新极坐标采样。"""
        if not self._rcs_recording:
            return
        rt = getattr(self.controller, "cluster_csv_runtime", None)
        if rt is not None and rt.is_recording():
            nf, nc = rt.snapshot_stats()
            tag = "圆周" if self._orbit_rcs_active else "距离"
            self.rcs_status_label.setText(f"Cluster RCS CSV ({tag}): 帧≈{nf} 簇≈{nc}")
            # 圆周段与 Cluster CSV 同时录制时也必须更新极坐标采样，否则「圆周-RCS」图始终为空
            if self._orbit_rcs_active:
                self._update_orbit_rcs_recording(targets, raw_clusters)
            else:
                self._draw_rcs(live=True)
            return
        if self._orbit_rcs_active:
            self._update_orbit_rcs_recording(targets, raw_clusters)

    def _is_car_moving(self) -> bool:
        car = self.controller.car
        if car is None:
            return False
        try:
            st = car.get_status()
        except Exception:
            return True
        if st.last_update <= 0:
            return True
        if time.time() - float(st.last_update) > 1.5:
            return True
        return abs(st.linear_speed) > 0.05 or abs(st.angular_speed) > 0.05

    def _get_radar_stop_threshold(self) -> float:
        val = float(self._radar_stop_threshold)
        if val <= 0:
            val = 0.1
        return val

    def _get_radar_safety_targets(self, fallback_targets: List) -> List:
        del fallback_targets
        rt = getattr(self.controller, "cluster_csv_runtime", None)
        if rt is None:
            return []
        try:
            _, clusters = rt.get_cluster_display_snapshot()
        except Exception:
            return []
        return [_ClusterSafetyProxy(c) for c in clusters]

    def _check_radar_emergency_stop(self, targets: List) -> None:
        if self.controller.car is None:
            return
        if not self._radar_emergency_stop_enabled:
            self._radar_emergency_active = False
            return
        if not self._motion_active:
            self._radar_emergency_active = False
            return
        if not targets:
            self._radar_emergency_active = False
            return

        if not self._is_car_moving():
            return

        threshold = self._get_radar_stop_threshold()
        near_target = None
        for t in targets:
            try:
                x = float(getattr(t, "x", 0.0))
            except (TypeError, ValueError):
                continue
            if 0.0 <= x < threshold:
                near_target = t
                break

        if near_target is None:
            self._radar_emergency_active = False
            return

        if not self._radar_emergency_active:
            self._radar_emergency_active = True
            reason = (
                f"雷达急停触发: x<{float(getattr(near_target, 'x', 0.0)):.2f}m "
                f"阈值={threshold:.2f}m"
            )
            self.controller.car.emergency_stop()
            self._handle_motion_session_interrupt(
                reason,
                popup_title="\u96f7\u8fbe\u6025\u505c",
            )
            print(
                f"[MainUI] Radar emergency stop: x<{float(getattr(near_target, 'x', 0.0)):.2f}m "
                f"(threshold={threshold:.2f}m)"
            )
            self._log(
                f"雷达急停触发: x<{float(getattr(near_target, 'x', 0.0)):.2f}m 阈值={threshold:.2f}m"
            )

    # ========= 事件处理函数 =========

    @staticmethod
    def _preset_name_sort_key(name: str) -> Tuple[Any, ...]:
        parts = re.split(r"(\d+(?:\.\d+)?)", str(name))
        key: List[Any] = []
        for part in parts:
            if not part:
                continue
            try:
                key.append((0, float(part)))
            except ValueError:
                key.append((1, part.lower()))
        return tuple(key)

    def _set_loaded_task_sequence(
        self,
        task_names: Optional[List[str]] = None,
        transition_count: int = 0,
    ) -> None:
        self._loaded_task_sequence_names = list(task_names or [])
        self._loaded_task_transition_count = int(max(0, transition_count))
        if self._loaded_task_sequence_names:
            self._set_current_path_name(" + ".join(self._loaded_task_sequence_names))

    @staticmethod
    def _radial_measurement_task_name(angle_deg: int) -> str:
        return f"星型{int(angle_deg) % 360}°"

    @staticmethod
    def _radial_measurement_cycle_name(angle_deg: int, cycle_index: int) -> str:
        return f"星型{int(angle_deg) % 360}°_第{int(cycle_index)}次"

    @staticmethod
    def _wrap_angle_deg(angle_deg: float) -> float:
        value = float(angle_deg)
        while value > 180.0:
            value -= 360.0
        while value <= -180.0:
            value += 360.0
        return value

    def _normalize_radial_measurement_spec(
        self,
        spec: RadialMeasurementSpec,
    ) -> RadialMeasurementSpec:
        seen: set[int] = set()
        angle_cycles: List[Tuple[int, int]] = []
        for raw_angle, raw_cycles in spec.angle_cycles:
            angle = int(raw_angle) % 360
            if angle in seen:
                continue
            seen.add(angle)
            angle_cycles.append((angle, max(1, int(raw_cycles))))
        inner_radius = self._normalize_positive_float(spec.inner_radius_m, 4.0)
        line_length = self._normalize_positive_float(getattr(spec, "line_length_m", 50.0), 50.0)
        speed_mps = self._normalize_speed_mps(
            spec.speed_mps,
            float(getattr(self, "_radial_default_speed_mps", DEFAULT_RADIAL_MEASUREMENT_SPEED_MPS)),
        )
        accel_dist_m = self._normalize_positive_float(
            spec.accel_dist_m,
            DEFAULT_ACCEL_DIST_M,
        )
        decel_dist_m = self._normalize_positive_float(
            spec.decel_dist_m,
            DEFAULT_DECEL_DIST_M,
        )
        return RadialMeasurementSpec(
            angle_cycles=angle_cycles,
            inner_radius_m=inner_radius,
            line_length_m=line_length,
            speed_mps=speed_mps,
            accel_dist_m=accel_dist_m,
            decel_dist_m=decel_dist_m,
        )

    def _order_radial_measurement_entries(
        self,
        spec: RadialMeasurementSpec,
    ) -> List[Tuple[int, int]]:
        pending = list(spec.angle_cycles)
        if len(pending) <= 1:
            return pending

        def wrapped_distance(a0: int, a1: int) -> float:
            return abs(self._wrap_angle_deg(float(a1) - float(a0)))

        start_index = min(
            range(len(pending)),
            key=lambda idx: (
                abs(self._wrap_angle_deg(float(pending[idx][0]))),
                pending[idx][0],
            ),
        )
        ordered = [pending.pop(start_index)]
        while pending:
            current_angle = ordered[-1][0]
            next_index = min(
                range(len(pending)),
                key=lambda idx: (
                    wrapped_distance(current_angle, pending[idx][0]),
                    abs(self._wrap_angle_deg(float(pending[idx][0]))),
                    pending[idx][0],
                ),
            )
            ordered.append(pending.pop(next_index))
        return ordered

    def _build_radial_measurement_geometry(
        self,
        center: Tuple[float, float],
        approach_heading: float,
        inner_radius_m: float,
        outer_radius_m: float,
        standoff_radius_m: float,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        ux = math.cos(approach_heading)
        uy = math.sin(approach_heading)
        cx = float(center[0])
        cy = float(center[1])
        standoff_point = (
            cx - standoff_radius_m * ux,
            cy - standoff_radius_m * uy,
        )
        outer_point = (
            cx - outer_radius_m * ux,
            cy - outer_radius_m * uy,
        )
        inner_point = (
            cx - inner_radius_m * ux,
            cy - inner_radius_m * uy,
        )
        return standoff_point, outer_point, inner_point

    @staticmethod
    def _append_plan_segment(
        plan_points: List[Tuple[float, float]],
        plan_ranges: List[SegmentRange],
        range_task_names: List[Optional[str]],
        segment_points: List[Tuple[float, float]],
        speed_sign: int,
        speed_mps: float,
        accel_dist_m: float,
        decel_dist_m: float,
        rcs_start: bool,
        task_name: Optional[str],
    ) -> None:
        if len(segment_points) < 2:
            return
        if not plan_points:
            plan_points.extend((float(x), float(y)) for x, y in segment_points)
            start_idx = 0
            end_idx = len(plan_points) - 1
        else:
            same_start = (
                math.hypot(
                    plan_points[-1][0] - float(segment_points[0][0]),
                    plan_points[-1][1] - float(segment_points[0][1]),
                )
                <= 1e-6
            )
            start_idx = len(plan_points) - 1 if same_start else len(plan_points)
            append_slice = segment_points[1:] if same_start else segment_points
            plan_points.extend((float(x), float(y)) for x, y in append_slice)
            end_idx = len(plan_points) - 1
        if end_idx <= start_idx:
            return
        plan_ranges.append(
            (
                int(start_idx),
                int(end_idx),
                int(-1 if speed_sign < 0 else 1),
                bool(rcs_start),
                float(speed_mps),
                float(accel_dist_m),
                float(decel_dist_m),
            )
        )
        range_task_names.append(task_name)

    def _plan_forward_transition(
        self,
        start_pose: Tuple[float, float, float],
        end_pose: Tuple[float, float, float],
        transition_speed_mps: float,
        close_threshold_m: float,
    ) -> Tuple[List[Tuple[float, float]], List[SegmentRange]]:
        candidate = None
        if hasattr(self.controller, "_plan_transition_candidate"):
            candidate = self.controller._plan_transition_candidate(
                start_pose=start_pose,
                end_pose=end_pose,
                motion_sign=1,
                close_threshold_m=close_threshold_m,
            )
        if candidate is None:
            return self.controller.plan_transition_path(
                start_pose=start_pose,
                end_pose=end_pose,
                start_speed_mps=transition_speed_mps,
                end_speed_mps=transition_speed_mps,
                close_threshold_m=close_threshold_m,
            )

        points, _, _ = candidate
        if len(points) < 2:
            return [], []
        transition_len = sum(
            math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
            for i in range(len(points) - 1)
        )
        accel_dist = min(0.9, max(0.2, 0.22 * transition_len))
        decel_dist = min(0.9, max(0.2, 0.22 * transition_len))
        ranges: List[SegmentRange] = [
            (
                0,
                len(points) - 1,
                1,
                False,
                float(transition_speed_mps),
                float(accel_dist),
                float(decel_dist),
            )
        ]
        return [(float(x), float(y)) for x, y in points], ranges

    def _build_radial_measurement_plan(
        self,
        anchor_pose: PoseSolution,
        spec: RadialMeasurementSpec,
        *,
        origin_x_m: float,
        origin_y_m: float,
    ) -> Tuple[RadialMeasurementPlan, RadialMeasurementSpec]:
        normalized = self._normalize_radial_measurement_spec(spec)
        if self.target_point is None:
            raise ValueError("请先标定目标物位置。")

        def transition_planner(
            start_pose: Tuple[float, float, float],
            end_pose: Tuple[float, float, float],
            transition_speed_mps: float,
            close_threshold_m: float,
        ) -> Tuple[List[Tuple[float, float]], List[SegmentRange]]:
            return self._plan_forward_transition(
                start_pose=start_pose,
                end_pose=end_pose,
                transition_speed_mps=transition_speed_mps,
                close_threshold_m=close_threshold_m,
            )

        plan = build_star_measurement_plan(
            target_point_xy=(float(self.target_point[0]), float(self.target_point[1])),
            origin_xy=(float(origin_x_m), float(origin_y_m)),
            spec=StarMeasurementSpec(
                angle_cycles=list(normalized.angle_cycles),
                inner_radius_m=float(normalized.inner_radius_m),
                line_length_m=float(normalized.line_length_m),
                speed_mps=float(normalized.speed_mps),
                accel_dist_m=float(normalized.accel_dist_m),
                decel_dist_m=float(normalized.decel_dist_m),
                max_step_m=0.08,
                enable_transitions=True,
                transition_heading_align_reserve_m=min(
                    6.0,
                    max(2.5, 0.06 * float(normalized.line_length_m)),
                ),
            ),
            transition_planner=transition_planner,
            transition_speed_mps=None,
            transition_close_threshold_m=max(
                0.8,
                0.18
                * (
                    (float(normalized.inner_radius_m) + float(normalized.line_length_m))
                    + max(
                        1.6,
                        0.55 * (float(normalized.inner_radius_m) + float(normalized.line_length_m)),
                        float(normalized.line_length_m) + 1.0,
                    )
                ),
            ),
        )
        return (
            RadialMeasurementPlan(
                task_names=list(plan.task_names),
                local_points=list(plan.local_points),
                ranges=list(plan.ranges),
                range_task_names=list(plan.range_task_names),
                transition_count=int(plan.transition_count),
            ),
            normalized,
        )

    def _apply_radial_measurement_spec(
        self,
        spec: RadialMeasurementSpec,
        anchor_pose: Optional[PoseSolution] = None,
        announce_virtual_preview: bool = True,
    ) -> bool:
        frame = self._get_path_anchor_reference_frame()
        resolved_frame = self._resolve_path_reference_frame(frame)

        resolved_pose = anchor_pose
        if resolved_pose is None:
            resolved_pose = get_robot_pose()
        using_virtual_pose = resolved_pose is None
        if resolved_pose is None:
            resolved_pose = self._build_virtual_preview_pose()

        try:
            plan, normalized = self._build_radial_measurement_plan(
                resolved_pose,
                spec,
                origin_x_m=resolved_frame.origin_x_m,
                origin_y_m=resolved_frame.origin_y_m,
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "星型测量无效", str(exc))
            self._log(f"星型测量生成失败: {exc}")
            return False

        local_points = list(plan.local_points)
        if len(local_points) < 2:
            QtWidgets.QMessageBox.warning(self, "星型测量无效", "生成的测量轨迹点数不足。")
            self._log("星型测量生成失败: 点数不足")
            return False

        def trim_star_forward_window(
            points: List[Tuple[float, float]],
            ranges: List[SegmentRange],
            range_task_names: List[Optional[str]],
            *,
            forward_dir_global: Tuple[float, float],
            window_m: float,
        ) -> Tuple[List[Tuple[float, float]], List[SegmentRange], List[Optional[str]]]:
            if len(points) < 2:
                return points, ranges, range_task_names
            fx, fy = float(forward_dir_global[0]), float(forward_dir_global[1])
            f_norm = math.hypot(fx, fy)
            if f_norm <= 1e-9:
                return points, ranges, range_task_names
            fx /= f_norm
            fy /= f_norm

            proj = [float(px) * fx + float(py) * fy for px, py in points]
            eps = 1e-6
            start_keep = None
            for i, s in enumerate(proj):
                if s >= -eps:
                    start_keep = i
                    break
            if start_keep is None:
                start_keep = 0
            end_keep = start_keep
            max_s = float(window_m) + eps
            for i in range(start_keep, len(proj)):
                if proj[i] <= max_s:
                    end_keep = i
                else:
                    break
            if end_keep - start_keep < 1:
                end_keep = min(len(points) - 1, start_keep + 1)

            new_points = list(points[start_keep : end_keep + 1])
            new_ranges: List[SegmentRange] = []
            new_range_names: List[Optional[str]] = []
            for idx, seg in enumerate(ranges):
                s0, e0, speed_sign, rcs_start, speed_mps, accel_dist, decel_dist = seg
                s1 = max(int(s0), int(start_keep))
                e1 = min(int(e0), int(end_keep))
                if e1 - s1 < 1:
                    continue
                new_ranges.append(
                    (
                        int(s1 - start_keep),
                        int(e1 - start_keep),
                        int(speed_sign),
                        bool(rcs_start),
                        float(speed_mps),
                        float(accel_dist),
                        float(decel_dist),
                    )
                )
                new_range_names.append(
                    range_task_names[idx] if idx < len(range_task_names) else None
                )
            if len(new_points) >= 2 and not new_ranges:
                new_ranges = [
                    (
                        0,
                        len(new_points) - 1,
                        1,
                        False,
                        float(self._path_speed),
                        float(self._traj_default_accel_dist),
                        float(self._traj_default_decel_dist),
                    )
                ]
                new_range_names = [None]
            return new_points, new_ranges, new_range_names

        # 单角度星型：沿锚点→目标方向做 [0, L] 投影裁剪，减少“后方”杂段在预览里的干扰。
        # 多角度时各向射线会绕目标分布，同一投影轴上很容易超过 L 或落在侧向，裁剪会把后续角度整段删掉，
        # 表现为只残留某一角度的局部轨迹；故仅对「只选一个角度」启用该裁剪。
        if self.target_point is not None and len(normalized.angle_cycles) <= 1:
            fdx = float(self.target_point[0]) - float(resolved_frame.origin_x_m)
            fdy = float(self.target_point[1]) - float(resolved_frame.origin_y_m)
            local_points, plan_ranges, plan_range_task_names = trim_star_forward_window(
                local_points,
                list(plan.ranges),
                list(plan.range_task_names),
                forward_dir_global=(fdx, fdy),
                window_m=float(normalized.line_length_m),
            )
        else:
            plan_ranges = list(plan.ranges)
            plan_range_task_names = list(plan.range_task_names)

        self._planned_ranges = list(plan_ranges) if self._should_use_segment_ranges(list(plan_ranges)) else None
        self._planned_range_task_names = (
            list(plan_range_task_names) if self._planned_ranges else []
        )
        self._planned_segment_kinds = None
        if not self._apply_planned_local_points(local_points, frame=resolved_frame):
            return False

        self._radial_measurement_spec = normalized
        self._radial_default_angle_cycles = {
            int(angle): int(cycles) for angle, cycles in normalized.angle_cycles
        }
        self._radial_default_inner_radius = float(normalized.inner_radius_m)
        self._radial_default_outer_radius = float(getattr(normalized, "inner_radius_m", 4.0)) + float(
            getattr(normalized, "line_length_m", 50.0)
        )
        self._radial_default_line_length_m = float(getattr(normalized, "line_length_m", 50.0))
        self._radial_default_speed_mps = float(normalized.speed_mps)
        self._path_speed = float(normalized.speed_mps)
        self._traj_default_accel_dist = float(normalized.accel_dist_m)
        self._traj_default_decel_dist = float(normalized.decel_dist_m)
        self._set_loaded_task_sequence(list(plan.task_names), int(plan.transition_count))

        angle_text = "/".join(str(name) for name in plan.task_names)
        self._set_current_path_name(f"星型测量[{angle_text}]")
        if using_virtual_pose and announce_virtual_preview:
            self._log("星型测量当前使用虚拟原点位姿进行预览，获取实时位姿后会自动重建。")
        self._log(
            f"星型测量已生成: 角度={angle_text} 点数={len(local_points)} 过渡段={int(plan.transition_count)}"
        )
        return True

    def _on_radial_measure_clicked(self) -> None:
        if self.target_point is None:
            QtWidgets.QMessageBox.information(
                self,
                "未标定目标",
                "请先点击“标定目标物位置”，在轨迹图上标定目标点后再生成星型测量轨迹。",
            )
            self._log("星型测量生成失败: 未标定目标物位置")
            return
        dialog = RadialMeasurementDialog(
            target_point=self.target_point,
            default_angle_cycles=self._radial_default_angle_cycles,
            default_inner_radius=self._radial_default_inner_radius,
            default_line_length_m=float(getattr(self, "_radial_default_line_length_m", 50.0)),
            default_speed=float(getattr(self, "_radial_default_speed_mps", DEFAULT_RADIAL_MEASUREMENT_SPEED_MPS)),
            default_accel_dist=self._traj_default_accel_dist,
            default_decel_dist=self._traj_default_decel_dist,
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        spec = dialog.values()
        self._apply_radial_measurement_spec(spec)

    def _load_preset_task_sequence(self, task_names: List[str]) -> bool:
        self._clear_radial_measurement_spec()
        queue_tasks: List[QueuedPathTask] = []
        task_frame: Optional[PathReferenceFrame] = None
        for name in task_names:
            local_points = self._preset_paths.get(name)
            if not local_points or len(local_points) < 2:
                QtWidgets.QMessageBox.warning(
                    self,
                    "预设轨迹无效",
                    f"轨迹“{name}”点数不足，无法加入任务序列。",
                )
                self._log(f"任务序列加载失败: {name} 点数不足")
                return False
            current_frame = self._resolve_path_reference_frame(
                self._preset_path_frames.get(name)
            )
            if task_frame is None:
                task_frame = current_frame
            elif not self._path_reference_frames_match(task_frame, current_frame):
                QtWidgets.QMessageBox.warning(
                    self,
                    "任务序列坐标系不一致",
                    "所选预设轨迹使用了不同的坐标参考系或固定原点，当前无法直接拼接成同一任务序列。",
                )
                self._log(
                    "任务序列加载失败: 预设轨迹坐标参考系不一致，无法拼接。"
                )
                return False
            ranges = [
                self._normalize_segment_range(seg_range)
                for seg_range in self._preset_ranges.get(name, [])
            ]
            queue_tasks.append(
                QueuedPathTask(
                    name=name,
                    local_points=list(local_points),
                    ranges=ranges,
                )
            )

        plan = self.controller.build_task_queue_plan(queue_tasks)
        if len(plan.local_points) < 2:
            QtWidgets.QMessageBox.warning(
                self,
                "任务序列无效",
                "勾选的轨迹无法组成有效任务序列，请检查轨迹内容。",
            )
            self._log("任务序列加载失败: 规划结果为空")
            return False

        self._planned_ranges = list(plan.ranges) if self._should_use_segment_ranges(list(plan.ranges)) else None
        self._planned_range_task_names = list(plan.range_task_names) if self._planned_ranges else []
        self._planned_segment_kinds = None
        if not self._apply_planned_local_points(
            list(plan.local_points),
            frame=task_frame,
        ):
            return False

        self._set_loaded_task_sequence(plan.task_names, plan.transition_count)
        self._apply_path_reference_frame_to_controls(task_frame)
        pair_text = "、".join(f"{a}->{b}" for a, b in plan.transition_pairs[:6])
        if plan.transition_count > 6:
            pair_text += "..."
        self._log(
            f"任务序列已加载: {' -> '.join(plan.task_names)} | "
            f"轨迹数={len(plan.task_names)} 过渡段={plan.transition_count} 点数={len(plan.local_points)}"
        )
        if task_frame is not None:
            self._log(
                f"任务序列参考系: {self._format_path_reference_frame(task_frame)}"
            )
        if pair_text:
            self._log(f"任务序列过渡: {pair_text}")
        return True

    def _on_preset_path_clicked(self) -> None:
        """选择预设轨迹"""
        if not self._preset_paths:
            self._import_preset_trajectory_from_file()
            return

        names = sorted(self._preset_paths.keys(), key=self._preset_name_sort_key)
        dialog = PresetSequenceDialog(
            preset_names=names,
            checked_names=self._loaded_task_sequence_names,
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        if dialog.import_requested():
            self._import_preset_trajectory_from_file()
            return
        checked_names = dialog.checked_names()
        if not checked_names:
            QtWidgets.QMessageBox.information(self, "未选择轨迹", "请至少勾选一条预设轨迹。")
            return
        self._load_preset_task_sequence(checked_names)

    def _on_delete_preset_clicked(self) -> None:
        """删除预设轨迹"""
        if not self._preset_paths:
            QtWidgets.QMessageBox.information(
                self,
                "无预设轨迹",
                "当前没有可删除的预设轨迹。",
            )
            self._log("删除预设轨迹失败: 列表为空")
            return
        names = sorted(self._preset_paths.keys())
        choice, ok = QtWidgets.QInputDialog.getItem(
            self,
            "删除预设轨迹",
            "选择要删除的预设轨迹",
            names,
            0,
            False,
        )
        if not ok:
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除预设轨迹“{choice}”？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            self._log(f"删除预设轨迹已取消: {choice}")
            return

        self._preset_paths.pop(choice, None)
        self._preset_ranges.pop(choice, None)
        self._preset_path_frames.pop(choice, None)
        if not self._persist_preset_paths():
            return
        self._log(f"预设轨迹已删除: {choice}")

    def _import_preset_trajectory_from_file(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "导入预设轨迹文件",
            "",
            "CSV / 文本 (*.csv *.txt);;所有文件 (*)",
        )
        if not filename:
            return

        try:
            local_points, ranges = self._parse_trajectory_file(filename)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "导入失败", f"读取轨迹文件失败: {e}")
            self._log(f"预设轨迹文件导入失败: {e}")
            return

        if len(local_points) < 2:
            QtWidgets.QMessageBox.warning(self, "轨迹无效", "轨迹点数量不足（至少需要 2 个点）。")
            self._log("预设轨迹文件导入失败: 点数不足")
            return

        frame = self._get_path_anchor_reference_frame()

        self._set_loaded_task_sequence([], 0)
        self._clear_radial_measurement_spec()
        self._planned_ranges = ranges if self._should_use_segment_ranges(ranges) else None
        self._planned_range_task_names = []
        self._planned_segment_kinds = None
        if not self._apply_planned_local_points(local_points, frame=frame):
            return

        default_name = Path(filename).stem or f"预设轨迹{len(self._preset_paths) + 1}"
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "保存为预设轨迹",
            "请输入预设轨迹名称（留空仅临时加载）",
            text=default_name,
        )
        if ok and name.strip():
            self._save_preset_path(name.strip(), local_points, ranges, frame=frame)
            self._set_current_path_name(name.strip())
        elif ok:
            self._log("导入轨迹未保存为预设轨迹（仅临时加载）")
            self._set_current_path_name(default_name)
        else:
            self._set_current_path_name(default_name)
        self._apply_path_reference_frame_to_controls(frame)

        self._log(
            f"轨迹文件已导入: {Path(filename).name} 点数={len(local_points)} "
            f"分段={len(ranges)} RCS触发段={sum(1 for r in ranges if r[3])} | "
            f"参考系={self._format_path_reference_frame(frame)}"
        )

    def _on_calib_clicked(self) -> None:
        """标定目标物位置"""
        if self.target_marker is None:
            return
        self._target_marking_mode = True
        self._log("开始标定目标物位置: 请在轨迹图上点击目标点")

    def _on_open_log_window_clicked(self) -> None:
        self._show_tool_dialog(self._log_viewer_dialog)

    def _on_radar_lock_check_clicked(self) -> None:
        self._place_radar_target_dialog_at_trajectory_top_right()
        self._show_tool_dialog(self._radar_target_dialog)
        self._log("已打开雷达目标检查窗口")

    def _on_enu_calibration_clicked(self) -> None:
        dialog = EnuCalibrationDialog(self._enu_calibration_points, parent=self)
        dialog.exec_()
        self._enu_calibration_points = dialog.points()
        if dialog.calibration_changed():
            self._use_calibration_plane_path_origin()
        else:
            self._update_path_coordinate_widgets()
        if dialog.calibration_changed():
            self._refresh_ui_after_enu_frame_change()

    def _refresh_data_analysis_window(
        self,
        focus_path: Optional[Path] = None,
        *,
        activate: bool = False,
    ) -> None:
        window = self._data_analysis_window
        if window is None:
            return
        try:
            if hasattr(window, "set_data_root"):
                window.set_data_root(self._data_save_root)
            if hasattr(window, "refresh_data"):
                window.refresh_data(focus_path=focus_path)
            if activate:
                window.show()
                window.raise_()
                window.activateWindow()
        except Exception as exc:
            self._log(f"刷新误差分析窗口失败: {exc}")

    def _on_data_analysis_clicked(self) -> None:
        try:
            if self._data_analysis_window is None:
                data_analysis_mod = _load_data_analysis_module()
                self._data_analysis_window = data_analysis_mod.DataAnalysisWindow(
                    data_root=self._data_save_root
                )
            self._refresh_data_analysis_window(activate=True)
            self._log(f"已打开误差分析窗口: {self._data_save_root}")
        except Exception as exc:
            self._log(f"打开误差分析窗口失败: {exc}")
            QtWidgets.QMessageBox.critical(
                self,
                "误差分析",
                f"打开误差分析界面失败：\n{exc}",
            )

    def _on_heading_calib_clicked(self) -> None:
        """清除历史轨迹"""
        self.path_x.clear()
        self.path_y.clear()
        self.traj_curve.setData([], [])
        self._log("历史轨迹已清除")

    def _on_traj_plot_clicked(self, event: QtCore.QEvent) -> None:
        if not self._target_marking_mode:
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        if self.target_marker is None:
            return
        vb = self.traj_plot.getViewBox()
        if vb is None:
            return
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        mouse_point = vb.mapSceneToView(event.scenePos())
        x = float(mouse_point.x())
        y = float(mouse_point.y())
        self._set_target_point_from_global(
            x,
            y,
            log_prefix="目标物位置标定",
        )
        self._log(f"目标物位置标定: x={x:.2f}m, y={y:.2f}m")
        if self._radial_measurement_spec is not None:
            anchor_pose = get_robot_pose()
            if anchor_pose is None:
                anchor_pose = self._build_virtual_preview_pose()
            if self._apply_radial_measurement_spec(
                self._radial_measurement_spec,
                anchor_pose=anchor_pose,
                announce_virtual_preview=False,
            ):
                self._log("目标物位置已更新，星型测量轨迹已自动重建。")

    @staticmethod
    def _is_nearly_straight_segment(points: List[Tuple[float, float]], ratio_thresh: float = 1.002) -> bool:
        if len(points) < 3:
            return True
        chord = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
        if chord <= 1e-6:
            return False
        length = 0.0
        for i in range(len(points) - 1):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            length += math.hypot(dx, dy)
        return (length / chord) <= max(1.0, float(ratio_thresh))

    @staticmethod
    def _estimate_path_curvature(points: List[Tuple[float, float]]) -> float:
        if len(points) < 3:
            return 0.0
        total_len = 0.0
        total_turn = 0.0
        prev_heading: Optional[float] = None
        for i in range(len(points) - 1):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-6:
                continue
            heading = math.atan2(dy, dx)
            if prev_heading is not None:
                d = heading - prev_heading
                while d > math.pi:
                    d -= 2.0 * math.pi
                while d < -math.pi:
                    d += 2.0 * math.pi
                total_turn += abs(d)
            total_len += seg_len
            prev_heading = heading
        if total_len <= 1e-6:
            return 0.0
        return total_turn / total_len

    def _estimate_expected_w_limit(
        self,
        points: List[Tuple[float, float]],
        speed_mps: float,
        is_straight: Optional[bool] = None,
    ) -> float:
        speed_abs = abs(float(speed_mps))
        if is_straight is None:
            is_straight = self._is_nearly_straight_segment(points)
        if is_straight:
            return 1.0 if speed_abs <= 1.0 else 1.2
        kappa = self._estimate_path_curvature(points)
        nominal_w = speed_abs * kappa
        return max(0.9, min(2.4, nominal_w * 2.6 + 0.45))

    @staticmethod
    def _segment_profile_floor_speed(speed_mps: float) -> float:
        speed_abs = max(MIN_SEGMENT_SPEED_MPS, abs(float(speed_mps)))
        return min(speed_abs, max(MIN_PROFILE_SPEED_MPS, 0.35 * speed_abs))

    @staticmethod
    def _segment_terminal_end_speed(speed_mps: float) -> float:
        speed_abs = max(MIN_SEGMENT_SPEED_MPS, abs(float(speed_mps)))
        return min(speed_abs, max(MIN_SEGMENT_SPEED_MPS, min(0.12, 0.18 * speed_abs)))

    @staticmethod
    def _segment_arrival_dist(speed_mps: float) -> float:
        speed_abs = max(MIN_SEGMENT_SPEED_MPS, abs(float(speed_mps)))
        speed_ratio = min(1.0, speed_abs / 0.8)
        return max(0.04, min(0.09, 0.04 + 0.04 * speed_ratio))

    @staticmethod
    def _estimate_segment_length(points: List[Tuple[float, float]]) -> float:
        total_len = 0.0
        for i in range(len(points) - 1):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            total_len += math.hypot(dx, dy)
        return total_len

    @staticmethod
    def _densify_polyline_for_tracking(
        points: List[Tuple[float, float]],
        max_step_m: float = 0.08,
    ) -> List[Tuple[float, float]]:
        """沿折线加密路径点，使相邻点间距不超过 max_step_m（用于圆弧段 Stanley 跟踪）。"""
        if len(points) < 2:
            return list(points)
        max_step_m = max(0.02, float(max_step_m))
        out: List[Tuple[float, float]] = [(float(points[0][0]), float(points[0][1]))]
        for i in range(len(points) - 1):
            x0, y0 = float(points[i][0]), float(points[i][1])
            x1, y1 = float(points[i + 1][0]), float(points[i + 1][1])
            dx, dy = x1 - x0, y1 - y0
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-9:
                continue
            n = max(1, int(math.ceil(seg_len / max_step_m)))
            for j in range(1, n + 1):
                t = j / float(n)
                out.append((x0 + t * dx, y0 + t * dy))
        return out

    @staticmethod
    def _infer_circle_orbit_params(
        points: List[Tuple[float, float]],
    ) -> Optional[Dict[str, Any]]:
        if len(points) < 6:
            return None

        filtered: List[Tuple[float, float]] = [points[0]]
        for px, py in points[1:]:
            if math.hypot(float(px) - filtered[-1][0], float(py) - filtered[-1][1]) > 1e-4:
                filtered.append((float(px), float(py)))
        if len(filtered) < 6:
            return None

        arr = np.asarray(filtered, dtype=float)
        x = arr[:, 0]
        y = arr[:, 1]
        mat = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
        rhs = x * x + y * y

        try:
            sol, _, _, _ = np.linalg.lstsq(mat, rhs, rcond=None)
        except np.linalg.LinAlgError:
            return None

        cx = float(sol[0])
        cy = float(sol[1])
        c = float(sol[2])
        radius_sq = c + cx * cx + cy * cy
        if (not math.isfinite(radius_sq)) or radius_sq <= 1e-6:
            return None
        radius_m = math.sqrt(radius_sq)

        radial_err = np.hypot(x - cx, y - cy) - radius_m
        radial_rmse_m = float(np.sqrt(np.mean(radial_err * radial_err)))
        radial_max_abs_m = float(np.max(np.abs(radial_err)))
        rmse_limit_m = max(0.025, min(0.18, 0.06 * radius_m))
        max_limit_m = max(0.060, min(0.35, 0.12 * radius_m))
        if radial_rmse_m > rmse_limit_m or radial_max_abs_m > max_limit_m:
            return None

        angles = np.unwrap(np.arctan2(y - cy, x - cx))
        total_angle_rad = float(angles[-1] - angles[0])
        if abs(total_angle_rad) < math.radians(6.0):
            return None

        arc_len_m = 0.0
        for idx in range(len(filtered) - 1):
            dx = filtered[idx + 1][0] - filtered[idx][0]
            dy = filtered[idx + 1][1] - filtered[idx][1]
            arc_len_m += math.hypot(dx, dy)
        expected_arc_len_m = radius_m * abs(total_angle_rad)
        if expected_arc_len_m <= 1e-6:
            return None
        arc_mismatch_m = abs(arc_len_m - expected_arc_len_m)
        mismatch_limit_m = max(0.25, min(1.2, 0.35 * expected_arc_len_m))
        if arc_mismatch_m > mismatch_limit_m:
            return None

        return {
            "center_x_m": cx,
            "center_y_m": cy,
            "radius_m": radius_m,
            "angle_deg": abs(math.degrees(total_angle_rad)),
            "clockwise": bool(total_angle_rad < 0.0),
            "fit_rmse_m": radial_rmse_m,
            "fit_max_error_m": radial_max_abs_m,
        }

    def _arc_entry_speed_cap_mps(
        self,
        arc_points: List[Tuple[float, float]],
        cruise_mps: float,
    ) -> float:
        """直线连续接入下一段圆弧时，按半径限制切入速度，减轻过冲与切弯跟不上。"""
        cruise_abs = max(MIN_SEGMENT_SPEED_MPS, abs(float(cruise_mps)))
        inf = self._infer_circle_orbit_params(arc_points)
        if inf is None:
            return max(MIN_SEGMENT_SPEED_MPS, min(cruise_abs, 0.50 * cruise_abs + 0.06))
        R = max(0.12, float(inf["radius_m"]))
        cap = min(cruise_abs, max(0.10, 0.36 + 0.44 * math.sqrt(R)))
        return float(cap)

    def _build_segment_speed_profile(
        self,
        points: List[Tuple[float, float]],
        start_speed_mps: float,
        cruise_speed_mps: float,
        end_speed_mps: float,
        accel_dist_m: float,
        decel_dist_m: float,
    ) -> List[Tuple[float, float]]:
        total_len = self._estimate_segment_length(points)
        cruise_speed = self._normalize_speed_mps(cruise_speed_mps)
        start_speed = max(0.0, abs(float(start_speed_mps)))
        end_speed = max(0.0, abs(float(end_speed_mps)))
        if total_len <= 1e-6:
            return [(0.0, cruise_speed)]

        accel_dist = min(
            total_len,
            self._normalize_positive_float(accel_dist_m, DEFAULT_ACCEL_DIST_M),
        )
        decel_dist = min(
            total_len,
            self._normalize_positive_float(decel_dist_m, DEFAULT_DECEL_DIST_M),
        )
        if accel_dist + decel_dist > total_len and (accel_dist + decel_dist) > 1e-6:
            scale = total_len / (accel_dist + decel_dist)
            accel_dist *= scale
            decel_dist *= scale
        cruise_end_s = max(accel_dist, total_len - decel_dist)

        raw_profile = [
            (0.0, start_speed),
            (accel_dist, cruise_speed),
            (cruise_end_s, cruise_speed),
            (total_len, end_speed),
        ]
        profile: List[Tuple[float, float]] = []
        for s, v in raw_profile:
            s = max(0.0, min(total_len, float(s)))
            v = max(0.0, float(v))
            if profile and abs(s - profile[-1][0]) <= 1e-6:
                profile[-1] = (s, v)
            else:
                profile.append((s, v))
        if not profile:
            profile.append((0.0, cruise_speed))
        if abs(profile[-1][0] - total_len) > 1e-6:
            profile.append((total_len, end_speed))
        return profile

    def _configure_motion_guard(
        self,
        design: str,
        nominal_speed: float,
        expected_w_limit: float,
        allow_spin_s: float = 0.0,
    ) -> None:
        self._motion_guard_design = str(design)
        self._motion_guard_nominal_speed = abs(float(nominal_speed))
        self._motion_guard_w_limit = max(0.5, min(2.5, abs(float(expected_w_limit))))
        if allow_spin_s > 0.0:
            self._motion_guard_allow_spin_until = max(
                self._motion_guard_allow_spin_until,
                time.time() + float(allow_spin_s),
            )
        self._motion_guard_spin_since = None
        self._motion_guard_jerk_since = None
        self._motion_guard_prev_ts = None
        self._motion_guard_prev_w = None

    def _reset_motion_guard_runtime(self, clear_profile: bool = False) -> None:
        self._motion_guard_spin_since = None
        self._motion_guard_jerk_since = None
        self._motion_guard_prev_ts = None
        self._motion_guard_prev_w = None
        if clear_profile:
            self._motion_guard_design = "unknown"
            self._motion_guard_nominal_speed = 0.0
            self._motion_guard_w_limit = 1.5
            self._motion_guard_allow_spin_until = 0.0

    def _check_motion_anomaly_emergency_stop(self) -> None:
        if not self._motion_guard_enabled:
            return
        car = self.controller.car
        if car is None:
            self._reset_motion_guard_runtime(clear_profile=True)
            return
        if not self._motion_active:
            self._reset_motion_guard_runtime(clear_profile=True)
            return
        try:
            st = car.get_status()
        except Exception:
            return

        now = time.time()
        if st.last_update <= 0:
            return
        fb_age = now - float(st.last_update)
        if fb_age > 0.4:
            # 底盘反馈过旧时不做判定，避免误报
            return

        w_signed = float(st.angular_speed)
        w_abs = abs(w_signed)
        v_abs = abs(float(st.linear_speed))

        prev_ts = self._motion_guard_prev_ts
        prev_w = self._motion_guard_prev_w
        self._motion_guard_prev_ts = now
        self._motion_guard_prev_w = w_signed
        if prev_ts is None or prev_w is None:
            return

        dt = max(1e-3, now - prev_ts)
        dw_dt = abs(w_signed - float(prev_w)) / dt

        if now < self._motion_guard_allow_spin_until:
            self._motion_guard_spin_since = None
            self._motion_guard_jerk_since = None
            return

        low_v_thresh = max(
            float(self._motion_guard_low_v_mps),
            0.12 * max(0.1, float(self._motion_guard_nominal_speed)),
        )
        spin_w_limit = max(
            float(self._motion_guard_spin_w_radps),
            float(self._motion_guard_w_limit) + 0.25,
        )
        jerk_w_limit = max(
            float(self._motion_guard_jerk_w_floor),
            float(self._motion_guard_w_limit),
        )

        spin_violation = (v_abs <= low_v_thresh) and (w_abs >= spin_w_limit)
        if spin_violation:
            if self._motion_guard_spin_since is None:
                self._motion_guard_spin_since = now
        else:
            self._motion_guard_spin_since = None

        jerk_violation = (w_abs >= jerk_w_limit) and (
            dw_dt >= float(self._motion_guard_dw_dt_thresh)
        )
        if jerk_violation:
            if self._motion_guard_jerk_since is None:
                self._motion_guard_jerk_since = now
        else:
            self._motion_guard_jerk_since = None

        reason: Optional[str] = None
        if (
            self._motion_guard_spin_since is not None
            and (now - self._motion_guard_spin_since) >= float(self._motion_guard_spin_hold_s)
        ):
            reason = (
                f"运动异常急停：检测到疑似非设计原地自转\n"
                f"v={v_abs:.2f} m/s, |w|={w_abs:.2f} rad/s, 允许上限≈{spin_w_limit:.2f} rad/s\n"
                f"当前段类型={self._motion_guard_design}"
            )
        elif (
            self._motion_guard_jerk_since is not None
            and (now - self._motion_guard_jerk_since) >= float(self._motion_guard_jerk_hold_s)
        ):
            reason = (
                f"运动异常急停：检测到角速度突变/抽搐\n"
                f"|w|={w_abs:.2f} rad/s, |dw/dt|={dw_dt:.2f} rad/s², "
                f"阈值={self._motion_guard_dw_dt_thresh:.2f} rad/s²\n"
                f"当前段类型={self._motion_guard_design}"
            )

        if reason is None:
            return

        car.emergency_stop()
        self._motion_guard_last_popup_ts = now
        self._handle_motion_session_interrupt(
            reason.replace("\n", " | "),
            popup_title="\u8fd0\u52a8\u5f02\u5e38\u6025\u505c",
            popup_text=(
                reason
                + "\n\n\u5df2\u81ea\u52a8\u6025\u505c\uff0c\u8bf7\u68c0\u67e5\u8f68\u8ff9\u53c2\u6570\u3001"
                "\u901f\u5ea6\u8bbe\u7f6e\u548c\u5b9a\u4f4d\u72b6\u6001\u3002"
            ),
        )

    def _on_load_path(self) -> None:
        self._log("打开轨迹规划")
        self._open_traj_planner()

    def _prompt_preset_name(self) -> Optional[str]:
        default_name = f"预设轨迹{len(self._preset_paths) + 1}"
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "预设轨迹名称",
            "请输入预设轨迹名称",
            text=default_name,
        )
        if not ok:
            return None
        name = name.strip()
        if not name:
            return None
        return name

    def _save_preset_path(
        self,
        name: str,
        local_points: List[Tuple[float, float]],
        ranges: Optional[List[SegmentRange]] = None,
        frame: Optional[PathReferenceFrame] = None,
    ) -> bool:
        if name in self._preset_paths:
            reply = QtWidgets.QMessageBox.question(
                self,
                "覆盖预设轨迹",
                f"预设轨迹“{name}”已存在，是否覆盖？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                self._log(f"预设轨迹保存已取消: {name}")
                return False
        self._preset_paths[name] = list(local_points)
        self._preset_ranges[name] = [
            self._normalize_segment_range(seg_range) for seg_range in (ranges or [])
        ]
        self._preset_path_frames[name] = self._resolve_path_reference_frame(
            frame if frame is not None else self._path_anchor_frame_from_ui()
        )
        if not self._persist_preset_paths():
            return False
        self._log(
            f"预设轨迹已保存: {name} 点数={len(local_points)} "
            f"分段={len(self._preset_ranges[name])} | "
            f"参考系={self._format_path_reference_frame(self._preset_path_frames[name])}"
        )
        return True

    def _persist_preset_paths(self) -> bool:
        try:
            with PRESET_PATHS_CSV.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "record_type",
                        "name",
                        "index",
                        "x",
                        "y",
                        "start_idx",
                        "end_idx",
                        "speed_sign",
                        "rcs_start",
                        "speed_mps",
                        "accel_dist",
                        "decel_dist",
                        "coord_mode",
                        "origin_key",
                        "origin_label",
                        "origin_x",
                        "origin_y",
                        "origin_z",
                    ]
                )
                for name, points in self._preset_paths.items():
                    frame = self._resolve_path_reference_frame(
                        self._preset_path_frames.get(name)
                    )
                    writer.writerow(
                        [
                            "meta",
                            name,
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            frame.mode,
                            frame.origin_key,
                            frame.origin_label,
                            f"{frame.origin_x_m:.6f}",
                            f"{frame.origin_y_m:.6f}",
                            f"{frame.origin_z_m:.6f}",
                        ]
                    )
                    for idx, (x, y) in enumerate(points):
                        writer.writerow(
                            [
                                "point",
                                name,
                                idx,
                                f"{x:.6f}",
                                f"{y:.6f}",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                            ]
                        )
                    ranges = self._preset_ranges.get(name, [])
                    for idx, (
                        start_idx,
                        end_idx,
                        speed_sign,
                        rcs_start,
                        speed_mps,
                        accel_dist,
                        decel_dist,
                    ) in enumerate(ranges):
                        writer.writerow(
                            [
                                "segment",
                                name,
                                idx,
                                "",
                                "",
                                int(start_idx),
                                int(end_idx),
                                int(speed_sign),
                                int(bool(rcs_start)),
                                f"{float(speed_mps):.3f}",
                                f"{float(accel_dist):.3f}",
                                f"{float(decel_dist):.3f}",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                            ]
                        )
            return True
        except Exception as e:
            self._log(f"预设轨迹保存失败: {e}")
            QtWidgets.QMessageBox.warning(self, "保存失败", f"预设轨迹保存失败: {e}")
            return False

    def _load_preset_paths(self) -> None:
        self._preset_paths.clear()
        self._preset_ranges.clear()
        self._preset_path_frames.clear()
        if not PRESET_PATHS_CSV.exists():
            return
        try:
            temp_points: Dict[str, List[Tuple[int, float, float]]] = {}
            temp_ranges: Dict[str, List[Tuple[int, SegmentRange]]] = {}
            temp_frames: Dict[str, PathReferenceFrame] = {}
            with PRESET_PATHS_CSV.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = [str(c or "").strip().lower() for c in (reader.fieldnames or [])]
                is_new_format = "record_type" in fieldnames
                for row in reader:
                    name = (row.get("name") or "").strip()
                    if not name:
                        continue
                    if is_new_format:
                        record_type = (row.get("record_type") or "").strip().lower()
                        if record_type == "meta":
                            try:
                                temp_frames[name] = self._normalize_path_reference_frame(
                                    PathReferenceFrame(
                                        mode=row.get("coord_mode", PATH_COORD_MODE_FIXED_ORIGIN),
                                        origin_key=row.get("origin_key", ""),
                                        origin_label=row.get("origin_label", ""),
                                        origin_x_m=float(row.get("origin_x", "0") or 0.0),
                                        origin_y_m=float(row.get("origin_y", "0") or 0.0),
                                        origin_z_m=float(row.get("origin_z", "0") or 0.0),
                                    )
                                )
                            except (TypeError, ValueError):
                                temp_frames[name] = PathReferenceFrame()
                        elif record_type == "point":
                            try:
                                idx = int(row.get("index", "0"))
                                x = float(row.get("x", ""))
                                y = float(row.get("y", ""))
                            except (TypeError, ValueError):
                                continue
                            temp_points.setdefault(name, []).append((idx, x, y))
                        elif record_type == "segment":
                            try:
                                idx = int(row.get("index", "0"))
                                start_idx = int(row.get("start_idx", "0"))
                                end_idx = int(row.get("end_idx", "0"))
                                speed_sign = self._normalize_speed_sign(row.get("speed_sign", "1"))
                                rcs_start = self._parse_bool_flag(row.get("rcs_start", "0"))
                            except (TypeError, ValueError):
                                continue
                            seg_range = self._normalize_segment_range(
                                (
                                    start_idx,
                                    end_idx,
                                    speed_sign,
                                    rcs_start,
                                    row.get("speed_mps", DEFAULT_SEGMENT_SPEED_MPS),
                                    row.get("accel_dist", DEFAULT_ACCEL_DIST_M),
                                    row.get("decel_dist", DEFAULT_DECEL_DIST_M),
                                )
                            )
                            temp_ranges.setdefault(name, []).append((idx, seg_range))
                        continue

                    try:
                        idx = int(row.get("index", "0"))
                        x = float(row.get("x", ""))
                        y = float(row.get("y", ""))
                    except (TypeError, ValueError):
                        continue
                    temp_points.setdefault(name, []).append((idx, x, y))

            for name, points in temp_points.items():
                points.sort(key=lambda item: item[0])
                self._preset_paths[name] = [(x, y) for _, x, y in points]
                self._preset_ranges[name] = []
                self._preset_path_frames[name] = self._resolve_path_reference_frame(
                    temp_frames.get(name)
                )

            for name, segments in temp_ranges.items():
                if name not in self._preset_paths:
                    continue
                segments.sort(key=lambda item: item[0])
                self._preset_ranges[name] = [
                    seg_range
                    for _, seg_range in segments
                    if seg_range[1] > seg_range[0]
                ]

            if self._preset_paths:
                self._log(f"已加载预设轨迹: {len(self._preset_paths)}个")
        except Exception as e:
            self._log(f"预设轨迹加载失败: {e}")

    @staticmethod
    def _parse_bool_flag(value: Any) -> bool:
        text = str(value).strip().lower()
        return text in {"1", "true", "yes", "y", "on", "t"}

    @staticmethod
    def _normalize_speed_sign(value: Any) -> int:
        try:
            return -1 if float(value) < 0 else 1
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _normalize_positive_float(value: Any, default: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(result):
            return float(default)
        return max(0.0, result)

    @staticmethod
    def _normalize_speed_mps(value: Any, default: float = DEFAULT_SEGMENT_SPEED_MPS) -> float:
        speed = MainWindow._normalize_positive_float(value, default)
        return max(MIN_SEGMENT_SPEED_MPS, speed)

    @staticmethod
    def _normalize_segment_range(value: Tuple[Any, ...]) -> SegmentRange:
        start_idx = int(value[0])
        end_idx = int(value[1])
        speed_sign = MainWindow._normalize_speed_sign(value[2] if len(value) >= 3 else 1)
        if len(value) >= 4 and isinstance(value[3], str):
            rcs_start = MainWindow._parse_bool_flag(value[3])
        else:
            rcs_start = bool(value[3]) if len(value) >= 4 else False
        speed_mps = MainWindow._normalize_speed_mps(
            value[4] if len(value) >= 5 else DEFAULT_SEGMENT_SPEED_MPS
        )
        accel_dist = MainWindow._normalize_positive_float(
            value[5] if len(value) >= 6 else DEFAULT_ACCEL_DIST_M,
            DEFAULT_ACCEL_DIST_M,
        )
        decel_dist = MainWindow._normalize_positive_float(
            value[6] if len(value) >= 7 else DEFAULT_DECEL_DIST_M,
            DEFAULT_DECEL_DIST_M,
        )
        return (
            start_idx,
            end_idx,
            speed_sign,
            rcs_start,
            speed_mps,
            accel_dist,
            decel_dist,
        )

    def _apply_forward_straight_rcs_flags(
        self,
        segment_ranges: List[SegmentRange],
        path_points: List[Tuple[float, float]],
    ) -> List[SegmentRange]:
        if self._radial_measurement_spec is not None:
            # 星型测量：强制覆盖所有带“星型角度_第N次”任务名的前进直线段；
            # 过渡、对齐短直线和倒车段没有“第N次”任务名，不纳入 RCS 采集。
            out: List[SegmentRange] = []
            names = list(self._planned_range_task_names or [])
            for si, seg in enumerate(segment_ranges):
                (
                    start_idx,
                    end_idx,
                    speed_sign,
                    rcs_start,
                    cruise_speed_mps,
                    accel_dist,
                    decel_dist,
                ) = seg
                pts = path_points[int(start_idx) : int(end_idx) + 1]
                name = (
                    str(names[si]).strip()
                    if si < len(names) and names[si]
                    else ""
                )
                forward_measure_line = (
                    int(speed_sign) > 0
                    and len(pts) >= 2
                    and self._is_radial_forward_rcs_task_name(name)
                )
                out.append(
                    (
                        start_idx,
                        end_idx,
                        speed_sign,
                        bool(rcs_start) or bool(forward_measure_line),
                        cruise_speed_mps,
                        accel_dist,
                        decel_dist,
                    )
                )
            return out
        chk = getattr(self, "chk_rcs_all_forward_straight", None)
        if chk is None or not chk.isChecked():
            return segment_ranges
        kinds = getattr(self, "_planned_segment_kinds", None) or []
        out: List[SegmentRange] = []
        for si, seg in enumerate(segment_ranges):
            (
                start_idx,
                end_idx,
                speed_sign,
                rcs_start,
                cruise_speed_mps,
                accel_dist,
                decel_dist,
            ) = seg
            forward = int(speed_sign) > 0
            pts = path_points[int(start_idx) : int(end_idx) + 1]
            kind = str(kinds[si]).strip().lower() if si < len(kinds) else ""
            if kind == "circle":
                straight = False
            elif kind == "line":
                straight = len(pts) >= 2
            else:
                straight = len(pts) >= 2 and MainWindow._is_nearly_straight_segment(pts)
            use_rcs = True if (forward and straight) else bool(rcs_start)
            out.append(
                (
                    start_idx,
                    end_idx,
                    speed_sign,
                    use_rcs,
                    cruise_speed_mps,
                    accel_dist,
                    decel_dist,
                )
            )
        return out

    @staticmethod
    def _should_use_segment_ranges(ranges: List[SegmentRange]) -> bool:
        return len(ranges) > 0

    @staticmethod
    def _compute_path_cumulative_lengths(
        points: List[Tuple[float, float]],
    ) -> List[float]:
        if not points:
            return []
        s_cum = [0.0]
        total_len = 0.0
        for i in range(len(points) - 1):
            dx = float(points[i + 1][0]) - float(points[i][0])
            dy = float(points[i + 1][1]) - float(points[i][1])
            total_len += math.hypot(dx, dy)
            s_cum.append(total_len)
        return s_cum

    @staticmethod
    def _compress_path_points(
        points: List[Tuple[float, float]],
        min_spacing_m: float = PATH_DUPLICATE_EPS_M,
    ) -> List[Tuple[float, float]]:
        if not points:
            return []
        min_spacing2 = max(0.0, float(min_spacing_m)) ** 2
        compacted: List[Tuple[float, float]] = [
            (float(points[0][0]), float(points[0][1]))
        ]
        for px, py in points[1:]:
            x = float(px)
            y = float(py)
            dx = x - compacted[-1][0]
            dy = y - compacted[-1][1]
            if dx * dx + dy * dy > min_spacing2:
                compacted.append((x, y))
        if len(compacted) == 1 and len(points) >= 2:
            last = (float(points[-1][0]), float(points[-1][1]))
            dx = last[0] - compacted[0][0]
            dy = last[1] - compacted[0][1]
            if dx * dx + dy * dy > 1e-12:
                compacted.append(last)
        return compacted

    @staticmethod
    def _choose_path_resample_step(
        points: List[Tuple[float, float]],
        ranges: Optional[List[SegmentRange]] = None,
    ) -> float:
        seg_lengths = [
            math.hypot(
                float(points[i + 1][0]) - float(points[i][0]),
                float(points[i + 1][1]) - float(points[i][1]),
            )
            for i in range(len(points) - 1)
        ]
        seg_lengths = [seg for seg in seg_lengths if seg > 1e-6]
        if not seg_lengths:
            return PATH_RESAMPLE_STEP_MIN_M

        step_m = float(
            np.clip(
                float(np.median(np.asarray(seg_lengths, dtype=float))),
                PATH_RESAMPLE_STEP_MIN_M,
                PATH_RESAMPLE_STEP_MAX_M,
            )
        )
        if not ranges:
            return step_m

        s_cum = MainWindow._compute_path_cumulative_lengths(points)
        boundary_s: List[float] = []
        last_idx = len(points) - 1
        for seg_range in ranges:
            start_idx = max(0, min(int(seg_range[0]), last_idx))
            end_idx = max(0, min(int(seg_range[1]), last_idx))
            if end_idx <= start_idx:
                continue
            boundary_s.append(float(s_cum[start_idx]))
            boundary_s.append(float(s_cum[end_idx]))
        boundary_s = sorted(set(boundary_s))
        min_gap = min(
            (boundary_s[i + 1] - boundary_s[i] for i in range(len(boundary_s) - 1)),
            default=0.0,
        )
        if min_gap > 1e-6:
            step_m = min(
                step_m,
                max(
                    PATH_RESAMPLE_STEP_FLOOR_M,
                    PATH_RESAMPLE_BOUNDARY_GAP_RATIO * min_gap,
                ),
            )
        return step_m

    @staticmethod
    def _resample_path_by_arclength(
        points: List[Tuple[float, float]],
        step_m: float,
    ) -> List[Tuple[float, float]]:
        compacted = MainWindow._compress_path_points(points, PATH_DUPLICATE_EPS_M)
        if len(compacted) < 2:
            return compacted

        s_cum = MainWindow._compute_path_cumulative_lengths(compacted)
        total_len = float(s_cum[-1]) if s_cum else 0.0
        if total_len <= 1e-6:
            return compacted

        step_m = max(PATH_RESAMPLE_STEP_FLOOR_M, float(step_m))
        target_s = [0.0]
        s = step_m
        while s < total_len - 1e-6:
            target_s.append(s)
            s += step_m
        if total_len > target_s[-1] + 1e-6:
            target_s.append(total_len)

        resampled: List[Tuple[float, float]] = []
        seg_idx = 0
        for s_val in target_s:
            while seg_idx < len(compacted) - 2 and s_cum[seg_idx + 1] < s_val - 1e-9:
                seg_idx += 1
            s0 = s_cum[seg_idx]
            s1 = s_cum[seg_idx + 1]
            x0, y0 = compacted[seg_idx]
            x1, y1 = compacted[seg_idx + 1]
            if s1 - s0 <= 1e-9:
                resampled.append((float(x0), float(y0)))
                continue
            t = max(0.0, min(1.0, (s_val - s0) / (s1 - s0)))
            resampled.append(
                (
                    float(x0 + (x1 - x0) * t),
                    float(y0 + (y1 - y0) * t),
                )
            )
        resampled[0] = compacted[0]
        resampled[-1] = compacted[-1]
        return resampled

    @staticmethod
    def _compute_local_turn_angle(
        points: List[Tuple[float, float]],
        idx: int,
    ) -> float:
        if idx <= 0 or idx >= len(points) - 1:
            return 0.0
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx]
        x2, y2 = points[idx + 1]
        h0 = math.atan2(y1 - y0, x1 - x0)
        h1 = math.atan2(y2 - y1, x2 - x1)
        d = h1 - h0
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        return abs(d)

    @staticmethod
    def _smooth_resampled_path(
        points: List[Tuple[float, float]],
        window_radius_m: float,
        blend: float,
        max_shift_m: float,
        corner_limit_deg: float = PATH_SMOOTH_CORNER_LIMIT_DEG,
    ) -> List[Tuple[float, float]]:
        if len(points) < 3:
            return list(points)

        s_cum = MainWindow._compute_path_cumulative_lengths(points)
        window_radius_m = max(PATH_RESAMPLE_STEP_FLOOR_M, float(window_radius_m))
        blend = max(0.0, min(1.0, float(blend)))
        max_shift_m = max(0.0, float(max_shift_m))
        corner_limit_rad = math.radians(max(1.0, float(corner_limit_deg)))

        smoothed = [points[0]]
        for idx in range(1, len(points) - 1):
            cur_s = s_cum[idx]
            start_idx = idx
            while start_idx > 0 and cur_s - s_cum[start_idx - 1] <= window_radius_m:
                start_idx -= 1
            end_idx = idx
            while end_idx + 1 < len(points) and s_cum[end_idx + 1] - cur_s <= window_radius_m:
                end_idx += 1

            window = points[start_idx : end_idx + 1]
            avg_x = sum(float(px) for px, _ in window) / len(window)
            avg_y = sum(float(py) for _, py in window) / len(window)
            cur_x = float(points[idx][0])
            cur_y = float(points[idx][1])

            turn_angle = MainWindow._compute_local_turn_angle(points, idx)
            corner_gain = max(0.0, 1.0 - turn_angle / corner_limit_rad)
            effective_blend = blend * corner_gain

            dx = (avg_x - cur_x) * effective_blend
            dy = (avg_y - cur_y) * effective_blend
            shift = math.hypot(dx, dy)
            if shift > max_shift_m and shift > 1e-9:
                scale = max_shift_m / shift
                dx *= scale
                dy *= scale
            smoothed.append((cur_x + dx, cur_y + dy))
        smoothed.append(points[-1])
        smoothed[0] = points[0]
        smoothed[-1] = points[-1]
        return smoothed

    @staticmethod
    def _find_index_for_path_s(
        s_cum: List[float],
        target_s: float,
        start_idx: int = 0,
    ) -> int:
        if not s_cum:
            return 0
        start_idx = max(0, min(int(start_idx), len(s_cum) - 1))
        target_s = max(0.0, min(float(target_s), float(s_cum[-1])))
        for idx in range(start_idx, len(s_cum)):
            if s_cum[idx] >= target_s:
                if idx == start_idx:
                    return idx
                prev_idx = idx - 1
                if prev_idx < start_idx:
                    return idx
                if abs(s_cum[prev_idx] - target_s) <= abs(s_cum[idx] - target_s):
                    return prev_idx
                return idx
        return len(s_cum) - 1

    def _stabilize_local_path_for_tracking(
        self,
        local_points: List[Tuple[float, float]],
        ranges: Optional[List[SegmentRange]] = None,
        segment_kinds: Optional[List[str]] = None,
    ) -> Tuple[List[Tuple[float, float]], List[SegmentRange], Optional[List[str]], Dict[str, Any]]:
        original_points = [(float(x), float(y)) for x, y in local_points]
        normalized_ranges = [
            self._normalize_segment_range(seg_range) for seg_range in (ranges or [])
        ]
        stats: Dict[str, Any] = {
            "input_points": len(original_points),
            "cleaned_points": len(original_points),
            "output_points": len(original_points),
            "total_length_m": 0.0,
            "resample_step_m": 0.0,
            "changed": False,
        }
        if len(original_points) < 2:
            return original_points, normalized_ranges, self._remap_segment_kinds_list(
                segment_kinds, normalized_ranges
            ), stats

        original_s = self._compute_path_cumulative_lengths(original_points)
        total_len = float(original_s[-1]) if original_s else 0.0
        stats["total_length_m"] = total_len
        if total_len <= 1e-6:
            return original_points, normalized_ranges, self._remap_segment_kinds_list(
                segment_kinds, normalized_ranges
            ), stats

        compacted = self._compress_path_points(original_points, PATH_DUPLICATE_EPS_M)
        stats["cleaned_points"] = len(compacted)
        if len(compacted) < 2:
            return original_points, normalized_ranges, self._remap_segment_kinds_list(
                segment_kinds, normalized_ranges
            ), stats

        step_m = self._choose_path_resample_step(original_points, normalized_ranges)
        stats["resample_step_m"] = float(step_m)
        processed_points = self._resample_path_by_arclength(compacted, step_m)
        if len(processed_points) >= 3:
            processed_points = self._smooth_resampled_path(
                processed_points,
                window_radius_m=max(
                    PATH_SMOOTH_WINDOW_MIN_M,
                    min(PATH_SMOOTH_WINDOW_MAX_M, PATH_SMOOTH_WINDOW_SCALE * step_m),
                ),
                blend=PATH_SMOOTH_BLEND,
                max_shift_m=max(
                    PATH_SMOOTH_MAX_SHIFT_MIN_M,
                    min(PATH_SMOOTH_MAX_SHIFT_MAX_M, PATH_SMOOTH_MAX_SHIFT_SCALE * step_m),
                ),
            )
        processed_points = self._compress_path_points(
            processed_points,
            min_spacing_m=max(PATH_DUPLICATE_EPS_M, 0.25 * step_m),
        )
        if len(processed_points) < 2:
            processed_points = [compacted[0], compacted[-1]]
        processed_points[0] = compacted[0]
        processed_points[-1] = compacted[-1]

        processed_s = self._compute_path_cumulative_lengths(processed_points)
        remapped_ranges: List[SegmentRange] = []
        remapped_kinds: List[str] = []
        kinds_parallel = bool(segment_kinds) and len(segment_kinds) == len(normalized_ranges)
        prev_end_s: Optional[float] = None
        prev_end_idx: Optional[int] = None
        last_input_idx = len(original_points) - 1
        for range_idx, seg_range in enumerate(normalized_ranges):
            start_idx = max(0, min(int(seg_range[0]), last_input_idx))
            end_idx = max(0, min(int(seg_range[1]), last_input_idx))
            if end_idx <= start_idx:
                continue

            start_s = float(original_s[start_idx])
            end_s = float(original_s[end_idx])
            if prev_end_s is not None and abs(start_s - prev_end_s) <= 1e-6 and prev_end_idx is not None:
                new_start_idx = prev_end_idx
            else:
                new_start_idx = self._find_index_for_path_s(processed_s, start_s)
            new_end_idx = self._find_index_for_path_s(
                processed_s,
                end_s,
                start_idx=new_start_idx,
            )
            if new_end_idx <= new_start_idx:
                new_end_idx = min(len(processed_points) - 1, new_start_idx + 1)
            if new_end_idx <= new_start_idx:
                continue
            remapped_ranges.append(
                (
                    new_start_idx,
                    new_end_idx,
                    seg_range[2],
                    seg_range[3],
                    seg_range[4],
                    seg_range[5],
                    seg_range[6],
                )
            )
            if kinds_parallel:
                k0 = str(segment_kinds[range_idx] or "line").strip().lower()
                remapped_kinds.append(k0 if k0 in ("line", "circle") else "line")
            prev_end_s = end_s
            prev_end_idx = new_end_idx

        stats["output_points"] = len(processed_points)
        stats["changed"] = (
            stats["cleaned_points"] != stats["input_points"]
            or stats["output_points"] != stats["input_points"]
        )
        out_kinds: Optional[List[str]] = None
        if kinds_parallel and len(remapped_kinds) == len(remapped_ranges):
            out_kinds = remapped_kinds
        return processed_points, remapped_ranges, out_kinds, stats

    @staticmethod
    def _remap_segment_kinds_list(
        kinds: Optional[List[str]],
        ranges: List[SegmentRange],
    ) -> Optional[List[str]]:
        if not kinds or len(kinds) != len(ranges):
            return None
        out: List[str] = []
        for k in kinds:
            k0 = str(k or "line").strip().lower()
            out.append(k0 if k0 in ("line", "circle") else "line")
        return out

    def _build_ranges_from_point_metadata(
        self,
        rows: List[
            Tuple[
                float,
                float,
                Optional[int],
                Optional[int],
                bool,
                Optional[float],
                Optional[float],
                Optional[float],
            ]
        ],
    ) -> List[SegmentRange]:
        if not rows:
            return []
        has_segment_id = any(seg is not None for _, _, seg, _, _, _, _, _ in rows)
        has_inline_meta = any(
            (seg_speed is not None)
            or rcs
            or (speed_mps is not None)
            or (accel_dist is not None)
            or (decel_dist is not None)
            for _, _, _, seg_speed, rcs, speed_mps, accel_dist, decel_dist in rows
        )
        if (not has_segment_id) and has_inline_meta and len(rows) >= 2:
            speed_sign = 1
            speed_mps = DEFAULT_SEGMENT_SPEED_MPS
            accel_dist = DEFAULT_ACCEL_DIST_M
            decel_dist = DEFAULT_DECEL_DIST_M
            for _, _, _, seg_speed, _, seg_speed_mps, seg_accel, seg_decel in rows:
                if seg_speed is not None:
                    speed_sign = self._normalize_speed_sign(seg_speed)
                    break
            for _, _, _, _, _, seg_speed_mps, seg_accel, seg_decel in rows:
                if seg_speed_mps is not None:
                    speed_mps = self._normalize_speed_mps(seg_speed_mps)
                    break
            for _, _, _, _, _, _, seg_accel, _ in rows:
                if seg_accel is not None:
                    accel_dist = self._normalize_positive_float(seg_accel, DEFAULT_ACCEL_DIST_M)
                    break
            for _, _, _, _, _, _, _, seg_decel in rows:
                if seg_decel is not None:
                    decel_dist = self._normalize_positive_float(seg_decel, DEFAULT_DECEL_DIST_M)
                    break
            rcs_start = any(rcs for _, _, _, _, rcs, _, _, _ in rows)
            return [
                (
                    0,
                    len(rows) - 1,
                    speed_sign,
                    rcs_start,
                    speed_mps,
                    accel_dist,
                    decel_dist,
                )
            ]
        if not has_segment_id:
            return []

        resolved_segment_ids: List[int] = []
        cur_seg = 1
        for _, _, seg, _, _, _, _, _ in rows:
            if seg is not None:
                cur_seg = int(seg)
            resolved_segment_ids.append(cur_seg)

        ranges: List[SegmentRange] = []
        start = 0
        while start < len(rows):
            seg_id = resolved_segment_ids[start]
            end = start
            while end + 1 < len(rows) and resolved_segment_ids[end + 1] == seg_id:
                end += 1

            if end > start:
                speed_sign = 1
                speed_mps = DEFAULT_SEGMENT_SPEED_MPS
                accel_dist = DEFAULT_ACCEL_DIST_M
                decel_dist = DEFAULT_DECEL_DIST_M
                for _, _, _, seg_speed, _, seg_speed_mps, seg_accel, seg_decel in rows[start : end + 1]:
                    if seg_speed is not None:
                        speed_sign = self._normalize_speed_sign(seg_speed)
                        break
                for _, _, _, _, _, seg_speed_mps, seg_accel, seg_decel in rows[start : end + 1]:
                    if seg_speed_mps is not None:
                        speed_mps = self._normalize_speed_mps(seg_speed_mps)
                        break
                for _, _, _, _, _, _, seg_accel, _ in rows[start : end + 1]:
                    if seg_accel is not None:
                        accel_dist = self._normalize_positive_float(seg_accel, DEFAULT_ACCEL_DIST_M)
                        break
                for _, _, _, _, _, _, _, seg_decel in rows[start : end + 1]:
                    if seg_decel is not None:
                        decel_dist = self._normalize_positive_float(seg_decel, DEFAULT_DECEL_DIST_M)
                        break
                rcs_start = any(rcs for _, _, _, _, rcs, _, _, _ in rows[start : end + 1])
                ranges.append(
                    (
                        start,
                        end,
                        speed_sign,
                        rcs_start,
                        speed_mps,
                        accel_dist,
                        decel_dist,
                    )
                )
            start = end + 1

        return ranges

    def _parse_trajectory_file(self, filename: str) -> Tuple[List[Tuple[float, float]], List[SegmentRange]]:
        with open(filename, "r", encoding="utf-8-sig") as f:
            lines = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
        if not lines:
            raise ValueError("文件为空。")

        first_tokens = [p.strip() for p in lines[0].replace("\t", ",").split(",")]
        is_header = True
        if len(first_tokens) >= 2:
            try:
                float(first_tokens[0])
                float(first_tokens[1])
                is_header = False
            except ValueError:
                is_header = True

        rows: List[
            Tuple[
                float,
                float,
                Optional[int],
                Optional[int],
                bool,
                Optional[float],
                Optional[float],
                Optional[float],
            ]
        ] = []
        if is_header:
            reader = csv.DictReader(lines)
            if reader.fieldnames is None:
                raise ValueError("无法识别CSV表头。")

            key_map = {str(name).strip().lower(): str(name) for name in reader.fieldnames}

            def pick(*candidates: str) -> Optional[str]:
                for c in candidates:
                    if c in key_map:
                        return key_map[c]
                return None

            x_key = pick("x", "xr", "x_m", "local_x")
            y_key = pick("y", "yr", "y_m", "local_y")
            seg_key = pick("segment", "segment_id", "seg")
            speed_key = pick("speed_sign", "dir", "direction")
            rcs_key = pick("rcs_start", "record_rcs", "rcs_record")
            speed_mps_key = pick("speed_mps", "speed", "velocity", "v")
            accel_key = pick("accel_dist", "accel_dist_m", "accel_m", "accel")
            decel_key = pick("decel_dist", "decel_dist_m", "decel_m", "decel")

            if x_key is None or y_key is None:
                raise ValueError("CSV需要至少包含 x,y 两列。")

            for row in reader:
                try:
                    x = float(str(row.get(x_key, "")).strip())
                    y = float(str(row.get(y_key, "")).strip())
                except ValueError:
                    continue
                seg_val: Optional[int] = None
                if seg_key:
                    raw_seg = str(row.get(seg_key, "")).strip()
                    if raw_seg:
                        try:
                            seg_val = int(float(raw_seg))
                        except ValueError:
                            seg_val = None
                speed_sign: Optional[int] = None
                if speed_key:
                    raw_speed = str(row.get(speed_key, "")).strip()
                    if raw_speed:
                        speed_sign = self._normalize_speed_sign(raw_speed)
                rcs_start = self._parse_bool_flag(row.get(rcs_key, "0")) if rcs_key else False
                speed_mps: Optional[float] = None
                accel_dist: Optional[float] = None
                decel_dist: Optional[float] = None
                if speed_mps_key:
                    raw_speed_mps = str(row.get(speed_mps_key, "")).strip()
                    if raw_speed_mps:
                        speed_mps = self._normalize_speed_mps(raw_speed_mps)
                if accel_key:
                    raw_accel = str(row.get(accel_key, "")).strip()
                    if raw_accel:
                        accel_dist = self._normalize_positive_float(raw_accel, DEFAULT_ACCEL_DIST_M)
                if decel_key:
                    raw_decel = str(row.get(decel_key, "")).strip()
                    if raw_decel:
                        decel_dist = self._normalize_positive_float(raw_decel, DEFAULT_DECEL_DIST_M)
                rows.append((x, y, seg_val, speed_sign, rcs_start, speed_mps, accel_dist, decel_dist))
        else:
            for line in lines:
                parts = [p for p in line.replace(",", " ").split() if p]
                if len(parts) < 2:
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                except ValueError:
                    continue
                seg_val: Optional[int] = None
                speed_sign: Optional[int] = None
                rcs_start = False
                speed_mps: Optional[float] = None
                accel_dist: Optional[float] = None
                decel_dist: Optional[float] = None
                if len(parts) >= 3:
                    try:
                        seg_val = int(float(parts[2]))
                    except ValueError:
                        seg_val = None
                if len(parts) >= 4:
                    speed_sign = self._normalize_speed_sign(parts[3])
                if len(parts) >= 5:
                    rcs_start = self._parse_bool_flag(parts[4])
                if len(parts) >= 6:
                    speed_mps = self._normalize_speed_mps(parts[5])
                if len(parts) >= 7:
                    accel_dist = self._normalize_positive_float(parts[6], DEFAULT_ACCEL_DIST_M)
                if len(parts) >= 8:
                    decel_dist = self._normalize_positive_float(parts[7], DEFAULT_DECEL_DIST_M)
                rows.append((x, y, seg_val, speed_sign, rcs_start, speed_mps, accel_dist, decel_dist))

        if len(rows) < 2:
            raise ValueError("轨迹点数量不足（至少需要 2 个点）。")

        local_points = [(x, y) for x, y, _, _, _, _, _, _ in rows]
        ranges = self._build_ranges_from_point_metadata(rows)
        return local_points, ranges

    def _open_traj_planner(self) -> None:
        dialog = TrajectoryPlannerDialog(
            self,
            default_dist=self._traj_default_dist,
            default_radius=self._traj_default_radius,
            default_angle=self._traj_default_angle,
            default_circle_direction=self._traj_default_circle_direction,
            default_line_speed=self._traj_default_line_speed,
            default_circle_speed=self._traj_default_circle_speed,
            default_accel_dist=self._traj_default_accel_dist,
            default_decel_dist=self._traj_default_decel_dist,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            self._log("轨迹规划已取消")
            return
        self._traj_default_dist = float(dialog.line_dist.value())
        self._traj_default_radius = float(dialog.circle_radius.value())
        self._traj_default_angle = float(dialog.circle_angle.value())
        self._traj_default_circle_direction = (
            "ccw" if dialog.circle_dir.currentIndex() == 0 else "cw"
        )
        self._traj_default_line_speed = float(dialog.line_speed.value())
        self._traj_default_circle_speed = float(dialog.circle_speed.value())
        self._traj_default_accel_dist = float(dialog.line_accel_dist.value())
        self._traj_default_decel_dist = float(dialog.line_decel_dist.value())
        self._path_speed = self._traj_default_line_speed
        segments = dialog.get_segments()
        if not segments:
            QtWidgets.QMessageBox.information(self, "无轨迹段", "请先添加直线或圆弧段。")
            self._log("轨迹规划失败: 未添加轨迹段")
            return
        frame = self._get_path_anchor_reference_frame()
        local_points, ranges, plan_kinds = self._build_planned_path(segments)
        self._set_loaded_task_sequence([], 0)
        self._clear_radial_measurement_spec()
        self._planned_ranges = ranges if self._should_use_segment_ranges(ranges) else None
        self._planned_segment_kinds = (
            list(plan_kinds)
            if (
                self._planned_ranges
                and plan_kinds
                and len(plan_kinds) == len(self._planned_ranges)
            )
            else None
        )
        self._planned_range_task_names = []
        self._log(f"轨迹规划生成: 段数={len(segments)} 点数={len(local_points)}")
        if len(local_points) < 2:
            QtWidgets.QMessageBox.information(self, "轨迹无效", "规划轨迹点数量不足。")
            self._log("轨迹规划失败: 点数不足")
            return
        preset_name = self._prompt_preset_name()
        if preset_name:
            self._save_preset_path(
                preset_name,
                local_points,
                ranges,
                frame=frame,
                segment_kinds=self._planned_segment_kinds,
            )
            self._set_current_path_name(preset_name)
        else:
            self._log("轨迹规划未保存为预设轨迹")
            self._set_current_path_name("规划轨迹")
        self._apply_path_reference_frame_to_controls(frame)
        self._apply_planned_local_points(local_points, frame=frame)

    def _apply_planned_local_points(
        self,
        local_points: List[Tuple[float, float]],
        frame: Optional[PathReferenceFrame] = None,
    ) -> bool:
        if len(local_points) < 2:
            QtWidgets.QMessageBox.information(self, "轨迹无效", "规划轨迹点数量不足。")
            self._log("轨迹规划失败: 点数不足")
            return False

        stabilized_points, stabilized_ranges, stabilized_kinds, stab_stats = (
            self._stabilize_local_path_for_tracking(
                local_points,
                self._planned_ranges,
                getattr(self, "_planned_segment_kinds", None),
            )
        )
        local_points = stabilized_points
        self._planned_ranges = (
            stabilized_ranges if self._should_use_segment_ranges(stabilized_ranges) else None
        )
        if self._planned_ranges and stabilized_kinds and len(stabilized_kinds) == len(
            self._planned_ranges
        ):
            self._planned_segment_kinds = list(stabilized_kinds)
        else:
            self._planned_segment_kinds = None
        if stab_stats["total_length_m"] > 1e-6:
            self._log(
                "轨迹稳定化: "
                f"点数 {stab_stats['input_points']} -> {stab_stats['output_points']} "
                f"(去重后={stab_stats['cleaned_points']}) | "
                f"弧长={stab_stats['total_length_m']:.2f}m | "
                f"重采样步长≈{stab_stats['resample_step_m']:.2f}m"
            )

        if frame is None:
            resolved_frame = self._resolve_path_reference_frame(
                self._path_anchor_frame_from_ui()
            )
        else:
            resolved_frame = self._resolve_path_reference_frame(frame)
        if not resolved_frame.is_fixed_origin():
            self._log("轨迹应用失败: 锚点帧无效")
            return False

        pose = self._build_virtual_preview_pose(
            x_m=resolved_frame.origin_x_m,
            y_m=resolved_frame.origin_y_m,
            z_m=resolved_frame.origin_z_m,
        )
        # 局部→全局只按「轨迹锚点」平移，绝不使用小车位置；无 INS 时标记虚拟预览，便于位姿就绪后仅刷新 UI。
        using_virtual_pose = get_robot_pose() is None

        global_points = self._apply_loaded_path_preview(
            local_points,
            pose,
            using_virtual_pose=using_virtual_pose,
            frame=resolved_frame,
        )
        self._log(
            f"轨迹规划应用完成: 全局点数={len(global_points)} | "
            f"参考系={self._format_path_reference_frame(resolved_frame)}"
        )
        return True

    def _reanchor_loaded_path(self) -> bool:
        if self._radial_measurement_spec is not None:
            pose = get_robot_pose()
            if pose is None:
                detail = (
                    "当前星型测量轨迹仅完成虚拟预览，无法直接执行。"
                    "请先获取实时 GNSS / INS 位姿后再执行轨迹。"
                )
                QtWidgets.QMessageBox.warning(self, "定位无效", detail)
                self._log("星型测量重新锚定失败: 无有效位姿")
                return False
            return self._apply_radial_measurement_spec(
                self._radial_measurement_spec,
                anchor_pose=pose,
                announce_virtual_preview=False,
            )

        local_points = self.loaded_path_local_points
        frame = self._resolve_path_reference_frame(self._path_anchor_frame_from_ui())
        if len(local_points) < 2:
            QtWidgets.QMessageBox.warning(
                self,
                "轨迹无效",
                "缺少本地轨迹点，无法重新锚定，请重新规划或加载轨迹。",
            )
            self._log("轨迹重新锚定失败: 缺少本地轨迹点")
            return False
        if not frame.is_fixed_origin():
            QtWidgets.QMessageBox.warning(
                self,
                "轨迹锚点无效",
                "请在主界面输入轨迹锚点的平面坐标 x、y（米），再重新锚定轨迹。",
            )
            self._log("轨迹重新锚定失败: 锚点无效")
            return False

        pose = self._build_virtual_preview_pose(
            x_m=frame.origin_x_m,
            y_m=frame.origin_y_m,
            z_m=frame.origin_z_m,
        )
        global_points = self._apply_loaded_path_preview(
            local_points,
            pose,
            using_virtual_pose=(get_robot_pose() is None),
            frame=frame,
        )
        self._log(
            f"轨迹按校准平面系与当前锚点下发: {self._format_path_reference_frame(frame)} | "
            f"全局点数={len(global_points)}"
        )
        return True

    def _build_planned_path(
        self,
        segments: List[dict],
    ) -> Tuple[List[Tuple[float, float]], List[SegmentRange], List[str]]:
        line_step = 1.2
        circle_step = 0.12
        points: List[Tuple[float, float]] = [(0.0, 0.0)]
        ranges: List[SegmentRange] = []
        segment_kinds: List[str] = []
        x, y = 0.0, 0.0
        heading = math.pi * 0.5

        def add_range(
            start_idx: int,
            end_idx: int,
            speed_sign: int,
            speed_mps: Any,
            accel_dist: Any,
            decel_dist: Any,
            rcs_start: bool = False,
            segment_kind: str = "line",
        ) -> None:
            if end_idx - start_idx >= 1:
                ranges.append(
                    (
                        start_idx,
                        end_idx,
                        self._normalize_speed_sign(speed_sign),
                        bool(rcs_start),
                        self._normalize_speed_mps(speed_mps),
                        self._normalize_positive_float(accel_dist, DEFAULT_ACCEL_DIST_M),
                        self._normalize_positive_float(decel_dist, DEFAULT_DECEL_DIST_M),
                    )
                )
                sk = str(segment_kind or "line").strip().lower()
                segment_kinds.append(sk if sk in ("line", "circle") else "line")

        seg_idx = 0
        while seg_idx < len(segments):
            seg = segments[seg_idx]
            if seg["type"] == "line":
                dist = float(seg.get("distance", 0.0))
                if abs(dist) < 1e-3:
                    seg_idx += 1
                    continue
                direction_sign = -1 if int(seg.get("direction_sign", 1)) < 0 else 1

                def append_line_with_direction(
                    speed_sign: int,
                    rcs_start: bool = False,
                ) -> None:
                    nonlocal x, y, heading
                    start_idx = len(points) - 1
                    heading_before = heading
                    x, y, heading = self._append_line_segment_global_y(
                        points,
                        x,
                        y,
                        dist * speed_sign,
                    )
                    end_idx = len(points) - 1
                    add_range(
                        start_idx,
                        end_idx,
                        speed_sign,
                        seg.get("speed_mps", DEFAULT_SEGMENT_SPEED_MPS),
                        seg.get("accel_dist", DEFAULT_ACCEL_DIST_M),
                        seg.get("decel_dist", DEFAULT_DECEL_DIST_M),
                        rcs_start,
                        segment_kind="line",
                    )
                    # 倒车不改变车头朝向；仅路径切向反向
                    if speed_sign < 0:
                        heading = heading_before

                if not seg.get("round_trip"):
                    append_line_with_direction(
                        direction_sign,
                        bool(seg.get("rcs_start")),
                    )
                    seg_idx += 1
                    continue
                round_trip_count = max(1, int(seg.get("round_trip_count", 1)))
                for i in range(round_trip_count):
                    append_line_with_direction(
                        direction_sign,
                        bool(seg.get("rcs_start")) and i == 0,
                    )
                    append_line_with_direction(-direction_sign, False)
            elif seg["type"] == "circle":
                radius = float(seg.get("radius", 0.0))
                angle = float(seg.get("angle", 0.0))
                direction = seg.get("direction", "ccw")
                if radius <= 1e-3 or abs(angle) < 1e-3:
                    seg_idx += 1
                    continue
                start_idx = len(points) - 1
                x, y, heading = self._append_circle_segment(
                    points,
                    x,
                    y,
                    heading,
                    radius,
                    angle,
                    direction,
                    circle_step,
                )
                end_idx = len(points) - 1
                add_range(
                    start_idx,
                    end_idx,
                    1,
                    seg.get("speed_mps", DEFAULT_SEGMENT_SPEED_MPS),
                    seg.get("accel_dist", DEFAULT_ACCEL_DIST_M),
                    seg.get("decel_dist", DEFAULT_DECEL_DIST_M),
                    bool(seg.get("rcs_start")),
                    segment_kind="circle",
                )
            seg_idx += 1
        return points, ranges, segment_kinds

    def _build_line_circle_transition(
        self,
        x: float,
        y: float,
        heading: float,
        line_dist: float,
        radius: float,
        angle_deg: float,
        direction: str,
        step: float,
    ) -> Optional[Dict[str, Any]]:
        line_dist = float(line_dist)
        radius = float(radius)
        total_phi = math.radians(abs(angle_deg))
        if line_dist <= max(3.0 * step, 0.8):
            return None
        if radius <= 1e-3 or total_phi < math.radians(10.0):
            return None

        max_beta = min(
            math.radians(25.0),
            total_phi * 0.35,
            0.8 * line_dist / max(radius, 1e-6),
        )
        beta = max_beta
        if beta < math.radians(5.0):
            return None

        remaining_angle_deg = abs(angle_deg) - math.degrees(beta)
        if remaining_angle_deg < 3.0:
            return None

        line_trim = min(0.35 * line_dist, 0.8 * radius * beta)
        line_trim = max(2.0 * step, line_trim)
        if line_trim >= line_dist - step:
            return None

        turn_sign = -1.0 if str(direction).lower() == "cw" else 1.0
        dir_x = math.cos(heading)
        dir_y = math.sin(heading)
        left_x = -dir_y
        left_y = dir_x

        junction_x = x + line_dist * dir_x
        junction_y = y + line_dist * dir_y
        start_x = junction_x - line_trim * dir_x
        start_y = junction_y - line_trim * dir_y

        entry_x = (
            junction_x
            + radius * math.sin(beta) * dir_x
            + turn_sign * radius * (1.0 - math.cos(beta)) * left_x
        )
        entry_y = (
            junction_y
            + radius * math.sin(beta) * dir_y
            + turn_sign * radius * (1.0 - math.cos(beta)) * left_y
        )
        end_heading = self._wrap_angle(heading + turn_sign * beta)
        end_dir_x = math.cos(end_heading)
        end_dir_y = math.sin(end_heading)

        chord = math.hypot(entry_x - start_x, entry_y - start_y)
        if chord <= step:
            return None

        start_handle = min(max(1.5 * step, 0.75 * line_trim), 0.6 * chord)
        end_handle = min(max(1.5 * step, 0.55 * radius * beta), 0.6 * chord)
        ctrl1_x = start_x + start_handle * dir_x
        ctrl1_y = start_y + start_handle * dir_y
        ctrl2_x = entry_x - end_handle * end_dir_x
        ctrl2_y = entry_y - end_handle * end_dir_y

        curve_len_est = line_trim + radius * beta
        sample_count = max(4, int(curve_len_est / step) + 1)
        transition_points: List[Tuple[float, float]] = []
        for i in range(1, sample_count + 1):
            u = i / sample_count
            one_minus_u = 1.0 - u
            px = (
                (one_minus_u ** 3) * start_x
                + 3.0 * (one_minus_u ** 2) * u * ctrl1_x
                + 3.0 * one_minus_u * (u ** 2) * ctrl2_x
                + (u ** 3) * entry_x
            )
            py = (
                (one_minus_u ** 3) * start_y
                + 3.0 * (one_minus_u ** 2) * u * ctrl1_y
                + 3.0 * one_minus_u * (u ** 2) * ctrl2_y
                + (u ** 3) * entry_y
            )
            transition_points.append((px, py))

        return {
            "line_dist": line_dist - line_trim,
            "transition_points": transition_points,
            "end_x": entry_x,
            "end_y": entry_y,
            "end_heading": end_heading,
            "remaining_angle_deg": remaining_angle_deg,
        }

    def _append_line_segment_global_y(
        self,
        points: List[Tuple[float, float]],
        x: float,
        y: float,
        dy: float,
    ) -> Tuple[float, float, float]:
        """在规划系中沿全局 +Y 平移（dy 可正可负），与车头切向无关。"""
        n = max(1, STRAIGHT_SEGMENT_POINT_COUNT - 1)
        x0, y0 = x, y
        for i in range(1, n + 1):
            t = i / n
            points.append((x0, y0 + dy * t))
        return x0, y0 + dy, math.pi * 0.5

    def _append_line_segment(
        self,
        points: List[Tuple[float, float]],
        x: float,
        y: float,
        heading: float,
        dist: float,
        step: float,
    ) -> Tuple[float, float, float]:
        del step
        n = max(1, STRAIGHT_SEGMENT_POINT_COUNT - 1)
        dx = dist * math.cos(heading)
        dy = dist * math.sin(heading)
        x0, y0 = x, y
        for i in range(1, n + 1):
            t = i / n
            points.append((x0 + dx * t, y0 + dy * t))
        x = x0 + dx
        y = y0 + dy
        if dist < 0:
            heading = self._wrap_angle(heading + math.pi)
        return x, y, heading

    def _append_circle_segment(
        self,
        points: List[Tuple[float, float]],
        x: float,
        y: float,
        heading: float,
        radius: float,
        angle_deg: float,
        direction: str,
        step: float,
    ) -> Tuple[float, float, float]:
        phi = math.radians(abs(angle_deg))
        if direction == "cw":
            phi = -phi

        if phi >= 0:
            cx = x - radius * math.sin(heading)
            cy = y + radius * math.cos(heading)
            start_angle = heading - math.pi / 2.0
        else:
            cx = x + radius * math.sin(heading)
            cy = y - radius * math.cos(heading)
            start_angle = heading + math.pi / 2.0

        arc_len = abs(phi) * radius
        n = max(12, min(720, int(arc_len / max(step, 1e-6)) + 1))
        for i in range(1, n + 1):
            ang = start_angle + phi * (i / n)
            px = cx + radius * math.cos(ang)
            py = cy + radius * math.sin(ang)
            points.append((px, py))

        x, y = points[-1]
        heading = self._wrap_angle(heading + phi)
        return x, y, heading

    def _wrap_angle(self, ang: float) -> float:
        while ang > math.pi:
            ang -= 2.0 * math.pi
        while ang < -math.pi:
            ang += 2.0 * math.pi
        return ang

    def _set_vehicle_model_hidden(self) -> None:
        self.vehicle_body_curve.setData([], [])
        self.vehicle_heading_curve.setData([], [])
        self.vehicle_center_marker.setData([], [])

    def _update_vehicle_model(self, x: float, y: float, yaw: float) -> None:
        half = 0.5 * VEHICLE_BODY_SIZE_M
        fx = math.cos(yaw)
        fy = math.sin(yaw)
        lx = -fy
        ly = fx

        rear_left = (x - half * fx + half * lx, y - half * fy + half * ly)
        front_left = (x + half * fx + half * lx, y + half * fy + half * ly)
        front_right = (x + half * fx - half * lx, y + half * fy - half * ly)
        rear_right = (x - half * fx - half * lx, y - half * fy - half * ly)

        body_x = [
            rear_left[0],
            front_left[0],
            front_right[0],
            rear_right[0],
            rear_left[0],
        ]
        body_y = [
            rear_left[1],
            front_left[1],
            front_right[1],
            rear_right[1],
            rear_left[1],
        ]
        self.vehicle_body_curve.setData(body_x, body_y)
        self.vehicle_center_marker.setData([x], [y])

        front_center_x = x + half * fx
        front_center_y = y + half * fy
        shaft_start_x = front_center_x + VEHICLE_HEADING_GAP_M * fx
        shaft_start_y = front_center_y + VEHICLE_HEADING_GAP_M * fy
        tip_x = shaft_start_x + VEHICLE_HEADING_SHAFT_M * fx
        tip_y = shaft_start_y + VEHICLE_HEADING_SHAFT_M * fy
        head_base_x = tip_x - VEHICLE_HEADING_HEAD_LEN_M * fx
        head_base_y = tip_y - VEHICLE_HEADING_HEAD_LEN_M * fy
        head_left_x = head_base_x + VEHICLE_HEADING_HEAD_HALF_WIDTH_M * lx
        head_left_y = head_base_y + VEHICLE_HEADING_HEAD_HALF_WIDTH_M * ly
        head_right_x = head_base_x - VEHICLE_HEADING_HEAD_HALF_WIDTH_M * lx
        head_right_y = head_base_y - VEHICLE_HEADING_HEAD_HALF_WIDTH_M * ly

        arrow_x = [
            shaft_start_x,
            tip_x,
            np.nan,
            tip_x,
            head_left_x,
            np.nan,
            tip_x,
            head_right_x,
        ]
        arrow_y = [
            shaft_start_y,
            tip_y,
            np.nan,
            tip_y,
            head_left_y,
            np.nan,
            tip_y,
            head_right_y,
        ]
        self.vehicle_heading_curve.setData(arrow_x, arrow_y)

    @staticmethod
    def _build_virtual_preview_pose(
        x_m: float = 0.0,
        y_m: float = 0.0,
        z_m: float = 0.0,
        yaw_rad: float = 0.0,
    ) -> PoseSolution:
        return PoseSolution(
            source=VIRTUAL_PREVIEW_POSE_SOURCE,
            gps_week=None,
            gps_sec=None,
            lat=0.0,
            lon=0.0,
            height=0.0,
            x=float(x_m),
            y=float(y_m),
            z=float(z_m),
            yaw=float(yaw_rad),
            pitch=0.0,
            roll=0.0,
            ins_status=None,
            ins_pos_type=None,
        )

    @staticmethod
    def _local_points_to_global(
        local_points: List[Tuple[float, float]],
        pose: PoseSolution,
        rotate_with_yaw: bool = True,
    ) -> List[Tuple[float, float]]:
        if not rotate_with_yaw:
            return [
                (
                    pose.x + float(xr),
                    pose.y + float(yr),
                )
                for xr, yr in local_points
            ]
        cos_yaw = math.cos(pose.yaw)
        sin_yaw = math.sin(pose.yaw)
        return [
            (
                pose.x + xr * cos_yaw - yr * sin_yaw,
                pose.y + xr * sin_yaw + yr * cos_yaw,
            )
            for xr, yr in local_points
        ]

    def _update_traj_axis_labels(self) -> None:
        summary = get_enu_calibration_summary()
        if summary.enabled:
            self.traj_plot.setLabel("left", "校准Y坐标", "m")
            self.traj_plot.setLabel("bottom", "校准X坐标", "m")
        else:
            self.traj_plot.setLabel("left", "北向坐标 Y", "m")
            self.traj_plot.setLabel("bottom", "东向坐标 X", "m")

    def _clear_target_marker_after_frame_change(self) -> None:
        self.target_point = None
        self._target_marking_mode = False
        if self.target_marker is not None:
            self.target_marker.setData([], [])

    def _refresh_ui_after_enu_frame_change(self) -> None:
        self._update_traj_axis_labels()
        self._update_path_coordinate_widgets(frame_override=self._loaded_path_frame)

        pose = get_robot_pose()
        self.path_x = []
        self.path_y = []
        self.traj_curve.setData([], [])
        self._clear_target_marker_after_frame_change()

        view_points: List[Tuple[float, float]] = []
        if pose is not None:
            self.path_x = [pose.x]
            self.path_y = [pose.y]
            self.traj_curve.setData(self.path_x, self.path_y)
            self._update_vehicle_model(pose.x, pose.y, pose.yaw)
            view_points.append((pose.x, pose.y))
        else:
            self._set_vehicle_model_hidden()

        preview_rebuilt = False
        if pose is not None and self._radial_measurement_spec is not None:
            preview_rebuilt = bool(
                self._apply_radial_measurement_spec(
                    self._radial_measurement_spec,
                    anchor_pose=pose,
                    announce_virtual_preview=False,
                )
            )
        elif len(self.loaded_path_local_points) >= 2 and self._loaded_path_frame.is_fixed_origin():
            frame_pose = self._build_virtual_preview_pose(
                x_m=self._loaded_path_frame.origin_x_m,
                y_m=self._loaded_path_frame.origin_y_m,
                z_m=self._loaded_path_frame.origin_z_m,
            )
            global_points = self._apply_loaded_path_preview(
                self.loaded_path_local_points,
                frame_pose,
                using_virtual_pose=(get_robot_pose() is None),
                frame=self._loaded_path_frame,
            )
            if global_points:
                preview_rebuilt = True
                view_points.extend(global_points)
        elif self.loaded_path_points:
            view_points.extend(self.loaded_path_points)

        if self.loaded_path_points:
            view_points.extend(self.loaded_path_points)
        if view_points:
            self._fit_traj_view_to_points(view_points)

        self._update_run_path_button_state()
        self._log(
            "ENU坐标系已更新: UI轨迹坐标已切换到校准后坐标系，旧历史轨迹和目标点标记已清除。"
        )
        if preview_rebuilt:
            self._log("已按校准后坐标系重新生成轨迹预览。")

    def _fit_traj_view_to_points(self, points: List[Tuple[float, float]]) -> None:
        if not points:
            return

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        margin = 0.5

        if max_x - min_x < 1.0:
            cx = 0.5 * (max_x + min_x)
            min_x = cx - 0.5
            max_x = cx + 0.5
        if max_y - min_y < 1.0:
            cy = 0.5 * (max_y + min_y)
            min_y = cy - 0.5
            max_y = cy + 0.5

        self.traj_plot.setXRange(min_x - margin, max_x + margin, padding=0)
        self.traj_plot.setYRange(min_y - margin, max_y + margin, padding=0)

    def _get_run_path_button_state(
        self,
        status: Optional[Any] = None,
    ) -> Tuple[bool, str]:
        if len(self.loaded_path_points) < 2:
            return False, "请先加载、导入或规划至少 2 个轨迹点。"

        if self.controller.car is None:
            return False, "底盘未连接，当前不能执行轨迹。"

        if status is None:
            status = get_status_summary()

        if status.mode != "INS":
            return False, f"已加载轨迹，但执行仍锁定：等待 INS 模式（当前: {status.mode}）。"

        age_inspvax = status.age_inspvax
        if age_inspvax is None:
            return False, "已加载轨迹，但执行仍锁定：尚未收到实时 INSPVAXA。"

        if age_inspvax >= 1.0:
            return False, f"已加载轨迹，但执行仍锁定：INSPVAXA 数据超时（{age_inspvax:.1f}s）。"

        return True, "轨迹、底盘和实时 INS 状态正常，可以执行。"

    def _update_run_path_button_state(self, status: Optional[Any] = None) -> None:
        enabled, tooltip = self._get_run_path_button_state(status)
        self.btn_run_path.setEnabled(enabled)
        self.btn_run_path.setToolTip(tooltip)

    def _has_virtual_path_preview(self) -> bool:
        return self._path_preview_uses_virtual_pose and len(self.loaded_path_local_points) >= 2

    def _planned_path_arrow_sample_indices(self) -> List[int]:
        """沿规划轨迹弧长按间距采样，用于放置方向箭头（避免过密）。"""
        xs, ys = self.planned_x, self.planned_y
        n = len(xs)
        if n < 2:
            return []
        segs = [
            math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) for i in range(1, n)
        ]
        total = sum(segs)
        if total < 1e-6:
            return [0, n - 1]
        min_sep_m = min(2.5, max(total / 14.0, 0.4))
        max_arrows = 26
        indices: List[int] = [0]
        acc = 0.0
        for i in range(1, n):
            acc += segs[i - 1]
            if acc >= min_sep_m:
                indices.append(i)
                acc = 0.0
        if indices[-1] != n - 1:
            indices.append(n - 1)
        deduped: List[int] = []
        for j in indices:
            if not deduped or j > deduped[-1]:
                deduped.append(j)
        if len(deduped) <= max_arrows:
            return deduped
        m = len(deduped)
        thinned: List[int] = []
        for k in range(max_arrows):
            idx = min(int(round(k * (m - 1) / (max_arrows - 1))), m - 1)
            v = deduped[idx]
            if not thinned or v > thinned[-1]:
                thinned.append(v)
        if thinned[-1] != deduped[-1]:
            thinned.append(deduped[-1])
        out: List[int] = []
        for v in thinned:
            if not out or v > out[-1]:
                out.append(v)
        return out

    def _refresh_planned_path_direction_arrows(self) -> None:
        for arrow in self._planned_path_curve_arrows:
            self.traj_plot.removeItem(arrow)
        self._planned_path_curve_arrows.clear()
        for idx in self._planned_path_arrow_sample_indices():
            arr = pg.CurveArrow(
                self.traj_planned_curve,
                index=idx,
                headLen=14,
                tipAngle=30,
                tailLen=None,
                pen=pg.mkPen("#E65100", width=1),
                brush=pg.mkBrush(255, 152, 0, 220),
            )
            self.traj_plot.addItem(arr)
            self._planned_path_curve_arrows.append(arr)

    def _apply_loaded_path_preview(
        self,
        local_points: List[Tuple[float, float]],
        pose: PoseSolution,
        using_virtual_pose: bool,
        frame: Optional[PathReferenceFrame] = None,
    ) -> List[Tuple[float, float]]:
        self.loaded_path_local_points = list(local_points)
        resolved_frame = self._resolve_path_reference_frame(frame or self._loaded_path_frame)
        self._loaded_path_frame = resolved_frame
        global_points = self._local_points_to_global(
            local_points,
            pose,
            rotate_with_yaw=False,
        )

        self.loaded_path_points = global_points
        self.planned_x = [p[0] for p in global_points]
        self.planned_y = [p[1] for p in global_points]
        self._path_preview_uses_virtual_pose = bool(using_virtual_pose)
        self.traj_planned_curve.setData(self.planned_x, self.planned_y)
        self._refresh_planned_path_direction_arrows()
        self._fit_traj_view_to_points(global_points)
        self._update_path_coordinate_widgets(frame_override=resolved_frame)
        self._update_run_path_button_state()
        return global_points

    def _clear_radial_measurement_spec(self) -> None:
        self._radial_measurement_spec = None
        self._clear_radial_rcs_session_state()

    def _clear_radial_rcs_session_state(self) -> None:
        self._radial_rcs_session_dir = None
        self._radial_rcs_expected_segments = []
        self._radial_rcs_started_segments.clear()
        self._radial_rcs_finished_segments.clear()
        self._radial_rcs_failed_segments.clear()

    @staticmethod
    def _format_radial_rcs_segment_label(angle_deg: int, cycle_index: int) -> str:
        return f"{int(angle_deg) % 360}度第{max(1, int(cycle_index))}次测量"

    @staticmethod
    def _parse_radial_rcs_task_name(task_name: Optional[str]) -> Optional[Tuple[int, Optional[int]]]:
        s = str(task_name or "").strip()
        if not s:
            return None
        m = re.search(r"星型\s*(\d+)\s*(?:°|度)(?:\s*[_\-]?\s*第\s*(\d+)\s*次)?", s)
        if not m:
            return None
        angle = int(m.group(1)) % 360
        cycle = int(m.group(2)) if m.group(2) else None
        return angle, cycle

    def _radial_rcs_segment_label_from_task(self, segment_task_name: Optional[str]) -> str:
        parsed = self._parse_radial_rcs_task_name(segment_task_name)
        if parsed is not None:
            angle, cycle = parsed
            return self._format_radial_rcs_segment_label(angle, cycle or 1)
        fallback = str(segment_task_name or "").strip().replace("°", "度")
        return fallback or "0度第1次测量"

    def _radial_rcs_segment_file_stem(self, segment_task_name: Optional[str]) -> str:
        return self._safe_filename_token(
            self._radial_rcs_segment_label_from_task(segment_task_name)
        )

    def _is_radial_forward_rcs_task_name(self, task_name: Optional[str]) -> bool:
        parsed = self._parse_radial_rcs_task_name(task_name)
        return bool(parsed is not None and parsed[1] is not None)

    def _expected_radial_rcs_segment_labels(self) -> List[str]:
        spec = self._radial_measurement_spec
        if spec is None:
            return []
        labels: List[str] = []
        for angle, cycles in list(spec.angle_cycles or []):
            for cycle in range(1, max(1, int(cycles)) + 1):
                labels.append(self._format_radial_rcs_segment_label(int(angle), cycle))
        return labels

    def _planned_radial_rcs_segment_labels(self) -> List[str]:
        if self._radial_measurement_spec is None:
            return []
        labels: List[str] = []
        ranges = list(self._planned_ranges or [])
        range_task_names = list(self._planned_range_task_names or [])
        for seg_idx, _seg_range in enumerate(ranges, start=1):
            name = (
                str(range_task_names[seg_idx - 1]).strip()
                if seg_idx - 1 < len(range_task_names) and range_task_names[seg_idx - 1]
                else ""
            )
            if not self._is_radial_forward_rcs_task_name(name):
                continue
            if self._segment_index_is_forward_straight_segment(seg_idx):
                labels.append(self._radial_rcs_segment_label_from_task(name))
        return labels

    def _begin_radial_rcs_session(self) -> Optional[Path]:
        if self._radial_measurement_spec is None:
            self._clear_radial_rcs_session_state()
            return None
        self._clear_radial_rcs_session_state()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = self._unique_output_path(
            Path(self._rcs_save_dir) / self._safe_filename_token(f"星型测量_{stamp}")
        )
        session_dir.mkdir(parents=True, exist_ok=True)
        self._radial_rcs_session_dir = session_dir
        self._radial_rcs_expected_segments = self._expected_radial_rcs_segment_labels()
        planned = self._planned_radial_rcs_segment_labels()
        missing_plan = [label for label in self._radial_rcs_expected_segments if label not in planned]
        msg = (
            f"星型RCS采集会话已创建: 目录={session_dir} | "
            f"计划前进直线段={len(self._radial_rcs_expected_segments)}"
        )
        if missing_plan:
            preview = "、".join(missing_plan[:8])
            if len(missing_plan) > 8:
                preview += "..."
            msg += f" | 规划覆盖检查缺失={preview}"
        self._log(msg)
        return session_dir

    def _ensure_radial_rcs_session_dir(self) -> Optional[Path]:
        if self._radial_measurement_spec is None:
            return None
        if self._radial_rcs_session_dir is None:
            return self._begin_radial_rcs_session()
        return self._radial_rcs_session_dir

    def _mark_radial_rcs_started(self, segment_task_name: Optional[str]) -> str:
        label = self._radial_rcs_segment_label_from_task(segment_task_name)
        self._radial_rcs_started_segments.add(label)
        return label

    def _mark_radial_rcs_finished(
        self,
        segment_task_name: Optional[str],
        *,
        success: bool,
    ) -> None:
        label = self._radial_rcs_segment_label_from_task(segment_task_name)
        if success:
            self._radial_rcs_finished_segments.add(label)
            self._radial_rcs_failed_segments.discard(label)
        else:
            self._radial_rcs_failed_segments.add(label)

    def _finish_radial_rcs_session_report(self) -> None:
        if (
            self._radial_rcs_session_dir is None
            and not self._radial_rcs_expected_segments
            and not self._radial_rcs_started_segments
        ):
            return
        session_dir = self._radial_rcs_session_dir
        expected = list(self._radial_rcs_expected_segments)
        finished = set(self._radial_rcs_finished_segments)
        missing = [label for label in expected if label not in finished]
        failed = [label for label in self._radial_rcs_failed_segments if label not in finished]
        csv_count = 0
        if session_dir is not None:
            try:
                csv_count = len([p for p in Path(session_dir).glob("*.csv") if p.is_file()])
            except Exception:
                csv_count = 0

        if missing:
            preview = "、".join(missing[:10])
            if len(missing) > 10:
                preview += "..."
            self._log(
                f"星型RCS采集结束: 目录={session_dir} | "
                f"已记录={len(finished)}/{len(expected)} | CSV={csv_count} | 缺失={preview}"
            )
        else:
            extra = f" | 失败重试项={len(failed)}" if failed else ""
            self._log(
                f"星型RCS采集结束: 目录={session_dir} | "
                f"已记录全部前进直线段={len(finished)}/{len(expected)} | CSV={csv_count}{extra}"
            )
        self._clear_radial_rcs_session_state()

    def _radial_rcs_group_file_trajectory_name(self, segment_task_name: str) -> str:
        """兼容旧的内存 RCS 聚合：同一角度多趟可合并到一个逻辑轨迹名。"""
        if self._radial_measurement_spec is None:
            return segment_task_name
        s = str(segment_task_name).strip()
        m = re.match(r"^星型(\d+)°", s)
        if not m:
            return segment_task_name
        ang = int(m.group(1)) % 360
        return f"星型{ang}°"

    def _build_tracking_tuning_kwargs(self) -> Dict[str, Any]:
        tracking_mode = self._get_path_tracking_mode()
        if tracking_mode not in {"stanley", "stanley_pid"}:
            tracking_mode = "stanley"
        kwargs: Dict[str, Any] = {
            "lateral_pid_kp": float(self._tracking_tuning["lateral_kp"]),
            "lateral_pid_ki": float(self._tracking_tuning["lateral_ki"]),
            "lateral_pid_kd": float(self._tracking_tuning["lateral_kd"]),
            "heading_pid_kp": float(self._tracking_tuning["heading_kp"]),
            "heading_pid_ki": float(self._tracking_tuning["heading_ki"]),
            "heading_pid_kd": float(self._tracking_tuning["heading_kd"]),
            "yaw_rate_pid_kp": 0.42,
            "yaw_rate_pid_ki": 0.07,
            "yaw_rate_pid_kd": 0.0,
        }
        if tracking_mode == "stanley":
            # 纯 Stanley：略降增益+软化。Stanley 横向 PD 暂不启用（kp/kd=0）；原值暂存在 _tracking_tuning["stanley_lateral_kp/kd"]。
            kwargs.update(
                {
                    "stanley_gain": 0.40,
                    "stanley_softening_speed_mps": 0.60,
                    "lookahead_heading_weight": 0.05,
                    "lookahead_heading_max_bias_deg": 52.0,
                    "max_w_step": 0.031,
                    "stanley_lateral_pd_kp": 0.0,
                    "stanley_lateral_pd_kd": 0.0,
                    "w_bias_tau": 0.92,
                    "w_bias_hf_gain": 0.24,
                }
            )
        elif tracking_mode == "stanley_pid":
            # 与底盘前进直线段 get_directional_straight_tracking_kwargs(speed_sign>=0) 完全一致
            kwargs.update(dict(FORWARD_STRAIGHT_TRACKING_KWARGS))
        return kwargs

    def _use_uniform_stanley_tracking_for_planned_path(self) -> bool:
        return (
            self._radial_measurement_spec is not None
            and self._get_path_tracking_mode() in {"stanley", "stanley_pid"}
        )

    @staticmethod
    def _build_uniform_stanley_tracking_kwargs() -> Dict[str, Any]:
        return {
            "stanley_gain": 0.30,
            "stanley_softening_speed_mps": 1.0,
            "prime_pid_on_first_cycle": True,
            "reverse_mirror_errors": False,
            "smoothing_strength": 0.82,
            "smoothing_strength_curve": 0.90,
            "w_bias_tau": 0.90,
            "w_bias_hf_gain": 0.22,
            "max_w_rate": 2.2,
            "max_w_step": 0.04,
            "heading_preview_threshold_deg": 18.0,
            "heading_preview_extra_gain": 1.35,
            "heading_preview_max_distance": 1.45,
        }

    def _on_path_tracking_mode_changed(self, _index: int) -> None:
        combo = getattr(self, "combo_path_tracking_mode", None)
        if combo is None:
            return
        selected = combo.currentData()
        self._path_tracking_mode = str(selected or "stanley")
        if self._path_tracking_mode not in {"stanley", "stanley_pid"}:
            self._path_tracking_mode = "stanley"

    def _get_path_tracking_mode(self) -> str:
        combo = getattr(self, "combo_path_tracking_mode", None)
        if combo is not None:
            selected = combo.currentData()
            if selected:
                value = str(selected)
                return value if value in {"stanley", "stanley_pid"} else "stanley"
        value = str(self._path_tracking_mode or "stanley")
        return value if value in {"stanley", "stanley_pid"} else "stanley"

    def _get_path_tracking_mode_label(self) -> str:
        return self._tracking_mode_label(self._get_path_tracking_mode())

    def _on_run_path(self) -> None:
        if not self._reanchor_loaded_path():
            return

        self._reset_rcs_trajectory_file_cache()
        if self._radial_measurement_spec is not None:
            self._begin_radial_rcs_session()
        else:
            self._clear_radial_rcs_session_state()
        tracking_mode = self._get_path_tracking_mode()
        tracking_label = self._get_path_tracking_mode_label()
        uniform_stanley_tracking = self._use_uniform_stanley_tracking_for_planned_path()
        self._reset_motion_stop_context()
        self._motion_active = True
        if self._planned_ranges and getattr(self, "chk_rcs_all_forward_straight", None):
            if self.chk_rcs_all_forward_straight.isChecked():
                self._log(
                    "Cluster RCS: 已启用「前进直线段均触发采集」；"
                    "每段前进直线单独 start/finish，CSV 为 Cluster 模式下 RCS 数据（0x701）。"
                )
        if self._planned_ranges:
            if self._loaded_task_sequence_names:
                self._log(
                    f"执行任务序列: {' -> '.join(self._loaded_task_sequence_names)} | "
                    f"过渡段={self._loaded_task_transition_count} | 模式={tracking_label}"
                )
            self._log(
                f"执行轨迹: 按规划速度分段执行 | 模式={tracking_label}"
            )
            if uniform_stanley_tracking:
                self._log(
                    f"星型测量执行: 所有分段统一使用 {tracking_label} + 角速度内环反馈。"
                )
            t = threading.Thread(
                target=self._run_planned_ranges,
                args=(tracking_mode,),
                daemon=True,
            )
            t.start()
        else:
            speed = max(MIN_SEGMENT_SPEED_MPS, float(self._path_speed))
            self._log(
                f"执行轨迹: 轨迹未携带速度，使用默认速度={speed:.2f}m/s | 模式={tracking_label}"
            )
            if self.loaded_path_points:
                self._configure_motion_guard(
                    design="path",
                    nominal_speed=speed,
                    expected_w_limit=self._estimate_expected_w_limit(self.loaded_path_points, speed),
                )
            self.controller.execute_loaded_path(
                self,
                speed,
                tracking_mode=tracking_mode,
                tracking_kwargs=self._build_tracking_tuning_kwargs(),
            )

    def _run_planned_ranges(self, tracking_mode: str) -> None:
        if self.controller.car is None:
            self._motion_active = False
            return
        if not self.loaded_path_points or not self._planned_ranges:
            self._motion_active = False
            return

        uniform_stanley_tracking = self._use_uniform_stanley_tracking_for_planned_path()
        lookat_target = None if self._radial_measurement_spec is not None else self.target_point
        default_trajectory_name = self._get_current_path_name()
        range_task_names = list(self._planned_range_task_names or [])
        segment_ranges = [
            self._normalize_segment_range(seg_range) for seg_range in self._planned_ranges
        ]
        segment_ranges = self._apply_forward_straight_rcs_flags(
            segment_ranges, self.loaded_path_points
        )
        carry_speed_abs: Optional[float] = None
        active_rcs_segment_idx: Optional[int] = None
        active_rcs_trajectory_name: Optional[str] = None
        try:
            self._motion_full_session_accumulator = []
            self._motion_full_session_active = True
            for seg_idx, seg_range in enumerate(segment_ranges, start=1):
                (
                    start_idx,
                    end_idx,
                    speed_sign,
                    rcs_start,
                    cruise_speed_mps,
                    accel_dist,
                    decel_dist,
                ) = seg_range
                segment_trajectory_name = (
                    str(range_task_names[seg_idx - 1]).strip()
                    if seg_idx - 1 < len(range_task_names) and range_task_names[seg_idx - 1]
                    else default_trajectory_name
                )
                if not self._motion_active:
                    break
                if end_idx <= start_idx:
                    continue
                if rcs_start:
                    active_rcs_segment_idx = seg_idx
                    active_rcs_trajectory_name = segment_trajectory_name
                    self.segment_rcs_start_requested.emit(seg_idx, segment_trajectory_name)
                segment_points = self.loaded_path_points[start_idx : end_idx + 1]
                if len(segment_points) < 2:
                    if rcs_start:
                        self.segment_rcs_finish_requested.emit(seg_idx, segment_trajectory_name)
                        active_rcs_segment_idx = None
                        active_rcs_trajectory_name = None
                    continue
                is_straight = self._segment_geometry_is_straight(seg_idx, segment_points)
                segment_len = self._estimate_segment_length(segment_points)
                seg_speed = speed_sign * cruise_speed_mps
                lookahead = 0.28 if is_straight else 0.48
                if seg_speed >= 0.0:
                    lookahead += 0.08
                arrival_dist = self._segment_arrival_dist(cruise_speed_mps)
                prev_straight_then_current_curve = False
                if seg_idx >= 2:
                    pr = segment_ranges[seg_idx - 2]
                    ps_i, pe_i = int(pr[0]), int(pr[1])
                    if pe_i > ps_i:
                        ppts = self.loaded_path_points[ps_i : pe_i + 1]
                        if len(ppts) >= 2:
                            prev_straight_then_current_curve = (
                                self._segment_geometry_is_straight(seg_idx - 1, ppts)
                                and (not is_straight)
                            )
                next_range = segment_ranges[seg_idx] if seg_idx < len(segment_ranges) else None
                next_points: List[Tuple[float, float]] = []
                next_is_straight = False
                if next_range is not None and next_range[1] > next_range[0]:
                    next_points = self.loaded_path_points[next_range[0] : next_range[1] + 1]
                    next_is_straight = self._segment_geometry_is_straight(seg_idx + 1, next_points)
                next_continuous = (
                    next_range is not None
                    and next_range[2] == speed_sign
                    and not (lookat_target is not None and next_is_straight)
                )
                straight_then_arc = (
                    is_straight
                    and next_continuous
                    and next_range is not None
                    and len(next_points) >= 2
                    and (not next_is_straight)
                )
                if straight_then_arc and seg_speed >= 0.0:
                    lookahead = max(lookahead + 0.12, 0.44)
                reverse_restart_required = bool(is_straight and speed_sign < 0.0)
                start_speed_abs = (
                    carry_speed_abs
                    if carry_speed_abs is not None
                    else self._segment_profile_floor_speed(cruise_speed_mps)
                )
                if next_continuous and next_range is not None:
                    end_speed_abs = max(
                        self._segment_profile_floor_speed(cruise_speed_mps),
                        abs(float(next_range[4])),
                    )
                else:
                    end_speed_abs = self._segment_terminal_end_speed(cruise_speed_mps)
                if straight_then_arc:
                    entry_cap = self._arc_entry_speed_cap_mps(next_points, cruise_speed_mps)
                    end_speed_abs = min(end_speed_abs, entry_cap)
                decel_dist_use = float(decel_dist)
                if straight_then_arc:
                    decel_dist_use = max(
                        decel_dist_use,
                        min(
                            segment_len * 0.44,
                            1.05 + 0.62 * abs(float(cruise_speed_mps)),
                        ),
                    )
                speed_profile = self._build_segment_speed_profile(
                    segment_points,
                    start_speed_mps=start_speed_abs,
                    cruise_speed_mps=cruise_speed_mps,
                    end_speed_mps=end_speed_abs,
                    accel_dist_m=accel_dist,
                    decel_dist_m=decel_dist_use,
                )
                stop_at_end = not next_continuous
                self._configure_motion_guard(
                    design="line" if is_straight else "circle",
                    nominal_speed=seg_speed,
                    expected_w_limit=self._estimate_expected_w_limit(
                        segment_points, seg_speed, is_straight=is_straight
                    ),
                )

                if reverse_restart_required:
                    try:
                        self.controller.car.stop()
                    except Exception as exc:
                        self._log(f"分段{seg_idx} 倒车起步前停车失败: {exc}")
                    self._log(f"分段{seg_idx}: 到达终点后已停车，开始倒车段。")
                    # 星型过渡后直线段：稍延长刹停间隔，避免速度/航向未稳就挂倒挡
                    time.sleep(
                        0.24 if self._radial_measurement_spec is not None else 0.09
                    )

                # 前进直线段前先对齐目标航向，降低“切段瞬间”车头偏差。
                # 倒车段不做这一步，避免把倒车姿态也硬拉到前向目标视线。
                if lookat_target is not None and is_straight and speed_sign > 0:
                    pose = get_robot_pose()
                    if pose is not None:
                        tyaw = math.atan2(lookat_target[1] - pose.y, lookat_target[0] - pose.x)
                        self._motion_guard_allow_spin_until = max(
                            self._motion_guard_allow_spin_until,
                            time.time() + 4.0,
                        )
                        car = getattr(self.controller, "car", None)
                        if car is not None and hasattr(car, "rotate_to_heading"):
                            try:
                                car.rotate_to_heading(tyaw)
                            except Exception as exc:
                                self._log(f"分段{seg_idx} 航向对齐失败: {exc}")
                        self._motion_guard_allow_spin_until = max(
                            self._motion_guard_allow_spin_until,
                            time.time() + 0.3,
                        )

                kwargs = {
                    "lookahead_distance": lookahead,
                    "tracking_mode": tracking_mode,
                    "metrics_callback": self._emit_tracking_metrics,
                    "sample_callback": self._append_tracking_sample,
                    "run_label": f"分段{seg_idx}",
                    "speed_profile": speed_profile,
                    "stop_at_end": stop_at_end,
                    "arrival_dist": arrival_dist,
                    "slow_down_dist": None,
                    "record_context": {
                        "segment_index": int(seg_idx),
                        "segment_kind": "line" if is_straight else "circle",
                        "segment_trajectory_name": segment_trajectory_name,
                        "segment_start_idx": int(start_idx),
                        "segment_end_idx": int(end_idx),
                        "segment_length_m": float(segment_len),
                        "segment_cruise_speed_mps": float(cruise_speed_mps),
                        "segment_accel_dist_m": float(accel_dist),
                        "segment_decel_dist_m": float(decel_dist),
                        "segment_start_speed_mps": float(start_speed_abs),
                        "segment_end_speed_mps": float(end_speed_abs),
                        "segment_stop_at_end": int(bool(stop_at_end)),
                    },
                }
                segment_forced_stop = False
                kwargs.update(self._build_tracking_tuning_kwargs())
                if is_straight and self.controller.car is not None:
                    # 直线段用底盘侧前进/倒车专用参数覆盖 UI 通用 PID/Stanley，避免倒车仍用前进大增益或缺积分
                    kwargs.update(
                        self.controller.car.get_directional_straight_tracking_kwargs(
                            float(speed_sign)
                        )
                    )
                if uniform_stanley_tracking:
                    is_transition_segment = not (
                        seg_idx - 1 < len(range_task_names)
                        and range_task_names[seg_idx - 1]
                    )
                    if tracking_mode != "stanley_pid":
                        kwargs.update(self._build_uniform_stanley_tracking_kwargs())
                    # 星型测量：直线段与过渡曲线段参数分开使用（都采用圆周同款的路径加密与角速度约束）。
                    # - 测量/返回直线段：gain=0.3 + softening=1.0，降低小 S 摆动
                    # - 过渡曲线：gain=0.1 + 更大的 softening + 适当降速（更“软”，减少切换时急转）
                    if is_transition_segment:
                        kwargs["stanley_gain"] = 0.1
                        kwargs["stanley_softening_speed_mps"] = max(
                            float(kwargs.get("stanley_softening_speed_mps", 0.55)),
                            1.60,
                        )
                        # 过渡段降速：速度在 segment range 里已可能更低，这里再做一次温和下限收敛
                        seg_speed = float(seg_speed)
                        seg_speed = math.copysign(
                            min(abs(seg_speed), max(0.12, 0.55 * float(cruise_speed_mps))),
                            seg_speed,
                        )
                    else:
                        kwargs["stanley_gain"] = 0.3
                        kwargs["stanley_softening_speed_mps"] = 1.0
                    seg_points_use = self._densify_polyline_for_tracking(
                        segment_points, max_step_m=0.08
                    )
                    self.controller.car.follow_path_with_pid(
                        seg_points_use,
                        seg_speed,
                        **kwargs,
                    )
                elif is_straight:
                    if lookat_target is not None and speed_sign > 0:
                        kwargs["target_point"] = lookat_target
                        kwargs["target_heading_weight"] = 0.92
                        kwargs["target_heading_max_bias_deg"] = 120.0
                else:
                    # 圆弧段不施加“朝向目标”约束，优先轨迹稳定与差动平顺
                    # stanley_pid 已采用前进直线段全套参数，此处不再覆盖（否则与直线不一致）
                    if tracking_mode != "stanley_pid":
                        kwargs["max_w_rate"] = 2.2
                        kwargs["max_w_step"] = 0.04
                        kwargs["smoothing_strength"] = 0.82
                        kwargs["smoothing_strength_curve"] = 0.90
                        kwargs["w_bias_tau"] = 0.90
                        kwargs["w_bias_hf_gain"] = 0.22
                        kwargs["heading_preview_threshold_deg"] = 18.0
                        kwargs["heading_preview_extra_gain"] = 1.35
                        kwargs["heading_preview_max_distance"] = 1.45
                        if prev_straight_then_current_curve:
                            kwargs["lookahead_distance"] = max(
                                float(kwargs["lookahead_distance"]), 0.58
                            )
                            kwargs["heading_preview_threshold_deg"] = 21.0
                            kwargs["heading_preview_extra_gain"] = 1.52
                            kwargs["heading_preview_max_distance"] = 1.78
                            kwargs["prime_pid_on_first_cycle"] = True
                            kwargs["smoothing_strength"] = 0.86
                    orbit_params = (
                        self._infer_circle_orbit_params(segment_points)
                        if seg_speed > 0.0
                        else None
                    )
                    if orbit_params is not None:
                        orbit_record_context = dict(kwargs["record_context"])
                        orbit_record_context.update(
                            {
                                "tracking_mode": "stanley",
                                "circle_control": "move_circle_stanley",
                                "circle_request_radius_m": float(orbit_params["radius_m"]),
                                "circle_request_angle_deg": float(orbit_params["angle_deg"]),
                                "circle_clockwise": int(bool(orbit_params["clockwise"])),
                                "circle_center_x_m": float(orbit_params["center_x_m"]),
                                "circle_center_y_m": float(orbit_params["center_y_m"]),
                                "circle_fit_rmse_m": float(orbit_params["fit_rmse_m"]),
                                "circle_fit_max_error_m": float(orbit_params["fit_max_error_m"]),
                            }
                        )
                        self._log(
                            f"分段{seg_idx}: 圆弧段 Stanley 跟踪 (move_circle, gain=0.4) "
                            f"(R={float(orbit_params['radius_m']):.2f}m, "
                            f"角度={float(orbit_params['angle_deg']):.1f}°, "
                            f"{'顺时针' if bool(orbit_params['clockwise']) else '逆时针'})"
                        )
                        orbit_speed_mps = abs(float(seg_speed))
                        if prev_straight_then_current_curve:
                            orbit_speed_mps = min(
                                orbit_speed_mps,
                                self._arc_entry_speed_cap_mps(
                                    segment_points, cruise_speed_mps
                                ),
                            )
                        self.controller.car.move_circle(
                            radius_m=float(orbit_params["radius_m"]),
                            angle_deg=float(orbit_params["angle_deg"]),
                            speed_mps=orbit_speed_mps,
                            clockwise=bool(orbit_params["clockwise"]),
                            metrics_callback=self._emit_tracking_metrics,
                            sample_callback=self._append_tracking_sample,
                            run_label=f"分段{seg_idx}",
                            record_context=orbit_record_context,
                        )
                        segment_forced_stop = True
                    else:
                        self._log(
                            f"分段{seg_idx}: 圆弧参数拟合失败，回退到 Stanley 轨迹跟踪（gain=0.4，路径加密）。"
                        )
                        kwargs["stanley_gain"] = 0.4
                        arc_points_dense = self._densify_polyline_for_tracking(
                            segment_points, max_step_m=0.08
                        )
                        self.controller.car.follow_path_with_pid(
                            arc_points_dense,
                            seg_speed,
                            **kwargs,
                        )
                if (not uniform_stanley_tracking) and is_straight:
                    self.controller.car.follow_path_with_pid(
                        segment_points,
                        seg_speed,
                        **kwargs,
                    )
                if rcs_start:
                    self.segment_rcs_finish_requested.emit(seg_idx, segment_trajectory_name)
                    active_rcs_segment_idx = None
                    active_rcs_trajectory_name = None
                carry_speed_abs = (
                    end_speed_abs
                    if (not stop_at_end and not segment_forced_stop)
                    else None
                )
        finally:
            if active_rcs_segment_idx is not None:
                self.segment_rcs_finish_requested.emit(
                    active_rcs_segment_idx,
                    active_rcs_trajectory_name or default_trajectory_name,
                )
            if getattr(self, "_motion_full_session_active", False):
                try:
                    self._finalize_full_motion_session_record()
                except Exception as exc:
                    self._log(f"完整运动记录合并失败: {exc}")
            self._motion_full_session_active = False
            self._motion_full_session_accumulator.clear()
            if self._radial_measurement_spec is not None:
                self._finish_radial_rcs_session_report()
            self._motion_active = False
            self._reset_motion_guard_runtime(clear_profile=False)
            self.motion_run_finished.emit("分段轨迹")

    def _on_segment_rcs_start_requested(self, seg_idx: int, trajectory_name: str) -> None:
        is_straight = self._segment_index_is_forward_straight_segment(seg_idx)
        is_curved = self._segment_index_is_forward_curved_segment(seg_idx)
        if not is_straight and not is_curved:
            self._log(
                f"第{seg_idx}段 RCS跳过: 仅在前进直线段或前进圆弧/曲线段记录（倒车或非前进不采集）"
            )
            return
        if self._rcs_recording:
            self._log(f"第{seg_idx}段开始前检测到上一段RCS仍在记录，先自动收尾")
            prev_traj_name = (
                str(self._rcs_active_path_name or self._get_current_path_name()).strip()
                or "轨迹"
            )
            auto_traj = (
                prev_traj_name
                if self._radial_measurement_spec is not None
                else self._radial_rcs_group_file_trajectory_name(prev_traj_name)
            )
            n_frames, raw_path, _fit_path = self._finalize_rcs_recording(
                save_raw=True,
                trajectory_name=auto_traj,
                segment_index=self._rcs_active_segment_index,
            )
            if self._radial_measurement_spec is not None:
                row_count = self._cluster_csv_data_row_count(Path(raw_path)) if raw_path else 0
                self._mark_radial_rcs_finished(
                    prev_traj_name,
                    success=bool(raw_path) and (int(n_frames) > 0 or row_count > 0),
                )
        if is_straight:
            label_text = ""
            if self._radial_measurement_spec is not None:
                label_text = self._mark_radial_rcs_started(trajectory_name)
                self._ensure_radial_rcs_session_dir()
            self._log(
                f"第{seg_idx}段开始: 前进直线段 Cluster(0x701) RCS CSV 采集"
                + (f" | {label_text}" if label_text else "")
            )
            if not self._on_rcs_start(segment_index=seg_idx, trajectory_name=trajectory_name):
                if self._radial_measurement_spec is not None:
                    self._mark_radial_rcs_finished(trajectory_name, success=False)
            return
        self._log(
            f"第{seg_idx}段开始: 前进圆弧/曲线段 圆周RCS（极坐标采样 CSV + 宽横向 Cluster CSV）"
        )
        self._on_orbit_rcs_start(segment_index=seg_idx, trajectory_name=trajectory_name)

    def _on_segment_rcs_finish_requested(self, seg_idx: int, trajectory_name: str) -> None:
        if self._rcs_active_segment_index is not None and self._rcs_active_segment_index != seg_idx:
            self._log(
                f"第{seg_idx}段 RCS自动保存忽略: 当前活跃分段={self._rcs_active_segment_index}"
            )
            if self._radial_measurement_spec is not None:
                self._mark_radial_rcs_finished(trajectory_name, success=False)
            return
        if not self._rcs_recording:
            self._log(f"第{seg_idx}段 RCS自动保存跳过: 当前未在记录")
            if self._radial_measurement_spec is not None:
                self._mark_radial_rcs_finished(trajectory_name, success=False)
            return
        save_traj_name = (
            trajectory_name
            if self._radial_measurement_spec is not None
            else self._radial_rcs_group_file_trajectory_name(trajectory_name)
        )
        n_frames, raw_path, _fit_path = self._finalize_rcs_recording(
            save_raw=True,
            trajectory_name=save_traj_name,
            segment_index=seg_idx,
        )
        if self._radial_measurement_spec is not None:
            row_count = self._cluster_csv_data_row_count(Path(raw_path)) if raw_path else 0
            self._mark_radial_rcs_finished(
                trajectory_name,
                success=bool(raw_path) and (int(n_frames) > 0 or row_count > 0),
            )

    def _on_radar_point_clicked(self, *args) -> None:
        points = []
        if len(args) >= 2:
            points = args[1]
        elif len(args) == 1:
            points = args[0]
        if not points:
            return

        spot = points[0]
        target = spot.data() if hasattr(spot, "data") else None
        if target is None:
            return
        self._set_tracked_radar_target(target, auto=False)
        projected = self._project_radar_target_to_global(target)
        if projected is not None:
            (gx, gy), x_rel, y_rel, using_virtual_pose = projected
            pose_text = "虚拟位姿" if using_virtual_pose else "当前位姿"
            self._set_target_point_from_global(
                gx,
                gy,
                log_prefix="目标物位置已由雷达锁定同步",
                detail=f"前向={x_rel:.2f}m 左向={y_rel:.2f}m | 基于{pose_text}",
            )
        else:
            self._log("目标物位置同步失败: 无法将雷达目标映射到轨迹图")

        if self._planned_ranges and any(r[3] for r in self._planned_ranges):
            self._log("已锁定目标，等待前进直线分段触发 Cluster RCS 采集")
        else:
            self._log(
                "已锁定目标。当前轨迹未配置 RCS 触发段，不会自动开始 Cluster 采集"
            )

    # ========= 定时器刷新 =========

    def _on_timer(self) -> None:
        # 1) 位姿 & 轨迹
        pose: Optional[PoseSolution] = get_robot_pose()
        if pose is not None and self._has_virtual_path_preview():
            if self._radial_measurement_spec is not None:
                if self._apply_radial_measurement_spec(
                    self._radial_measurement_spec,
                    anchor_pose=pose,
                    announce_virtual_preview=False,
                ):
                    self._log("已获取到实时位姿，星型测量轨迹已切换为实时对齐。")
            else:
                # 普通规划/导入轨迹：全局位置只由主界面「轨迹锚点」决定，不因小车平移；此处禁止用 get_robot_pose() 作平移原点。
                frame = self._resolve_path_reference_frame(self._loaded_path_frame)
                anchor_pose = self._build_virtual_preview_pose(
                    x_m=frame.origin_x_m,
                    y_m=frame.origin_y_m,
                    z_m=frame.origin_z_m,
                )
                if frame.is_fixed_origin() and len(self.loaded_path_local_points) >= 2:
                    global_points = self._apply_loaded_path_preview(
                        self.loaded_path_local_points,
                        anchor_pose,
                        using_virtual_pose=False,
                        frame=frame,
                    )
                    self._log(
                        f"INS 已可用；规划轨迹仍锚定在设定原点（非小车位置），全局点数={len(global_points)}"
                    )
                else:
                    self._path_preview_uses_virtual_pose = False
        if pose is not None:
            self._update_vehicle_model(pose.x, pose.y, pose.yaw)

            # 更新轨迹记录
            if self.traj_enabled:
                append_point = False
                if not self.path_x:
                    append_point = True
                else:
                    dx = pose.x - self.path_x[-1]
                    dy = pose.y - self.path_y[-1]
                    if dx * dx + dy * dy > (0.02 ** 2):
                        append_point = True

                if append_point:
                    self.path_x.append(pose.x)
                    self.path_y.append(pose.y)
                    if len(self.path_x) > 5000:
                        self.path_x = self.path_x[-5000:]
                        self.path_y = self.path_y[-5000:]
                    self.traj_curve.setData(self.path_x, self.path_y)
                    
                    # 自动缩放视图
                    if len(self.path_x) >= 2:
                        min_x = min(self.path_x)
                        max_x = max(self.path_x)
                        min_y = min(self.path_y)
                        max_y = max(self.path_y)
                        margin = 0.5

                        if max_x - min_x < 1.0:
                            cx = 0.5 * (max_x + min_x)
                            min_x = cx - 0.5
                            max_x = cx + 0.5
                        if max_y - min_y < 1.0:
                            cy = 0.5 * (max_y + min_y)
                            min_y = cy - 0.5
                            max_y = cy + 0.5

                        self.traj_plot.setXRange(min_x - margin, max_x + margin, padding=0)
                        self.traj_plot.setYRange(min_y - margin, max_y + margin, padding=0)
            
            # 更新状态文本
            yaw_deg = math.degrees(pose.yaw)
            pose_frame_text = "校准坐标" if get_enu_calibration_summary().enabled else "ENU"
            # 方位角：以正北为 0°，顺时针为正
            azimuth_deg = (90.0 - yaw_deg) % 360.0
            text = (
                f"源: {pose.source} | "
                f"lat={pose.lat:.7f}, lon={pose.lon:.7f} | "
                f"x={pose.x:.2f} m, y={pose.y:.2f} m ({pose_frame_text}) | "
                f"yaw={pose.yaw:.2f} rad ({yaw_deg:.1f}°)"
            )
            if pose.ins_status:
                text += f" | INS_STATUS={pose.ins_status}"
            if pose.ins_pos_type:
                text += f" | INS_POS={pose.ins_pos_type}"
            self.label_pose.setText(text)
            
            # 更新航向角显示
            self.label_heading.setText(f"方位角: {azimuth_deg:.1f}°")
        elif self._has_virtual_path_preview():
            preview_pose = self._build_virtual_preview_pose()
            self._update_vehicle_model(preview_pose.x, preview_pose.y, preview_pose.yaw)
            yaw_deg = math.degrees(preview_pose.yaw)
            pose_frame_text = "校准坐标" if get_enu_calibration_summary().enabled else "ENU"
            azimuth_deg = (90.0 - yaw_deg) % 360.0
            self.label_pose.setText(
                "源: VIRTUAL_PREVIEW | 未获取实时 GNSS / INS 位姿，当前使用原点虚拟位姿预览执行轨迹 | "
                f"x={preview_pose.x:.2f} m, y={preview_pose.y:.2f} m ({pose_frame_text}) | "
                f"yaw={preview_pose.yaw:.2f} rad ({yaw_deg:.1f}°)"
            )
            self.label_heading.setText(f"方位角: {azimuth_deg:.1f}° (虚拟)")
        else:
            self.label_pose.setText("等待 INSPVAXA / INS 对准 / 解算中...")
            self._set_vehicle_model_hidden()
            self.label_heading.setText("方位角: --")

        enu_summary = get_enu_calibration_summary()
        if enu_summary.enabled:
            enu_rms_text = "--" if enu_summary.rms_error_m is None else f"{enu_summary.rms_error_m:.3f}m"
            self.label_enu_calibration.setText(
                "ENU校准: 已启用 | "
                f"点数={enu_summary.point_count} | "
                f"rot={enu_summary.rotation_deg:.4f}° | "
                f"tx={enu_summary.translation_x_m:.3f}m | "
                f"ty={enu_summary.translation_y_m:.3f}m | "
                f"rms={enu_rms_text}"
            )
        else:
            self.label_enu_calibration.setText("ENU校准: 未启用")

        # 2) IMU 状态栏
        status = get_status_summary()

        def fmt_age(a: Optional[float]) -> str:
            return "—" if a is None else f"{a:.1f}s"

        def fmt_freq(f: Optional[float]) -> str:
            return "—" if f is None or f <= 0 else f"{f:.1f}Hz"

        imu_text = (
            f"IMU模式: {status.mode} | "
            f"INS: {status.ins_status}/{status.ins_pos_type} | "
            f"INSPVAXA age={fmt_age(status.age_inspvax)}, f={fmt_freq(status.freq_inspvax)}"
        )
        self.label_imu_status.setText(imu_text)

        self._update_run_path_button_state(status)

        # 3) 电量显示（树莓派 + 小车）
        power_snapshot = self.controller.get_power_snapshot()
        self._handle_car_control_mode_transition(power_snapshot)
        self._update_power_indicators(power_snapshot)
        self.label_power.setText(
            power_snapshot["pi"]["text"] + " | " + power_snapshot["car"]["text"]
        )

        # 4) CAN 初始化状态
        can_text = self.controller.get_can_status()
        self.label_can.setText(can_text)

        # 4.1) 运动异常急停（非设计原地自转 / 角速度抽搐）
        self._check_motion_anomaly_emergency_stop()

        # 5) 雷达 Cluster（0x701）显示：按目标物合并后再画散点；急停仍用原始簇（保守）
        targets: List[Any] = []
        clusters_raw: List[Any] = []
        rt_cluster = getattr(self.controller, "cluster_csv_runtime", None)
        wide_lateral = self._orbit_rcs_active or (
            rt_cluster is not None
            and bool(getattr(rt_cluster, "_orbit_roi_session", False))
        )
        if wide_lateral != self._radar_plot_wide_lateral_active:
            self._radar_plot_wide_lateral_active = wide_lateral
            if self.radar_plot is not None:
                if wide_lateral:
                    self.radar_plot.setXRange(-11.0, 11.0)
                else:
                    self.radar_plot.setXRange(-2.5, 2.5)
        if rt_cluster is not None:
            ts_snap, clusters_raw = rt_cluster.get_cluster_display_snapshot()
            merged = MainWindow._group_cluster_dicts_for_individual_targets(
                clusters_raw,
                RCS_DISPLAY_TARGET_MERGE_RADIUS_M,
            )
            targets = [
                _ClusterDisplayTarget(int(oid), dx, dy, rcs, ts_snap)
                for oid, dx, dy, rcs in merged
            ]

        tracked_text = self.controller.get_radar_status()
        self.label_tracked.setText(tracked_text)

        if targets:
            self._maybe_auto_relock_selected_target(targets, now_ts=time.time())
            spots = []
            for t in targets:
                oid = int(getattr(t, "oid", getattr(t, "id", getattr(t, "cid", 0))))
                is_selected = self.tracked_target_id is not None and oid == self.tracked_target_id
                brush = pg.mkBrush(0, 180, 0, 180) if is_selected else pg.mkBrush(255, 0, 0, 120)
                size = 12 if is_selected else 9
                plot_xy = (
                    t.xy_raw()
                    if hasattr(t, "xy_raw")
                    else (float(getattr(t, "x", 0.0)), float(getattr(t, "y", 0.0)))
                )
                px, py = float(plot_xy[0]), float(plot_xy[1])
                spots.append({"pos": (py, px), "data": t, "brush": brush, "size": size})
            self.radar_scatter.setData(spots)
            safety_targets = [_ClusterSafetyProxy(c) for c in clusters_raw]
            self._check_radar_emergency_stop(safety_targets)
            self._update_rcs_recording(
                targets, clusters_raw if rt_cluster is not None else None
            )
        else:
            self.radar_scatter.setData([])
            safety_targets = []
            self._check_radar_emergency_stop(safety_targets)
            self._update_rcs_recording([], None)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
