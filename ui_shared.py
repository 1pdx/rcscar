# renamed: ui_shared.py
# role: 主界面共享依赖和共享类型定义。
# contains: 标准库/第三方/项目导入、全局常量、dataclass、对话框类、字体配置、动态加载 Radar/RCS/DataAnalysis 模块。
# used by: main_ui.py 与所有 main_ui_* mixin 通过本文件获得共同符号。
# notes: 不放 MainWindow 实例方法；只放多个功能模块都需要复用的定义。
# -*- coding: utf-8 -*-

# Shared imports, constants, data classes, and dialogs for the main UI.
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
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import matplotlib.cm as mpl_cm

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "#0D47A1")
pg.setConfigOption("antialias", True)

from imu_gnss_pose import (
    get_robot_pose,
    get_absolute_robot_pose,
    PoseSolution,
    get_status_summary,
    clear_enu_calibration,
    get_enu_calibration_summary,
    define_local_frame_from_geodetic_y_axis_second_origin,
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

    module_path = Path(__file__).resolve().parent / "gongju" / "data_analysis_ui.py"
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
DEFAULT_REVERSE_LINE_PLAN_SPEED_MPS = 2.0
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
RCS_MAX_DISTANCE_M = 50.0
# 直线距离-RCS：绘图与拟合有效距离门（与 Radar Signal Processing 中 RCS_FILTER_X_* 一致）
RCS_STRAIGHT_X_MIN_M = 4.0
RCS_STRAIGHT_X_MAX_M = 50.0
RCS_STRAIGHT_EMA_ALPHA = 0.5  # 0~1，越小越平滑（建议 0.2~0.35）
RCS_FIT_GRID_STEP_M = 0.1
RCS_FIT_LINE_WIDTH = 1.8
RCS_EXPORT_DPI = 320
RADAR_TARGET_CHECK_TITLE = (
    "雷达 Cluster 检查图（0x701）：直线默认 前方 4–60m、左右 ±3m；"
    "圆周 RCS 采集进行中为 ±10m"
)
ACTION_SAVE_MOTION_DATA_TEXT = "保存运动数据"
ACTION_EMERGENCY_STOP_TEXT = "紧急停止"
PATH_COORD_MODE_FIXED_ORIGIN = "fixed_origin"
# 历史预设 CSV 中可能出现，导入时按「无固定原点」处理
_LEGACY_PATH_COORD_MODE_CURRENT_POSE = "current_pose"
# 轨迹参考原点：UI 不再提供手动输入，默认使用校准平面系 (0,0)
PATH_ORIGIN_MANUAL_KEY = "__manual_xy_anchor__"
# 旧版预设/会话中可能出现，解析时与校准平面原点同等对待（坐标以帧内数值为准）
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
# 圆周 RCS：惯导航向优先、时间平铺兜底，按角度分箱后取箱内 RCS 算术平均
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
    segment_display_labels: Optional[List[str]] = None


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
    collect_all_forward_straight: bool = True


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
        default_reverse_line_speed: float = DEFAULT_REVERSE_LINE_PLAN_SPEED_MPS,
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
        self.line_forward_speed = QtWidgets.QDoubleSpinBox()
        self.line_forward_speed.setRange(0.05, 3.0)
        self.line_forward_speed.setDecimals(2)
        self.line_forward_speed.setValue(default_line_speed)
        self.line_forward_speed.setSuffix(" m/s")
        self.line_reverse_speed = QtWidgets.QDoubleSpinBox()
        self.line_reverse_speed.setRange(0.05, 3.0)
        self.line_reverse_speed.setDecimals(2)
        self.line_reverse_speed.setValue(default_reverse_line_speed)
        self.line_reverse_speed.setSuffix(" m/s")
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
        line_form.addRow("前进段速度", self.line_forward_speed)
        line_form.addRow("倒退段速度", self.line_reverse_speed)
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
                "speed_mps": float(self.line_forward_speed.value()),
                "forward_speed_mps": float(self.line_forward_speed.value()),
                "reverse_speed_mps": float(self.line_reverse_speed.value()),
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
                forward_speed = float(
                    seg.get("forward_speed_mps", seg.get("speed_mps", DEFAULT_LINE_PLAN_SPEED_MPS))
                )
                reverse_speed = float(
                    seg.get("reverse_speed_mps", seg.get("speed_mps", DEFAULT_REVERSE_LINE_PLAN_SPEED_MPS))
                )
                params = (
                    f"沿+Y 距离={seg['distance']:.2f}m 前进v={forward_speed:.2f}m/s 倒退v={reverse_speed:.2f}m/s "
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
        default_collect_all_forward_straight: bool = True,
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
            "0° 为目标物点朝向平面 -Y 的方向（左侧为正30°），每个角度都可以单独设置往返测量次数。"
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

        self.chk_collect_all_forward_straight = QtWidgets.QCheckBox(
            "前进直线段均触发 Cluster RCS 采集"
        )
        self.chk_collect_all_forward_straight.setChecked(bool(default_collect_all_forward_straight))
        self.chk_collect_all_forward_straight.setToolTip(
            "勾选后：星型测量中的每段前进测量直线都会自动触发 Cluster(0x701) RCS CSV；"
            "过渡段、对齐段和倒车段不采集。"
        )

        form.addRow("距离目标最近距离", self.inner_radius)
        form.addRow("固定直线长度", self.line_length)
        form.addRow("巡航速度", self.speed_spin)
        form.addRow("加速段", self.accel_dist_spin)
        form.addRow("减速段", self.decel_dist_spin)
        form.addRow("RCS采集", self.chk_collect_all_forward_straight)
        layout.addLayout(form)

        hint = QtWidgets.QLabel(
            "说明: 0° 为目标物点朝向平面 -Y 的方向，角度按“向左(逆时针)为正”增加。"
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
            collect_all_forward_straight=bool(self.chk_collect_all_forward_straight.isChecked()),
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
            "3. “应用两点ENU校准”：使用最后两次采集的经纬度定系，第一个点指向第二个点为 +Y 方向，第二个点为平面原点。 "
            "可双击编辑经纬度列（适合地图量测或外业坐标）；局部轨迹原点固定使用校准平面系 (0,0)。"
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

        button_row = QtWidgets.QHBoxLayout()
        self.btn_record = QtWidgets.QPushButton("记录当前点")
        self.btn_delete = QtWidgets.QPushButton("删除选中")
        self.btn_apply = QtWidgets.QPushButton("应用两点ENU校准")
        self.btn_clear = QtWidgets.QPushButton("清除ENU校准")
        self.btn_close = QtWidgets.QPushButton("关闭")
        self.btn_record.clicked.connect(self._record_current_point)
        self.btn_delete.clicked.connect(self._delete_selected_point)
        self.btn_apply.clicked.connect(self._apply_geodetic_frame)
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
            "说明: 系统按二维局部水平面处理。校准使用表格中最后两行经纬度：点1→点2 对齐到平面坐标 +Y，点2 设为 (0,0) 原点。"
            "该规则与当前车位无关，INS 航向会叠加同一平面旋转。"
            "轨迹规划/跟踪在同一平面系下使用 (x,y)；航向由路径几何与惯导闭环得到。"
            "CSV 为局部轨迹；局部轨迹原点固定映射到校准平面系 (0,0)。"
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

        try:
            summary = define_local_frame_from_geodetic_y_axis_second_origin(
                lat_a,
                lon_a,
                lat_b,
                lon_b,
                first_height_m=h_a,
                second_height_m=h_b,
            )
            o_lat, o_lon = lat_b, lon_b
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
            f"点1→点2为+Y | 点2原点≈({o_lat:.8f},{o_lon:.8f}) | rot={summary.rotation_deg:.4f}° | "
            f"tx={summary.translation_x_m:.3f} m | ty={summary.translation_y_m:.3f} m | "
            "已根据经纬度刷新表中系统坐标（解析失败的行未修改）。"
        )
        QtWidgets.QMessageBox.information(
            self,
            "经纬度定系完成",
            "已用最后两点的经纬度设定平面系：点1→点2 为 +Y，点2 为平面原点。\n"
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


__all__ = [name for name in globals() if not (name.startswith("__") and name.endswith("__"))]
