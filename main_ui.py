 # main_ui.py
# -*- coding: utf-8 -*-
import csv
import importlib.util
import platform
import sys
import math
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "#0D47A1")
pg.setConfigOption("antialias", True)

from imu_gnss_pose import (
    get_robot_pose,
    PoseSolution,
    get_status_summary,
)
from main_controller import MainController


_RADAR_MODULE = None
_RCS_REF_MODULE = None


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


try:
    _rcs_ref_mod = _load_rcs_reference_module()
except Exception as _rcs_ref_err:
    _rcs_ref_mod = None
    print(f"[main_ui] Failed to load rcs reference module: {_rcs_ref_err}")

PRESET_PATHS_CSV = Path(__file__).with_name("preset_paths.csv")
SegmentRange = Tuple[int, int, int, bool]


def _apply_matplotlib_font() -> None:
    # Prefer installed CJK-capable fonts; also load any local font files under ./fonts.
    from matplotlib import font_manager

    matplotlib.rcParams["axes.unicode_minus"] = False

    candidates = [
        # Windows
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "SimSun",
        "NSimSun",
        "Microsoft JhengHei",
        "Microsoft JhengHei UI",
        # macOS
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "Hiragino Sans GB",
        # Linux
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans SC",
        "Source Han Serif SC",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "AR PL UMing CN",
        "AR PL UKai CN",
    ]

    fonts_dir = Path(__file__).with_name("fonts")
    if fonts_dir.is_dir():
        for ext in ("*.ttf", "*.otf", "*.ttc"):
            for font_path in fonts_dir.glob(ext):
                try:
                    font_manager.fontManager.addfont(str(font_path))
                except Exception:
                    continue

    available = {f.name for f in font_manager.fontManager.ttflist}
    preferred = [name for name in candidates if name in available]

    if not preferred:
        try:
            from matplotlib import ft2font

            sample_char = "中"
            font_paths = []
            for ext in ("ttf", "otf", "ttc"):
                font_paths.extend(font_manager.findSystemFonts(fontext=ext))
            for font_path in font_paths:
                try:
                    if ft2font.FT2Font(font_path).get_char_index(ord(sample_char)) != 0:
                        name = font_manager.FontProperties(fname=font_path).get_name()
                        if name:
                            preferred = [name]
                            break
                except Exception:
                    continue
        except Exception:
            preferred = []

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = preferred or candidates


class TrajectoryPlannerDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        default_dist: float = 2.0,
        default_radius: float = 1.0,
        default_angle: float = 360.0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("轨迹规划")
        self.setMinimumWidth(700)

        self._segments: List[dict] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

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
        self.line_round = QtWidgets.QCheckBox("往返")
        self.line_rcs_start = QtWidgets.QCheckBox("本段开始RCS记录")
        self.btn_add_line = QtWidgets.QPushButton("添加直线段")
        self.btn_add_line.clicked.connect(self._add_line_segment)
        line_form.addRow("距离", self.line_dist)
        line_form.addRow("往返", self.line_round)
        line_form.addRow("RCS", self.line_rcs_start)
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
        self.circle_dir = QtWidgets.QComboBox()
        self.circle_dir.addItems(["逆时针", "顺时针"])
        self.circle_rcs_start = QtWidgets.QCheckBox("本段开始RCS记录")
        self.btn_add_circle = QtWidgets.QPushButton("添加圆弧段")
        self.btn_add_circle.clicked.connect(self._add_circle_segment)
        circle_form.addRow("半径", self.circle_radius)
        circle_form.addRow("角度", self.circle_angle)
        circle_form.addRow("方向", self.circle_dir)
        circle_form.addRow("RCS", self.circle_rcs_start)
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
        self._segments.append(
            {
                "type": "line",
                "distance": dist,
                "round_trip": round_trip,
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
                params = f"距离={seg['distance']:.2f}m"
                if seg.get("round_trip"):
                    params += " 往返"
            else:
                direction = "逆时针" if seg.get("direction") == "ccw" else "顺时针"
                params = f"R={seg['radius']:.2f}m, θ={seg['angle']:.1f}° {direction}"
            rcs_text = "起始触发" if seg.get("rcs_start") else "--"
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(type_text))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(params))
            self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(rcs_text))

    def get_segments(self) -> List[dict]:
        return list(self._segments)


class MainWindow(QtWidgets.QMainWindow):
    segment_rcs_start_requested = QtCore.pyqtSignal(int)
    """
    RCS 雷达小车一体化调试 UI：

      - 上方：左侧状态信息，中间轨迹规划图，右侧控制按钮
      - 下方：左侧雷达目标检查图，右侧雷达跟踪图
      - 左侧：INSPVAXA/INS 状态 + 电量 + CAN 状态 + 锁定目标信息
    """

    def __init__(self) -> None:
        super().__init__()
        _apply_matplotlib_font()
        self.setWindowTitle("RCS 雷达小车调试 UI")
        self._apply_initial_geometry()

        self._is_linux = platform.system().lower() == "linux"

        # 初始化控制器
        self.controller = MainController(self._is_linux)

        self.rcs_recorder = RcsRunRecorder(max_segments=10)
        self.rcs_lock = AssocLock()
        self._rcs_recording = False
        self._rcs_fitted: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._rcs_show_only_fitted = False
        self._rcs_target_name: Optional[str] = None
        self._rcs_ref_class_options: List[str] = []
        self._rcs_ref_angle_options: List[str] = []
        self._rcs_ref_labels: Dict[str, str] = {}
        self._rcs_ref_limits: Optional[Dict[str, np.ndarray]] = None
        self._rcs_ref_class: Optional[str] = None
        self._rcs_ref_angle: Optional[str] = None
        self._load_rcs_reference_options()
        self._radar_emergency_active = False
        self._motion_active = False
        self._radar_stop_threshold = 1.0
        self._path_speed = 0.5
        self._traj_default_dist = 2.0
        self._traj_default_radius = 1.0
        self._traj_default_angle = 360.0
        self._preset_paths: Dict[str, List[Tuple[float, float]]] = {}
        self._preset_ranges: Dict[str, List[SegmentRange]] = {}
        self._planned_ranges: Optional[List[SegmentRange]] = None
        self.segment_rcs_start_requested.connect(self._on_segment_rcs_start_requested)

        # ================== UI 结构 ==================
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(16, 16, 16, 16)
        vbox.setSpacing(14)

        self._build_control_panel(vbox)
        self._build_plots(vbox)
        self._apply_theme()
        self._load_preset_paths()

        # 轨迹缓存（历史 + 规划）
        self.path_x: List[float] = []
        self.path_y: List[float] = []
        self.planned_x: List[float] = []
        self.planned_y: List[float] = []
        self.loaded_path_points: List[Tuple[float, float]] = []
        self.loaded_path_local_points: List[Tuple[float, float]] = []
        self._target_marking_mode = False
        self.target_point: Optional[Tuple[float, float]] = None

        self.traj_enabled = True
        self.tracked_target_id: Optional[int] = None

        # 默认禁用运动按钮，等待 IMU 状态良好后自动解锁
        self.btn_run_path.setEnabled(False)

        # ================== 定时更新 ==================
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(50)  # 20 Hz

        # 启动时全屏显示
        self.setWindowState(self.windowState() | QtCore.Qt.WindowMaximized)

        self._log("UI已启动")
        self._log(f"系统: {platform.system()} | 雷达跟踪: {'启用' if self.controller.radar is not None else '未启用'}")
        self._log(f"CAN状态: {self.controller.get_can_status()}")

    def _apply_initial_geometry(self) -> None:
        screen = None
        if hasattr(QtGui.QGuiApplication, "screenAt"):
            screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())
        if screen is None:
            screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            self.resize(1600, 1000)
            return

        available = screen.availableGeometry()
        if available.width() <= 0 or available.height() <= 0:
            self.resize(1600, 1000)
            return

        target_w = min(1400, int(available.width() * 0.85))
        target_h = min(900, int(available.height() * 0.85))
        self.resize(target_w, target_h)

        x = available.x() + max(0, (available.width() - target_w) // 2)
        y = available.y() + max(0, (available.height() - target_h) // 2)
        self.move(x, y)

    def _build_control_panel(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """构建控制面板"""
        panel = QtWidgets.QFrame()
        panel.setObjectName("ControlPanel")
        hbox = QtWidgets.QHBoxLayout(panel)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(16)

        # 左侧：状态信息
        info_frame = QtWidgets.QFrame()
        info_frame.setObjectName("InfoCard")
        vinfo = QtWidgets.QVBoxLayout(info_frame)
        vinfo.setContentsMargins(16, 16, 16, 16)
        vinfo.setSpacing(8)

        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setColor(QtGui.QColor(0, 0, 0, 140))
        shadow.setOffset(0, 6)
        info_frame.setGraphicsEffect(shadow)

        self.label_status_header = QtWidgets.QLabel("系统状态")
        self.label_status_header.setObjectName("InfoHeader")

        self.label_pose = QtWidgets.QLabel("等待 INSPVAXA / INS 数据...")
        self.label_pose.setWordWrap(True)
        self.label_pose.setMinimumWidth(380)

        self.label_imu_status = QtWidgets.QLabel("IMU: 初始化中...")
        self.label_imu_status.setWordWrap(True)
        self.label_imu_status.setMinimumWidth(380)

        self.label_power = QtWidgets.QLabel("树莓派电量: -- | 小车电量: --")
        self.label_power.setMinimumWidth(380)

        self.label_can = QtWidgets.QLabel("CAN 状态: 未知")
        self.label_can.setMinimumWidth(380)

        self.label_tracked = QtWidgets.QLabel("锁定目标: --")
        self.label_tracked.setMinimumWidth(380)

        self.label_heading = QtWidgets.QLabel("航向角: --")
        self.label_heading.setMinimumWidth(380)

        vinfo.addWidget(self.label_status_header)
        vinfo.addWidget(self.label_pose)
        vinfo.addWidget(self.label_imu_status)
        vinfo.addWidget(self.label_power)
        vinfo.addWidget(self.label_can)
        vinfo.addWidget(self.label_tracked)
        vinfo.addWidget(self.label_heading)
        vinfo.addStretch(1)

        hbox.addWidget(info_frame, 2)

        # 中间：轨迹规划图
        traj_frame = QtWidgets.QFrame()
        traj_frame.setObjectName("PlotFrame")
        traj_vbox = QtWidgets.QVBoxLayout(traj_frame)
        traj_vbox.setContentsMargins(8, 8, 8, 8)

        traj_label = QtWidgets.QLabel("轨迹规划图 (蓝色:实际轨迹, 橙色:规划轨迹, 红色箭头:车头方向)")
        traj_label.setObjectName("PlotLabel")
        traj_vbox.addWidget(traj_label)

        self.traj_plot = pg.PlotWidget()
        self.traj_plot.setLabel('left', '北向坐标 Y', 'm')
        self.traj_plot.setLabel('bottom', '东向坐标 X', 'm')
        self.traj_plot.setAspectLocked(True)
        self.traj_plot.showGrid(x=True, y=True, alpha=0.3)

        # 轨迹曲线
        self.traj_curve = self.traj_plot.plot([], [], pen=pg.mkPen(color='#0D47A1', width=2), name="实际轨迹")
        self.traj_planned_curve = self.traj_plot.plot([], [], pen=pg.mkPen(color='#FF9800', width=2, style=QtCore.Qt.DashLine), name="规划轨迹")

        # 车头方向箭头
        self.heading_arrow = pg.ArrowItem(
            angle=0,
            tipAngle=30,
            baseAngle=20,
            headLen=20,
            tailLen=0,
            tailWidth=5,
            pen={'color': 'r', 'width': 2},
            brush='r',
        )
        self.traj_plot.addItem(self.heading_arrow)
        self.heading_arrow.setPos(0, 0)

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

        traj_vbox.addWidget(self.traj_plot)
        hbox.addWidget(traj_frame, 3)

        # 右侧：按钮控制
        btn_frame = QtWidgets.QFrame()
        btn_frame.setObjectName("ButtonCard")
        btn_frame.setMinimumWidth(230)
        btn_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        vbtn = QtWidgets.QVBoxLayout(btn_frame)
        vbtn.setContentsMargins(12, 12, 12, 12)
        vbtn.setSpacing(12)

        self.btn_load_path = QtWidgets.QPushButton("轨迹规划")
        self.btn_run_path = QtWidgets.QPushButton("执行轨迹")
        self.btn_preset_path = QtWidgets.QPushButton("预定轨迹/导入")
        self.btn_delete_preset = QtWidgets.QPushButton("删除预定轨迹")
        self.btn_radar_stop = QtWidgets.QPushButton("雷达急停阈值")
        self.btn_calib = QtWidgets.QPushButton("标定目标物位置")
        self.btn_stop = QtWidgets.QPushButton("倒回")
        self.btn_save_traj = QtWidgets.QPushButton("保存轨迹图")
        self.btn_heading_calib = QtWidgets.QPushButton("清除历史轨迹")

        self.btn_load_path.clicked.connect(self._on_load_path)
        self.btn_run_path.clicked.connect(self._on_run_path)
        self.btn_preset_path.clicked.connect(self._on_preset_path_clicked)
        delete_handler = getattr(self, "_on_delete_preset_clicked", None)
        if callable(delete_handler):
            self.btn_delete_preset.clicked.connect(delete_handler)
        else:
            self.btn_delete_preset.setEnabled(False)
        self.btn_radar_stop.clicked.connect(self._on_radar_stop_threshold)
        self.btn_calib.clicked.connect(self._on_calib_clicked)
        self.btn_stop.clicked.connect(self._on_return_clicked)
        self.btn_save_traj.clicked.connect(self._on_save_traj)
        self.btn_heading_calib.clicked.connect(self._on_heading_calib_clicked)

        vbtn.addWidget(self.btn_load_path)
        vbtn.addWidget(self.btn_run_path)
        vbtn.addWidget(self.btn_preset_path)
        vbtn.addWidget(self.btn_delete_preset)
        vbtn.addWidget(self.btn_radar_stop)
        vbtn.addWidget(self.btn_calib)
        vbtn.addWidget(self.btn_heading_calib)
        vbtn.addWidget(self.btn_save_traj)
        vbtn.addWidget(self.btn_stop)
        vbtn.addStretch(1)

        hbox.addWidget(btn_frame, 1)

        parent_layout.addWidget(panel)

    def _build_plots(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """构建绘图区域 - 雷达目标检查 + 雷达跟踪"""
        plots_container = QtWidgets.QFrame()
        plots_container.setObjectName("PlotsContainer")
        plots_hbox = QtWidgets.QHBoxLayout(plots_container)
        plots_hbox.setContentsMargins(0, 0, 0, 0)
        plots_hbox.setSpacing(12)

        log_frame = self._create_log_panel()
        plots_hbox.addWidget(log_frame, 3)

        # 雷达目标检查图（左侧）
        radar_frame = QtWidgets.QFrame()
        radar_frame.setObjectName("PlotFrame")
        radar_vbox = QtWidgets.QVBoxLayout(radar_frame)
        radar_vbox.setContentsMargins(8, 8, 8, 8)

        radar_label = QtWidgets.QLabel("雷达目标检查图 (前方60m, 左右±20m)")
        radar_label.setObjectName("PlotLabel")
        radar_vbox.addWidget(radar_label)

        self.radar_plot = pg.PlotWidget()
        self.radar_plot.setLabel('left', '前方距离', 'm')
        self.radar_plot.setLabel('bottom', '横向距离', 'm')
        self.radar_plot.setXRange(-20, 20)
        self.radar_plot.setYRange(0, 60)
        self.radar_plot.setAspectLocked(True)
        self.radar_plot.showGrid(x=True, y=True, alpha=0.3)
        self.radar_plot.setMinimumSize(320, 320)

        # 雷达散点
        self.radar_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))
        self.radar_scatter.sigClicked.connect(self._on_radar_point_clicked)
        self.radar_plot.addItem(self.radar_scatter)

        radar_vbox.addWidget(self.radar_plot)
        plots_hbox.addWidget(radar_frame, 5)

        # 雷达跟踪图（右侧）
        rcs_frame = self._create_rcs_frame()
        plots_hbox.addWidget(rcs_frame, 5)

        parent_layout.addWidget(plots_container, 7)

    def _create_rcs_frame(self) -> QtWidgets.QFrame:
        """构建雷达跟踪图 + RCS 控件"""
        rcs_frame = QtWidgets.QFrame()
        rcs_frame.setObjectName("PlotFrame")
        rcs_vbox = QtWidgets.QVBoxLayout(rcs_frame)
        rcs_vbox.setContentsMargins(8, 8, 8, 8)

        rcs_label = QtWidgets.QLabel("雷达跟踪图 (点击雷达点锁定)")
        rcs_label.setObjectName("PlotLabel")
        rcs_vbox.addWidget(rcs_label)

        self.rcs_canvas = FigureCanvas(Figure(figsize=(4.8, 4.8), dpi=100))
        self.rcs_canvas.setMinimumSize(320, 320)
        self.rcs_canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.rcs_ax = self.rcs_canvas.figure.add_subplot(111)
        self.rcs_canvas.figure.subplots_adjust(bottom=0.18)
        rcs_vbox.addWidget(self.rcs_canvas, 1)

        name_row = QtWidgets.QHBoxLayout()
        self.rcs_target_combo = QtWidgets.QComboBox()
        self.rcs_target_combo.setMinimumWidth(140)
        self.rcs_target_combo.addItem("选择目标名称")
        for name in self._rcs_ref_class_options:
            self.rcs_target_combo.addItem(name)
        self.rcs_target_combo.currentIndexChanged[str].connect(self._on_rcs_ref_class_changed)

        self.rcs_angle_combo = QtWidgets.QComboBox()
        self.rcs_angle_combo.setMinimumWidth(100)
        self.rcs_angle_combo.addItem("选择角度")
        for ang in self._rcs_ref_angle_options:
            self.rcs_angle_combo.addItem(ang)
        self.rcs_angle_combo.currentIndexChanged[str].connect(self._on_rcs_ref_angle_changed)

        if not self._rcs_ref_class_options:
            self.rcs_target_combo.setEnabled(False)
        if not self._rcs_ref_angle_options:
            self.rcs_angle_combo.setEnabled(False)

        self.btn_rcs_set_name = QtWidgets.QPushButton("自定义名称")
        self.btn_rcs_set_name.clicked.connect(self._on_rcs_set_target_name)

        name_row.addWidget(self.rcs_target_combo)
        name_row.addWidget(self.rcs_angle_combo)
        name_row.addWidget(self.btn_rcs_set_name)
        name_row.addStretch(1)
        rcs_vbox.addLayout(name_row)

        btns = QtWidgets.QHBoxLayout()
        self.btn_rcs_end = QtWidgets.QPushButton("结束并拟合")
        self.btn_rcs_save_pdf = QtWidgets.QPushButton("保存拟合PDF")
        self.btn_rcs_save_raw = QtWidgets.QPushButton("保存原始txt")
        self.btn_rcs_end.clicked.connect(self._on_rcs_end_fit)
        self.btn_rcs_save_pdf.clicked.connect(self._on_rcs_save_pdf)
        self.btn_rcs_save_raw.clicked.connect(self._on_rcs_save_raw)
        for b in (self.btn_rcs_end, self.btn_rcs_save_pdf, self.btn_rcs_save_raw):
            btns.addWidget(b)
        btns.addStretch(1)
        rcs_vbox.addLayout(btns)

        self.rcs_status_label = QtWidgets.QLabel("RCS录制: 未开始")
        rcs_vbox.addWidget(self.rcs_status_label)
        rcs_vbox.addStretch(1)

        self._draw_rcs()
        return rcs_frame

    def _build_rcs_panel(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """构建 RCS 采集/展示区域"""
        rcs_frame = self._create_rcs_frame()
        parent_layout.addWidget(rcs_frame, 2)

    def _create_log_panel(self) -> QtWidgets.QFrame:
        log_frame = QtWidgets.QFrame()
        log_frame.setObjectName("LogFrame")
        log_vbox = QtWidgets.QVBoxLayout(log_frame)
        log_vbox.setContentsMargins(8, 8, 8, 8)

        log_label = QtWidgets.QLabel("运行日志")
        log_label.setObjectName("PlotLabel")
        log_vbox.addWidget(log_label)

        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.log_edit.setMaximumBlockCount(800)
        self.log_edit.setMinimumHeight(140)
        log_vbox.addWidget(self.log_edit, 1)

        log_btns = QtWidgets.QHBoxLayout()
        self.btn_clear_log = QtWidgets.QPushButton("清空日志")
        self.btn_clear_log.clicked.connect(self.log_edit.clear)
        log_btns.addWidget(self.btn_clear_log)
        log_btns.addStretch(1)
        log_vbox.addLayout(log_btns)
        return log_frame

    def _build_log_panel(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        log_frame = self._create_log_panel()
        parent_layout.addWidget(log_frame, 1)

    def _apply_theme(self) -> None:
        """应用主题样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            #ControlPanel {
                background-color: transparent;
            }
            #InfoCard, #ButtonCard {
                background-color: white;
                border-radius: 8px;
                padding: 8px;
            }
            #PlotFrame, #LogFrame {
                background-color: white;
                border-radius: 8px;
            }
            #InfoHeader {
                font-size: 16px;
                font-weight: bold;
                color: #0D47A1;
                padding-bottom: 8px;
            }
            #PlotLabel {
                font-size: 14px;
                font-weight: bold;
                color: #0D47A1;
                padding: 4px;
            }
            QPushButton {
                background-color: #0D47A1;
                color: white;
                border: none;
                padding: 4px 10px;
                border-radius: 4px;
                font-weight: bold;
                min-height: 16px;
                font-size: 11px;
            }
            #ButtonCard QPushButton {
                padding: 6px 12px;
                min-height: 28px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QPushButton:pressed {
                background-color: #003C8F;
            }
            QPushButton:disabled {
                background-color: #90CAF9;
                color: #E3F2FD;
            }
            QLineEdit {
                padding: 6px 8px;
                border: 1px solid #BBDEFB;
                border-radius: 4px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #0D47A1;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #BBDEFB;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QRadioButton {
                spacing: 5px;
            }
            QRadioButton::indicator {
                width: 13px;
                height: 13px;
            }
            QRadioButton::indicator:unchecked {
                border: 1px solid #0D47A1;
                border-radius: 7px;
                background-color: white;
            }
            QRadioButton::indicator:checked {
                border: 1px solid #0D47A1;
                border-radius: 7px;
                background-color: #0D47A1;
            }
        """)

    def _log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        if hasattr(self, "log_edit") and self.log_edit is not None:
            self.log_edit.appendPlainText(line)
            sb = self.log_edit.verticalScrollBar()
            sb.setValue(sb.maximum())
        else:
            print(line)

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

    def _on_rcs_ref_class_changed(self, text: str) -> None:
        if not text or text.startswith("选择"):
            self._rcs_ref_class = None
            self._rcs_target_name = None
            self._update_rcs_reference_limits()
            return
        self._rcs_ref_class = text
        self._rcs_target_name = text
        self._update_rcs_reference_limits()
        self._log(f"测试目标名称选择: {text}")

    def _on_rcs_ref_angle_changed(self, text: str) -> None:
        if not text or text.startswith("选择"):
            self._rcs_ref_angle = None
            self._update_rcs_reference_limits()
            return
        self._rcs_ref_angle = text
        self._update_rcs_reference_limits()
        self._log(f"测试角度选择: {text}")

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

    def _draw_rcs(self, live: bool = False) -> None:
        """刷新 RCS 曲线图"""
        ax = self.rcs_ax
        ax.clear()
        ax.set_title(self._get_rcs_title())
        ax.set_xlabel("Front X (m)")
        ax.set_ylabel("RCS (dBsm)")
        ax.set_axisbelow(True)
        ax.grid(False)
        ax.yaxis.grid(True, color="#e0e0e0", linestyle="-", linewidth=0.6)
        ax.set_xlim(0, 60)

        if self._rcs_show_only_fitted and self._rcs_fitted is not None:
            xg, yg = self._rcs_fitted
            ax.plot(xg, yg, linewidth=3, label="拟合")
            ax.legend(loc="best")
            self.rcs_canvas.draw_idle()
            return

        for i, seg in enumerate(self.rcs_recorder.segments[:10]):
            xs = [p.x for p in seg]
            ys = [p.rcs_filt for p in seg]
            ax.plot(xs, ys, label=f"run{i}")
        if live and getattr(self.rcs_recorder, "_cur", None):
            xs = [p.x for p in self.rcs_recorder._cur]
            ys = [p.rcs_filt for p in self.rcs_recorder._cur]
            ax.plot(xs, ys, linestyle="--", label=f"run{len(self.rcs_recorder.segments)}(cur)")
        if self._rcs_fitted is not None:
            xg, yg = self._rcs_fitted
            ax.plot(xg, yg, linewidth=3, label="拟合")

        if self._rcs_ref_limits is not None:
            limits = self._rcs_ref_limits
            xs = limits.get("x")
            lower = limits.get("lower")
            upper = limits.get("upper")
            ref = limits.get("ref")
            if xs is not None and lower is not None:
                ax.plot(xs, lower, color="k", linewidth=2, label="下界")
            if xs is not None and upper is not None:
                ax.plot(xs, upper, color="k", linewidth=2, label="上界")
            if xs is not None and ref is not None:
                ax.plot(xs, ref, color="b", linewidth=2, label="参考值")

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="best")
        self.rcs_canvas.draw_idle()

    def _get_rcs_title(self) -> str:
        base = (self._rcs_target_name or "RCS").strip() or "RCS"
        if self._rcs_ref_angle:
            base = f"{base} {self._rcs_ref_angle}"
        title = base
        if self.rcs_recorder.oid is not None:
            title += f" (ID={self.rcs_recorder.oid})"
        return title

    def _on_rcs_set_target_name(self) -> None:
        current = self._rcs_target_name or ""
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            "设置测试目标名称",
            "目标名称(留空恢复默认):",
            text=current,
        )
        if not ok:
            return
        name = text.strip()
        self._rcs_target_name = name if name else None
        self._draw_rcs()
        if self._rcs_target_name:
            self._log(f"测试目标名称设置: {self._rcs_target_name}")
        else:
            self._log("测试目标名称清空，恢复默认标题")

    def _on_rcs_start(self) -> None:
        """开始 RCS 录制（需先锁定目标）"""
        if self.tracked_target_id is None:
            QtWidgets.QMessageBox.information(self, "未锁定目标", "请在雷达图上点击稳定目标点进行锁定。")
            self._log("RCS录制启动失败: 未锁定目标")
            return
        self.rcs_recorder.arm(int(self.tracked_target_id))
        self.rcs_lock.disarm()
        self._rcs_recording = True
        self._rcs_fitted = None
        self._rcs_show_only_fitted = False
        self.rcs_status_label.setText(f"RCS录制: 进行中 | 目标ID={self.tracked_target_id}")
        self._log(f"RCS录制开始: 目标ID={self.tracked_target_id}")
        self._draw_rcs()

    def _on_rcs_end_fit(self) -> None:
        """结束录制并拟合"""
        if self.rcs_recorder.oid is None:
            QtWidgets.QMessageBox.information(self, "未开始录制", "请先开始录制后再结束。")
            self._log("RCS录制结束失败: 未开始录制")
            return
        self.rcs_recorder.finalize()
        self._rcs_recording = False
        grid = np.linspace(0.0, 60.0, 181)
        self._rcs_fitted = self.rcs_recorder.fitted_curve(grid)
        out_path = self._save_rcs_fit_image()
        if out_path:
            self.rcs_status_label.setText(
                f"RCS录制: 已结束 | 段数={len(self.rcs_recorder.segments)} | 曲线已保存 {out_path}"
            )
            self._log(f"RCS录制结束: 段数={len(self.rcs_recorder.segments)} 曲线={out_path}")
        else:
            self.rcs_status_label.setText(f"RCS录制: 已结束 | 段数={len(self.rcs_recorder.segments)}")
            self._log(f"RCS录制结束: 段数={len(self.rcs_recorder.segments)}")
        self._draw_rcs()

    def _save_rcs_fit_image(self) -> Optional[str]:
        if self._rcs_fitted is None or self.rcs_recorder.oid is None:
            return None
        xg, yg = self._rcs_fitted
        if np.all(np.isnan(yg)):
            return None
        stamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"rcs_fit_id{self.rcs_recorder.oid}_{stamp}.png"
        path = Path.cwd() / filename

        fig = Figure(figsize=(7.5, 4.5), dpi=120)
        ax = fig.add_subplot(111)
        ax.set_title(f"RCS Fit (ID={self.rcs_recorder.oid})")
        ax.set_xlabel("Front X (m)")
        ax.set_ylabel("RCS (dBsm)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(0, 60)
        ax.plot(xg, yg, linewidth=2.5, color="#1565C0")
        fig.tight_layout()
        fig.savefig(str(path), format="png")
        return str(path)

    def _on_rcs_save_pdf(self) -> None:
        """保存拟合曲线为 PDF"""
        if self._rcs_fitted is None or self.rcs_recorder.oid is None:
            QtWidgets.QMessageBox.information(self, "未拟合", "请先结束录制并完成拟合。")
            self._log("保存拟合PDF失败: 未拟合")
            return

        default_name = f"fitted_id{self.rcs_recorder.oid}.pdf"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存拟合PDF", default_name, "PDF (*.pdf)")
        if not path:
            return

        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title(default_name.replace(".pdf", ""))
        ax.set_xlabel("Front X (m)")
        ax.set_ylabel("RCS (dBsm)")
        ax.grid(True)
        ax.set_xlim(0, 60)
        xg, yg = self._rcs_fitted
        ax.plot(xg, yg, linewidth=3)
        fig.savefig(path, format="pdf")

        self._rcs_show_only_fitted = True
        self._draw_rcs()
        self.rcs_status_label.setText(f"RCS录制: 拟合曲线已保存 {path}")
        self._log(f"拟合PDF已保存: {path}")

    def _on_rcs_save_raw(self) -> None:
        """保存原始分段数据"""
        if not self.rcs_recorder.segments:
            QtWidgets.QMessageBox.information(self, "无数据", "当前没有可保存的分段数据。")
            self._log("保存原始数据失败: 无数据")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存原始txt", "rcs_raw.txt", "Text (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.rcs_recorder.raw_text())
        self.rcs_status_label.setText(f"RCS录制: 原始数据已保存 {path}")
        self._log(f"原始数据已保存: {path}")

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
            return ObjMeas(
                oid=oid,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                dyn=dyn,
                rcs_db=float(rcs),
                t=now_ts,
            )
        except Exception:
            return None

    def _update_rcs_recording(self, targets: List) -> None:
        """在定时器中基于当前雷达目标更新 RCS 录制"""
        if not self._rcs_recording or self.rcs_recorder.oid is None:
            return

        now_ts = time.time()
        candidates: List[ObjMeas] = []
        for t in targets:
            m = self._convert_target_to_objmeas(t, now_ts)
            if m is not None:
                candidates.append(m)

        if not candidates:
            self.rcs_status_label.setText("RCS录制: 丢失目标(无候选)")
            self._draw_rcs(live=True)
            return

        if not self.rcs_lock.armed:
            m0 = next((c for c in candidates if c.oid == self.rcs_recorder.oid), None)
            if m0 is not None:
                self.rcs_lock.arm_from(m0)

        m = self.rcs_lock.associate(candidates, now_ts) if self.rcs_lock.armed else None
        if m is None:
            self.rcs_status_label.setText("RCS录制: 丢失(hold)")
            self._draw_rcs(live=True)
            return

        self.rcs_recorder.oid = int(m.oid)
        self.rcs_recorder.add_point(m)

        if self.rcs_recorder.ended():
            self._on_rcs_end_fit()
            return

        self.rcs_status_label.setText(
            f"RCS录制: 进行中 | id={m.oid} x={m.x:.1f}m y={m.y:.1f}m rcs={m.rcs_db:.1f} dBsm"
        )
        self._draw_rcs(live=True)

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
        radar = self.controller.radar
        if radar is None:
            return []
        tracker = getattr(radar, "tracker", None)
        if tracker is None or not hasattr(tracker, "snapshot_tracks"):
            return list(fallback_targets)
        try:
            tracks = tracker.snapshot_tracks()
        except Exception:
            return list(fallback_targets)
        results = []
        for tr in tracks:
            last = getattr(tr, "last", None)
            if last is not None:
                results.append(last)
        return results

    def _check_radar_emergency_stop(self, targets: List) -> None:
        if self.controller.car is None:
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
            self.controller.car.emergency_stop()
            self._motion_active = False
            print(
                f"[MainUI] Radar emergency stop: x<{float(getattr(near_target, 'x', 0.0)):.2f}m "
                f"(threshold={threshold:.2f}m)"
            )
            self._log(
                f"雷达急停触发: x<{float(getattr(near_target, 'x', 0.0)):.2f}m 阈值={threshold:.2f}m"
            )

    # ========= 事件处理函数 =========

    def _on_preset_path_clicked(self) -> None:
        """选择预定轨迹"""
        import_choice = "从文件导入..."
        names = [import_choice] + sorted(self._preset_paths.keys())
        choice, ok = QtWidgets.QInputDialog.getItem(
            self,
            "预定轨迹",
            "选择预定轨迹",
            names,
            0,
            False,
        )
        if not ok:
            return
        if choice == import_choice:
            self._import_preset_trajectory_from_file()
            return
        local_points = self._preset_paths.get(choice)
        if not local_points or len(local_points) < 2:
            QtWidgets.QMessageBox.warning(
                self,
                "预定轨迹无效",
                "所选预定轨迹点数不足，请重新规划。",
            )
            self._log(f"预定轨迹加载失败: {choice}")
            return
        ranges = list(self._preset_ranges.get(choice, []))
        self._planned_ranges = ranges if self._should_use_segment_ranges(ranges) else None
        if self._apply_planned_local_points(local_points):
            self._log(
                f"预定轨迹已加载: {choice} 点数={len(local_points)} "
                f"分段={len(ranges)} RCS触发段={sum(1 for r in ranges if r[3])}"
            )

    def _on_delete_preset_clicked(self) -> None:
        """删除预定轨迹"""
        if not self._preset_paths:
            QtWidgets.QMessageBox.information(
                self,
                "无预定轨迹",
                "当前没有可删除的预定轨迹。",
            )
            self._log("删除预定轨迹失败: 列表为空")
            return
        names = sorted(self._preset_paths.keys())
        choice, ok = QtWidgets.QInputDialog.getItem(
            self,
            "删除预定轨迹",
            "选择要删除的预定轨迹",
            names,
            0,
            False,
        )
        if not ok:
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除预定轨迹“{choice}”？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            self._log(f"删除预定轨迹已取消: {choice}")
            return

        self._preset_paths.pop(choice, None)
        self._preset_ranges.pop(choice, None)
        if not self._persist_preset_paths():
            return
        self._log(f"预定轨迹已删除: {choice}")

    def _import_preset_trajectory_from_file(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "导入预定轨迹文件",
            "",
            "CSV / 文本 (*.csv *.txt);;所有文件 (*)",
        )
        if not filename:
            return

        try:
            local_points, ranges = self._parse_trajectory_file(filename)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "导入失败", f"读取轨迹文件失败: {e}")
            self._log(f"预定轨迹文件导入失败: {e}")
            return

        if len(local_points) < 2:
            QtWidgets.QMessageBox.warning(self, "轨迹无效", "轨迹点数量不足（至少需要 2 个点）。")
            self._log("预定轨迹文件导入失败: 点数不足")
            return

        self._planned_ranges = ranges if self._should_use_segment_ranges(ranges) else None
        if not self._apply_planned_local_points(local_points):
            return

        default_name = Path(filename).stem or f"预定轨迹{len(self._preset_paths) + 1}"
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "保存为预定轨迹",
            "请输入预定轨迹名称（留空仅临时加载）",
            text=default_name,
        )
        if ok and name.strip():
            self._save_preset_path(name.strip(), local_points, ranges)
        elif ok:
            self._log("导入轨迹未保存为预定轨迹（仅临时加载）")

        self._log(
            f"轨迹文件已导入: {Path(filename).name} 点数={len(local_points)} "
            f"分段={len(ranges)} RCS触发段={sum(1 for r in ranges if r[3])}"
        )

    def _on_radar_stop_threshold(self) -> None:
        value, ok = QtWidgets.QInputDialog.getDouble(
            self,
            "雷达急停阈值",
            "阈值X (m)",
            float(self._radar_stop_threshold),
            0.1,
            50.0,
            2,
        )
        if not ok:
            return
        self._radar_stop_threshold = max(0.1, float(value))
        self._log(f"雷达急停阈值设置: {self._radar_stop_threshold:.2f}m")

    def _ensure_car_ready(self) -> bool:
        return self.controller.ensure_car_ready(self)

    def _on_line_clicked(self) -> None:
        dist = float(self._traj_default_dist)
        speed = float(self._path_speed)
        if dist <= 0 or speed <= 0:
            QtWidgets.QMessageBox.warning(self, "参数错误", "请检查直线距离和线速度的输入。")
            self._log("直线运动启动失败: 参数解析错误")
            return

        self._motion_active = True
        self._log(f"直线运动启动: 距离={dist:.2f}m 速度={speed:.2f}m/s")
        self.controller.execute_line_movement(self, dist, speed)

    def _on_circle_clicked(self) -> None:
        radius = float(self._traj_default_radius)
        angle = float(self._traj_default_angle)
        speed = float(self._path_speed)
        if radius <= 0 or speed <= 0:
            QtWidgets.QMessageBox.warning(self, "参数错误", "请检查半径、角度和线速度的输入。")
            self._log("圆周运动启动失败: 参数解析错误")
            return

        self._motion_active = True
        self._log(f"圆周运动启动: 半径={radius:.2f}m 角度={angle:.1f}deg 速度={speed:.2f}m/s")
        self.controller.execute_circle_movement(self, radius, angle, speed)

    def _on_calib_clicked(self) -> None:
        """标定目标物位置"""
        if self.target_marker is None:
            return
        self._target_marking_mode = True
        self._log("开始标定目标物位置: 请在轨迹图上点击目标点")

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
        self.target_point = (x, y)
        self.target_marker.setData([x], [y])
        self._target_marking_mode = False
        self._log(f"目标物位置标定: x={x:.2f}m, y={y:.2f}m")

    def _on_return_clicked(self) -> None:
        """倒回（沿原轨迹返回）"""
        if not self.controller.ensure_car_ready(self):
            return

        return_points = self._build_return_path()
        if len(return_points) < 2:
            QtWidgets.QMessageBox.information(
                self,
                "无法倒回",
                "缺少可用的历史/规划轨迹，请先执行一次轨迹或记录实际轨迹。",
            )
            self._log("倒回失败: 无可用轨迹")
            return

        if self.controller.car is None:
            return

        # 先停止当前运动线程
        self.controller.car.stop()
        self._motion_active = True
        self._radar_emergency_active = False

        speed = -max(0.05, float(self._path_speed))
        self._log(f"倒回启动: 点数={len(return_points)} 速度={speed:.2f}m/s (倒车)")

        t = threading.Thread(
            target=self.controller.car.follow_path_with_pid,
            args=(return_points, speed),
            kwargs={"lookahead_distance": 0.4},
            daemon=True,
        )
        t.start()

    def _build_return_path(self) -> List[Tuple[float, float]]:
        """
        优先使用实际轨迹(path_x/path_y)作为倒回路径；若不足，则回退到已加载/规划轨迹。
        """
        points: List[Tuple[float, float]] = []

        pose = get_robot_pose()

        # 1) 已加载的全局轨迹（优先：预定轨迹）
        if len(self.loaded_path_points) >= 2:
            points = list(self.loaded_path_points)
            if pose is not None:
                nearest_idx = self._find_nearest_index(points, pose.x, pose.y)
                points = points[: max(1, nearest_idx + 1)]
        # 2) 实际轨迹
        elif len(self.path_x) >= 2:
            points = list(zip(self.path_x, self.path_y))
            if pose is not None:
                if not points or math.hypot(points[-1][0] - pose.x, points[-1][1] - pose.y) > 0.05:
                    points.append((pose.x, pose.y))
        # 3) 当前规划轨迹（显示用）
        elif len(self.planned_x) >= 2 and len(self.planned_x) == len(self.planned_y):
            points = list(zip(self.planned_x, self.planned_y))

        if len(points) < 2:
            return []

        # 反向并下采样，避免过密
        reversed_points = list(reversed(points))
        return self._downsample_path(reversed_points, min_step=0.1)

    @staticmethod
    def _find_nearest_index(points: List[Tuple[float, float]], x: float, y: float) -> int:
        min_d2 = float("inf")
        best_idx = 0
        for i, (px, py) in enumerate(points):
            dx = px - x
            dy = py - y
            d2 = dx * dx + dy * dy
            if d2 < min_d2:
                min_d2 = d2
                best_idx = i
        return best_idx

    @staticmethod
    def _downsample_path(points: List[Tuple[float, float]], min_step: float = 0.05) -> List[Tuple[float, float]]:
        if len(points) < 2:
            return list(points)
        result = [points[0]]
        last_x, last_y = points[0]
        min_step2 = min_step * min_step
        for x, y in points[1:]:
            dx = x - last_x
            dy = y - last_y
            if dx * dx + dy * dy >= min_step2:
                result.append((x, y))
                last_x, last_y = x, y
        if result[-1] != points[-1]:
            result.append(points[-1])
        return result

    def _on_save_traj(self) -> None:
        """保存轨迹图像"""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存轨迹图像",
            "",
            "PNG 图像 (*.png);;JPEG 图片 (*.jpg *.jpeg);;BMP 图像 (*.bmp);;TIFF 图像 (*.tif *.tiff)",
        )
        if not filename:
            return
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            filename += ".png"
        try:
            exporter = ImageExporter(self.traj_plot.plotItem)
            exporter.parameters()["width"] = 1200
            exporter.export(filename)
            self._log(f"??????: {filename}")
            QtWidgets.QMessageBox.information(self, "保存成功", f"轨迹图已保存至:\n{filename}")
        except Exception as e:
            self._log(f"???????: {e}")
            QtWidgets.QMessageBox.warning(self, "保存失败", f"导出轨迹图像失败: {e}")

    def _on_select_target(self) -> None:
        self._log("提示: 请在雷达图上点击稳定目标点进行锁定")
        QtWidgets.QMessageBox.information(self, "提示", "请在雷达图上点击稳定目标点进行锁定。")

    def _on_load_path(self) -> None:
        self._log("打开轨迹规划")
        self._open_traj_planner()

    def _prompt_path_speed(self) -> Optional[float]:
        speed, ok = QtWidgets.QInputDialog.getDouble(
            self,
            "线速度",
            "线速度 (m/s)",
            float(self._path_speed),
            0.01,
            10.0,
            2,
        )
        if not ok:
            return None
        self._path_speed = float(speed)
        return self._path_speed

    def _prompt_preset_name(self) -> Optional[str]:
        default_name = f"预定轨迹{len(self._preset_paths) + 1}"
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "预定轨迹名称",
            "请输入预定轨迹名称",
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
    ) -> bool:
        if name in self._preset_paths:
            reply = QtWidgets.QMessageBox.question(
                self,
                "覆盖预定轨迹",
                f"预定轨迹“{name}”已存在，是否覆盖？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                self._log(f"预定轨迹保存已取消: {name}")
                return False
        self._preset_paths[name] = list(local_points)
        self._preset_ranges[name] = list(ranges or [])
        if not self._persist_preset_paths():
            return False
        self._log(
            f"预定轨迹已保存: {name} 点数={len(local_points)} "
            f"分段={len(self._preset_ranges[name])}"
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
                    ]
                )
                for name, points in self._preset_paths.items():
                    for idx, (x, y) in enumerate(points):
                        writer.writerow(
                            ["point", name, idx, f"{x:.6f}", f"{y:.6f}", "", "", "", ""]
                        )
                    ranges = self._preset_ranges.get(name, [])
                    for idx, (start_idx, end_idx, speed_sign, rcs_start) in enumerate(ranges):
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
                            ]
                        )
            return True
        except Exception as e:
            self._log(f"预定轨迹保存失败: {e}")
            QtWidgets.QMessageBox.warning(self, "保存失败", f"预定轨迹保存失败: {e}")
            return False

    def _load_preset_paths(self) -> None:
        self._preset_paths.clear()
        self._preset_ranges.clear()
        if not PRESET_PATHS_CSV.exists():
            return
        try:
            temp_points: Dict[str, List[Tuple[int, float, float]]] = {}
            temp_ranges: Dict[str, List[Tuple[int, int, int, int, bool]]] = {}
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
                        if record_type == "point":
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
                            temp_ranges.setdefault(name, []).append(
                                (idx, start_idx, end_idx, speed_sign, rcs_start)
                            )
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

            for name, segments in temp_ranges.items():
                if name not in self._preset_paths:
                    continue
                segments.sort(key=lambda item: item[0])
                self._preset_ranges[name] = [
                    (start_idx, end_idx, speed_sign, rcs_start)
                    for _, start_idx, end_idx, speed_sign, rcs_start in segments
                    if end_idx > start_idx
                ]

            if self._preset_paths:
                self._log(f"已加载预定轨迹: {len(self._preset_paths)}个")
        except Exception as e:
            self._log(f"预定轨迹加载失败: {e}")

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
    def _should_use_segment_ranges(ranges: List[SegmentRange]) -> bool:
        return any((speed_sign < 0) or rcs_start for _, _, speed_sign, rcs_start in ranges)

    def _build_ranges_from_point_metadata(
        self,
        rows: List[Tuple[float, float, Optional[int], Optional[int], bool]],
    ) -> List[SegmentRange]:
        if not rows or not any(seg is not None for _, _, seg, _, _ in rows):
            return []

        resolved_segment_ids: List[int] = []
        cur_seg = 1
        for _, _, seg, _, _ in rows:
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
                for _, _, _, seg_speed, _ in rows[start : end + 1]:
                    if seg_speed is not None:
                        speed_sign = self._normalize_speed_sign(seg_speed)
                        break
                rcs_start = any(rcs for _, _, _, _, rcs in rows[start : end + 1])
                ranges.append((start, end, speed_sign, rcs_start))
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

        rows: List[Tuple[float, float, Optional[int], Optional[int], bool]] = []
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
                rows.append((x, y, seg_val, speed_sign, rcs_start))
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
                if len(parts) >= 3:
                    try:
                        seg_val = int(float(parts[2]))
                    except ValueError:
                        seg_val = None
                if len(parts) >= 4:
                    speed_sign = self._normalize_speed_sign(parts[3])
                if len(parts) >= 5:
                    rcs_start = self._parse_bool_flag(parts[4])
                rows.append((x, y, seg_val, speed_sign, rcs_start))

        if len(rows) < 2:
            raise ValueError("轨迹点数量不足（至少需要 2 个点）。")

        local_points = [(x, y) for x, y, _, _, _ in rows]
        ranges = self._build_ranges_from_point_metadata(rows)
        return local_points, ranges

    def _open_traj_planner(self) -> None:
        dialog = TrajectoryPlannerDialog(
            self,
            default_dist=self._traj_default_dist,
            default_radius=self._traj_default_radius,
            default_angle=self._traj_default_angle,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            self._log("轨迹规划已取消")
            return
        self._traj_default_dist = float(dialog.line_dist.value())
        self._traj_default_radius = float(dialog.circle_radius.value())
        self._traj_default_angle = float(dialog.circle_angle.value())
        segments = dialog.get_segments()
        if not segments:
            QtWidgets.QMessageBox.information(self, "无轨迹段", "请先添加直线或圆弧段。")
            self._log("轨迹规划失败: 未添加轨迹段")
            return
        local_points, ranges = self._build_planned_path(segments)
        self._planned_ranges = ranges if self._should_use_segment_ranges(ranges) else None
        self._log(f"轨迹规划生成: 段数={len(segments)} 点数={len(local_points)}")
        if len(local_points) < 2:
            QtWidgets.QMessageBox.information(self, "轨迹无效", "规划轨迹点数量不足。")
            self._log("轨迹规划失败: 点数不足")
            return
        preset_name = self._prompt_preset_name()
        if preset_name:
            self._save_preset_path(preset_name, local_points, ranges)
        else:
            self._log("轨迹规划未保存为预定轨迹")
        self._apply_planned_local_points(local_points)

    def _apply_planned_path(self, segments: List[dict]) -> None:
        local_points, ranges = self._build_planned_path(segments)
        self._planned_ranges = ranges if self._should_use_segment_ranges(ranges) else None
        self._log(f"轨迹规划生成: 段数={len(segments)} 点数={len(local_points)}")
        if len(local_points) < 2:
            QtWidgets.QMessageBox.information(self, "轨迹无效", "规划轨迹点数量不足。")
            self._log("轨迹规划失败: 点数不足")
            return
        self._apply_planned_local_points(local_points)

    def _apply_planned_local_points(self, local_points: List[Tuple[float, float]]) -> bool:
        pose = get_robot_pose()
        if pose is None:
            QtWidgets.QMessageBox.warning(
                self,
                "定位无效",
                "当前未获取有效 GNSS / INS 位姿，无法将规划轨迹对齐到车头。",
            )
            self._log("轨迹规划失败: 无有效位姿")
            return False
        if len(local_points) < 2:
            QtWidgets.QMessageBox.information(self, "轨迹无效", "规划轨迹点数量不足。")
            self._log("轨迹规划失败: 点数不足")
            return False
        self.loaded_path_local_points = list(local_points)

        cos_yaw = math.cos(pose.yaw)
        sin_yaw = math.sin(pose.yaw)
        global_points = []
        for xr, yr in local_points:
            gx = pose.x + xr * cos_yaw - yr * sin_yaw
            gy = pose.y + xr * sin_yaw + yr * cos_yaw
            global_points.append((gx, gy))

        self.loaded_path_points = global_points
        self.planned_x = [p[0] for p in global_points]
        self.planned_y = [p[1] for p in global_points]
        self.traj_planned_curve.setData(self.planned_x, self.planned_y)

        if self.planned_x and self.planned_y:
            min_x, max_x = min(self.planned_x), max(self.planned_x)
            min_y, max_y = min(self.planned_y), max(self.planned_y)
            margin = 0.5
            self.traj_plot.setXRange(min_x - margin, max_x + margin, padding=0)
            self.traj_plot.setYRange(min_y - margin, max_y + margin, padding=0)
        self._log(f"轨迹规划应用完成: 全局点数={len(global_points)}")
        return True

    def _reanchor_loaded_path(self) -> bool:
        local_points = self.loaded_path_local_points
        if len(local_points) < 2:
            QtWidgets.QMessageBox.warning(
                self,
                "轨迹无效",
                "缺少本地轨迹点，无法重新锚定，请重新规划或加载轨迹。",
            )
            self._log("轨迹重新锚定失败: 缺少本地轨迹点")
            return False
        pose = get_robot_pose()
        if pose is None:
            QtWidgets.QMessageBox.warning(
                self,
                "定位无效",
                "无法获取当前位姿，无法重新锚定轨迹。",
            )
            self._log("轨迹重新锚定失败: 无有效位姿")
            return False

        cos_yaw = math.cos(pose.yaw)
        sin_yaw = math.sin(pose.yaw)
        global_points = []
        for xr, yr in local_points:
            gx = pose.x + xr * cos_yaw - yr * sin_yaw
            gy = pose.y + xr * sin_yaw + yr * cos_yaw
            global_points.append((gx, gy))

        self.loaded_path_points = global_points
        self.planned_x = [p[0] for p in global_points]
        self.planned_y = [p[1] for p in global_points]
        self.traj_planned_curve.setData(self.planned_x, self.planned_y)

        if self.planned_x and self.planned_y:
            min_x, max_x = min(self.planned_x), max(self.planned_x)
            min_y, max_y = min(self.planned_y), max(self.planned_y)
            margin = 0.5
            self.traj_plot.setXRange(min_x - margin, max_x + margin, padding=0)
            self.traj_plot.setYRange(min_y - margin, max_y + margin, padding=0)

        self._log(f"轨迹重新锚定完成: 全局点数={len(global_points)}")
        return True
        return True

    def _build_planned_path(
        self,
        segments: List[dict],
    ) -> Tuple[List[Tuple[float, float]], List[SegmentRange]]:
        step = 0.2
        points: List[Tuple[float, float]] = [(0.0, 0.0)]
        ranges: List[SegmentRange] = []
        x, y = 0.0, 0.0
        heading = 0.0

        def add_range(start_idx: int, end_idx: int, speed_sign: int, rcs_start: bool = False) -> None:
            if end_idx - start_idx >= 1:
                ranges.append((start_idx, end_idx, speed_sign, bool(rcs_start)))

        for seg in segments:
            if seg["type"] == "line":
                dist = float(seg.get("distance", 0.0))
                if abs(dist) < 1e-3:
                    continue
                start_idx = len(points) - 1
                x, y, heading = self._append_line_segment(points, x, y, heading, dist, step)
                end_idx = len(points) - 1
                add_range(start_idx, end_idx, 1, bool(seg.get("rcs_start")))
                if seg.get("round_trip"):
                    start_idx = len(points) - 1
                    heading_before = heading
                    x, y, heading = self._append_line_segment(points, x, y, heading, -dist, step)
                    end_idx = len(points) - 1
                    add_range(start_idx, end_idx, -1, False)
                    heading = heading_before
            elif seg["type"] == "circle":
                radius = float(seg.get("radius", 0.0))
                angle = float(seg.get("angle", 0.0))
                direction = seg.get("direction", "ccw")
                if radius <= 1e-3 or abs(angle) < 1e-3:
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
                    step,
                )
                end_idx = len(points) - 1
                add_range(start_idx, end_idx, 1, bool(seg.get("rcs_start")))
        return points, ranges

    def _append_line_segment(
        self,
        points: List[Tuple[float, float]],
        x: float,
        y: float,
        heading: float,
        dist: float,
        step: float,
    ) -> Tuple[float, float, float]:
        n = max(2, int(abs(dist) / step) + 1)
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
        n = max(2, int(arc_len / step) + 1)
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

    def _on_run_path(self) -> None:
        speed = self._prompt_path_speed()
        if speed is None:
            return
        if not self._reanchor_loaded_path():
            return

        self._motion_active = True
        if self._planned_ranges:
            self._log(f"执行轨迹: 速度={speed:.2f}m/s 分段执行")
            t = threading.Thread(
                target=self._run_planned_ranges,
                args=(speed,),
                daemon=True,
            )
            t.start()
        else:
            self._log(f"执行轨迹: 速度={speed:.2f}m/s PID修正")
            self.controller.execute_loaded_path(self, speed)

    def _run_planned_ranges(self, speed: float) -> None:
        if self.controller.car is None:
            return
        if not self.loaded_path_points or not self._planned_ranges:
            return

        for seg_idx, (start_idx, end_idx, speed_sign, rcs_start) in enumerate(self._planned_ranges, start=1):
            if end_idx <= start_idx:
                continue
            if rcs_start:
                self.segment_rcs_start_requested.emit(seg_idx)
            segment_points = self.loaded_path_points[start_idx : end_idx + 1]
            if len(segment_points) < 2:
                continue
            seg_speed = speed * speed_sign
            lookahead = 0.4 if seg_speed < 0 else 0.6
            self.controller.car.follow_path_with_pid(
                segment_points,
                seg_speed,
                lookahead_distance=lookahead,
            )
        self._motion_active = False

    def _on_segment_rcs_start_requested(self, seg_idx: int) -> None:
        if self.tracked_target_id is None:
            self._log(f"第{seg_idx}段 RCS触发失败: 请先锁定雷达目标")
            return
        self._log(f"第{seg_idx}段开始: 触发RCS记录 目标ID={self.tracked_target_id}")
        self._on_rcs_start()

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
        oid = int(getattr(target, "oid", getattr(target, "id", getattr(target, "cid", 0))))
        self.tracked_target_id = oid
        if hasattr(self.controller, "set_selected_radar_id"):
            self.controller.set_selected_radar_id(oid)

        try:
            self._log(
                f"雷达目标锁定: id={oid} x={float(getattr(target, 'x', 0.0)):.2f}m "
                f"y={float(getattr(target, 'y', 0.0)):.2f}m rcs={float(getattr(target, 'rcs_db', 0.0)):.1f}dBsm"
            )
        except Exception:
            self._log(f"雷达目标锁定: id={oid}")

        if self._planned_ranges and any(r[3] for r in self._planned_ranges):
            self._log("已锁定目标，等待轨迹分段触发RCS记录")
        else:
            self._log("已锁定目标。当前轨迹未配置RCS触发段，不会自动开始记录")

    # ========= 定时器刷新 =========

    def _on_timer(self) -> None:
        # 1) 位姿 & 轨迹
        pose: Optional[PoseSolution] = get_robot_pose()
        if pose is not None:
            # 更新车头方向箭头
            arrow_length = 0.5  # 箭头长度
            self.heading_arrow.setPos(pose.x, pose.y)
            # 轨迹箭头旋转 180° 修正方向
            self.heading_arrow.setStyle(angle=180 - math.degrees(pose.yaw))  # 转换为度，并调整方向
            
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
            # 航向角真实值：以正北为 0°，顺时针为正
            heading_deg = (90.0 - yaw_deg) % 360.0
            text = (
                f"源: {pose.source} | "
                f"lat={pose.lat:.7f}, lon={pose.lon:.7f}, h={pose.height:.2f} m | "
                f"x={pose.x:.2f} m, y={pose.y:.2f} m (ENU) | "
                f"yaw={pose.yaw:.2f} rad ({yaw_deg:.1f}°)"
            )
            if pose.ins_status:
                text += f" | INS_STATUS={pose.ins_status}"
            if pose.ins_pos_type:
                text += f" | INS_POS={pose.ins_pos_type}"
            self.label_pose.setText(text)
            
            # 更新航向角显示
            self.label_heading.setText(f"航向角: {heading_deg:.1f}°")
        else:
            self.label_pose.setText("等待 INSPVAXA / INS 对准 / 解算中...")
            self.heading_arrow.setPos(0, 0)
            self.label_heading.setText("航向角: --")

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

        imu_ok = (
            status.mode == "INS"
            and status.age_inspvax is not None
            and status.age_inspvax < 1.0
        )
        car_ok = self.controller.car is not None
        self.btn_run_path.setEnabled(imu_ok and car_ok and len(self.loaded_path_points) >= 2)

        # 3) 电量显示（树莓派 + 小车）
        pi_text, car_text = self.controller.get_power_status()
        self.label_power.setText(pi_text + " | " + car_text)

        # 4) CAN 初始化状态
        can_text = self.controller.get_can_status()
        self.label_can.setText(can_text)

        # 5) 雷达目标 + 锁定目标信息
        tracked_text = self.controller.get_radar_status()
        self.label_tracked.setText(tracked_text)
        
        # 更新雷达散点图
        if self.controller.radar is not None:
            targets = self.controller.radar.get_targets_snapshot()
            spots = []
            for t in targets:
                oid = int(getattr(t, "oid", getattr(t, "id", getattr(t, "cid", 0))))
                is_selected = self.tracked_target_id is not None and oid == self.tracked_target_id
                brush = pg.mkBrush(0, 180, 0, 180) if is_selected else pg.mkBrush(255, 0, 0, 120)
                size = 12 if is_selected else 9
                spots.append({"pos": (t.y, t.x), "data": t, "brush": brush, "size": size})
            self.radar_scatter.setData(spots)
            safety_targets = self._get_radar_safety_targets(targets)
            self._check_radar_emergency_stop(safety_targets)
            self._update_rcs_recording(targets)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
