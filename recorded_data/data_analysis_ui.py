# -*- coding: utf-8 -*-
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg


def _configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                continue


_configure_utf8_stdio()

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "#0D47A1")
pg.setConfigOption("antialias", True)

_FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
    "PingFang SC",
]

_HEADER_LABELS: Dict[str, str] = {
    "row_index": "行号",
    "travel_distance_m": "当前运动距离(m)",
    "motion_distance_m": "当前运动距离(m)",
    "motion_distance_signed_m": "带方向运动距离(m)",
    "motion_distance_total_m": "总运动距离(m)",
    "motion_distance_total_signed_m": "带方向总运动距离(m)",
    "motion_distance_kind": "运动距离类型",
    "motion_distance_label": "运动距离类型中文",
    "record_index": "记录序号",
    "sample_index": "采样序号",
    "run_key": "运行标识",
    "timestamp": "时间戳",
    "time_local": "本地时间",
    "started_at": "开始时间",
    "relative_time_s": "相对时间(s)",
    "dt_s": "控制周期(s)",
    "pose_age_s": "位姿时延(s)",
    "dist_to_goal_m": "距终点距离(m)",
    "run_label": "运行名称",
    "tracking_mode": "控制模式",
    "tracking_mode_label": "控制模式中文",
    "speed_mps": "请求速度(m/s)",
    "speed_sign": "方向符号",
    "motion_direction": "运动方向",
    "motion_direction_label": "运动方向中文",
    "nominal_speed_abs_mps": "标称速度绝对值(m/s)",
    "lookahead_base_m": "基础前瞻距离(m)",
    "arrival_dist_m": "到点阈值(m)",
    "slow_down_dist_m": "减速距离(m)",
    "stanley_gain": "Stanley增益",
    "stanley_softening_distance_m": "Stanley软化距离(m)",
    "stanley_max_bias_deg": "Stanley最大偏角(deg)",
    "lookahead_heading_weight": "前瞻视线航向权重",
    "lookahead_heading_max_bias_deg": "前瞻视线最大偏角(deg)",
    "target_heading_weight": "目标点航向权重",
    "target_heading_max_bias_deg": "目标点最大偏角(deg)",
    "heading_preview_threshold_deg": "航向预瞄阈值(deg)",
    "heading_preview_extra_gain": "航向预瞄额外增益",
    "heading_preview_max_distance_m": "航向预瞄最大距离(m)",
    "yaw_rate_feedback_timeout_s": "角速度反馈超时(s)",
    "max_v_rate_mps2": "线速度变化率上限(m/s^2)",
    "max_w_rate_radps2": "角速度变化率上限(rad/s^2)",
    "max_w_step_radps": "单周期角速度步长(rad/s)",
    "gain_schedule_start_mps": "增益调度起始速度(m/s)",
    "gain_schedule_end_mps": "增益调度结束速度(m/s)",
    "min_gain_scale": "最小增益缩放",
    "reverse_heading_soft_start_s": "倒车航向软启动时间(s)",
    "reverse_heading_soft_start_gain_min": "倒车航向软启动最小增益",
    "reverse_heading_soft_start_w_scale_min": "倒车角速度软启动最小比例",
    "lateral_pid_kp": "横向PID-Kp",
    "lateral_pid_ki": "横向PID-Ki",
    "lateral_pid_kd": "横向PID-Kd",
    "heading_pid_kp": "航向PID-Kp",
    "heading_pid_ki": "航向PID-Ki",
    "heading_pid_kd": "航向PID-Kd",
    "yaw_rate_pid_kp": "角速度内环PID-Kp",
    "yaw_rate_pid_ki": "角速度内环PID-Ki",
    "yaw_rate_pid_kd": "角速度内环PID-Kd",
    "segment_index": "分段序号",
    "segment_kind": "分段类型",
    "segment_trajectory_name": "分段轨迹名",
    "segment_start_idx": "分段起点索引",
    "segment_end_idx": "分段终点索引",
    "segment_length_m": "分段长度(m)",
    "segment_cruise_speed_mps": "分段巡航速度(m/s)",
    "segment_accel_dist_m": "分段加速距离(m)",
    "segment_decel_dist_m": "分段减速距离(m)",
    "segment_start_speed_mps": "分段起始速度(m/s)",
    "segment_end_speed_mps": "分段末端速度(m/s)",
    "segment_stop_at_end": "段末是否停车",
    "motion_request_distance_m": "请求直线距离(m)",
    "waypoints_count": "轨迹点数",
    "duration_s": "持续时间(s)",
    "samples": "采样次数",
    "feedback_samples": "反馈样本数",
    "peak_abs_lateral_error_m": "横向误差峰值(m)",
    "peak_abs_heading_error_deg": "航向误差峰值(deg)",
    "peak_abs_yaw_rate_error_radps": "角速度跟踪误差峰值(rad/s)",
    "peak_abs_cmd_w_radps": "命令角速度峰值(rad/s)",
    "peak_abs_feedback_w_radps": "反馈角速度峰值(rad/s)",
    "completed": "是否完成",
    "exit_reason": "退出原因",
    "exit_reason_text": "退出原因中文",
    "stop_reason": "停止原因",
    "current_x_m": "当前位置X(m)",
    "current_y_m": "当前位置Y(m)",
    "current_yaw_rad": "当前航向(rad)",
    "motion_yaw_rad": "运动方向航向(rad)",
    "path_s_m": "路径进度(m)",
    "nearest_idx": "最近点索引",
    "nearest_seg_idx": "最近线段索引",
    "lookahead_idx": "前瞻点索引",
    "lookahead_ctrl_m": "动态前瞻距离(m)",
    "path_heading_rad": "路径切向角(rad)",
    "heading_basis_rad": "航向基准(rad)",
    "target_heading_rad": "目标航向(rad)",
    "stanley_term_rad": "Stanley航向修正(rad)",
    "stanley_lookahead_distance_m": "Stanley前瞻距离(m)",
    "path_curvature_inv_m": "路径曲率(1/m)",
    "path_ff_w_radps": "曲率前馈角速度(rad/s)",
    "path_ff_w_adjusted_radps": "速度修正后前馈角速度(rad/s)",
    "nearest_x_m": "最近点X(m)",
    "nearest_y_m": "最近点Y(m)",
    "lookahead_x_m": "前瞻点X(m)",
    "lookahead_y_m": "前瞻点Y(m)",
    "lateral_error_m": "横向/径向误差(m)：直线/Stanley 为路径右正左负；圆弧 orbit 为半径误差 current_r-R",
    "heading_error_rad": "航向误差(rad)",
    "heading_error_deg": "航向误差(deg)",
    "lateral_assist_scale": "横向辅助缩放",
    "gain_scale": "增益缩放",
    "reverse_heading_soft_scale": "倒车航向软启动比例",
    "reverse_w_limit_scale": "倒车角速度限幅比例",
    "outer_desired_w_raw_radps": "外环原始目标角速度(rad/s)",
    "outer_desired_w_pre_filter_radps": "外环滤波前目标角速度(rad/s)",
    "angular_speed_limit_radps": "角速度限幅(rad/s)",
    "curvature_excess": "附加曲率负担",
    "speed_factor": "误差降速系数",
    "speed_scale": "最终速度缩放",
    "stale_speed_scale": "位姿老化降速比例",
    "lateral_pid_error_m": "横向PID误差(m)",
    "lateral_pid_dt_s": "横向PID周期(s)",
    "lateral_pid_integral_m_s": "横向PID积分(m*s)",
    "lateral_pid_derivative_mps": "横向PID微分(m/s)",
    "lateral_pid_p_term_radps": "横向PID-P输出(rad/s)",
    "lateral_pid_i_term_radps": "横向PID-I输出(rad/s)",
    "lateral_pid_d_term_radps": "横向PID-D输出(rad/s)",
    "lateral_pid_output_radps": "横向PID总输出(rad/s)",
    "heading_pid_error_rad": "航向PID误差(rad)",
    "heading_pid_dt_s": "航向PID周期(s)",
    "heading_pid_integral_rad_s": "航向PID积分(rad*s)",
    "heading_pid_derivative_radps": "航向PID微分(rad/s)",
    "heading_pid_p_term_radps": "航向PID-P输出(rad/s)",
    "heading_pid_i_term_radps": "航向PID-I输出(rad/s)",
    "heading_pid_d_term_radps": "航向PID-D输出(rad/s)",
    "heading_pid_output_radps": "航向PID总输出(rad/s)",
    "feedback_v_mps": "反馈线速度(m/s)",
    "feedback_w_radps": "反馈角速度(rad/s)",
    "feedback_age_s": "角速度反馈时延(s)",
    "yaw_rate_feedback_valid": "角速度反馈有效",
    "yaw_rate_error_radps": "角速度跟踪误差(rad/s)",
    "yaw_rate_correction_radps": "角速度内环修正(rad/s)",
    "yaw_rate_pid_dt_s": "角速度PID周期(s)",
    "yaw_rate_pid_integral_rad": "角速度PID积分(rad)",
    "yaw_rate_pid_derivative_radps2": "角速度PID微分(rad/s^2)",
    "yaw_rate_pid_p_term_radps": "角速度PID-P输出(rad/s)",
    "yaw_rate_pid_i_term_radps": "角速度PID-I输出(rad/s)",
    "yaw_rate_pid_d_term_radps": "角速度PID-D输出(rad/s)",
    "yaw_rate_pid_output_radps": "角速度PID总输出(rad/s)",
    "outer_desired_w_radps": "外环目标角速度(rad/s)",
    "profile_speed_mps": "速度曲线参考(m/s)",
    "desired_v_mps": "目标线速度(m/s)",
    "desired_w_radps": "目标角速度(rad/s)",
    "cmd_v_mps": "下发线速度(m/s)",
    "cmd_w_radps": "下发角速度(rad/s)",
    "circle_center_x_m": "圆心X(m)",
    "circle_center_y_m": "圆心Y(m)",
    "circle_radius_m": "目标圆半径(m)",
    "circle_radius_error_m": "半径误差(m)",
    "circle_tangent_heading_rad": "圆切线航向(rad)",
    "circle_progress_deg": "圆弧完成角度(deg)",
    "circle_remaining_deg": "圆弧剩余角度(deg)",
    "circle_nominal_w_radps": "圆弧名义角速度(rad/s)",
}


def _header_label(key: str) -> str:
    text = str(key or "").strip()
    return _HEADER_LABELS.get(text, text)


def _header_option_text(key: str) -> str:
    text = str(key or "").strip()
    label = _header_label(text)
    if not label or label == text:
        return text
    return f"{label} ({text})"


def _iter_font_dirs() -> List[Path]:
    here = Path(__file__).resolve().parent
    return [
        here / "fonts",
        here.parent / "fonts",
    ]


def _apply_cjk_font(app: QtWidgets.QApplication) -> None:
    db = QtGui.QFontDatabase()
    for font_dir in _iter_font_dirs():
        if not font_dir.is_dir():
            continue
        for ext in ("*.ttf", "*.otf", "*.ttc"):
            for font_path in font_dir.glob(ext):
                try:
                    QtGui.QFontDatabase.addApplicationFont(str(font_path))
                except Exception:
                    continue

    families = {str(name) for name in db.families()}
    family = next((name for name in _FONT_CANDIDATES if name in families), None)
    if not family:
        return

    font = QtGui.QFont(app.font())
    font.setFamily(family)
    if font.pointSizeF() <= 0:
        font.setPointSizeF(10.0)
    app.setFont(font)


def _parse_float(value: object) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return float(result)


def _read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as fp:
                reader = csv.DictReader(fp)
                if reader.fieldnames is None:
                    raise ValueError("CSV 缺少表头")
                headers = [str(name or "").strip() for name in reader.fieldnames]
                rows = [{str(k or "").strip(): str(v or "") for k, v in row.items()} for row in reader]
                return headers, rows
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"读取 CSV 失败: {last_error}")


def _append_derived_motion_distance(
    headers: List[str],
    rows: List[Dict[str, str]],
) -> List[str]:
    if "travel_distance_m" in headers:
        return headers
    if "motion_distance_m" in headers:
        for row in rows:
            row["travel_distance_m"] = str(row.get("motion_distance_m", "") or "")
        return [*headers, "travel_distance_m"]
    if "motion_distance_signed_m" in headers:
        for row in rows:
            value = _parse_float(row.get("motion_distance_signed_m", ""))
            row["travel_distance_m"] = "" if value is None else f"{abs(value):.6f}"
        return [*headers, "travel_distance_m"]
    if "current_x_m" not in headers or "current_y_m" not in headers:
        return headers

    prev_x: Optional[float] = None
    prev_y: Optional[float] = None
    total_distance = 0.0
    for row in rows:
        x = _parse_float(row.get("current_x_m", ""))
        y = _parse_float(row.get("current_y_m", ""))
        if x is None or y is None:
            row["travel_distance_m"] = ""
            continue
        if prev_x is not None and prev_y is not None:
            total_distance += math.hypot(x - prev_x, y - prev_y)
        row["travel_distance_m"] = f"{total_distance:.6f}"
        prev_x = x
        prev_y = y

    return [*headers, "travel_distance_m"]


@dataclass
class CsvDataset:
    path: Path
    headers: List[str]
    rows: List[Dict[str, str]]
    numeric_headers: List[str]

    @classmethod
    def load(cls, path: Path) -> "CsvDataset":
        headers, rows = _read_csv_rows(path)
        headers = _append_derived_motion_distance(headers, rows)
        numeric_headers = ["row_index"]
        for header in headers:
            numeric_count = 0
            for row in rows:
                if _parse_float(row.get(header, "")) is not None:
                    numeric_count += 1
            if numeric_count >= 2 or (numeric_count >= 1 and len(rows) <= 1):
                numeric_headers.append(header)
        return cls(path=path, headers=headers, rows=rows, numeric_headers=numeric_headers)

    def row_count(self) -> int:
        return len(self.rows)

    def xy_series(self, x_key: str, y_key: str) -> Tuple[List[float], List[float]]:
        xs: List[float] = []
        ys: List[float] = []
        for row_index, row in enumerate(self.rows, start=1):
            x_value = float(row_index) if x_key == "row_index" else _parse_float(row.get(x_key, ""))
            y_value = float(row_index) if y_key == "row_index" else _parse_float(row.get(y_key, ""))
            if x_value is None or y_value is None:
                continue
            xs.append(x_value)
            ys.append(y_value)
        return xs, ys

    def group_headers(self) -> List[str]:
        preferred = [
            "segment_index",
            "segment_trajectory_name",
            "motion_distance_label",
            "motion_direction_label",
            "motion_direction",
            "run_label",
            "run_key",
            "tracking_mode",
        ]
        candidates: List[str] = []
        for header in self.headers:
            unique_values = {
                str(row.get(header, "")).strip()
                for row in self.rows
                if str(row.get(header, "")).strip()
            }
            if 2 <= len(unique_values) <= 32:
                candidates.append(header)
        ordered: List[str] = []
        for header in preferred:
            if header in candidates:
                ordered.append(header)
        for header in candidates:
            if header not in ordered:
                ordered.append(header)
        return ordered

    def grouped_xy_series(
        self,
        x_key: str,
        y_key: str,
        group_key: str,
    ) -> Dict[str, Tuple[List[float], List[float]]]:
        groups: Dict[str, Tuple[List[float], List[float]]] = {}
        for row_index, row in enumerate(self.rows, start=1):
            x_value = float(row_index) if x_key == "row_index" else _parse_float(row.get(x_key, ""))
            y_value = float(row_index) if y_key == "row_index" else _parse_float(row.get(y_key, ""))
            if x_value is None or y_value is None:
                continue
            group_value = str(row.get(group_key, "")).strip() or "(空)"
            xs, ys = groups.setdefault(group_value, ([], []))
            xs.append(x_value)
            ys.append(y_value)
        return groups


class DataAnalysisWindow(QtWidgets.QMainWindow):
    def __init__(self, data_root: Optional[Path] = None) -> None:
        super().__init__()
        self._data_root = (data_root or Path(__file__).resolve().parent).resolve()
        self._current_dataset: Optional[CsvDataset] = None
        self._current_file_path: Optional[Path] = None
        self._dataset_cache: Dict[Path, CsvDataset] = {}
        self._current_plot_item = None
        self._y_axis_actions: Dict[str, QtWidgets.QAction] = {}
        self._updating_y_axis_actions = False

        self.setWindowTitle("运动数据分析 UI")
        self.resize(1480, 920)

        self._build_ui()
        self._refresh_file_list()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        header = QtWidgets.QLabel(
            f"数据根目录: {self._data_root}"
        )
        header.setWordWrap(True)
        main_layout.addWidget(header)

        top_bar = QtWidgets.QHBoxLayout()
        self.file_type_combo = QtWidgets.QComboBox()
        self.file_type_combo.addItem("全部 CSV", "all")
        self.file_type_combo.addItem("运动轨迹", "motion")
        self.file_type_combo.addItem("汇总指标", "metrics")
        self.file_type_combo.addItem("其他 CSV", "other")
        self.file_type_combo.currentIndexChanged.connect(self._refresh_file_list)

        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("按文件名过滤")
        self.search_edit.textChanged.connect(self._refresh_file_list)

        self.refresh_button = QtWidgets.QPushButton("刷新列表")
        self.refresh_button.clicked.connect(self._refresh_file_list)

        self.open_button = QtWidgets.QPushButton("打开CSV...")
        self.open_button.clicked.connect(self._open_external_csv)

        top_bar.addWidget(self.file_type_combo, 0)
        top_bar.addWidget(self.search_edit, 1)
        top_bar.addWidget(self.refresh_button, 0)
        top_bar.addWidget(self.open_button, 0)
        main_layout.addLayout(top_bar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter, 1)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.file_list = QtWidgets.QListWidget()
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.file_list.currentItemChanged.connect(self._on_file_selected)
        self.file_list.itemSelectionChanged.connect(self._on_file_selection_changed)
        left_layout.addWidget(self.file_list, 1)

        self.file_list_status = QtWidgets.QLabel("未扫描文件")
        self.file_list_status.setWordWrap(True)
        left_layout.addWidget(self.file_list_status)

        splitter.addWidget(left_panel)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        controls = QtWidgets.QHBoxLayout()
        self.overlay_mode_combo = QtWidgets.QComboBox()
        self.overlay_mode_combo.addItem("单文件", "single")
        self.overlay_mode_combo.addItem("当前文件按字段叠加", "group")
        self.overlay_mode_combo.addItem("多文件叠加", "files")
        self.overlay_mode_combo.currentIndexChanged.connect(self._on_overlay_mode_changed)

        self.group_combo = QtWidgets.QComboBox()
        self.group_combo.currentIndexChanged.connect(self._update_plot)

        self.x_combo = QtWidgets.QComboBox()
        self.x_combo.currentIndexChanged.connect(self._update_plot)

        self.y_axis_button = QtWidgets.QToolButton()
        self.y_axis_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.y_axis_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.y_axis_button.setArrowType(QtCore.Qt.DownArrow)
        self.y_axis_button.setText("请选择Y轴")
        self.y_axis_button.setMinimumWidth(220)
        self.y_axis_menu = QtWidgets.QMenu(self.y_axis_button)
        self.y_axis_button.setMenu(self.y_axis_menu)

        self.equal_aspect_check = QtWidgets.QCheckBox("等比例坐标")
        self.equal_aspect_check.stateChanged.connect(self._update_plot)

        self.plot_button = QtWidgets.QPushButton("重新绘图")
        self.plot_button.clicked.connect(self._update_plot)

        controls.addWidget(QtWidgets.QLabel("叠加"))
        controls.addWidget(self.overlay_mode_combo, 0)
        controls.addWidget(QtWidgets.QLabel("分组"))
        controls.addWidget(self.group_combo, 1)
        controls.addWidget(QtWidgets.QLabel("X轴"))
        controls.addWidget(self.x_combo, 1)
        controls.addWidget(QtWidgets.QLabel("Y轴(多选)"))
        controls.addWidget(self.y_axis_button, 1)
        controls.addWidget(self.equal_aspect_check, 0)
        controls.addWidget(self.plot_button, 0)
        right_layout.addLayout(controls)

        self.dataset_info = QtWidgets.QLabel("请选择左侧 CSV 文件")
        self.dataset_info.setWordWrap(True)
        right_layout.addWidget(self.dataset_info)

        scope_row = QtWidgets.QHBoxLayout()
        scope_row.addWidget(QtWidgets.QLabel("二次分段"))
        self.segment_scope_combo = QtWidgets.QComboBox()
        self.segment_scope_combo.setMinimumWidth(260)
        self.segment_scope_combo.currentIndexChanged.connect(self._on_segment_scope_changed)
        scope_row.addWidget(self.segment_scope_combo, 2)
        scope_row.addWidget(QtWidgets.QLabel("行号"))
        self.row_start_spin = QtWidgets.QSpinBox()
        self.row_start_spin.setMinimum(1)
        self.row_start_spin.setMaximum(1)
        self.row_start_spin.setValue(1)
        self.row_end_spin = QtWidgets.QSpinBox()
        self.row_end_spin.setMinimum(1)
        self.row_end_spin.setMaximum(1)
        self.row_end_spin.setValue(1)
        self.row_start_spin.valueChanged.connect(self._on_manual_row_range_changed)
        self.row_end_spin.valueChanged.connect(self._on_manual_row_range_changed)
        scope_row.addWidget(self.row_start_spin, 0)
        scope_row.addWidget(QtWidgets.QLabel("—"), 0)
        scope_row.addWidget(self.row_end_spin, 0)
        self.apply_scope_button = QtWidgets.QPushButton("应用范围")
        self.apply_scope_button.clicked.connect(self._update_plot)
        scope_row.addWidget(self.apply_scope_button, 0)
        right_layout.addLayout(scope_row)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.addLegend()
        right_layout.addWidget(self.plot_widget, 1)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)

        self.statusBar().showMessage("就绪")
        self._update_overlay_controls()

    def _classify_file(self, path: Path) -> str:
        name = path.name.lower()
        if name.startswith("motion_trace_"):
            return "motion"
        if name.startswith("tracking_metrics_"):
            return "metrics"
        return "other"

    def _scan_csv_files(self) -> List[Path]:
        return sorted(
            self._data_root.rglob("*.csv"),
            key=lambda p: (p.stat().st_mtime, str(p).lower()),
            reverse=True,
        )

    def _refresh_file_list(self) -> None:
        selected_path_texts = {
            str(item.data(QtCore.Qt.UserRole) or "")
            for item in self.file_list.selectedItems()
        }
        current_path_text = ""
        current_item = self.file_list.currentItem()
        if current_item is not None:
            current_path_text = str(current_item.data(QtCore.Qt.UserRole) or "")

        self.file_list.blockSignals(True)
        self.file_list.clear()

        file_type = str(self.file_type_combo.currentData() or "all")
        keyword = self.search_edit.text().strip().lower()
        all_files = self._scan_csv_files()
        filtered_files: List[Path] = []
        for path in all_files:
            if file_type != "all" and self._classify_file(path) != file_type:
                continue
            rel_text = str(path.relative_to(self._data_root)).replace("\\", "/")
            if keyword and keyword not in rel_text.lower():
                continue
            filtered_files.append(path)

        for path in filtered_files:
            rel_text = str(path.relative_to(self._data_root)).replace("\\", "/")
            stat = path.stat()
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
            size_kb = stat.st_size / 1024.0
            item = QtWidgets.QListWidgetItem(f"{rel_text}\n{stamp} | {size_kb:.1f} KB")
            item.setData(QtCore.Qt.UserRole, str(path))
            item.setToolTip(str(path))
            self.file_list.addItem(item)

        self.file_list.blockSignals(False)
        self.file_list_status.setText(
            f"共扫描 {len(all_files)} 个 CSV，当前显示 {len(filtered_files)} 个"
        )

        target_item = None
        restored_items: List[QtWidgets.QListWidgetItem] = []
        if current_path_text:
            for index in range(self.file_list.count()):
                item = self.file_list.item(index)
                path_text = str(item.data(QtCore.Qt.UserRole) or "")
                if path_text in selected_path_texts:
                    restored_items.append(item)
                if path_text == current_path_text:
                    target_item = item
        if target_item is None and self.file_list.count() > 0:
            target_item = self.file_list.item(0)

        for item in restored_items:
            item.setSelected(True)
        if target_item is not None:
            self.file_list.setCurrentItem(target_item)
            self._on_file_selected(target_item)
        else:
            self._clear_dataset_state()
            self.statusBar().showMessage("未找到可分析的 CSV 文件")

    def _clear_dataset_state(self) -> None:
        self._current_dataset = None
        self._current_file_path = None
        self.x_combo.clear()
        self._populate_y_axis_menu([], [])
        self.group_combo.clear()
        self.segment_scope_combo.blockSignals(True)
        self.segment_scope_combo.clear()
        self.segment_scope_combo.addItem("全部行（1–N）", "__all__")
        self.segment_scope_combo.blockSignals(False)
        self.row_start_spin.blockSignals(True)
        self.row_end_spin.blockSignals(True)
        self.row_start_spin.setMaximum(1)
        self.row_end_spin.setMaximum(1)
        self.row_start_spin.setValue(1)
        self.row_end_spin.setValue(1)
        self.row_start_spin.blockSignals(False)
        self.row_end_spin.blockSignals(False)
        self.plot_widget.clear()
        if self.plot_widget.plotItem.legend is not None:
            self.plot_widget.plotItem.legend.clear()
        self.dataset_info.setText("请选择左侧 CSV 文件")
        self._update_overlay_controls()

    def _populate_segment_scope_combo(self, dataset: CsvDataset) -> None:
        self.segment_scope_combo.blockSignals(True)
        self.segment_scope_combo.clear()
        self.segment_scope_combo.addItem("全部行（1–N）", "__all__")
        if "segment_index" in dataset.headers:
            spans: Dict[str, Tuple[int, int]] = {}
            for row_index, row in enumerate(dataset.rows, start=1):
                raw = str(row.get("segment_index", "") or "").strip()
                key = raw if raw else "(空)"
                lo, hi = spans.get(key, (10**9, 0))
                spans[key] = (min(lo, row_index), max(hi, row_index))

            def _sort_seg_key(k: str) -> Tuple[int, float | str]:
                num = _parse_float(k)
                if num is not None:
                    return (0, float(num))
                return (1, k)

            for key in sorted(spans.keys(), key=_sort_seg_key):
                lo, hi = spans[key]
                self.segment_scope_combo.addItem(
                    f"仅 segment_index = {key}（CSV 行 {lo}–{hi}）",
                    ("seg", lo, hi),
                )
        self.segment_scope_combo.setCurrentIndex(0)
        self.segment_scope_combo.blockSignals(False)

    def _on_segment_scope_changed(self, _index: int = 0) -> None:
        dataset = self._current_dataset
        if dataset is None:
            return
        data = self.segment_scope_combo.currentData()
        n = max(1, dataset.row_count())
        self.row_start_spin.blockSignals(True)
        self.row_end_spin.blockSignals(True)
        if data == "__all__" or data is None:
            self.row_start_spin.setValue(1)
            self.row_end_spin.setValue(n)
        elif isinstance(data, tuple) and len(data) == 3 and str(data[0]) == "seg":
            _, lo, hi = data
            self.row_start_spin.setValue(max(1, min(n, int(lo))))
            self.row_end_spin.setValue(max(1, min(n, int(hi))))
        self.row_start_spin.blockSignals(False)
        self.row_end_spin.blockSignals(False)
        self._update_plot()

    def _on_manual_row_range_changed(self, _value: int = 0) -> None:
        self.segment_scope_combo.blockSignals(True)
        self.segment_scope_combo.setCurrentIndex(0)
        self.segment_scope_combo.blockSignals(False)

    def _slice_dataset_for_plot(self, dataset: CsvDataset) -> CsvDataset:
        n = dataset.row_count()
        if n <= 0:
            return dataset
        lo = int(self.row_start_spin.value())
        hi = int(self.row_end_spin.value())
        if lo > hi:
            lo, hi = hi, lo
        lo = max(1, min(lo, n))
        hi = max(1, min(hi, n))
        if lo == 1 and hi == n:
            return dataset
        i0 = lo - 1
        i1 = hi
        sliced_rows = dataset.rows[i0:i1]
        return CsvDataset(
            path=dataset.path,
            headers=list(dataset.headers),
            rows=sliced_rows,
            numeric_headers=list(dataset.numeric_headers),
        )

    def _plot_scope_caption(self, plot_rows: int, total_rows: int) -> str:
        lo = int(self.row_start_spin.value())
        hi = int(self.row_end_spin.value())
        if lo > hi:
            lo, hi = hi, lo
        if plot_rows >= total_rows:
            return f"全文件 {total_rows} 行"
        return f"当前分析: CSV 行 {lo}–{hi}（共 {plot_rows}/{total_rows} 行）"

    def _get_selected_paths(self) -> List[Path]:
        paths: List[Path] = []
        seen: set[str] = set()
        for item in self.file_list.selectedItems():
            path_text = str(item.data(QtCore.Qt.UserRole) or "").strip()
            if not path_text or path_text in seen:
                continue
            seen.add(path_text)
            paths.append(Path(path_text))
        return paths

    def _get_cached_dataset(self, path: Path) -> CsvDataset:
        resolved = path.resolve()
        cached = self._dataset_cache.get(resolved)
        if cached is not None:
            return cached
        dataset = CsvDataset.load(resolved)
        self._dataset_cache[resolved] = dataset
        return dataset

    def _on_file_selection_changed(self) -> None:
        self._update_overlay_controls()
        if str(self.overlay_mode_combo.currentData() or "single") == "files":
            self._update_plot()

    def _open_external_csv(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "打开 CSV",
            str(self._data_root),
            "CSV 文件 (*.csv);;所有文件 (*)",
        )
        if not file_path:
            return
        self._load_dataset(Path(file_path))

    def _on_file_selected(self, current: Optional[QtWidgets.QListWidgetItem], previous=None) -> None:
        del previous
        if current is None:
            return
        path_text = str(current.data(QtCore.Qt.UserRole) or "").strip()
        if not path_text:
            return
        self._load_dataset(Path(path_text))

    def _load_dataset(self, path: Path) -> None:
        try:
            dataset = self._get_cached_dataset(path)
        except Exception as exc:
            self._clear_dataset_state()
            self.dataset_info.setText(f"读取失败: {path}\n{exc}")
            self.statusBar().showMessage(f"读取失败: {exc}")
            return

        self._current_dataset = dataset
        self._current_file_path = path

        previous_y_keys = [
            key for key in self._selected_y_keys() if key in dataset.numeric_headers
        ]
        self.x_combo.blockSignals(True)
        self.group_combo.blockSignals(True)
        self.x_combo.clear()
        self.group_combo.clear()
        self._populate_header_combo(self.x_combo, dataset.numeric_headers)
        self.group_combo.addItem("自动分组", "__auto__")
        for header in dataset.group_headers():
            self.group_combo.addItem(_header_option_text(header), header)

        x_default, y_default = self._pick_default_axes(dataset.numeric_headers)
        self._set_combo_current_key(self.x_combo, x_default)
        selected_y_keys = previous_y_keys or [y_default]
        self._populate_y_axis_menu(dataset.numeric_headers, selected_y_keys)
        self.x_combo.blockSignals(False)
        self._pick_default_group(dataset)
        self.group_combo.blockSignals(False)

        rel_path = self._safe_relpath(path)
        n_rows = dataset.row_count()
        self.row_start_spin.blockSignals(True)
        self.row_end_spin.blockSignals(True)
        self.row_start_spin.setMaximum(max(1, n_rows))
        self.row_end_spin.setMaximum(max(1, n_rows))
        self.row_start_spin.setValue(1)
        self.row_end_spin.setValue(max(1, n_rows))
        self.row_start_spin.blockSignals(False)
        self.row_end_spin.blockSignals(False)
        self._populate_segment_scope_combo(dataset)
        self.dataset_info.setText(
            f"文件: {rel_path}\n"
            f"总行数: {n_rows} | 原始字段: {len(dataset.headers)} | 可绘图数字字段: {len(dataset.numeric_headers)}\n"
            "提示：一次完整运动 CSV 可在下方「二次分段」按 segment_index 或行号截取后再绘图。"
        )
        self.statusBar().showMessage(f"已加载: {path}")
        self._update_overlay_controls()
        self._update_plot()

    @staticmethod
    def _combo_current_key(combo: QtWidgets.QComboBox) -> str:
        data = combo.currentData()
        if data is not None:
            return str(data).strip()
        return combo.currentText().strip()

    @staticmethod
    def _set_combo_current_key(combo: QtWidgets.QComboBox, key: str) -> None:
        index = combo.findData(key)
        if index < 0:
            index = combo.findText(key)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _populate_header_combo(combo: QtWidgets.QComboBox, headers: Sequence[str]) -> None:
        for header in headers:
            combo.addItem(_header_option_text(header), header)

    def _populate_y_axis_menu(
        self,
        headers: Sequence[str],
        selected_keys: Sequence[str],
    ) -> None:
        selected_set = {str(key).strip() for key in selected_keys if str(key).strip()}
        self._updating_y_axis_actions = True
        self.y_axis_menu.clear()
        self._y_axis_actions.clear()

        select_all_action = self.y_axis_menu.addAction("全选Y轴")
        select_all_action.triggered.connect(self._select_all_y_axes)
        clear_action = self.y_axis_menu.addAction("清空Y轴")
        clear_action.triggered.connect(self._clear_y_axis_selection)
        if headers:
            self.y_axis_menu.addSeparator()

        for header in headers:
            action = QtWidgets.QAction(_header_option_text(header), self.y_axis_menu)
            action.setCheckable(True)
            action.setData(header)
            action.setChecked(header in selected_set)
            action.toggled.connect(self._on_y_axis_selection_changed)
            self.y_axis_menu.addAction(action)
            self._y_axis_actions[str(header)] = action

        self._updating_y_axis_actions = False
        self._update_y_axis_button_text()

    def _selected_y_keys(self) -> List[str]:
        keys: List[str] = []
        for key, action in self._y_axis_actions.items():
            if action.isChecked():
                keys.append(str(key))
        return keys

    def _set_selected_y_keys(self, keys: Sequence[str]) -> None:
        key_set = {str(key).strip() for key in keys if str(key).strip()}
        self._updating_y_axis_actions = True
        for key, action in self._y_axis_actions.items():
            action.setChecked(key in key_set)
        self._updating_y_axis_actions = False
        self._update_y_axis_button_text()

    def _select_all_y_axes(self) -> None:
        if not self._y_axis_actions:
            return
        self._set_selected_y_keys(list(self._y_axis_actions.keys()))
        self._update_plot()

    def _clear_y_axis_selection(self) -> None:
        self._set_selected_y_keys([])
        self._update_plot()

    def _on_y_axis_selection_changed(self, _checked: bool) -> None:
        if self._updating_y_axis_actions:
            return
        self._update_y_axis_button_text()
        self._update_plot()

    def _update_y_axis_button_text(self) -> None:
        keys = self._selected_y_keys()
        if not keys:
            text = "请选择Y轴"
            tooltip = "请选择一个或多个Y轴字段"
        elif len(keys) == 1:
            text = _header_option_text(keys[0])
            tooltip = text
        elif len(keys) == 2:
            text = f"{_header_label(keys[0])} + {_header_label(keys[1])}"
            tooltip = "\n".join(_header_option_text(key) for key in keys)
        else:
            text = f"{_header_label(keys[0])} 等 {len(keys)} 项"
            tooltip = "\n".join(_header_option_text(key) for key in keys)
        self.y_axis_button.setText(text)
        self.y_axis_button.setToolTip(tooltip)

    def _safe_relpath(self, path: Path) -> str:
        try:
            return str(path.relative_to(self._data_root)).replace("\\", "/")
        except Exception:
            return str(path)

    def _pick_default_axes(self, headers: Sequence[str]) -> Tuple[str, str]:
        header_set = set(headers)
        x_candidates = [
            "motion_distance_signed_m",
            "motion_distance_m",
            "travel_distance_m",
            "relative_time_s",
            "motion_distance_total_signed_m",
            "motion_distance_total_m",
            "record_index",
            "timestamp",
            "path_s_m",
            "row_index",
        ]
        y_candidates = [
            "lateral_error_m",
            "heading_error_deg",
            "yaw_rate_error_radps",
            "cmd_w_radps",
            "feedback_w_radps",
            "cmd_v_mps",
            "peak_abs_lateral_error_m",
            "speed_mps",
            "current_y_m",
            "row_index",
        ]
        x_key = next((name for name in x_candidates if name in header_set), headers[0] if headers else "row_index")
        y_key = next((name for name in y_candidates if name in header_set and name != x_key), None)
        if not y_key:
            for name in headers:
                if name != x_key:
                    y_key = name
                    break
        if not y_key:
            y_key = x_key
        return x_key, y_key

    def _pick_default_group(self, dataset: CsvDataset) -> None:
        preferred = [
            "segment_index",
            "segment_trajectory_name",
            "motion_direction_label",
            "motion_direction",
            "run_label",
            "run_key",
        ]
        available = {str(self.group_combo.itemData(i) or "") for i in range(self.group_combo.count())}
        target = next((name for name in preferred if name in available), "__auto__")
        self._set_combo_current_key(self.group_combo, target)

    def _on_overlay_mode_changed(self, _index: int) -> None:
        self._update_overlay_controls()
        self._update_plot()

    def _update_overlay_controls(self) -> None:
        mode = str(self.overlay_mode_combo.currentData() or "single")
        has_dataset = self._current_dataset is not None
        group_enabled = has_dataset and mode == "group" and self.group_combo.count() > 1
        self.group_combo.setEnabled(group_enabled)
        if mode == "files":
            count = len(self._get_selected_paths())
            self.file_list_status.setToolTip("按住 Ctrl / Shift 可多选文件叠加分析")
            self.statusBar().showMessage(f"多文件叠加模式 | 已选 {count} 个文件")
        else:
            self.file_list_status.setToolTip("")

    def _resolve_group_key(self, dataset: CsvDataset) -> Optional[str]:
        raw_group = str(self.group_combo.currentData() or "__auto__")
        if raw_group and raw_group != "__auto__":
            return raw_group
        group_headers = dataset.group_headers()
        return group_headers[0] if group_headers else None

    @staticmethod
    def _sort_group_items(items: Sequence[Tuple[str, Tuple[List[float], List[float]]]]) -> List[Tuple[str, Tuple[List[float], List[float]]]]:
        def sort_key(item: Tuple[str, Tuple[List[float], List[float]]]) -> Tuple[int, float | str]:
            label = item[0]
            number = _parse_float(label)
            if number is not None:
                return (0, number)
            return (1, label)
        return sorted(items, key=sort_key)

    @staticmethod
    def _series_palette(index: int) -> Tuple[QtGui.QColor, QtGui.QColor]:
        palette = [
            ("#1565C0", (21, 101, 192, 120)),
            ("#C62828", (198, 40, 40, 120)),
            ("#2E7D32", (46, 125, 50, 120)),
            ("#6A1B9A", (106, 27, 154, 120)),
            ("#EF6C00", (239, 108, 0, 120)),
            ("#00838F", (0, 131, 143, 120)),
            ("#5D4037", (93, 64, 55, 120)),
            ("#AD1457", (173, 20, 87, 120)),
        ]
        line_hex, brush_rgba = palette[index % len(palette)]
        return QtGui.QColor(line_hex), QtGui.QColor(*brush_rgba)

    @staticmethod
    def _y_axis_title(y_keys: Sequence[str]) -> str:
        keys = [str(key).strip() for key in y_keys if str(key).strip()]
        if not keys:
            return "Y轴"
        if len(keys) == 1:
            return _header_option_text(keys[0])
        if len(keys) <= 3:
            return " / ".join(_header_label(key) for key in keys)
        return f"多个Y轴 ({len(keys)})"

    @staticmethod
    def _y_axis_summary(y_keys: Sequence[str]) -> str:
        keys = [str(key).strip() for key in y_keys if str(key).strip()]
        if not keys:
            return "(未选择)"
        if len(keys) <= 3:
            return ", ".join(_header_option_text(key) for key in keys)
        return ", ".join(_header_option_text(key) for key in keys[:3]) + f" 等 {len(keys)} 项"

    def _plot_series(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        *,
        name: str,
        color_index: int,
    ) -> None:
        line_color, brush_color = self._series_palette(color_index)
        symbol = "o" if len(xs) <= 2000 else None
        symbol_size = 5 if len(xs) <= 500 else 3
        self.plot_widget.plot(
            list(xs),
            list(ys),
            pen=pg.mkPen(line_color, width=2),
            symbol=symbol,
            symbolBrush=pg.mkBrush(brush_color),
            symbolPen=pg.mkPen(None),
            symbolSize=symbol_size,
            name=name,
        )

    def _reset_plot(self, x_key: str, y_keys: Sequence[str], title: str) -> None:
        self.plot_widget.clear()
        if self.plot_widget.plotItem.legend is not None:
            self.plot_widget.plotItem.legend.clear()
        self.plot_widget.setLabel("bottom", _header_option_text(x_key))
        self.plot_widget.setLabel("left", self._y_axis_title(y_keys))
        self.plot_widget.setTitle(title)

    def _apply_plot_view(self) -> None:
        if self.equal_aspect_check.isChecked():
            self.plot_widget.setAspectLocked(True, 1.0)
        else:
            self.plot_widget.setAspectLocked(False)
        self.plot_widget.enableAutoRange()

    def _update_plot(self) -> None:
        dataset = self._current_dataset
        if dataset is None:
            return
        plot_dataset = self._slice_dataset_for_plot(dataset)

        x_key = self._combo_current_key(self.x_combo)
        y_keys = self._selected_y_keys()
        if not x_key:
            return
        if not y_keys:
            self.plot_widget.clear()
            if self.plot_widget.plotItem.legend is not None:
                self.plot_widget.plotItem.legend.clear()
            self.dataset_info.setText(
                f"文件: {self._safe_relpath(dataset.path)}\n"
                "请至少选择一个Y轴字段"
            )
            self.statusBar().showMessage("未选择Y轴字段")
            return

        mode = str(self.overlay_mode_combo.currentData() or "single")
        if mode == "group":
            self._update_group_overlay_plot(plot_dataset, dataset, x_key, y_keys)
        elif mode == "files":
            self._update_files_overlay_plot(plot_dataset, dataset, x_key, y_keys)
        else:
            self._update_single_plot(plot_dataset, dataset, x_key, y_keys)

    def _update_single_plot(
        self,
        dataset: CsvDataset,
        full_dataset: CsvDataset,
        x_key: str,
        y_keys: Sequence[str],
    ) -> None:
        self._reset_plot(x_key, y_keys, self._safe_relpath(dataset.path))
        x_text = _header_option_text(x_key)
        y_text = self._y_axis_summary(y_keys)
        plotted_count = 0
        total_points = 0
        skipped: List[str] = []
        for index, y_key in enumerate(y_keys):
            xs, ys = dataset.xy_series(x_key, y_key)
            if not xs:
                skipped.append(_header_option_text(y_key))
                continue
            self._plot_series(
                xs,
                ys,
                name=f"{_header_option_text(y_key)} vs {x_text}",
                color_index=index,
            )
            plotted_count += 1
            total_points += len(xs)
        if plotted_count == 0:
            scope = self._plot_scope_caption(dataset.row_count(), full_dataset.row_count())
            self.dataset_info.setText(
                f"文件: {self._safe_relpath(dataset.path)}\n"
                f"{scope}\n"
                f"字段 {x_text} 与 {y_text} 没有可同时绘图的数值行"
            )
            self.statusBar().showMessage("当前字段组合无有效数值点")
            return

        self._apply_plot_view()
        all_xs: List[float] = []
        all_ys: List[float] = []
        for y_key in y_keys:
            xs, ys = dataset.xy_series(x_key, y_key)
            if not xs:
                continue
            all_xs.extend(xs)
            all_ys.extend(ys)
        x_min = min(all_xs)
        x_max = max(all_xs)
        y_min = min(all_ys)
        y_max = max(all_ys)
        scope = self._plot_scope_caption(dataset.row_count(), full_dataset.row_count())
        self.dataset_info.setText(
            f"文件: {self._safe_relpath(dataset.path)}\n"
            f"{scope}\n"
            f"已绘Y轴: {plotted_count} 项 | 有效绘图点: {total_points}\n"
            f"X={x_text}: [{x_min:.6f}, {x_max:.6f}] | "
            f"Y={y_text}: [{y_min:.6f}, {y_max:.6f}]"
        )
        if skipped:
            self.dataset_info.setText(self.dataset_info.text() + f"\n跳过: {', '.join(skipped[:6])}")
        self.statusBar().showMessage(
            f"已绘图: Y轴 {plotted_count} 项 | 总点数 {total_points} | X={x_text}"
        )

    def _update_group_overlay_plot(
        self,
        dataset: CsvDataset,
        full_dataset: CsvDataset,
        x_key: str,
        y_keys: Sequence[str],
    ) -> None:
        group_key = self._resolve_group_key(dataset)
        self._reset_plot(x_key, y_keys, f"{self._safe_relpath(dataset.path)} | 多段叠加")
        x_text = _header_option_text(x_key)
        y_text = self._y_axis_summary(y_keys)
        group_text = _header_option_text(group_key or "")
        if not group_key:
            scope = self._plot_scope_caption(dataset.row_count(), full_dataset.row_count())
            self.dataset_info.setText(
                f"文件: {self._safe_relpath(dataset.path)}\n"
                f"{scope}\n"
                "当前文件没有可用于叠加分析的分组字段"
            )
            self.statusBar().showMessage("没有可用分组字段")
            return

        grouped_items: List[Tuple[str, str, Tuple[List[float], List[float]]]] = []
        for y_key in y_keys:
            grouped = dataset.grouped_xy_series(x_key, y_key, group_key)
            for label, series in self._sort_group_items(list(grouped.items())):
                if not series[0]:
                    continue
                grouped_items.append((y_key, label, series))
        if not grouped_items:
            scope = self._plot_scope_caption(dataset.row_count(), full_dataset.row_count())
            self.dataset_info.setText(
                f"文件: {self._safe_relpath(dataset.path)}\n"
                f"{scope}\n"
                f"字段 {x_text} 与 {y_text} 在分组 {group_text} 下没有可绘图数据"
            )
            self.statusBar().showMessage("分组叠加无有效数值点")
            return

        total_points = 0
        scope = self._plot_scope_caption(dataset.row_count(), full_dataset.row_count())
        info_lines = [
            f"文件: {self._safe_relpath(dataset.path)}",
            scope,
            f"叠加模式: 当前文件按 {group_text} 分组",
            f"分组数量: {len(grouped_items)}",
        ]
        multi_y = len(y_keys) > 1
        for idx, (y_key, label, (xs, ys)) in enumerate(grouped_items):
            name = f"{group_text}={label}"
            if multi_y:
                name += f" | {_header_label(y_key)}"
            self._plot_series(xs, ys, name=name, color_index=idx)
            total_points += len(xs)
            info_label = label if not multi_y else f"{label} | {_header_label(y_key)}"
            info_lines.append(f"{info_label}: {len(xs)} 点")
        self._apply_plot_view()
        self.dataset_info.setText("\n".join(info_lines))
        self.statusBar().showMessage(
            f"分组叠加完成: {len(grouped_items)} 组 | 总点数 {total_points} | X={x_text} | Y={y_text}"
        )

    def _update_files_overlay_plot(
        self,
        plot_dataset: CsvDataset,
        full_dataset: CsvDataset,
        x_key: str,
        y_keys: Sequence[str],
    ) -> None:
        selected_paths = self._get_selected_paths()
        if not selected_paths and self._current_file_path is not None:
            selected_paths = [self._current_file_path]

        self._reset_plot(x_key, y_keys, "多文件叠加分析")
        x_text = _header_option_text(x_key)
        y_text = self._y_axis_summary(y_keys)
        if not selected_paths:
            self.dataset_info.setText("请先在左侧选择至少一个 CSV 文件")
            self.statusBar().showMessage("未选择叠加文件")
            return

        plotted_count = 0
        skipped: List[str] = []
        info_lines = [
            f"叠加模式: 多文件叠加",
            f"已选文件: {len(selected_paths)}",
        ]
        if len(selected_paths) == 1:
            info_lines.append(
                self._plot_scope_caption(plot_dataset.row_count(), full_dataset.row_count())
            )
        for index, path in enumerate(selected_paths):
            try:
                if path.resolve() == plot_dataset.path.resolve():
                    current_dataset = plot_dataset
                else:
                    current_dataset = self._get_cached_dataset(path)
            except Exception as exc:
                skipped.append(f"{self._safe_relpath(path)} (读取失败: {exc})")
                continue
            if x_key not in current_dataset.numeric_headers:
                skipped.append(f"{self._safe_relpath(path)} (缺少字段)")
                continue
            file_points = 0
            file_plotted = 0
            for y_offset, y_key in enumerate(y_keys):
                if y_key not in current_dataset.numeric_headers:
                    skipped.append(f"{self._safe_relpath(path)} | {_header_option_text(y_key)} (缺少字段)")
                    continue
                xs, ys = current_dataset.xy_series(x_key, y_key)
                if not xs:
                    skipped.append(f"{self._safe_relpath(path)} | {_header_option_text(y_key)} (无有效点)")
                    continue
                color_index = index * max(1, len(y_keys)) + y_offset
                name = self._safe_relpath(path)
                if len(y_keys) > 1:
                    name += f" | {_header_label(y_key)}"
                self._plot_series(xs, ys, name=name, color_index=color_index)
                plotted_count += 1
                file_plotted += 1
                file_points += len(xs)
            if file_plotted > 0:
                info_lines.append(f"{self._safe_relpath(path)}: {file_plotted} 条曲线 / {file_points} 点")

        if plotted_count == 0:
            message = "所选文件没有可叠加的有效数据"
            if skipped:
                message += "\n" + "\n".join(skipped[:6])
            self.dataset_info.setText(message)
            self.statusBar().showMessage("多文件叠加失败")
            return

        if skipped:
            info_lines.append(f"跳过文件: {len(skipped)}")
        self._apply_plot_view()
        self.dataset_info.setText("\n".join(info_lines))
        self.statusBar().showMessage(
            f"多文件叠加完成: {plotted_count}/{len(selected_paths)} 个文件 | X={x_text} | Y={y_text}"
        )


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    _apply_cjk_font(app)
    window = DataAnalysisWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
