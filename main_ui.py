# renamed: main_ui.py
# role: 主窗口入口和 MainWindow 组合层。
# contains: MainWindow 类声明、Qt 信号、__init__ 状态初始化、mixin 组合、程序入口 main()。
# moved out: 具体界面布局、RCS 绘图/采集、轨迹规划、运动日志、雷达安全和定时刷新均在功能文件中。
# notes: 这里尽量只保留跨模块装配代码；新增业务逻辑优先放到 ui_layout/rcs_workflow/trajectory_workflow 等功能模块。
# -*- coding: utf-8 -*-

from ui_shared import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Path,
    QtCore,
    QtWidgets,
    np,
    platform,
    sys,
    threading,
    AggregatedRcsFile,
    AssocLock,
    CurvePoint,
    DATA_SAVE_ROOT_DIR_NAME,
    DEFAULT_ACCEL_DIST_M,
    DEFAULT_DECEL_DIST_M,
    DEFAULT_LINE_PLAN_DIST_M,
    DEFAULT_LINE_PLAN_SPEED_MPS,
    DEFAULT_RADIAL_MEASUREMENT_SPEED_MPS,
    DEFAULT_REVERSE_LINE_PLAN_SPEED_MPS,
    DEFAULT_SEGMENT_SPEED_MPS,
    LoadedRcsCurve,
    MOTION_DATA_DIR_NAME,
    MainController,
    ORBIT_RCS_NOMINAL_FORWARD_M,
    PathReferenceFrame,
    RADAR_EMERGENCY_STOP_DEFAULT_ENABLED,
    RADAR_TARGET_FRESH_S,
    RCS_DATA_DIR_NAME,
    RadialMeasurementSpec,
    RcsRunRecorder,
    SegmentRange,
    _apply_matplotlib_font,
)
import ui_layout as _ui_layout
import motion_records as _motion_records
import target_management as _target_management
import rcs_workflow as _rcs_workflow
import trajectory_workflow as _trajectory_workflow
import radar_safety as _radar_safety
import runtime_refresh as _runtime_refresh
from ui_layout import MainWindowLayoutMixin
from motion_records import MainWindowMotionMixin
from target_management import MainWindowTargetMixin
from rcs_workflow import MainWindowRcsMixin
from trajectory_workflow import MainWindowPathPlanningMixin
from radar_safety import MainWindowRadarSafetyMixin
from runtime_refresh import MainWindowRuntimeMixin


class MainWindow(
    MainWindowLayoutMixin,
    MainWindowMotionMixin,
    MainWindowTargetMixin,
    MainWindowRcsMixin,
    MainWindowPathPlanningMixin,
    MainWindowRadarSafetyMixin,
    MainWindowRuntimeMixin,
    QtWidgets.QMainWindow,
):
    segment_rcs_start_requested = QtCore.pyqtSignal(int, str)
    segment_rcs_finish_requested = QtCore.pyqtSignal(int, str)
    tracking_metrics_reported = QtCore.pyqtSignal(object)
    motion_run_finished = QtCore.pyqtSignal(str)
    """
    RCS 雷达小车一体化调试 UI：

      - 主窗口：状态、轨迹预览和控制入口
      - 具体布局、轨迹规划、RCS 绘图、运动记录等功能由 mixin 模块提供
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
        self._loaded_path_frame = PathReferenceFrame(
            origin_key="__calib_plane_origin__",
            origin_label="校准平面原点",
            origin_x_m=0.0,
            origin_y_m=0.0,
            origin_z_m=0.0,
        )
        self._radial_rcs_session_dir_name: Optional[str] = None
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
        self._orbit_rcs_heading0_rad: Optional[float] = None
        self._orbit_rcs_clockwise: Optional[bool] = None
        # 前进直线段：合并记录（不按 ID 分开），将所有目标点写入同一 CSV 并统一拟合直线
        self._straight_rcs_collect_all: bool = False
        self._straight_rcs_last_by_oid: Dict[int, float] = {}
        self._straight_rcs_rcs_ema_by_oid: Dict[int, float] = {}
        # 前进直线段：UI 叠加显示“第N次”测量数据（不依赖保存文件解析）
        self._straight_rcs_runs: List[List[CurvePoint]] = []
        self._straight_rcs_max_runs: int = 30
        # 与 rcs_recorder.segments 对齐：第 i 段对应的“次数”显示（通常由 CSV 的 SegIdx 列解析得到）
        self._rcs_segment_run_labels: Optional[List[int]] = None
        self._rcs_segment_display_labels: Optional[List[str]] = None
        # RCS plotting workflow state
        self._rcs_base_file_path: Optional[str] = None
        self._rcs_curve_csv_source_paths: List[str] = []
        # When True, do not overlay reference limit bands/lines on the plot.
        # Used for "选择数据绘RCS图" as requested.
        self._rcs_hide_reference_limits: bool = False
        # 绘图时叠加到测量 RCS（dB），不改动落盘原始数据；参考上下限不偏移。
        self._rcs_plot_calibration_db: float = 0.0
        self._rcs_all_forward_straight_default = True
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
        self._traj_default_reverse_line_speed = DEFAULT_REVERSE_LINE_PLAN_SPEED_MPS
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
        self._radial_collect_all_forward_straight = True
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
        self._latest_radar_targets: List[Any] = []
        self._latest_radar_clusters_raw: List[Dict[str, Any]] = []
        self._latest_radar_snapshot_ts: float = 0.0
        self._static_orbit_rcs_active: bool = False
        self._static_orbit_rcs_points: List[CurvePoint] = []
        self._static_orbit_rcs_start_ts: float = 0.0
        self._static_orbit_rcs_name: str = "静态圆周"
        self._static_orbit_rcs_radius_m: float = ORBIT_RCS_NOMINAL_FORWARD_M
        self._static_orbit_rcs_oid: Optional[int] = None

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



for _module in (
    _ui_layout,
    _motion_records,
    _target_management,
    _rcs_workflow,
    _trajectory_workflow,
    _radar_safety,
    _runtime_refresh,
):
    _module.MainWindow = MainWindow
del _module


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
