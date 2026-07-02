# renamed: ui_layout.py
# role: 主界面布局、控件创建和视觉样式。
# contains: 控制面板、轨迹/RCS/雷达图容器、雷达目标弹窗、主题样式、窗口缩放与屏幕适配。
# depends on: MainWindow 的状态字段和回调方法；由 main_ui.py 以 mixin 方式组合。
# notes: 只处理“界面长什么样”和控件连接；具体按钮业务逻辑放到 RCS/轨迹/运动等模块。
# -*- coding: utf-8 -*-

from ui_shared import *
from ui_shared import _layout_compact_v, _resolve_cjk_font_family


MainWindow = None  # 主入口回填真实 MainWindow，供 mixin 内静态方法引用。


class MainWindowLayoutMixin:
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

        self.label_position_accuracy = QtWidgets.QLabel("位置精度估计: Lat σ=-- | Lon σ=-- | Hgt σ=--")
        self.label_position_accuracy.setWordWrap(True)
        self.label_position_accuracy.setMinimumWidth(status_label_min_width)

        self.label_power = QtWidgets.QLabel("树莓派电量: -- | 小车电量: --")
        self.label_power.setWordWrap(True)
        self.label_power.setMinimumWidth(status_label_min_width)

        self.label_can = QtWidgets.QLabel("CAN 状态: 未知")
        self.label_can.setMinimumWidth(status_label_min_width)

        self.label_tracked = QtWidgets.QLabel("锁定目标: --")
        self.label_tracked.setMinimumWidth(status_label_min_width)

        self.label_heading = QtWidgets.QLabel("方位角: --")
        self.label_heading.setMinimumWidth(status_label_min_width)

        vinfo.addLayout(power_strip)
        vinfo.addWidget(self.label_status_header)
        vinfo.addWidget(self.label_pose)
        vinfo.addWidget(self.label_enu_calibration)
        vinfo.addWidget(self.label_imu_status)
        vinfo.addWidget(self.label_position_accuracy)
        vinfo.addWidget(self.label_power)
        vinfo.addWidget(self.label_can)
        vinfo.addWidget(self.label_tracked)
        vinfo.addWidget(self.label_heading)
        vinfo.addStretch(1)

        side_vbox.addWidget(info_frame, 2)

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
        self.traj_entry_transition_curve = self.traj_plot.plot(
            [],
            [],
            pen=pg.mkPen(color="#78909C", width=2, style=QtCore.Qt.DashLine),
            name="入口过渡预览",
        )
        self.traj_entry_transition_curve.setZValue(1)
        self.traj_planned_curve.setZValue(2)

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
        btn_frame.setMinimumWidth(max(340, self._scaled_px(360)))
        btn_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        btn_card_shadow = QtWidgets.QGraphicsDropShadowEffect(btn_frame)
        btn_card_shadow.setBlurRadius(24)
        btn_card_shadow.setColor(QtGui.QColor(0, 0, 0, 118))
        btn_card_shadow.setOffset(0, 5)
        btn_frame.setGraphicsEffect(btn_card_shadow)
        vbtn = QtWidgets.QVBoxLayout(btn_frame)
        vbtn.setContentsMargins(
            self._compact_v(8),
            self._compact_v(8),
            self._compact_v(8),
            self._compact_v(8),
        )
        vbtn.setSpacing(self._compact_v(8))

        self.btn_load_path = QtWidgets.QPushButton("轨迹规划")
        self.btn_radial_measure = QtWidgets.QPushButton("星型测量")
        self.btn_run_path = QtWidgets.QPushButton("执行轨迹")
        self.btn_preset_path = QtWidgets.QPushButton("预设轨迹 / 导入")
        self.btn_delete_preset = QtWidgets.QPushButton("删除预设轨迹")
        self.btn_calib = QtWidgets.QPushButton("标定目标物位置")
        self.btn_enu_calib = QtWidgets.QPushButton("ENU坐标校准")
        self.btn_data_analysis = QtWidgets.QPushButton("误差分析")
        self.btn_save_traj = QtWidgets.QPushButton("锁定雷达目标/检查锁定")
        self.btn_static_orbit_rcs = QtWidgets.QPushButton("静态圆周测试")
        self.btn_select_rcs_save_dir = QtWidgets.QPushButton("选择数据绘RCS图")
        self.btn_select_rcs_reference = QtWidgets.QPushButton("选择RCS参考产品")
        self.btn_plot_saved_rcs = QtWidgets.QPushButton("导入数据对比")
        self.btn_heading_calib = QtWidgets.QPushButton("清除历史轨迹")
        self.btn_save_motion_data = QtWidgets.QPushButton(ACTION_SAVE_MOTION_DATA_TEXT)
        self.btn_emergency_stop = QtWidgets.QPushButton(ACTION_EMERGENCY_STOP_TEXT)
        self.btn_emergency_stop.setObjectName("DangerButton")
        self.btn_emergency_stop.setToolTip("立即停止当前轨迹/分段控制循环，并下发底盘零速")
        self.btn_emergency_stop.setStyleSheet(
            "background-color: #C62828; color: #FFFFFF; font-weight: bold;"
        )

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
        self.btn_static_orbit_rcs.clicked.connect(self._on_static_orbit_rcs_clicked)
        self.btn_select_rcs_save_dir.clicked.connect(self._on_select_rcs_save_dir)
        self.btn_select_rcs_reference.clicked.connect(self._on_select_rcs_reference)
        self.btn_plot_saved_rcs.clicked.connect(self._on_plot_saved_rcs_clicked)
        self.btn_heading_calib.clicked.connect(self._on_heading_calib_clicked)
        self.btn_save_motion_data.clicked.connect(self._on_save_motion_data)
        self.btn_emergency_stop.clicked.connect(self._on_emergency_stop_clicked)

        primary_buttons = [
            self.btn_load_path,
            self.btn_radial_measure,
            self.btn_run_path,
            self.btn_preset_path,
            self.btn_delete_preset,
            self.btn_calib,
            self.btn_enu_calib,
            self.btn_data_analysis,
            self.btn_save_traj,
            self.btn_static_orbit_rcs,
            self.btn_select_rcs_save_dir,
            self.btn_select_rcs_reference,
            self.btn_plot_saved_rcs,
            self.btn_heading_calib,
            self.btn_save_motion_data,
            self.btn_emergency_stop,
        ]
        button_min_h = max(63, self._compact_v(69))
        for btn in primary_buttons:
            btn.setFixedHeight(button_min_h)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.label_path_tracking_mode = QtWidgets.QLabel("轨迹跟踪算法")
        self.combo_path_tracking_mode = QtWidgets.QComboBox()
        self.combo_path_tracking_mode.addItem("Stanley", "stanley")
        self.combo_path_tracking_mode.addItem("Stanley + PID", "stanley_pid")
        self.combo_path_tracking_mode.currentIndexChanged.connect(self._on_path_tracking_mode_changed)
        tracking_mode_index = self.combo_path_tracking_mode.findData(self._path_tracking_mode)
        if tracking_mode_index < 0:
            tracking_mode_index = 0
        self.combo_path_tracking_mode.setCurrentIndex(tracking_mode_index)
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
        self.btn_static_orbit_rcs.setToolTip(
            "记录当前锁定雷达目标的 RCS，并按圆周-RCS方式平铺为 360° 静态测试数据。"
        )

        vbtn.addWidget(self.label_path_tracking_mode)
        vbtn.addWidget(self.combo_path_tracking_mode)
        # 隐藏轨迹跟踪算法选择，固定使用初始化时的默认（见 _path_tracking_mode）
        self.label_path_tracking_mode.setVisible(False)
        self.combo_path_tracking_mode.setVisible(False)
        path_btn_grid = QtWidgets.QGridLayout()
        path_btn_grid.setContentsMargins(0, 0, 0, 0)
        path_btn_grid.setHorizontalSpacing(self._scaled_px(10))
        path_btn_grid.setVerticalSpacing(self._compact_v(6))
        path_btn_grid.setColumnStretch(0, 1)
        path_btn_grid.setColumnStretch(1, 1)
        path_btn_grid.addWidget(self.btn_load_path, 0, 0)
        path_btn_grid.addWidget(self.btn_radial_measure, 0, 1)
        path_action_pairs = [
            (self.btn_run_path, self.btn_preset_path),
            (self.btn_delete_preset, self.btn_calib),
            (self.btn_enu_calib, self.btn_data_analysis),
            (self.btn_heading_calib, self.btn_save_traj),
            (self.btn_static_orbit_rcs, self.btn_select_rcs_save_dir),
            (self.btn_select_rcs_reference, self.btn_plot_saved_rcs),
            (self.btn_save_motion_data, self.btn_emergency_stop),
        ]
        for i, (left_btn, right_btn) in enumerate(path_action_pairs):
            path_btn_grid.addWidget(left_btn, i + 1, 0)
            path_btn_grid.addWidget(right_btn, i + 1, 1)
        vbtn.addLayout(path_btn_grid)
        vbtn.addWidget(self.label_rcs_reference_summary)
        vbtn.addStretch(1)
        self._update_path_coordinate_widgets()
        self._update_rcs_reference_summary_label()

        side_vbox.addWidget(btn_frame, 5)
        main_splitter.addWidget(traj_frame)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 5)
        sw = max(800, self._screen_available_geometry.width())
        main_splitter.setSizes(
            [
                max(380, self._scaled_px(460), int(sw * 0.34)),
                max(760, self._scaled_px(880), int(sw * 0.58)),
            ]
        )
        hbox.addWidget(main_splitter, 1)

        # 主窗口仅保留上方轨迹与控制区；雷达检查 / RCS 图仅在弹窗中构建
        parent_layout.addWidget(panel, 1)


    def _build_plots(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """雷达目标检查、RCS 绘图均在独立弹窗中，不占用主窗口底部。"""
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
        mode_row.addWidget(QtWidgets.QLabel("标定值"))
        self.rcs_calibration_spin = QtWidgets.QDoubleSpinBox()
        self.rcs_calibration_spin.setRange(-80.0, 80.0)
        self.rcs_calibration_spin.setDecimals(2)
        self.rcs_calibration_spin.setSingleStep(0.5)
        self.rcs_calibration_spin.setSuffix(" dB")
        self.rcs_calibration_spin.setValue(float(self._rcs_plot_calibration_db))
        self.rcs_calibration_spin.setToolTip(
            "叠加到当前RCS绘图数据上；不会修改已保存的原始CSV。"
        )
        self.rcs_calibration_spin.valueChanged.connect(self._on_rcs_plot_calibration_changed)
        mode_row.addWidget(self.rcs_calibration_spin)
        rcs_vbox.addLayout(mode_row)

        self.rcs_canvas = FigureCanvas(Figure(figsize=(4.6, 3.8), dpi=100))
        self.rcs_canvas.setMinimumSize(self._scaled_px(360), self._compact_v(220))
        self.rcs_canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.rcs_ax = self.rcs_canvas.figure.add_subplot(111)
        self.rcs_canvas.figure.subplots_adjust(bottom=0.18)
        rcs_vbox.addWidget(self.rcs_canvas, 1)

        self.rcs_status_label = QtWidgets.QLabel("Cluster RCS 采集: 未开始")
        rcs_vbox.addWidget(self.rcs_status_label)

        static_group = QtWidgets.QGroupBox("静态圆周测试")
        static_layout = QtWidgets.QGridLayout(static_group)
        static_layout.setContentsMargins(
            self._compact_v(8),
            self._compact_v(8),
            self._compact_v(8),
            self._compact_v(8),
        )
        static_layout.setHorizontalSpacing(self._scaled_px(8))
        static_layout.setVerticalSpacing(self._compact_v(6))
        self.edit_static_orbit_name = QtWidgets.QLineEdit("静态圆周")
        self.edit_static_orbit_name.setPlaceholderText("输入保存名称")
        self.spin_static_orbit_radius = QtWidgets.QDoubleSpinBox()
        self.spin_static_orbit_radius.setRange(0.1, 1000.0)
        self.spin_static_orbit_radius.setDecimals(2)
        self.spin_static_orbit_radius.setSuffix(" m")
        self.spin_static_orbit_radius.setValue(float(ORBIT_RCS_NOMINAL_FORWARD_M))
        self.btn_static_orbit_start = QtWidgets.QPushButton("开始")
        self.btn_static_orbit_finish = QtWidgets.QPushButton("结束并保存")
        self.btn_static_orbit_finish.setEnabled(False)
        self.btn_static_orbit_start.clicked.connect(self._on_static_orbit_rcs_start_clicked)
        self.btn_static_orbit_finish.clicked.connect(self._on_static_orbit_rcs_finish_clicked)
        static_layout.addWidget(QtWidgets.QLabel("名称"), 0, 0)
        static_layout.addWidget(self.edit_static_orbit_name, 0, 1)
        static_layout.addWidget(QtWidgets.QLabel("半径"), 0, 2)
        static_layout.addWidget(self.spin_static_orbit_radius, 0, 3)
        static_layout.addWidget(self.btn_static_orbit_start, 1, 0, 1, 2)
        static_layout.addWidget(self.btn_static_orbit_finish, 1, 2, 1, 2)
        rcs_vbox.addWidget(static_group)

        rcs_btns = QtWidgets.QHBoxLayout()
        self.btn_save_rcs_image = QtWidgets.QPushButton("保存RCS图片")
        self.btn_save_rcs_image.clicked.connect(self._on_rcs_save_image)
        rcs_btns.addWidget(self.btn_save_rcs_image)
        rcs_btns.addStretch(1)
        rcs_vbox.addLayout(rcs_btns)

        self._draw_rcs()
        return rcs_frame


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
        side_button_min_height = self._compact_v(45)
        side_button_font = self._scaled_px(10)
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

