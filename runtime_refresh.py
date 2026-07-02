# renamed: runtime_refresh.py
# role: 主界面定时刷新和实时数据同步。
# contains: 轨迹图点击雷达目标、定时读取 IMU/电源/雷达/车辆状态、刷新车辆模型、目标列表、RCS 实时采样和安全检查。
# depends on: target、rcs、path_planning、radar_safety、motion 等 mixin 提供的业务方法。
# notes: 定时器入口在 MainWindow.__init__ 中连接到本文件的 _on_timer。
# -*- coding: utf-8 -*-

from ui_shared import *
from ui_shared import _ClusterDisplayTarget, _ClusterSafetyProxy


MainWindow = None  # 主入口回填真实 MainWindow，供 mixin 内静态方法引用。


class MainWindowRuntimeMixin:
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
                # 普通规划/导入轨迹：全局位置只由主界面「轨迹参考原点」决定，不因小车平移；此处禁止用 get_robot_pose() 作平移原点。
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
            if len(getattr(self, "loaded_path_points", []) or []) >= 2:
                self._update_entry_transition_preview(self.loaded_path_points)

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

        def fmt_sigma(v: Optional[float]) -> str:
            return "--" if v is None or v < 0 else f"{v:.3f}m"

        imu_text = (
            f"IMU模式: {status.mode} | "
            f"INS: {status.ins_status}/{status.ins_pos_type} | "
            f"INSPVAXA age={fmt_age(status.age_inspvax)}, f={fmt_freq(status.freq_inspvax)}"
        )
        self.label_imu_status.setText(imu_text)
        self.label_position_accuracy.setText(
            "位置精度估计: "
            f"Lat σ={fmt_sigma(status.lat_sigma_m)} | "
            f"Lon σ={fmt_sigma(status.lon_sigma_m)} | "
            f"Hgt σ={fmt_sigma(status.hgt_sigma_m)}"
        )

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
            self._latest_radar_snapshot_ts = float(ts_snap)
            self._latest_radar_clusters_raw = [
                dict(c) for c in clusters_raw if isinstance(c, dict)
            ]
            self._latest_radar_targets = list(targets)
        else:
            self._latest_radar_targets = []
            self._latest_radar_clusters_raw = []
            self._latest_radar_snapshot_ts = 0.0

        tracked_text = self.controller.get_radar_status()
        self.label_tracked.setText(tracked_text)

        if self._static_orbit_rcs_active:
            static_target = self._find_current_locked_radar_target()
            if static_target is not None:
                self._append_static_orbit_rcs_sample(static_target)
            else:
                self.rcs_status_label.setText(
                    f"静态圆周RCS: 采集中 | 等待锁定目标ID{self._static_orbit_rcs_oid}的新鲜数据 "
                    f"点={len(self._static_orbit_rcs_points)}"
                )

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


