# renamed: trajectory_workflow.py
# role: 轨迹规划、预设轨迹、路径执行和分段调度。
# contains: 直线/圆弧/星型测量路径生成、路径导入导出、预设任务、固定参考原点、路径平滑/重采样、分段运行、Stanley/PID 参数组织。
# also contains: 分段开始/结束时触发直线 RCS 或圆周 RCS 的调度逻辑。
# notes: 不直接解析 RCS Raw 或绘制 RCS 图；这些在 rcs_workflow.py。
# -*- coding: utf-8 -*-

from ui_shared import *
from ui_shared import _LEGACY_PATH_COORD_MODE_CURRENT_POSE, _load_data_analysis_module


MainWindow = None  # 主入口回填真实 MainWindow，供 mixin 内静态方法引用。


class MainWindowPathPlanningMixin:
    _SEGMENT_KIND_VALUES = {"line", "circle", "transition"}
    _MIN_TRANSITION_TURN_RADIUS_M = 2.0

    def _planned_segment_kind_for_index(self, seg_idx: int) -> Optional[str]:
        """返回规划器逐段类型：line / circle / transition；无元数据时返回 None（由几何判定）。"""
        if seg_idx < 1:
            return None
        kinds = getattr(self, "_planned_segment_kinds", None) or []
        if seg_idx > len(kinds):
            return None
        raw = str(kinds[seg_idx - 1] or "").strip().lower()
        return raw if raw in self._SEGMENT_KIND_VALUES else None


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
        if kind == "transition":
            return False
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
        if self._planned_segment_kind_for_index(seg_idx) == "transition":
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


    def _planned_orbit_clockwise_for_segment(self, seg_idx: Optional[int]) -> Optional[bool]:
        if seg_idx is None:
            return None
        ranges = getattr(self, "_planned_ranges", None) or []
        if int(seg_idx) < 1 or int(seg_idx) > len(ranges):
            return None
        try:
            sr = self._normalize_segment_range(ranges[int(seg_idx) - 1])
            start_idx, end_idx = int(sr[0]), int(sr[1])
            pts = self.loaded_path_points[start_idx : end_idx + 1]
            inf = self._infer_circle_orbit_params(pts)
        except Exception:
            inf = None
        if inf is None:
            return None
        try:
            return bool(inf["clockwise"])
        except (KeyError, TypeError, ValueError):
            return None


    def _segment_index_is_forward_straight_segment(self, seg_idx: Optional[int]) -> bool:
        if seg_idx is None:
            return False
        ranges = getattr(self, "_planned_ranges", None) or []
        if seg_idx < 1 or seg_idx > len(ranges):
            return False
        kind = self._planned_segment_kind_for_index(seg_idx)
        if kind in {"circle", "transition"}:
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
        kind = self._planned_segment_kind_for_index(seg_idx)
        if kind == "transition":
            return False
        if kind == "line":
            return False
        pts = self.loaded_path_points[start_idx : end_idx + 1]
        if len(pts) < 2:
            return False
        return not self._is_nearly_straight_segment(pts)


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
        # 参考原点平面坐标以帧内 origin_x/y/z 为准（默认 0,0 或预设保存值），
        # 不再根据 origin_key 去反查 ENU 表里「记录控制点」（与当前系 x,y 无直接对应）。
        key = str(normalized.origin_key or "").strip()
        label = str(normalized.origin_label or "").strip() or "校准平面原点"
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
            return "轨迹参考原点未就绪（请检查界面初始化）"
        label = normalized.origin_label or "校准平面原点"
        return (
            f"{label} | x₀={normalized.origin_x_m:.3f} m, "
            f"y₀={normalized.origin_y_m:.3f} m"
        )


    def _default_path_reference_frame(self) -> PathReferenceFrame:
        return PathReferenceFrame(
            mode=PATH_COORD_MODE_FIXED_ORIGIN,
            origin_key=PATH_ORIGIN_CALIB_PLANE_KEY,
            origin_label="校准平面原点",
            origin_x_m=0.0,
            origin_y_m=0.0,
            origin_z_m=0.0,
        )


    def _get_path_reference_frame(self) -> PathReferenceFrame:
        frame = self._resolve_path_reference_frame(getattr(self, "_loaded_path_frame", None))
        if frame.is_fixed_origin():
            return frame
        return self._default_path_reference_frame()


    def _apply_path_reference_frame_to_controls(
        self,
        frame: Optional[PathReferenceFrame],
    ) -> None:
        normalized = self._resolve_path_reference_frame(frame)
        self._loaded_path_frame = normalized if normalized.is_fixed_origin() else self._default_path_reference_frame()
        self._update_path_coordinate_widgets(frame_override=self._loaded_path_frame)


    def _update_path_coordinate_widgets(
        self,
        frame_override: Optional[PathReferenceFrame] = None,
    ) -> None:
        if frame_override is not None:
            nf = self._normalize_path_reference_frame(frame_override)
            if nf.is_fixed_origin():
                self._loaded_path_frame = nf
        else:
            self._loaded_path_frame = self._get_path_reference_frame()


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
            collect_all_forward_straight=bool(
                getattr(spec, "collect_all_forward_straight", True)
            ),
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
        del close_threshold_m
        points = self._plan_entry_transition_from_current_heading(
            start_pose=start_pose,
            end_pose=end_pose,
            nominal_speed_mps=transition_speed_mps,
        )
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
        frame = self._get_path_reference_frame()
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

        # 星型 0° 已固定为目标物点朝向 -Y；不再按“参考原点→目标点”做投影裁剪。
        collect_forward_rcs = bool(
            getattr(normalized, "collect_all_forward_straight", True)
        )
        plan_ranges = []
        for seg_range in list(plan.ranges):
            s0, e0, speed_sign, rcs_start, speed_mps, accel_dist, decel_dist = seg_range
            plan_ranges.append(
                (
                    int(s0),
                    int(e0),
                    int(speed_sign),
                    bool(rcs_start) if collect_forward_rcs else False,
                    float(speed_mps),
                    float(accel_dist),
                    float(decel_dist),
                )
            )
        plan_range_task_names = list(plan.range_task_names)

        self._planned_ranges = list(plan_ranges) if self._should_use_segment_ranges(list(plan_ranges)) else None
        self._planned_range_task_names = (
            list(plan_range_task_names) if self._planned_ranges else []
        )
        self._planned_segment_kinds = None
        if not self._apply_planned_local_points(
            local_points,
            frame=resolved_frame,
            stabilize=False,
        ):
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
        self._radial_collect_all_forward_straight = bool(
            getattr(normalized, "collect_all_forward_straight", True)
        )
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
            frame = self._resolve_path_reference_frame(self._get_path_reference_frame())
            self._set_target_point_from_global(
                float(frame.origin_x_m),
                float(frame.origin_y_m),
                log_prefix="目标物位置已使用轨迹原点预览",
                detail="当前未标定目标物，先以原点生成预览",
            )
        dialog = RadialMeasurementDialog(
            target_point=self.target_point,
            default_angle_cycles=self._radial_default_angle_cycles,
            default_inner_radius=self._radial_default_inner_radius,
            default_line_length_m=float(getattr(self, "_radial_default_line_length_m", 50.0)),
            default_speed=float(getattr(self, "_radial_default_speed_mps", DEFAULT_RADIAL_MEASUREMENT_SPEED_MPS)),
            default_accel_dist=self._traj_default_accel_dist,
            default_decel_dist=self._traj_default_decel_dist,
            default_collect_all_forward_straight=bool(
                getattr(self, "_radial_collect_all_forward_straight", True)
            ),
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
        if len(plan.task_names) == 1 and self._planned_ranges:
            self._planned_segment_kinds = self._remap_segment_kinds_list(
                self._preset_segment_kinds.get(plan.task_names[0]),
                list(plan.ranges),
            )
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
        self._preset_segment_kinds.pop(choice, None)
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

        frame = self._get_path_reference_frame()

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
        """采集当前位置作为目标物位置。"""
        if self.target_marker is None:
            return
        pose = get_robot_pose()
        if pose is None:
            self._target_marking_mode = False
            frame = self._resolve_path_reference_frame(self._get_path_reference_frame())
            self._set_target_point_from_global(
                float(frame.origin_x_m),
                float(frame.origin_y_m),
                log_prefix="目标物位置已使用轨迹原点预览",
                detail="当前无实时定位，仅用于生成轨迹预览",
            )
            return

        self._set_target_point_from_global(
            float(pose.x),
            float(pose.y),
            log_prefix="目标物位置已采集当前位置",
            detail=f"source={pose.source}",
        )

    def _on_radar_lock_check_clicked(self) -> None:
        self._place_radar_target_dialog_at_trajectory_top_right()
        self._show_tool_dialog(self._radar_target_dialog)
        self._log("已打开雷达目标检查窗口")


    def _on_enu_calibration_clicked(self) -> None:
        dialog = EnuCalibrationDialog(self._enu_calibration_points, parent=self)
        dialog.exec_()
        self._enu_calibration_points = dialog.points()
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
        segment_kinds: Optional[List[str]] = None,
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
        normalized_ranges = [
            self._normalize_segment_range(seg_range) for seg_range in (ranges or [])
        ]
        self._preset_paths[name] = list(local_points)
        self._preset_ranges[name] = normalized_ranges
        normalized_segment_kinds = self._remap_segment_kinds_list(
            segment_kinds,
            normalized_ranges,
        )
        if normalized_segment_kinds:
            self._preset_segment_kinds[name] = normalized_segment_kinds
        else:
            self._preset_segment_kinds.pop(name, None)
        self._preset_path_frames[name] = self._resolve_path_reference_frame(
            frame if frame is not None else self._default_path_reference_frame()
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
                        "segment_kind",
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
                                "",
                            ]
                        )
                    ranges = self._preset_ranges.get(name, [])
                    kinds = self._preset_segment_kinds.get(name, [])
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
                                self._normalize_segment_kind(
                                    kinds[idx] if idx < len(kinds) else ""
                                ),
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
        self._preset_segment_kinds.clear()
        self._preset_path_frames.clear()
        if not PRESET_PATHS_CSV.exists():
            return
        try:
            temp_points: Dict[str, List[Tuple[int, float, float]]] = {}
            temp_ranges: Dict[str, List[Tuple[int, SegmentRange]]] = {}
            temp_segment_kinds: Dict[str, List[Tuple[int, str]]] = {}
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
                            temp_segment_kinds.setdefault(name, []).append(
                                (idx, self._normalize_segment_kind(row.get("segment_kind", "")))
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
                self._preset_path_frames[name] = self._resolve_path_reference_frame(
                    temp_frames.get(name)
                )

            for name, segments in temp_ranges.items():
                if name not in self._preset_paths:
                    continue
                segments.sort(key=lambda item: item[0])
                normalized_ranges = [
                    seg_range
                    for _, seg_range in segments
                    if seg_range[1] > seg_range[0]
                ]
                self._preset_ranges[name] = normalized_ranges
                kind_rows = temp_segment_kinds.get(name, [])
                if kind_rows:
                    kind_rows.sort(key=lambda item: item[0])
                    ordered_kinds = [kind for _, kind in kind_rows]
                    if (
                        len(ordered_kinds) == len(normalized_ranges)
                        and all(kind in self._SEGMENT_KIND_VALUES for kind in ordered_kinds)
                    ):
                        self._preset_segment_kinds[name] = ordered_kinds

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
    def _normalize_segment_kind(value: Any) -> str:
        kind = str(value or "").strip().lower()
        return kind if kind in MainWindowPathPlanningMixin._SEGMENT_KIND_VALUES else ""


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
            collect_forward_rcs = bool(
                getattr(self._radial_measurement_spec, "collect_all_forward_straight", True)
            )
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
                    collect_forward_rcs
                    and
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
        if not bool(getattr(self, "_rcs_all_forward_straight_default", True)):
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
            if kind in {"circle", "transition"}:
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
                remapped_kinds.append(
                    k0 if k0 in self._SEGMENT_KIND_VALUES else "line"
                )
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
            out.append(k0 if k0 in MainWindowPathPlanningMixin._SEGMENT_KIND_VALUES else "line")
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
            default_reverse_line_speed=self._traj_default_reverse_line_speed,
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
        self._traj_default_line_speed = float(dialog.line_forward_speed.value())
        self._traj_default_reverse_line_speed = float(dialog.line_reverse_speed.value())
        self._traj_default_circle_speed = float(dialog.circle_speed.value())
        self._traj_default_accel_dist = float(dialog.line_accel_dist.value())
        self._traj_default_decel_dist = float(dialog.line_decel_dist.value())
        self._path_speed = self._traj_default_line_speed
        segments = dialog.get_segments()
        if not segments:
            QtWidgets.QMessageBox.information(self, "无轨迹段", "请先添加直线或圆弧段。")
            self._log("轨迹规划失败: 未添加轨迹段")
            return
        frame = self._get_path_reference_frame()
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
        *,
        stabilize: bool = True,
    ) -> bool:
        if len(local_points) < 2:
            QtWidgets.QMessageBox.information(self, "轨迹无效", "规划轨迹点数量不足。")
            self._log("轨迹规划失败: 点数不足")
            return False

        if stabilize:
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
        else:
            self._log("轨迹稳定化: 已跳过，保留规划器原始几何。")

        if frame is None:
            resolved_frame = self._resolve_path_reference_frame(
                self._default_path_reference_frame()
            )
        else:
            resolved_frame = self._resolve_path_reference_frame(frame)
        if not resolved_frame.is_fixed_origin():
            self._log("轨迹应用失败: 参考原点帧无效")
            return False

        pose = self._build_virtual_preview_pose(
            x_m=resolved_frame.origin_x_m,
            y_m=resolved_frame.origin_y_m,
            z_m=resolved_frame.origin_z_m,
        )
        # 局部→全局只按「轨迹参考原点」平移，绝不使用小车位置；无 INS 时标记虚拟预览，便于位姿就绪后仅刷新 UI。
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
        frame = self._resolve_path_reference_frame(self._default_path_reference_frame())
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
                "轨迹参考系无效",
                "当前轨迹参考系无效，请重新规划或加载轨迹。",
            )
            self._log("轨迹重新锚定失败: 参考系无效")
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
            f"轨迹按校准平面系与当前参考原点下发: {self._format_path_reference_frame(frame)} | "
            f"全局点数={len(global_points)}"
        )
        return True


    def _build_planned_path(
        self,
        segments: List[dict],
    ) -> Tuple[List[Tuple[float, float]], List[SegmentRange], List[str]]:
        line_step = 1.2
        circle_step = 0.12
        target_xy = (
            (float(self.target_point[0]), float(self.target_point[1]))
            if self.target_point is not None
            else None
        )
        points: List[Tuple[float, float]] = []
        ranges: List[SegmentRange] = []
        segment_kinds: List[str] = []
        x, y = (target_xy[0], target_xy[1]) if target_xy is not None else (0.0, 0.0)
        heading = math.pi * 0.5

        def ensure_start(pt: Tuple[float, float]) -> None:
            nonlocal x, y
            if not points:
                points.append((float(pt[0]), float(pt[1])))
                x, y = float(pt[0]), float(pt[1])

        def sample_line_between(
            p0: Tuple[float, float],
            p1: Tuple[float, float],
        ) -> List[Tuple[float, float]]:
            dist_m = math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1]))
            step = max(0.05, float(line_step))
            n = max(1, int(math.ceil(dist_m / step)))
            return [
                (
                    float(p0[0]) + (float(p1[0]) - float(p0[0])) * (i / n),
                    float(p0[1]) + (float(p1[1]) - float(p0[1])) * (i / n),
                )
                for i in range(0, n + 1)
            ]

        def append_polyline(polyline: List[Tuple[float, float]]) -> Tuple[int, int]:
            nonlocal x, y
            if len(polyline) < 2:
                return len(points) - 1, len(points) - 1
            if not points:
                points.extend((float(px), float(py)) for px, py in polyline)
                start_idx = 0
            else:
                start_idx = len(points) - 1
                if math.hypot(points[-1][0] - polyline[0][0], points[-1][1] - polyline[0][1]) <= 1e-6:
                    points.extend((float(px), float(py)) for px, py in polyline[1:])
                else:
                    points.extend((float(px), float(py)) for px, py in polyline)
                    start_idx = len(points) - len(polyline)
            end_idx = len(points) - 1
            x, y = points[-1]
            return start_idx, end_idx

        def target_line_endpoints(length_m: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
            if target_xy is None:
                return (x, y), (x, y + float(length_m))
            tx, ty = target_xy
            length = abs(float(length_m))
            return (tx, ty - length), (tx, ty)

        def target_circle_points(
            radius_m: float,
            angle_deg: float,
            direction: str,
        ) -> Tuple[List[Tuple[float, float]], float]:
            if target_xy is None:
                return [], heading
            tx, ty = target_xy
            radius = abs(float(radius_m))
            phi = math.radians(abs(float(angle_deg)))
            if str(direction).lower() == "cw":
                phi = -phi
            start_angle = -math.pi * 0.5
            arc_len = abs(phi) * radius
            n = max(12, min(720, int(arc_len / max(circle_step, 1e-6)) + 1))
            polyline = []
            for i in range(0, n + 1):
                ang = start_angle + phi * (i / n)
                polyline.append((tx + radius * math.cos(ang), ty + radius * math.sin(ang)))
            end_heading = self._wrap_angle((0.0 if phi >= 0 else math.pi) + phi)
            return polyline, end_heading

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
                segment_kinds.append(sk if sk in self._SEGMENT_KIND_VALUES else "line")

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
                    heading_before = heading
                    if target_xy is not None:
                        outer, inner = target_line_endpoints(dist)
                        polyline = sample_line_between(
                            outer if speed_sign >= 0 else inner,
                            inner if speed_sign >= 0 else outer,
                        )
                        start_idx, end_idx = append_polyline(polyline)
                        heading = math.pi * 0.5
                    else:
                        if not points:
                            ensure_start((x, y))
                        start_idx = len(points) - 1
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
                        seg.get(
                            "forward_speed_mps",
                            seg.get("speed_mps", DEFAULT_LINE_PLAN_SPEED_MPS),
                        )
                        if speed_sign >= 0
                        else seg.get(
                            "reverse_speed_mps",
                            seg.get("speed_mps", DEFAULT_REVERSE_LINE_PLAN_SPEED_MPS),
                        ),
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
                if target_xy is not None:
                    polyline, heading = target_circle_points(radius, angle, direction)
                    start_idx, end_idx = append_polyline(polyline)
                else:
                    if not points:
                        ensure_start((x, y))
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
        if not points:
            points.append((0.0, 0.0))
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


    def _clear_entry_transition_preview(self) -> None:
        curve = getattr(self, "traj_entry_transition_curve", None)
        if curve is not None:
            curve.setData([], [])


    def _preview_pose_for_entry_transition(self) -> PoseSolution:
        pose = get_robot_pose()
        if pose is not None:
            return pose
        return self._build_virtual_preview_pose()


    def _formal_path_entry_pose(
        self,
        points: List[Tuple[float, float]],
        fallback_yaw: float,
    ) -> Optional[Tuple[float, float, float]]:
        if len(points) < 2:
            return None

        start_idx = 0
        end_idx = len(points) - 1
        ranges = getattr(self, "_planned_ranges", None) or []
        if ranges:
            try:
                first = self._normalize_segment_range(ranges[0])
                start_idx = max(0, min(int(first[0]), len(points) - 1))
                end_idx = max(start_idx + 1, min(int(first[1]), len(points) - 1))
            except Exception:
                start_idx = 0
                end_idx = len(points) - 1

        if start_idx >= len(points) - 1:
            return None
        next_idx = min(start_idx + 1, end_idx, len(points) - 1)
        formal_start = points[start_idx]
        formal_next = points[next_idx]
        formal_heading = self._heading_from_points(formal_start, formal_next, fallback_yaw)
        return (float(formal_start[0]), float(formal_start[1]), float(formal_heading))


    def _update_entry_transition_preview(
        self,
        global_points: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        curve = getattr(self, "traj_entry_transition_curve", None)
        if curve is None:
            return []
        if not global_points or len(global_points) < 2:
            self._clear_entry_transition_preview()
            return []
        if self._planned_segment_kind_for_index(1) == "transition":
            self._clear_entry_transition_preview()
            return []

        pose = self._preview_pose_for_entry_transition()
        formal = self._formal_path_entry_pose(global_points, float(pose.yaw))
        if formal is None:
            self._clear_entry_transition_preview()
            return []

        formal_x, formal_y, formal_heading = formal
        dist_to_start = math.hypot(formal_x - float(pose.x), formal_y - float(pose.y))
        heading_delta = abs(self._wrap_angle(formal_heading - float(pose.yaw)))
        if dist_to_start < 0.35 and heading_delta < math.radians(10.0):
            self._clear_entry_transition_preview()
            return []

        first_speed = float(getattr(self, "_path_speed", MIN_SEGMENT_SPEED_MPS))
        ranges = getattr(self, "_planned_ranges", None) or []
        if ranges:
            try:
                first_speed = abs(float(self._normalize_segment_range(ranges[0])[4]))
            except Exception:
                first_speed = float(getattr(self, "_path_speed", MIN_SEGMENT_SPEED_MPS))
        transition_speed = max(0.18, min(max(MIN_SEGMENT_SPEED_MPS, first_speed), 0.75))
        transition_points = self._plan_entry_transition_from_current_heading(
            start_pose=(float(pose.x), float(pose.y), float(pose.yaw)),
            end_pose=(formal_x, formal_y, formal_heading),
            nominal_speed_mps=transition_speed,
        )
        if len(transition_points) < 2:
            self._clear_entry_transition_preview()
            return []

        curve.setData(
            [p[0] for p in transition_points],
            [p[1] for p in transition_points],
        )
        return transition_points


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
        entry_transition_points = self._update_entry_transition_preview(global_points)
        self._refresh_planned_path_direction_arrows()
        self._fit_traj_view_to_points(global_points + entry_transition_points)
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


    def _wait_for_segment_rcs_started(self, seg_idx: int, timeout_s: float = 2.0) -> bool:
        deadline = time.time() + max(0.0, float(timeout_s))
        while time.time() < deadline:
            if (
                self._rcs_recording
                and self._rcs_active_segment_index is not None
                and int(self._rcs_active_segment_index) == int(seg_idx)
            ):
                return True
            time.sleep(0.02)
        return False


    def _wait_for_segment_rcs_finished(self, seg_idx: int, timeout_s: float = 4.0) -> bool:
        deadline = time.time() + max(0.0, float(timeout_s))
        while time.time() < deadline:
            if (not self._rcs_recording) or (
                self._rcs_active_segment_index is not None
                and int(self._rcs_active_segment_index) != int(seg_idx)
            ):
                return True
            time.sleep(0.02)
        return False


    def _radial_rcs_group_file_trajectory_name(self, segment_task_name: str) -> str:
        """星型测量：文件名保留角度与次数标记，不再按角度合并命名。"""
        return segment_task_name


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
            # 纯 Stanley：略降增益+软化；横向 PD 参数保留在 _tracking_tuning["stanley_lateral_kp/kd"]。
            kwargs.update(
                {
                    "stanley_gain": 0.40,
                    "stanley_softening_speed_mps": 0.60,
                    "lookahead_heading_weight": 0.05,
                    "lookahead_heading_max_bias_deg": 52.0,
                    "max_w_step": 0.031,
                    "stanley_lateral_pd_kp": 0.0,
                    "stanley_lateral_pd_ki": 0.02,
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
        if self._planned_ranges:
            if self._radial_measurement_spec is not None:
                if bool(getattr(self._radial_measurement_spec, "collect_all_forward_straight", True)):
                    self._log(
                        "星型测量 Cluster RCS: 已启用「前进测量直线段均触发采集」。"
                    )
                else:
                    self._log("星型测量 Cluster RCS: 已关闭自动前进直线段采集。")
            elif bool(getattr(self, "_rcs_all_forward_straight_default", True)):
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
                    f"星型测量执行: 直线往返沿用普通直线控制；仅过渡曲线使用 {tracking_label} 软化跟踪。"
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


    @staticmethod
    def _heading_from_points(
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        fallback: float = 0.0,
    ) -> float:
        dx = float(p1[0]) - float(p0[0])
        dy = float(p1[1]) - float(p0[1])
        if math.hypot(dx, dy) <= 1e-6:
            return float(fallback)
        return math.atan2(dy, dx)


    @staticmethod
    def _sample_cubic_bezier_points(
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        step_m: float = 0.16,
    ) -> List[Tuple[float, float]]:
        ctrl_len = (
            math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1]))
            + math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))
            + math.hypot(float(p3[0]) - float(p2[0]), float(p3[1]) - float(p2[1]))
        )
        n = max(8, min(240, int(ctrl_len / max(1e-3, float(step_m))) + 1))
        out: List[Tuple[float, float]] = []
        for i in range(n + 1):
            t = i / n
            omt = 1.0 - t
            x = (
                (omt ** 3) * float(p0[0])
                + 3.0 * (omt ** 2) * t * float(p1[0])
                + 3.0 * omt * (t ** 2) * float(p2[0])
                + (t ** 3) * float(p3[0])
            )
            y = (
                (omt ** 3) * float(p0[1])
                + 3.0 * (omt ** 2) * t * float(p1[1])
                + 3.0 * omt * (t ** 2) * float(p2[1])
                + (t ** 3) * float(p3[1])
            )
            out.append((x, y))
        return out


    @staticmethod
    def _polyline_min_turn_radius(points: List[Tuple[float, float]]) -> Optional[float]:
        min_radius: Optional[float] = None
        if len(points) < 3:
            return None
        for a, b, c in zip(points, points[1:], points[2:]):
            ax, ay = float(a[0]), float(a[1])
            bx, by = float(b[0]), float(b[1])
            cx, cy = float(c[0]), float(c[1])
            ab = math.hypot(bx - ax, by - ay)
            bc = math.hypot(cx - bx, cy - by)
            ca = math.hypot(ax - cx, ay - cy)
            if ab <= 1e-5 or bc <= 1e-5 or ca <= 1e-5:
                continue
            cross = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
            if cross <= 1e-7:
                continue
            radius = (ab * bc * ca) / (2.0 * cross)
            if math.isfinite(radius):
                min_radius = radius if min_radius is None else min(min_radius, radius)
        return min_radius


    @staticmethod
    def _mod2pi(angle: float) -> float:
        return float(angle) % (2.0 * math.pi)


    def _sample_dubins_transition_points(
        self,
        start_pose: Tuple[float, float, float],
        end_pose: Tuple[float, float, float],
        *,
        radius_m: float,
        step_m: float = 0.12,
    ) -> List[Tuple[float, float]]:
        x0, y0, yaw0 = (float(start_pose[0]), float(start_pose[1]), float(start_pose[2]))
        x3, y3, yaw3 = (float(end_pose[0]), float(end_pose[1]), float(end_pose[2]))
        radius_m = max(0.1, float(radius_m))
        dx = x3 - x0
        dy = y3 - y0
        c0 = math.cos(yaw0)
        s0 = math.sin(yaw0)
        local_x = (c0 * dx + s0 * dy) / radius_m
        local_y = (-s0 * dx + c0 * dy) / radius_m
        local_yaw = self._wrap_angle(yaw3 - yaw0)
        d = math.hypot(local_x, local_y)
        theta = math.atan2(local_y, local_x)
        alpha = self._mod2pi(-theta)
        beta = self._mod2pi(local_yaw - theta)
        sa, sb = math.sin(alpha), math.sin(beta)
        ca, cb = math.cos(alpha), math.cos(beta)
        cab = math.cos(alpha - beta)

        candidates: List[Tuple[float, Tuple[str, str, str], Tuple[float, float, float]]] = []

        def add_candidate(modes: Tuple[str, str, str], values: Optional[Tuple[float, float, float]]) -> None:
            if values is None:
                return
            t, p, q = values
            if min(t, p, q) < -1e-9:
                return
            candidates.append((float(t + p + q), modes, (float(t), float(p), float(q))))

        tmp = d + sa - sb
        p2 = 2.0 + d * d - 2.0 * cab + 2.0 * d * (sa - sb)
        if p2 >= 0.0:
            p = math.sqrt(p2)
            phi = math.atan2(cb - ca, tmp)
            add_candidate(("L", "S", "L"), (self._mod2pi(-alpha + phi), p, self._mod2pi(beta - phi)))

        tmp = d - sa + sb
        p2 = 2.0 + d * d - 2.0 * cab + 2.0 * d * (-sa + sb)
        if p2 >= 0.0:
            p = math.sqrt(p2)
            phi = math.atan2(ca - cb, tmp)
            add_candidate(("R", "S", "R"), (self._mod2pi(alpha - phi), p, self._mod2pi(-beta + phi)))

        p2 = -2.0 + d * d + 2.0 * cab + 2.0 * d * (sa + sb)
        if p2 >= 0.0:
            p = math.sqrt(p2)
            phi = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
            add_candidate(("L", "S", "R"), (self._mod2pi(-alpha + phi), p, self._mod2pi(-beta + phi)))

        p2 = -2.0 + d * d + 2.0 * cab - 2.0 * d * (sa + sb)
        if p2 >= 0.0:
            p = math.sqrt(p2)
            phi = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
            add_candidate(("R", "S", "L"), (self._mod2pi(alpha - phi), p, self._mod2pi(beta - phi)))

        tmp = (6.0 - d * d + 2.0 * cab + 2.0 * d * (sa - sb)) / 8.0
        if abs(tmp) <= 1.0:
            p = self._mod2pi(2.0 * math.pi - math.acos(tmp))
            phi = math.atan2(ca - cb, d - sa + sb)
            add_candidate(("R", "L", "R"), (self._mod2pi(alpha - phi + 0.5 * p), p, self._mod2pi(alpha - beta - phi + 0.5 * p)))

        tmp = (6.0 - d * d + 2.0 * cab + 2.0 * d * (-sa + sb)) / 8.0
        if abs(tmp) <= 1.0:
            p = self._mod2pi(2.0 * math.pi - math.acos(tmp))
            phi = math.atan2(ca - cb, d + sa - sb)
            add_candidate(("L", "R", "L"), (self._mod2pi(-alpha - phi + 0.5 * p), p, self._mod2pi(beta - alpha - phi + 0.5 * p)))

        if not candidates:
            return []
        _, modes, lengths = min(candidates, key=lambda item: item[0])

        points: List[Tuple[float, float]] = [(x0, y0)]
        x, y, yaw = x0, y0, yaw0
        step_m = max(0.04, float(step_m))
        for mode, value in zip(modes, lengths):
            if value <= 1e-8:
                continue
            if mode == "S":
                seg_len = value * radius_m
                n = max(1, int(math.ceil(seg_len / step_m)))
                sx, sy = x, y
                for i in range(1, n + 1):
                    s = seg_len * i / n
                    points.append((sx + s * math.cos(yaw), sy + s * math.sin(yaw)))
                x = sx + seg_len * math.cos(yaw)
                y = sy + seg_len * math.sin(yaw)
            else:
                sign = 1.0 if mode == "L" else -1.0
                cx = x - sign * radius_m * math.sin(yaw)
                cy = y + sign * radius_m * math.cos(yaw)
                radial0 = math.atan2(y - cy, x - cx)
                n = max(1, int(math.ceil((value * radius_m) / step_m)))
                for i in range(1, n + 1):
                    a = radial0 + sign * value * i / n
                    points.append((cx + radius_m * math.cos(a), cy + radius_m * math.sin(a)))
                yaw = self._wrap_angle(yaw + sign * value)
                x = cx + radius_m * math.cos(radial0 + sign * value)
                y = cy + radius_m * math.sin(radial0 + sign * value)

        points[-1] = (x3, y3)
        return points


    def _plan_entry_transition_from_current_heading(
        self,
        *,
        start_pose: Tuple[float, float, float],
        end_pose: Tuple[float, float, float],
        nominal_speed_mps: float,
    ) -> List[Tuple[float, float]]:
        x0, y0, yaw0 = (float(start_pose[0]), float(start_pose[1]), float(start_pose[2]))
        x3, y3, yaw3 = (float(end_pose[0]), float(end_pose[1]), float(end_pose[2]))
        dist = math.hypot(x3 - x0, y3 - y0)
        heading_delta = abs(self._wrap_angle(float(yaw3) - float(yaw0)))
        if dist <= 1e-6:
            return []

        min_radius_m = float(self._MIN_TRANSITION_TURN_RADIUS_M)
        points = self._sample_dubins_transition_points(
            start_pose=(x0, y0, yaw0),
            end_pose=(x3, y3, yaw3),
            radius_m=min_radius_m,
            step_m=0.12,
        )

        # Remove accidental duplicates while preserving order.
        compact: List[Tuple[float, float]] = []
        for pt in points:
            if compact and math.hypot(compact[-1][0] - pt[0], compact[-1][1] - pt[1]) <= 1e-5:
                continue
            compact.append((float(pt[0]), float(pt[1])))
        return compact


    def _prepend_runtime_entry_transition(self) -> None:
        pose = get_robot_pose()
        if pose is None:
            self._log("轨迹入口过渡跳过: 当前无实时位姿")
            return
        if not self.loaded_path_points or not self._planned_ranges:
            return
        if self._planned_segment_kind_for_index(1) == "transition":
            return

        ranges = [self._normalize_segment_range(r) for r in self._planned_ranges]
        first = ranges[0]
        first_start = max(0, int(first[0]))
        first_end = max(0, int(first[1]))
        if first_end <= first_start or first_start >= len(self.loaded_path_points):
            return
        formal_start = self.loaded_path_points[first_start]
        formal_next = self.loaded_path_points[min(first_start + 1, first_end)]
        formal_heading = self._heading_from_points(formal_start, formal_next, float(pose.yaw))
        dist_to_start = math.hypot(float(formal_start[0]) - float(pose.x), float(formal_start[1]) - float(pose.y))
        heading_delta = abs(self._wrap_angle(formal_heading - float(pose.yaw)))
        if dist_to_start < 0.35 and heading_delta < math.radians(10.0):
            self._log("轨迹入口过渡: 当前已接近正式轨迹起点，直接开始测量")
            return

        first_speed = max(MIN_SEGMENT_SPEED_MPS, abs(float(first[4])))
        transition_speed = max(0.18, min(first_speed, 0.75))
        transition_points = self._plan_entry_transition_from_current_heading(
            start_pose=(float(pose.x), float(pose.y), float(pose.yaw)),
            end_pose=(float(formal_start[0]), float(formal_start[1]), float(formal_heading)),
            nominal_speed_mps=transition_speed,
        )
        if len(transition_points) < 2:
            self._log("轨迹入口过渡规划失败: 直接执行正式轨迹")
            return

        old_points = list(self.loaded_path_points)
        skip_formal_first = math.hypot(
            float(transition_points[-1][0]) - float(old_points[0][0]),
            float(transition_points[-1][1]) - float(old_points[0][1]),
        ) <= 1e-6
        if skip_formal_first:
            merged_points = list(transition_points) + old_points[1:]
            offset = len(transition_points) - 1
        else:
            merged_points = list(transition_points) + old_points
            offset = len(transition_points)

        shifted_ranges: List[SegmentRange] = []
        for seg in ranges:
            shifted_ranges.append(
                (
                    int(seg[0]) + offset,
                    int(seg[1]) + offset,
                    int(seg[2]),
                    bool(seg[3]),
                    float(seg[4]),
                    float(seg[5]),
                    float(seg[6]),
                )
            )

        transition_len = self._estimate_segment_length(transition_points)
        transition_accel = min(0.9, max(0.25, 0.22 * transition_len))
        transition_decel = min(1.2, max(0.35, 0.30 * transition_len))
        runtime_ranges = [
            (
                0,
                len(transition_points) - 1,
                1,
                False,
                float(transition_speed),
                float(transition_accel),
                float(transition_decel),
            )
        ] + shifted_ranges

        old_kinds = self._remap_segment_kinds_list(
            getattr(self, "_planned_segment_kinds", None),
            ranges,
        ) or [
            "circle" if self._segment_index_is_arc_segment(i + 1) else "line"
            for i in range(len(ranges))
        ]
        self.loaded_path_points = [(float(x), float(y)) for x, y in merged_points]
        self._planned_ranges = runtime_ranges
        self._planned_segment_kinds = ["transition"] + list(old_kinds)
        self._planned_range_task_names = [None] + list(self._planned_range_task_names or [])
        self._log(
            "轨迹入口过渡已生成: "
            f"当前位置→正式起点 距离={dist_to_start:.2f}m | "
            f"过渡点数={len(transition_points)} | 正式分段数={len(ranges)}"
        )


    def _run_planned_ranges(self, tracking_mode: str) -> None:
        if self.controller.car is None:
            self._motion_active = False
            return
        if not self.loaded_path_points or not self._planned_ranges:
            self._motion_active = False
            return

        self._prepend_runtime_entry_transition()
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
                    if not self._wait_for_segment_rcs_started(seg_idx):
                        self._log(f"第{seg_idx}段 RCS启动确认超时，继续执行运动")
                segment_points = self.loaded_path_points[start_idx : end_idx + 1]
                if len(segment_points) < 2:
                    if rcs_start:
                        self.segment_rcs_finish_requested.emit(seg_idx, segment_trajectory_name)
                        self._wait_for_segment_rcs_finished(seg_idx)
                        active_rcs_segment_idx = None
                        active_rcs_trajectory_name = None
                    continue
                segment_kind_meta = self._planned_segment_kind_for_index(seg_idx)
                is_transition = segment_kind_meta == "transition"
                is_straight = self._segment_geometry_is_straight(seg_idx, segment_points)
                segment_len = self._estimate_segment_length(segment_points)
                seg_speed = speed_sign * cruise_speed_mps
                lookahead = 0.62 if is_transition else (0.28 if is_straight else 0.48)
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
                if is_transition:
                    next_continuous = (
                        next_range is not None and next_range[2] == speed_sign
                    )
                else:
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
                    design="transition" if is_transition else ("line" if is_straight else "circle"),
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

                prev_is_transition = (
                    seg_idx >= 2
                    and self._planned_segment_kind_for_index(seg_idx - 1) == "transition"
                )
                # 前进直线段前先对齐目标航向，降低“切段瞬间”车头偏差。
                # 倒车段不做这一步，避免把倒车姿态也硬拉到前向目标视线。
                if (
                    lookat_target is not None
                    and is_straight
                    and speed_sign > 0
                    and not prev_is_transition
                ):
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
                        "segment_kind": (
                            "transition" if is_transition else ("line" if is_straight else "circle")
                        ),
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
                use_uniform_stanley_for_segment = bool(
                    uniform_stanley_tracking and (not is_straight)
                )
                if is_transition:
                    kwargs["lookahead_distance"] = max(float(kwargs["lookahead_distance"]), 0.64)
                    kwargs["stanley_gain"] = 0.18
                    kwargs["stanley_softening_speed_mps"] = max(
                        float(kwargs.get("stanley_softening_speed_mps", 0.60)),
                        1.45,
                    )
                    kwargs["max_w_rate"] = 1.65
                    kwargs["max_w_step"] = 0.032
                    kwargs["smoothing_strength"] = 0.88
                    kwargs["smoothing_strength_curve"] = 0.92
                    kwargs["w_bias_tau"] = 1.05
                    kwargs["prime_pid_on_first_cycle"] = True
                    transition_points_dense = self._densify_polyline_for_tracking(
                        segment_points, max_step_m=0.08
                    )
                    self._log(
                        f"分段{seg_idx}: 入口/衔接过渡曲线跟踪 | "
                        f"长度={segment_len:.2f}m 点数={len(transition_points_dense)}"
                    )
                    self.controller.car.follow_path_with_pid(
                        transition_points_dense,
                        seg_speed,
                        **kwargs,
                    )
                elif use_uniform_stanley_for_segment:
                    is_transition_segment = not (
                        seg_idx - 1 < len(range_task_names)
                        and range_task_names[seg_idx - 1]
                    )
                    if tracking_mode != "stanley_pid":
                        kwargs.update(self._build_uniform_stanley_tracking_kwargs())
                    # 星型测量：仅过渡曲线使用统一 Stanley 软化参数。
                    if is_transition_segment:
                        kwargs["lookahead_distance"] = lookahead
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
                        circle_radius_for_gain = abs(float(orbit_params["radius_m"]))
                        circle_stanley_gain = max(
                            0.25,
                            min(0.40, 0.20 + 0.005 * circle_radius_for_gain),
                        )
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
                            f"分段{seg_idx}: 圆弧段 Stanley 跟踪 "
                            f"(move_circle, adaptive gain={circle_stanley_gain:.3f}) "
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
                if is_straight:
                    self.controller.car.follow_path_with_pid(
                        segment_points,
                        seg_speed,
                        **kwargs,
                    )
                if rcs_start:
                    self.segment_rcs_finish_requested.emit(seg_idx, segment_trajectory_name)
                    if not self._wait_for_segment_rcs_finished(seg_idx):
                        self._log(f"第{seg_idx}段 RCS保存确认超时，后续分段启动时会先自动收尾")
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
                self._wait_for_segment_rcs_finished(active_rcs_segment_idx)
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
        self._on_orbit_rcs_start(
            segment_index=seg_idx,
            trajectory_name=trajectory_name,
            clockwise=self._planned_orbit_clockwise_for_segment(seg_idx),
        )


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

