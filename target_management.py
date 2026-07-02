# renamed: target_management.py
# role: 雷达目标选择、目标点标定和目标状态管理。
# contains: 当前 RCS 目标名、雷达目标 ID 提取、新鲜度判断、自动重锁、轨迹图目标点同步、雷达局部坐标到全局坐标投影。
# used by: RCS 采集、运行时雷达点击、星型测量目标点更新。
# notes: 不做雷达安全停车判断；安全逻辑在 radar_safety.py。
# -*- coding: utf-8 -*-

from ui_shared import *


MainWindow = None  # 主入口回填真实 MainWindow，供 mixin 内静态方法引用。


class MainWindowTargetMixin:
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

