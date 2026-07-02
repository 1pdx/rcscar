# renamed: radar_safety.py
# role: 雷达障碍物安全检查和紧急停车。
# contains: 小车运动状态判断、安全目标列表选择、停车阈值读取、前方障碍物触发急停和日志记录。
# used by: runtime_refresh.py 定时刷新循环。
# notes: 只负责安全停车判定；雷达目标显示、锁定和 RCS 关联在其它模块。
# -*- coding: utf-8 -*-

from ui_shared import *
from ui_shared import _ClusterSafetyProxy


MainWindow = None  # 主入口回填真实 MainWindow，供 mixin 内静态方法引用。


class MainWindowRadarSafetyMixin:
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

