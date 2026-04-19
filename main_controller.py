# main_controller.py
# -*- coding: utf-8 -*-
import importlib.util
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

from PyQt5 import QtWidgets
from imu_gnss_pose import get_robot_pose, calibrate_pose_to_current, PoseSolution
from car_control import ScoutMiniCAN
from pi_power import PiPowerMonitor
from can_init import init_can, CanInitResult

try:
    from ars40x_cluster_logger import ClusterCsvRuntime
except ImportError:  # pragma: no cover
    ClusterCsvRuntime = None  # type: ignore


_RADAR_MODULE = None
PathPoint = Tuple[float, float]
PathSegmentRange = Tuple[int, int, int, bool, float, float, float]
STRAIGHT_POINT_COUNT = 2
DEFAULT_ACCEL_DIST_M = 1.5
DEFAULT_DECEL_DIST_M = 1.5


@dataclass
class QueuedPathTask:
    name: str
    local_points: List[PathPoint]
    ranges: List[PathSegmentRange]


@dataclass
class PlannedTaskSequence:
    task_names: List[str]
    local_points: List[PathPoint]
    ranges: List[PathSegmentRange]
    range_task_names: List[Optional[str]]
    transition_count: int
    transition_pairs: List[Tuple[str, str]]


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


class MainController:
    """
    主控制器类，负责处理业务逻辑和设备控制
    """
    
    def __init__(self, is_linux: bool) -> None:
        self._is_linux = is_linux
        
        # ================== CAN 初始化 ==================
        self.can_init_result: Optional[CanInitResult] = None
        if self._is_linux:
            try:
                self.can_init_result = init_can(interface="can0", bitrate=500000)
                print(f"[MainController] CAN 初始化结果: {self.can_init_result}")
            except Exception as e:
                print(f"[MainController] 调用 init_can 失败: {e}")
                self.can_init_result = None
        else:
            print("[MainController] 非 Linux 系统，跳过 socketcan 初始化。")

        # ================== 底层设备实例 ==================
        self.car: Optional[ScoutMiniCAN] = None
        self.pi_power: Optional[PiPowerMonitor] = None

        # 小车底盘
        try:
            if self._is_linux:
                self.car = ScoutMiniCAN(channel="can0", interface="socketcan", bitrate=500000)
            else:
                print("[MainController] 非 Linux 系统，使用虚拟小车实例")
                # 创建虚拟小车实例用于测试
                self.car = ScoutMiniCAN(channel="virtual", interface="virtual", bitrate=500000)
        except Exception as e:
            print(f"[MainController] 初始化小车失败: {e}")
            self.car = None

        # 树莓派电量监测
        try:
            if self._is_linux:
                self.pi_power = PiPowerMonitor(i2c_bus=1, addr=0x41)
            else:
                print("[MainController] 非 Linux 系统，跳过树莓派电量监控初始化")
                self.pi_power = None
        except Exception as e:
            print(f"[MainController] 初始化树莓派电量监控失败: {e}")
            self.pi_power = None

        self._radar_selected_id: Optional[int] = None
        self.cluster_csv_runtime: Optional[Any] = None
        radar_mod = None
        if self._is_linux:
            try:
                radar_mod = _load_radar_processing_module()
            except Exception as e:
                print(f"[MainController] Radar module load failed: {e}")
            if radar_mod is not None and self.can_init_result is not None and self.can_init_result.ok:
                try:
                    radar_mod.init_radar_cluster_output(
                        iface="can0",
                        bitrate=500000,
                        sensor_id=0,
                        send_count=20,
                        verify_timeout_s=1.2,
                        retries=2,
                    )
                    radar_mod.print_ars40x_terminal_mode_commands(sensor_id=0)
                    print(
                        "[MainController] 雷达已配置为 Cluster 模式；Object 列表输出已停用。"
                    )
                except Exception as e:
                    print(f"[MainController] Radar cluster output init failed: {e}")
            if (
                ClusterCsvRuntime is not None
                and self.can_init_result is not None
                and self.can_init_result.ok
            ):
                try:
                    self.cluster_csv_runtime = ClusterCsvRuntime(
                        iface="can0",
                        bitrate=500000,
                        sensor_id=0,
                        enable_ins=True,
                    )
                    self.cluster_csv_runtime.start()
                    print("[MainController] Cluster CSV 接收线程已启动（与底盘共用 can0）。")
                except Exception as e:
                    print(f"[MainController] Cluster CSV runtime start failed: {e}")
                    self.cluster_csv_runtime = None
        else:
            print("[MainController] Non-Linux system, skip radar cluster init.")

        # Battery percent mapping for the chassis (adjust if needed).
        self._car_batt_v_min = 23.0
        self._car_batt_v_max = 29.25

    def _estimate_car_battery_percent(self, voltage: float) -> Optional[float]:
        if voltage <= 1e-3:
            return None
        if self._car_batt_v_max <= self._car_batt_v_min:
            return None
        percent = (voltage - self._car_batt_v_min) / (
            self._car_batt_v_max - self._car_batt_v_min
        ) * 100.0
        return max(0.0, min(100.0, percent))

    @staticmethod
    def _describe_car_control_mode(mode: int) -> str:
        return {
            0: "待机",
            1: "CAN",
            3: "遥控",
        }.get(int(mode), f"未知({int(mode)})")

    def evaluate_circle_motion(self, radius_m: float, speed_mps: float) -> Dict[str, Any]:
        planner = self.car if self.car is not None else ScoutMiniCAN
        return planner.evaluate_circle_command(radius_m, speed_mps)

    def ensure_car_ready(self, parent_window) -> bool:
        """确保小车准备就绪"""
        if self.car is None:
            QtWidgets.QMessageBox.warning(
                parent_window,
                "CAN 错误",
                "小车 CAN 未连接，请检查 USB-CAN 适配器、接线或驱动程序。",
            )
            return False
        return True

    def begin_cluster_rcs_capture(
        self,
        output_dir: str,
        stem: str,
        run_number: int = 1,
        calibration: Optional[Any] = None,
    ) -> bool:
        """开始一段 Cluster RCS CSV 写入（与 ars40x_cluster_logger / DRI Raw 元数据格式一致）。"""
        if self.cluster_csv_runtime is None:
            return False
        try:
            self.cluster_csv_runtime.begin_recording(
                output_dir,
                run_number=run_number,
                stem=stem,
                calibration=calibration,
            )
            return True
        except Exception as e:
            print(f"[MainController] begin_cluster_rcs_capture failed: {e}")
            return False

    def end_cluster_rcs_capture(self) -> Optional[str]:
        """结束当前 CSV 采集，返回文件路径。"""
        if self.cluster_csv_runtime is None:
            return None
        try:
            return self.cluster_csv_runtime.end_recording()
        except Exception as e:
            print(f"[MainController] end_cluster_rcs_capture failed: {e}")
            return None

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

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

    @classmethod
    def _normalize_speed_mps(cls, value: Any, default: float = 0.5) -> float:
        return max(0.08, cls._normalize_positive_float(value, default))

    @classmethod
    def _normalize_segment_range(cls, value: Tuple[Any, ...]) -> PathSegmentRange:
        start_idx = int(value[0])
        end_idx = int(value[1])
        speed_sign = cls._normalize_speed_sign(value[2] if len(value) >= 3 else 1)
        rcs_start = bool(value[3]) if len(value) >= 4 else False
        speed_mps = cls._normalize_speed_mps(value[4] if len(value) >= 5 else 0.5)
        accel_dist = cls._normalize_positive_float(
            value[5] if len(value) >= 6 else DEFAULT_ACCEL_DIST_M,
            DEFAULT_ACCEL_DIST_M,
        )
        decel_dist = cls._normalize_positive_float(
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

    @staticmethod
    def _path_length(points: List[PathPoint]) -> float:
        total_len = 0.0
        for i in range(len(points) - 1):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            total_len += math.hypot(dx, dy)
        return total_len

    @staticmethod
    def _find_heading(points: List[PathPoint], at_start: bool) -> float:
        if len(points) < 2:
            return 0.0
        if at_start:
            indices = range(len(points) - 1)
        else:
            indices = range(len(points) - 2, -1, -1)
        for idx in indices:
            x0, y0 = points[idx]
            x1, y1 = points[idx + 1]
            dx = x1 - x0
            dy = y1 - y0
            if dx * dx + dy * dy > 1e-8:
                return math.atan2(dy, dx)
        return 0.0

    @classmethod
    def _path_pose(cls, points: List[PathPoint], at_start: bool) -> Tuple[float, float, float]:
        if not points:
            return (0.0, 0.0, 0.0)
        heading = cls._find_heading(points, at_start=at_start)
        if at_start:
            x, y = points[0]
        else:
            x, y = points[-1]
        return (float(x), float(y), float(heading))

    @classmethod
    def _ensure_task_ranges(
        cls,
        points: List[PathPoint],
        ranges: List[PathSegmentRange],
    ) -> List[PathSegmentRange]:
        normalized = [
            cls._normalize_segment_range(seg_range)
            for seg_range in (ranges or [])
            if len(seg_range) >= 2 and int(seg_range[1]) > int(seg_range[0])
        ]
        if normalized:
            return normalized
        if len(points) < 2:
            return []
        return [
            (
                0,
                len(points) - 1,
                1,
                False,
                0.5,
                DEFAULT_ACCEL_DIST_M,
                DEFAULT_DECEL_DIST_M,
            )
        ]

    @staticmethod
    def _task_start_speed_abs(ranges: List[PathSegmentRange]) -> float:
        if not ranges:
            return 0.5
        return max(0.08, abs(float(ranges[0][4])))

    @staticmethod
    def _task_end_speed_abs(ranges: List[PathSegmentRange]) -> float:
        if not ranges:
            return 0.5
        return max(0.08, abs(float(ranges[-1][4])))

    @staticmethod
    def _sample_line(
        start: PathPoint,
        end: PathPoint,
        step: float = 0.35,
    ) -> List[PathPoint]:
        del step
        x0, y0 = start
        x1, y1 = end
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist <= 1e-8:
            return [(float(x0), float(y0))]
        n = max(1, STRAIGHT_POINT_COUNT - 1)
        points: List[PathPoint] = []
        for i in range(n + 1):
            t = i / n
            points.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
        return points

    @staticmethod
    def _sample_cubic_bezier(
        p0: PathPoint,
        p1: PathPoint,
        p2: PathPoint,
        p3: PathPoint,
        step: float = 0.18,
    ) -> List[PathPoint]:
        chord = math.hypot(p3[0] - p0[0], p3[1] - p0[1])
        ctrl_len = (
            math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            + math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            + math.hypot(p3[0] - p2[0], p3[1] - p2[1])
        )
        est_len = max(chord, ctrl_len)
        n = max(6, min(160, int(est_len / max(1e-3, step)) + 1))
        points: List[PathPoint] = []
        for i in range(n + 1):
            t = i / n
            omt = 1.0 - t
            x = (
                (omt ** 3) * p0[0]
                + 3.0 * (omt ** 2) * t * p1[0]
                + 3.0 * omt * (t ** 2) * p2[0]
                + (t ** 3) * p3[0]
            )
            y = (
                (omt ** 3) * p0[1]
                + 3.0 * (omt ** 2) * t * p1[1]
                + 3.0 * omt * (t ** 2) * p2[1]
                + (t ** 3) * p3[1]
            )
            points.append((x, y))
        return points

    def _plan_transition_candidate(
        self,
        start_pose: Tuple[float, float, float],
        end_pose: Tuple[float, float, float],
        motion_sign: int,
        close_threshold_m: float,
    ) -> Optional[Tuple[List[PathPoint], int, float]]:
        x0, y0, yaw0 = start_pose
        x1, y1, yaw1 = end_pose
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)
        motion_yaw0 = yaw0 if motion_sign > 0 else self._wrap_angle(yaw0 + math.pi)
        motion_yaw1 = yaw1 if motion_sign > 0 else self._wrap_angle(yaw1 + math.pi)
        if dist <= 1e-6:
            heading_delta = self._wrap_angle(motion_yaw1 - motion_yaw0)
            if abs(heading_delta) <= math.radians(8.0):
                return None
            tangent_len = max(0.45, min(1.2, 0.55 * abs(heading_delta) + 0.2))
            ctrl1 = (
                x0 + tangent_len * math.cos(motion_yaw0),
                y0 + tangent_len * math.sin(motion_yaw0),
            )
            ctrl2 = (
                x1 - tangent_len * math.cos(motion_yaw1),
                y1 - tangent_len * math.sin(motion_yaw1),
            )
            points = self._sample_cubic_bezier((x0, y0), ctrl1, ctrl2, (x1, y1), step=0.12)
            path_len = self._path_length(points)
            if path_len <= 1e-6:
                return None
            cost = path_len + 0.25 * abs(heading_delta)
            return points, motion_sign, cost

        chord_heading = math.atan2(dy, dx)
        heading_penalty = abs(self._wrap_angle(chord_heading - motion_yaw0)) + abs(
            self._wrap_angle(motion_yaw1 - chord_heading)
        )

        if dist < close_threshold_m and heading_penalty <= math.radians(30.0):
            points = self._sample_line((x0, y0), (x1, y1), step=close_threshold_m / 2.0)
            cost = dist + 0.15 * heading_penalty
            return points, motion_sign, cost

        tangent_len = max(0.35, min(2.8, 0.45 * dist + 0.28 * heading_penalty))
        ctrl1 = (
            x0 + tangent_len * math.cos(motion_yaw0),
            y0 + tangent_len * math.sin(motion_yaw0),
        )
        ctrl2 = (
            x1 - tangent_len * math.cos(motion_yaw1),
            y1 - tangent_len * math.sin(motion_yaw1),
        )
        points = self._sample_cubic_bezier((x0, y0), ctrl1, ctrl2, (x1, y1))
        path_len = self._path_length(points)
        if path_len <= 1e-6:
            return None
        reverse_penalty = 0.08 if motion_sign < 0 else 0.0
        cost = path_len + 0.55 * heading_penalty + reverse_penalty
        return points, motion_sign, cost

    def plan_transition_path(
        self,
        start_pose: Tuple[float, float, float],
        end_pose: Tuple[float, float, float],
        start_speed_mps: float,
        end_speed_mps: float,
        close_threshold_m: float = 0.5,
    ) -> Tuple[List[PathPoint], List[PathSegmentRange]]:
        x0, y0, _ = start_pose
        x1, y1, _ = end_pose
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist <= 1e-6:
            return [], []

        candidates = []
        for motion_sign in (1, -1):
            candidate = self._plan_transition_candidate(
                start_pose=start_pose,
                end_pose=end_pose,
                motion_sign=motion_sign,
                close_threshold_m=close_threshold_m,
            )
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            points = self._sample_line((x0, y0), (x1, y1), step=0.3)
            motion_sign = 1
        else:
            points, motion_sign, _ = min(candidates, key=lambda item: item[2])
        if len(points) < 2:
            return [], []

        transition_len = self._path_length(points)
        transition_speed = max(
            0.18,
            min(
                max(abs(float(start_speed_mps)), abs(float(end_speed_mps)), 0.25),
                0.9,
            ),
        )
        accel_dist = min(0.8, max(0.2, 0.25 * transition_len))
        decel_dist = min(0.8, max(0.2, 0.25 * transition_len))
        ranges: List[PathSegmentRange] = [
            (
                0,
                len(points) - 1,
                motion_sign,
                False,
                transition_speed,
                accel_dist,
                decel_dist,
            )
        ]
        return points, ranges

    @staticmethod
    def _append_points_and_ranges(
        base_points: List[PathPoint],
        base_ranges: List[PathSegmentRange],
        append_points: List[PathPoint],
        append_ranges: List[PathSegmentRange],
    ) -> None:
        if not append_points:
            return
        if not base_points:
            base_points.extend((float(x), float(y)) for x, y in append_points)
            base_ranges.extend(append_ranges)
            return

        skip_first = (
            math.hypot(
                base_points[-1][0] - append_points[0][0],
                base_points[-1][1] - append_points[0][1],
            )
            <= 1e-6
        )
        offset = len(base_points) - 1 if skip_first else len(base_points)
        if skip_first:
            base_points.extend((float(x), float(y)) for x, y in append_points[1:])
        else:
            base_points.extend((float(x), float(y)) for x, y in append_points)
        for start_idx, end_idx, speed_sign, rcs_start, speed_mps, accel_dist, decel_dist in append_ranges:
            base_ranges.append(
                (
                    offset + int(start_idx),
                    offset + int(end_idx),
                    int(speed_sign),
                    bool(rcs_start),
                    float(speed_mps),
                    float(accel_dist),
                    float(decel_dist),
                )
            )

    def build_task_queue_plan(
        self,
        tasks: List[QueuedPathTask],
        close_threshold_m: float = 0.5,
    ) -> PlannedTaskSequence:
        valid_tasks: List[QueuedPathTask] = []
        for task in tasks:
            local_points = [(float(x), float(y)) for x, y in task.local_points]
            if len(local_points) < 2:
                continue
            valid_tasks.append(
                QueuedPathTask(
                    name=str(task.name),
                    local_points=local_points,
                    ranges=self._ensure_task_ranges(local_points, list(task.ranges or [])),
                )
            )
        if not valid_tasks:
            return PlannedTaskSequence([], [], [], [], 0, [])

        combined_points: List[PathPoint] = []
        combined_ranges: List[PathSegmentRange] = []
        range_task_names: List[Optional[str]] = []
        transition_pairs: List[Tuple[str, str]] = []

        first_task = valid_tasks[0]
        self._append_points_and_ranges(
            combined_points,
            combined_ranges,
            first_task.local_points,
            first_task.ranges,
        )
        range_task_names.extend([first_task.name] * len(first_task.ranges))

        for idx in range(1, len(valid_tasks)):
            prev_task = valid_tasks[idx - 1]
            next_task = valid_tasks[idx]
            prev_end_pose = self._path_pose(prev_task.local_points, at_start=False)
            next_start_pose = self._path_pose(next_task.local_points, at_start=True)
            transition_points, transition_ranges = self.plan_transition_path(
                start_pose=prev_end_pose,
                end_pose=next_start_pose,
                start_speed_mps=self._task_end_speed_abs(prev_task.ranges),
                end_speed_mps=self._task_start_speed_abs(next_task.ranges),
                close_threshold_m=close_threshold_m,
            )
            if transition_points and transition_ranges:
                self._append_points_and_ranges(
                    combined_points,
                    combined_ranges,
                    transition_points,
                    transition_ranges,
                )
                range_task_names.extend([None] * len(transition_ranges))
                transition_pairs.append((prev_task.name, next_task.name))
            self._append_points_and_ranges(
                combined_points,
                combined_ranges,
                next_task.local_points,
                next_task.ranges,
            )
            range_task_names.extend([next_task.name] * len(next_task.ranges))

        return PlannedTaskSequence(
            task_names=[task.name for task in valid_tasks],
            local_points=combined_points,
            ranges=combined_ranges,
            range_task_names=range_task_names,
            transition_count=len(transition_pairs),
            transition_pairs=transition_pairs,
        )

    def execute_line_movement(self, parent_window, dist: float, speed: float) -> None:
        """执行直线运动"""
        if not self.ensure_car_ready(parent_window):
            return
            
        if abs(dist) < 1e-3:
            QtWidgets.QMessageBox.warning(parent_window, "参数错误", "直线距离需要非零值。")
            return
        if speed <= 0:
            QtWidgets.QMessageBox.warning(parent_window, "参数错误", "线速度必须为正值。")
            return

        # 基于当前位置生成规划直线轨迹（ENU）
        pose = get_robot_pose()
        if pose is not None:
            self._update_planned_line(parent_window, pose, dist)
            # 启动直线运动线程
            t = threading.Thread(
                target=self.car.move_straight,
                args=(dist, speed),
                kwargs={
                    "metrics_callback": getattr(parent_window, "_emit_tracking_metrics", None),
                    "sample_callback": getattr(parent_window, "_append_tracking_sample", None),
                    "run_label": "直线",
                },
                daemon=True,
            )
            t.start()
        else:
            QtWidgets.QMessageBox.warning(parent_window, "定位无效", "无法获取当前位姿，请检查 IMU/GNSS 连接。")

    def execute_circle_movement(
        self,
        parent_window,
        radius: float,
        angle: float,
        speed: float,
        *,
        clockwise: Optional[bool] = None,
    ) -> None:
        """执行圆周运动"""
        if not self.ensure_car_ready(parent_window):
            return
            
        if radius <= 0:
            QtWidgets.QMessageBox.warning(parent_window, "参数错误", "半径必须为正值。")
            return
        if angle <= 0:
            QtWidgets.QMessageBox.warning(parent_window, "参数错误", "角度必须为正值。")
            return
        if speed <= 0:
            QtWidgets.QMessageBox.warning(parent_window, "参数错误", "线速度必须为正值。")
            return

        circle_plan = self.evaluate_circle_motion(radius, speed)
        speed = float(circle_plan["adjusted_speed_mps"])
        if speed <= 0:
            QtWidgets.QMessageBox.warning(parent_window, "参数错误", "圆周运动速度无效。")
            return
        if bool(circle_plan["adjusted"]):
            print(
                "[MainController] Circle motion speed adjusted for feasibility: "
                f"R={float(circle_plan['radius_m']):.2f}m, "
                f"v={float(circle_plan['requested_speed_mps']):.2f}->{speed:.2f}m/s, "
                f"nominal |w|={float(circle_plan['requested_nominal_w_radps']):.2f}"
                f"->{float(circle_plan['adjusted_nominal_w_radps']):.2f}rad/s"
            )

        # 基于当前位置生成规划圆周轨迹
        pose = get_robot_pose()
        if pose is not None:
            if clockwise is None:
                clockwise = angle > 0
            clockwise = bool(clockwise)
            self._update_planned_circle(parent_window, pose, radius, abs(angle), clockwise)
            # 启动圆周运动线程（Stanley 路径跟踪，见 car_control.move_circle）
            t = threading.Thread(
                target=self.car.move_circle,
                args=(radius, angle, speed, clockwise),
                kwargs={
                    "metrics_callback": getattr(parent_window, "_emit_tracking_metrics", None),
                    "sample_callback": getattr(parent_window, "_append_tracking_sample", None),
                    "run_label": "圆周",
                },
                daemon=True,
            )
            t.start()
        else:
            QtWidgets.QMessageBox.warning(parent_window, "定位无效", "无法获取当前位姿，请检查 IMU/GNSS 连接。")

    def calibrate_coordinate_system(self, parent_window) -> None:
        """坐标系校准"""
        try:
            calibrate_pose_to_current()
            QtWidgets.QMessageBox.information(parent_window, "校准成功", "已将当前位姿设为 ENU 坐标系原点 (0,0,0)。")
        except Exception as e:
            QtWidgets.QMessageBox.warning(parent_window, "校准失败", f"校准过程中出现错误: {e}")

    def calibrate_heading(self, parent_window) -> None:
        """航向角校准"""
        pose = get_robot_pose()
        if pose is None:
            QtWidgets.QMessageBox.warning(parent_window, "校准失败", "无法获取当前位姿")
            return
            
        yaw_deg = math.degrees(pose.yaw)
        QtWidgets.QMessageBox.information(
            parent_window, 
            "当前航向角", 
            f"当前航向角: {pose.yaw:.3f} rad ({yaw_deg:.1f}°)\n"
            f"东向: {math.cos(pose.yaw):.3f}, 北向: {math.sin(pose.yaw):.3f}"
        )

    def select_radar_target(self, parent_window) -> None:
        """ARS40X 仅 Cluster 输出：在主界面雷达图中点击簇点可选中序号。"""
        QtWidgets.QMessageBox.information(
            parent_window,
            "选择簇目标",
            "请在主界面「雷达 Cluster 检查图」中直接点击散点；"
            "选中序号会显示在状态栏（不再使用 CAN Object 列表）。",
        )

    def set_selected_radar_id(self, oid: int) -> None:
        self._radar_selected_id = int(oid)

    def load_path_file(self, parent_window) -> None:
        """
        加载路径文件。

        文件中 x、y（米）与 imu_gnss_pose.get_robot_pose() 使用同一套校准后平面坐标：
        原点在 ENU 校准原点，轴向与界面轨迹图一致（通常 X 东向、Y 北向），
        不按车头旋转，也不平移到“当前车位”。
        """
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent_window,
            "选择轨迹文件（校准局部坐标系，单位 m）",
            "",
            "CSV / 文本 (*.csv *.txt);;所有文件 (*)",
        )
        if not filename:
            return

        raw_points = []
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip() or line.lstrip().startswith('#'):
                        continue
                    parts = line.replace(',', ' ').split()
                    if len(parts) < 2:
                        continue
                    try:
                        xr = float(parts[0])
                        yr = float(parts[1])
                    except ValueError:
                        continue
                    raw_points.append((xr, yr))
        except Exception as e:
            QtWidgets.QMessageBox.warning(parent_window, "加载失败", f"读取轨迹文件失败: {e}")
            return

        if len(raw_points) < 2:
            QtWidgets.QMessageBox.warning(parent_window, "轨迹无效", "轨迹点数量不足（至少需要 2 个点）。")
            return
        parent_window.loaded_path_local_points = list(raw_points)
        if hasattr(parent_window, "_planned_ranges"):
            parent_window._planned_ranges = None
        if hasattr(parent_window, "_planned_range_task_names"):
            parent_window._planned_range_task_names = []

        # 与预设轨迹里 rotate_with_yaw=False 的语义一致：直接使用局部平面坐标作为跟踪路径
        path_points = [(float(x), float(y)) for x, y in raw_points]
        if hasattr(parent_window, "_loaded_path_frame"):
            f = parent_window._loaded_path_frame
            if not f.is_fixed_origin() and hasattr(
                parent_window, "_use_calibration_plane_path_origin"
            ):
                parent_window._use_calibration_plane_path_origin()
            elif not f.is_fixed_origin():
                f.origin_key = ""
                f.origin_label = ""
                f.origin_x_m = 0.0
                f.origin_y_m = 0.0
                f.origin_z_m = 0.0
        if hasattr(parent_window, "_path_preview_uses_virtual_pose"):
            parent_window._path_preview_uses_virtual_pose = False

        parent_window.loaded_path_points = path_points
        parent_window.planned_x = [p[0] for p in path_points]
        parent_window.planned_y = [p[1] for p in path_points]
        parent_window.traj_planned_curve.setData(parent_window.planned_x, parent_window.planned_y)

        if parent_window.planned_x and parent_window.planned_y:
            min_x, max_x = min(parent_window.planned_x), max(parent_window.planned_x)
            min_y, max_y = min(parent_window.planned_y), max(parent_window.planned_y)
            margin = 0.5
            parent_window.traj_plot.setXRange(min_x - margin, max_x + margin, padding=0)
            parent_window.traj_plot.setYRange(min_y - margin, max_y + margin, padding=0)
        if hasattr(parent_window, "_update_path_coordinate_widgets"):
            parent_window._update_path_coordinate_widgets(
                frame_override=getattr(parent_window, "_loaded_path_frame", None)
            )
        if hasattr(parent_window, "_update_run_path_button_state"):
            parent_window._update_run_path_button_state()
        else:
            parent_window.btn_run_path.setEnabled(True)

        ins_hint = ""
        if get_robot_pose() is None:
            ins_hint = "\n\n当前尚无有效位姿；执行轨迹前请等待 INS 可用。"

        QtWidgets.QMessageBox.information(
            parent_window,
            "加载成功",
            f"已读取 {len(path_points)} 个轨迹点。\n"
            "说明：坐标为校准后局部平面（与 get_robot_pose 一致），"
            "例如 (0,0)→(0,10) 为沿 +Y 走 10m，(0,0)→(5,5) 为沿 XY 45° 方向。"
            f"{ins_hint}",
        )

    def execute_loaded_path(
        self,
        parent_window,
        speed: float,
        tracking_mode: str = "pid",
        tracking_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """执行已加载的路径"""
        if not self.ensure_car_ready(parent_window):
            return
        if not hasattr(parent_window, 'loaded_path_points') or len(parent_window.loaded_path_points) < 2:
            QtWidgets.QMessageBox.warning(parent_window, "轨迹未加载", "请先加载包含至少 2 个点的轨迹文件。")
            return
        if speed <= 0:
            QtWidgets.QMessageBox.warning(parent_window, "参数错误", "线速度必须为正值。")
            return

        path_tracking_kwargs = dict(tracking_kwargs or {})
        path_tracking_kwargs.setdefault(
            "record_context",
            {
                "segment_index": 1,
                "segment_kind": "path",
                "segment_trajectory_name": "整条轨迹",
                "segment_start_idx": 0,
                "segment_end_idx": max(0, len(parent_window.loaded_path_points) - 1),
                "segment_cruise_speed_mps": float(speed),
            },
        )
        follow_kw: Dict[str, Any] = {
            "tracking_mode": tracking_mode,
            "metrics_callback": getattr(parent_window, "_emit_tracking_metrics", None),
            "sample_callback": getattr(parent_window, "_append_tracking_sample", None),
            "run_label": "整条轨迹",
            "lookahead_distance": 0.6,
            "max_w_rate": 3.0,
            "max_w_step": 0.06,
        }
        # 纯 Stanley：略增大前视、限制单步角速度跳变，与 UI 侧较低 stanley_gain 配套
        if tracking_mode == "stanley":
            follow_kw["lookahead_distance"] = 0.78
            follow_kw["max_w_rate"] = 2.6
            follow_kw["max_w_step"] = 0.048
        follow_kw.update(path_tracking_kwargs)
        t = threading.Thread(
            target=self.car.follow_path_with_pid,
            args=(parent_window.loaded_path_points, speed),
            kwargs=follow_kw,
            daemon=True,
        )
        print(
            f"[MainController] 使用 {tracking_mode} 控制执行路径跟踪，速度: {speed} m/s"
        )
        t.start()

    def get_power_snapshot(self) -> Dict[str, Dict[str, Any]]:
        snapshot: Dict[str, Dict[str, Any]] = {
            "pi": {
                "name": "树莓派",
                "available": False,
                "percent": None,
                "voltage": None,
                "power": None,
                "status": "unavailable",
                "text": "树莓派电量: 不可用",
            },
            "car": {
                "name": "小车底盘",
                "available": False,
                "percent": None,
                "voltage": None,
                "power": None,
                "status": "unavailable",
                "mode": None,
                "mode_text": "--",
                "feedback_age_s": None,
                "text": "小车电量: 不可用",
            },
        }

        if self.pi_power is not None:
            try:
                if hasattr(self.pi_power, "available") and self.pi_power.available:
                    p_status = self.pi_power.read_status()
                    snapshot["pi"]["available"] = True
                    if p_status is not None:
                        snapshot["pi"].update(
                            {
                                "percent": float(p_status.percent),
                                "voltage": float(p_status.voltage),
                                "power": float(p_status.power),
                                "status": "ok",
                                "text": (
                                    f"树莓派电量: {p_status.voltage:.2f} V "
                                    f"({p_status.percent:.0f}%, {p_status.power:.2f} W)"
                                ),
                            }
                        )
                    else:
                        snapshot["pi"]["status"] = "read_failed"
                        snapshot["pi"]["text"] = "树莓派电量: 读取失败"
            except Exception as e:
                print(f"[MainController] 读取树莓派电量失败: {e}")
                snapshot["pi"]["status"] = "read_error"
                snapshot["pi"]["text"] = "树莓派电量: 读取异常"

        if self.car is not None:
            snapshot["car"]["available"] = True
            try:
                st = self.car.get_status()
                mode = int(st.control_mode)
                mode_text = self._describe_car_control_mode(mode)
                sys_status_ts = float(getattr(st, "sys_status_update", 0.0) or 0.0)
                feedback_age_s = None
                if sys_status_ts > 0.0:
                    feedback_age_s = max(0.0, time.time() - sys_status_ts)

                snapshot["car"].update(
                    {
                        "mode": mode,
                        "mode_text": mode_text,
                        "feedback_age_s": feedback_age_s,
                    }
                )

                if sys_status_ts <= 0.0:
                    snapshot["car"]["status"] = "no_feedback"
                    snapshot["car"]["text"] = "小车电量: 未收到 0x211 系统状态反馈"
                elif feedback_age_s is not None and feedback_age_s > 1.5:
                    snapshot["car"]["status"] = "stale"
                    snapshot["car"]["text"] = (
                        f"小车电量: 0x211 反馈过期 ({feedback_age_s:.1f} s)"
                    )
                elif st.battery_voltage <= 1e-3:
                    snapshot["car"]["status"] = "invalid"
                    snapshot["car"]["text"] = "小车电量: 0x211 已收到，但电压字段无效"
                else:
                    percent = self._estimate_car_battery_percent(float(st.battery_voltage))
                    base_text = f"小车电量: {st.battery_voltage:.1f} V"
                    if percent is not None:
                        base_text += f" ({percent:.0f}%)"

                    status = "ok"
                    if mode == 0:
                        status = "standby"
                        base_text += " | 模式: 待机，需先发送 0x421 使能 CAN"
                    elif mode == 1:
                        base_text += " | 模式: CAN"
                    elif mode == 3:
                        status = "remote"
                        base_text += " | 模式: 遥控优先"
                    else:
                        status = "mode_unknown"
                        base_text += f" | 模式: {mode_text}"

                    snapshot["car"].update(
                        {
                            "percent": percent,
                            "voltage": float(st.battery_voltage),
                            "status": status,
                            "text": base_text,
                        }
                    )
            except Exception as e:
                print(f"[MainController] 读取小车电量失败: {e}")
                snapshot["car"]["status"] = "read_error"
                snapshot["car"]["text"] = "小车电量: 读取异常"

        return snapshot

    def get_power_status(self) -> Tuple[str, str]:
        """获取电源状态"""
        snapshot = self.get_power_snapshot()
        return snapshot["pi"]["text"], snapshot["car"]["text"]

    def get_can_status(self) -> str:
        """获取CAN状态"""
        can_text = "CAN 初始化: 未执行"
        if self.can_init_result is not None:
            if self.can_init_result.ok:
                kbps = self.can_init_result.bitrate // 1000
                can_text = (
                    f"CAN 初始化: 成功 ({self.can_init_result.interface}@{kbps} kbps)"
                )
            else:
                msg = self.can_init_result.message or ""
                if len(msg) > 60:
                    msg = msg[:60] + "..."
                can_text = f"CAN 初始化: 失败 ({msg})"
        elif not self._is_linux:
            can_text = "CAN 初始化: 当前系统非 Linux，socketcan 功能已禁用"
        return can_text

    def get_radar_status(self) -> str:
        if self.cluster_csv_runtime is not None:
            base = "雷达: Cluster(0x600/0x701) | 跟踪图为簇解析"
            if self._radar_selected_id is not None:
                return f"{base} | 选中序号={int(self._radar_selected_id)}"
            return f"{base} | 点击散点可选中簇序号"
        return "雷达: Cluster 接收未启动（检查 CAN）"

    def get_stable_radar_targets(self, min_confidence: float = 0.7) -> List:
        del min_confidence
        return []

    # ========= 规划轨迹（ENU） =========

    def _update_planned_line(self, parent_window, pose: PoseSolution, dist_m: float) -> None:
        """
        根据当前位置 + 直线参数生成规划直线路径（ENU 坐标）。
        起点为当前 ENU (x, y)，方向为当前 yaw。
        """
        parent_window.planned_x.clear()
        parent_window.planned_y.clear()

        if abs(dist_m) < 1e-3:
            parent_window.traj_planned_curve.setData([], [])
            return

        x0, y0, yaw = pose.x, pose.y, pose.yaw
        n = max(1, STRAIGHT_POINT_COUNT - 1)
        for i in range(n + 1):
            s = dist_m * i / n
            x = x0 + s * math.cos(yaw)
            y = y0 + s * math.sin(yaw)
            parent_window.planned_x.append(x)
            parent_window.planned_y.append(y)

        parent_window.traj_planned_curve.setData(parent_window.planned_x, parent_window.planned_y)

    def _update_planned_circle(
        self,
        parent_window,
        pose: PoseSolution,
        radius_m: float,
        angle_deg: float,
        clockwise: bool,
    ) -> None:
        """
        根据当前位置 + 圆弧参数生成规划圆周轨迹（ENU 坐标）。
        圆心计算与 car_control.move_circle 完全对齐。
        """
        parent_window.planned_x.clear()
        parent_window.planned_y.clear()

        radius_m = max(0.1, abs(radius_m))
        angle_deg = abs(angle_deg)
        if angle_deg < 1e-2:
            parent_window.traj_planned_curve.setData([], [])
            return

        x0, y0, yaw0 = pose.x, pose.y, pose.yaw

        if clockwise:
            cx = x0 + radius_m * math.sin(yaw0)
            cy = y0 - radius_m * math.cos(yaw0)
            sign = -1.0
        else:
            cx = x0 - radius_m * math.sin(yaw0)
            cy = y0 + radius_m * math.cos(yaw0)
            sign = 1.0

        phi0 = math.atan2(y0 - cy, x0 - cx)
        total_rad = math.radians(angle_deg)

        n = 80
        for i in range(n + 1):
            dphi = sign * total_rad * i / n
            phi = phi0 + dphi
            x = cx + radius_m * math.cos(phi)
            y = cy + radius_m * math.sin(phi)
            parent_window.planned_x.append(x)
            parent_window.planned_y.append(y)

        parent_window.traj_planned_curve.setData(parent_window.planned_x, parent_window.planned_y)
