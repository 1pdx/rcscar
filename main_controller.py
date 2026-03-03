# main_controller.py
# -*- coding: utf-8 -*-
import importlib.util
import math
import threading
from pathlib import Path
import sys
from typing import List, Optional, Tuple

from PyQt5 import QtWidgets
from imu_gnss_pose import get_robot_pose, calibrate_pose_to_current, PoseSolution
from car_control import ScoutMiniCAN
from pi_power import PiPowerMonitor
from can_init import init_can, CanInitResult


_RADAR_MODULE = None


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

        # Radar object tracker (ARS408 0x60A/0x60B)
        self.radar: Optional[object] = None
        self._radar_selected_id: Optional[int] = None
        radar_mod = None
        if self._is_linux:
            try:
                radar_mod = _load_radar_processing_module()
            except Exception as e:
                print(f"[MainController] Radar module load failed: {e}")
            if radar_mod is not None:
                if self.can_init_result is not None and self.can_init_result.ok:
                    try:
                        radar_mod.init_radar_object_output(
                            iface="can0",
                            bitrate=500000,
                            sensor_id=0,
                            send_count=20,
                            verify_timeout_s=1.2,
                            retries=2,
                        )
                    except Exception as e:
                        print(f"[MainController] Radar object output init failed: {e}")
                try:
                    self.radar = radar_mod.RadarObjectTracker(
                        iface="can0",
                        bitrate=500000,
                        roi_front_abs=80.0,
                        roi_lat_abs=20.0,
                    )
                except Exception as e:
                    print(f"[MainController] Radar tracker init failed: {e}")
                    self.radar = None
        else:
            print("[MainController] Non-Linux system, skip radar tracker init.")

        # Battery percent mapping for the chassis (adjust if needed).
        self._car_batt_v_min = 23.0
        self._car_batt_v_max = 29.25

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
                daemon=True,
            )
            t.start()
        else:
            QtWidgets.QMessageBox.warning(parent_window, "定位无效", "无法获取当前位姿，请检查 IMU/GNSS 连接。")

    def execute_circle_movement(self, parent_window, radius: float, angle: float, speed: float) -> None:
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

        # 基于当前位置生成规划圆周轨迹
        pose = get_robot_pose()
        if pose is not None:
            clockwise = angle > 0  # 正角度为顺时针
            self._update_planned_circle(parent_window, pose, radius, abs(angle), clockwise)
            # 启动圆周运动线程
            t = threading.Thread(
                target=self.car.move_circle,
                args=(radius, angle, speed, clockwise),
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

    def emergency_stop(self, parent_window) -> None:
        """紧急停止"""
        if self.car is not None:
            self.car.emergency_stop()
            QtWidgets.QMessageBox.information(parent_window, "已停止", "小车已紧急停止。")

    def select_radar_target(self, parent_window) -> None:
        """Select a stable radar target for RCS recording."""
        if self.radar is None:
            QtWidgets.QMessageBox.information(
                parent_window,
                "雷达未启用",
                "当前未接入雷达，无法选择或锁定目标。",
            )
            return

        target = self.radar.get_best_stable_target()
        if target is None:
            QtWidgets.QMessageBox.information(
                parent_window,
                "无稳定目标",
                "当前未检测到稳定目标，请稍后再试。",
            )
            return

        self._radar_selected_id = int(target.oid)
        parent_window.tracked_target_id = int(target.oid)
        QtWidgets.QMessageBox.information(
            parent_window,
            "目标已选择",
            f"已选择目标 ID={target.oid} (x={target.x:.1f}m, y={target.y:.1f}m, rcs={target.rcs_db:.1f}dBsm)",
        )

    def set_selected_radar_id(self, oid: int) -> None:
        self._radar_selected_id = int(oid)

    def load_path_file(self, parent_window) -> None:
        """加载路径文件"""
        pose = get_robot_pose()
        if pose is None:
            QtWidgets.QMessageBox.warning(parent_window, "定位无效", "当前未获取有效 GNSS / INS 位姿，无法将轨迹对齐到车头。")
            return

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent_window,
            "选择轨迹文件（车辆坐标系，单位 m）",
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

        cos_yaw = math.cos(pose.yaw)
        sin_yaw = math.sin(pose.yaw)
        global_points = []
        for xr, yr in raw_points:
            gx = pose.x + xr * cos_yaw - yr * sin_yaw
            gy = pose.y + xr * sin_yaw + yr * cos_yaw
            global_points.append((gx, gy))

        parent_window.loaded_path_points = global_points
        parent_window.planned_x = [p[0] for p in global_points]
        parent_window.planned_y = [p[1] for p in global_points]
        parent_window.traj_planned_curve.setData(parent_window.planned_x, parent_window.planned_y)

        if parent_window.planned_x and parent_window.planned_y:
            min_x, max_x = min(parent_window.planned_x), max(parent_window.planned_x)
            min_y, max_y = min(parent_window.planned_y), max(parent_window.planned_y)
            margin = 0.5
            parent_window.traj_plot.setXRange(min_x - margin, max_x + margin, padding=0)
            parent_window.traj_plot.setYRange(min_y - margin, max_y + margin, padding=0)
        parent_window.btn_run_path.setEnabled(True)

        QtWidgets.QMessageBox.information(
            parent_window,
            "加载成功",
            f"已读取 {len(global_points)} 个轨迹点。\n"
            "说明：文件中的 x、y 单位为米，默认为车辆坐标系（x 向前、y 向左），已按当前车头姿态转换到 ENU 全局坐标。",
        )

    def execute_loaded_path(self, parent_window, speed: float) -> None:
        """执行已加载的路径"""
        if not self.ensure_car_ready(parent_window):
            return
        if not hasattr(parent_window, 'loaded_path_points') or len(parent_window.loaded_path_points) < 2:
            QtWidgets.QMessageBox.warning(parent_window, "轨迹未加载", "请先加载包含至少 2 个点的轨迹文件。")
            return
        if speed <= 0:
            QtWidgets.QMessageBox.warning(parent_window, "参数错误", "线速度必须为正值。")
            return

        t = threading.Thread(
            target=self.car.follow_path_with_pid,
            args=(parent_window.loaded_path_points, speed),
            daemon=True,
        )
        print(f"[MainController] 使用PID控制执行路径跟踪，速度: {speed} m/s")
        t.start()

    def get_power_status(self) -> Tuple[str, str]:
        """获取电源状态"""
        pi_text = "树莓派电量: 不可用"
        if self.pi_power is not None:
            try:
                # 检查是否有可用的电源监控
                if hasattr(self.pi_power, 'available') and self.pi_power.available:
                    p_status = self.pi_power.read_status()
                    if p_status is not None:
                        pi_text = (
                            f"树莓派电量: {p_status.voltage:.2f} V "
                            f"({p_status.percent:.0f}%, {p_status.power:.2f} W)"
                        )
                    else:
                        pi_text = "树莓派电量: 读取失败"
            except Exception as e:
                print(f"[MainController] 读取树莓派电量失败: {e}")
                pi_text = "树莓派电量: 读取异常"

        car_text = "小车电量: 不可用"
        if self.car is not None:
            try:
                st = self.car.get_status()
                if st.battery_voltage > 1e-3:
                    percent = None
                    if self._car_batt_v_max > self._car_batt_v_min:
                        percent = (st.battery_voltage - self._car_batt_v_min) / (
                            self._car_batt_v_max - self._car_batt_v_min
                        ) * 100.0
                        percent = max(0.0, min(100.0, percent))
                    if percent is None:
                        car_text = f"小车电量: {st.battery_voltage:.1f} V"
                    else:
                        car_text = f"小车电量: {st.battery_voltage:.1f} V ({percent:.0f}%)"
                else:
                    car_text = "小车电量: 无反应"
            except Exception as e:
                print(f"[MainController] 读取小车电量失败: {e}")
                car_text = "小车电量: 读取异常"

        return pi_text, car_text

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

    # 雷达功能已移除，保留兼容接口
    def get_radar_status(self) -> str:
        if self.radar is None:
            return "锁定目标: 雷达未启用"

        targets = self.radar.get_targets_snapshot()
        if not targets:
            return "锁定目标: --"

        if self._radar_selected_id is not None:
            t = next((m for m in targets if m.oid == self._radar_selected_id), None)
            if t is not None:
                return f"锁定目标: ID={t.oid} x={t.x:.1f}m y={t.y:.1f}m rcs={t.rcs_db:.1f}dBsm"

        best = self.radar.get_best_stable_target()
        if best is None:
            return f"锁定目标: {len(targets)}个"
        return f"锁定目标: ID={best.oid} x={best.x:.1f}m y={best.y:.1f}m rcs={best.rcs_db:.1f}dBsm"

    def get_stable_radar_targets(self, min_confidence: float = 0.7) -> List:
        if self.radar is None:
            return []
        # min_confidence maps to stability score threshold.
        return self.radar.get_targets_snapshot(min_score=min_confidence)

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
        n = 50
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
