import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable, List, Tuple

import can

from imu_gnss_pose import (
    get_robot_pose,
    PoseSolution,
    set_cluster_csv_straight_path_remaining_m,
)

#
# UI 侧会导入该常量用于复用直线前进段的跟踪参数。
# 若需要调整直线前进段参数，请优先改这里并同步相关调用处。
#
# Stanley 横向 PD（在 follow_path_with_pid 中与 stanley_term 叠加）：当前默认不启用（运行时 kp/kd=0）。
# 以下为历史默认，暂存备恢复：把 FORWARD 里对应项改回这些值即可。
STANLEY_LATERAL_PD_KP_STORED: float = 0.04
STANLEY_LATERAL_PD_KD_STORED: float = 0.0

FORWARD_STRAIGHT_TRACKING_KWARGS: Dict[str, Any] = {
    "lookahead_distance": 4.0,
    "stanley_gain": 0.4,
    "stanley_softening_speed_mps": 1.0,
    "straight_switch_pause_s": 0.12,
    "stanley_lateral_pd_kp": 0.0,
    "stanley_lateral_pd_ki": 0.02,
    "stanley_lateral_pd_kd": 0.0,
    "speed_pid_kp": 0.95,
    "speed_pid_ki": 0.20,
    "speed_pid_kd": 0.03,
    "speed_pid_output_limit_mps": 0.45,
    "lateral_pid_kp": 2.08,
    "lateral_pid_ki": 0.50,
    "lateral_pid_kd": 0.42,
    "heading_pid_kp": 1.58,
    "heading_pid_ki": 0.25,
    "heading_pid_kd": 0.44,
    "yaw_rate_pid_kp": 0.42,
    "yaw_rate_pid_ki": 0.07,
    "yaw_rate_pid_kd": 0.0,
    "max_w_rate": 2.05,
    "max_w_step": 0.031,
    "smoothing_strength": 0.55,
    "smoothing_strength_curve": 0.88,
    "w_bias_tau": 0.60,
    "w_bias_hf_gain": 0.35,
}


@dataclass
class ScoutStatus:
    """
    SCOUT MINI 底盘状态反馈（由 0x211 & 0x221 帧解析而来）
    """
    sys_status: int = 0          # 0 正常, 1 急停, 2 故障等 (详见手册)
    control_mode: int = 0        # 0 遥控, 1 CAN, 2 串口等
    battery_voltage: float = 0.0 # 单位: V
    linear_speed: float = 0.0    # 反馈线速度, m/s
    angular_speed: float = 0.0   # 反馈角速度, rad/s
    last_update: float = 0.0     # 最近一次收到反馈的时间戳


def _sat(value: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, value))


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle



class PIDController:
    """PID控制器"""
    
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limits: Tuple[float, float] = (-2.5, 2.5),
        i_output_limit: float = 2.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.i_output_limit = abs(float(i_output_limit))
        
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()
        
    def reset(self):
        """重置控制器"""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()
        
    def update(self, error: float, dt: Optional[float] = None) -> float:
        """更新PID控制器"""
        current_time = time.time()
        if dt is None:
            dt = current_time - self._prev_time
            if dt <= 0:
                dt = 0.01  # 默认10ms
        
        # 积分项（带抗饱和）
        self._integral += error * dt
        # 积分限幅
        max_integral = self.i_output_limit / max(1e-6, self.ki) if self.ki > 0 else 1000.0
        self._integral = max(-max_integral, min(max_integral, self._integral))
        
        # 微分项
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        
        # PID输出
        output = (self.kp * error + 
                 self.ki * self._integral + 
                 self.kd * derivative)
        
        # 输出限幅
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        
        # 更新状态
        self._prev_error = error
        self._prev_time = current_time
        
        return output


class ScoutMiniCAN:
    """
    松灵 SCOUT MINI 底盘 CAN 控制封装 (基于《用户手册 V2.0.4》)

    CAN ID 定义：
      - 0x111: 运动控制命令帧 (发送)
      - 0x211: 系统状态反馈帧 (接收)
      - 0x221: 运动状态反馈帧 (接收)
      - 0x421: 控制模式设定帧 (发送，切换到 CAN 控制)
    """

    ID_MOTION_CMD = 0x111
    ID_SYS_STATUS = 0x211
    ID_MOTION_FB = 0x221
    ID_CONTROL_MODE = 0x421

    # 这里用手册给出的最大线/角速度做饱和值
    MAX_LINEAR_MPS = 3.0          # m/s
    MAX_ANGULAR_RADPS = 2.5       # rad/s

    @staticmethod
    def _adaptive_circle_stanley_gain(radius_m: float) -> float:
        """Radius-adaptive Stanley gain for circle tracking."""
        try:
            radius = abs(float(radius_m))
        except (TypeError, ValueError):
            radius = 40.0
        raw_gain = 0.20 + 0.005 * radius
        return _sat(raw_gain, 0.25, 0.40)

    def _is_nearly_straight_segment(self, waypoints: List[Tuple[float, float]]) -> bool:
        if len(waypoints) < 3:
            return True
        x0, y0 = waypoints[0]
        x1, y1 = waypoints[-1]
        chord = math.hypot(x1 - x0, y1 - y0)
        if chord <= 1e-6:
            return True
        max_cross = 0.0
        dx = x1 - x0
        dy = y1 - y0
        for x, y in waypoints[1:-1]:
            cross = abs(dx * (y - y0) - dy * (x - x0))
            max_cross = max(max_cross, cross)
        max_dev = max_cross / chord
        return max_dev <= 0.06

    def __init__(
        self,
        channel: str = "can0",
        interface: str = "socketcan",
        bitrate: int = 500000,
        auto_enable_can_control: bool = True,
    ) -> None:
        """
        :param channel:   CAN 通道, Linux 默认 "can0"
        :param interface: python-can 接口类型, Linux 使用 "socketcan"
        :param bitrate:   波特率, SCOUT MINI 默认 500 kbit/s
        """
        self.channel = channel
        self.interface = interface
        self.bitrate = bitrate

        self.bus: Optional[can.BusABC] = None
        self._status = ScoutStatus()
        self._status_lock = threading.Lock()

        self._stop_flag = threading.Event()
        self._running = True

        # 发送计数（如需用作序号，这里预留）
        self._tx_count = 0
        # 控制周期 & 字节序（手册标注 Motorola -> big-endian）
        self._control_period_s = 0.02
        self._byteorder = "big"
        # Optional CAN TX debug (default enabled per request; can disable via SCOUT_CAN_DEBUG=0)
        debug_env = os.getenv("SCOUT_CAN_DEBUG", "").strip().lower()
        if debug_env in ("0", "false", "no", "off"):
            self._debug_tx = False
        elif debug_env in ("1", "true", "yes", "on"):
            self._debug_tx = True
        else:
            self._debug_tx = True
        self._debug_tx_interval_s = float(os.getenv("SCOUT_CAN_DEBUG_INTERVAL", "1.0") or "1.0")
        self._last_debug_tx_ts = 0.0
        status_env = os.getenv("SCOUT_CAN_STATUS_LOG", "").strip().lower()
        if status_env in ("0", "false", "no", "off"):
            self._debug_status = False
        elif status_env in ("1", "true", "yes", "on"):
            self._debug_status = True
        else:
            self._debug_status = True
        self._status_logged = False

        motion_env = os.getenv("SCOUT_CAN_MOTION_LOG", "").strip().lower()
        if motion_env in ("0", "false", "no", "off"):
            self._debug_motion = False
        elif motion_env in ("1", "true", "yes", "on"):
            self._debug_motion = True
        else:
            self._debug_motion = True
        self._motion_logged = False
        self._rx_drain_max = int(os.getenv("SCOUT_CAN_RX_DRAIN_MAX", "32") or "32")

        # Stanley 输出的角速度 w 闭环 PID：用底盘反馈角速度跟踪 w_stanley
        # 参数按需求：P=1, I=0, D=0.5
        self.stanley_w_pid = PIDController(
            kp=0.5,
            ki=0.0,
            kd=-0.5,
            output_limits=(-self.MAX_ANGULAR_RADPS, self.MAX_ANGULAR_RADPS),
            i_output_limit=self.MAX_ANGULAR_RADPS,
        )

        # 虚拟模式支持
        self._virtual_mode = (channel == "virtual" or interface == "virtual")
        
        if not self._virtual_mode:
            self._open_bus()
        else:
            print(f"[ScoutMiniCAN] 虚拟模式: {channel}, {interface}")
            self.bus = None

        if auto_enable_can_control:
            ok = self.enable_can_control()
            if not ok:
                print("[ScoutMiniCAN] 自动使能 CAN 控制失败，可能需要检查拨档/固件。")

        # 后台接收线程（仅在非虚拟模式下启动）
        if not self._virtual_mode:
            self._recv_thread = threading.Thread(
                target=self._recv_loop, daemon=True
            )
            self._recv_thread.start()
        else:
            self._recv_thread = None

    # ================= 高层接口 =================

    def enable_can_control(self) -> bool:
        """
        发送 0x421 控制模式设定帧，将底盘切换到 CAN 控制模式。

        帧格式 (参考用户手册 V2.0.4)：
          ID: 0x421
          Data[0]: 控制模式 (0: 遥控, 1: CAN, 2: UART ... )
          Data[1..7]: 预留 (0)
        """
        data = bytes([0x01])  # 切换到 CAN 控制模式
        data_hex = data.hex()
        pretty = " ".join(data_hex[i:i + 2].upper() for i in range(0, len(data_hex), 2))

        if self._virtual_mode:
            print("[ScoutMiniCAN] 虚拟模式: 已切换到CAN控制模式")
            print(f"[ScoutMiniCAN] TX 0x{self.ID_CONTROL_MODE:03X} bytes={pretty}")
            return True
            
        if not self._ensure_bus():
            print("[ScoutMiniCAN] enable_can_control: CAN bus 无法打开.")
            return False

        msg = can.Message(
            arbitration_id=self.ID_CONTROL_MODE,
            is_extended_id=False,
            data=data,
        )
        try:
            self.bus.send(msg)
            print("[ScoutMiniCAN] 已发送控制模式帧 0x421: 切换到底盘 CAN 控制模式.")
            print(f"[ScoutMiniCAN] TX 0x{self.ID_CONTROL_MODE:03X} bytes={pretty}")
            return True
        except can.CanError as e:
            print(f"[ScoutMiniCAN] enable_can_control 发送失败: {e}")
            return False

    def get_status(self) -> ScoutStatus:
        with self._status_lock:
            return ScoutStatus(**self._status.__dict__)

    def set_can_endianness(self, byteorder: str) -> None:
        byteorder = byteorder.strip().lower()
        if byteorder not in ("big", "little"):
            raise ValueError("byteorder must be 'big' or 'little'")
        self._byteorder = byteorder
        print(f"[ScoutMiniCAN] CAN byteorder set to {self._byteorder}")

    def get_can_endianness(self) -> str:
        return self._byteorder

    def set_debug_tx(self, enabled: bool, interval_s: float = 1.0) -> None:
        self._debug_tx = bool(enabled)
        self._debug_tx_interval_s = max(0.0, float(interval_s))
        self._last_debug_tx_ts = 0.0

    def send_test_motion_once(
        self,
        linear_mps: float = 0.15,
        angular_radps: float = 0.0,
        byteorder: Optional[str] = None,
    ) -> None:
        """Send one 0x111 frame for endianness probing (non-holonomic: vy=0)."""
        prev = self._byteorder
        if byteorder is not None:
            byteorder = byteorder.strip().lower()
            if byteorder not in ("big", "little"):
                raise ValueError("byteorder must be 'big' or 'little'")
            self._byteorder = byteorder
        try:
            self._send_motion_command(linear_mps, angular_radps)
        finally:
            if byteorder is not None:
                self._byteorder = prev

    def stop(self) -> None:
        """
        紧急停止：立即下发 0 速度，并置位 stop_flag 终止当前运动控制循环。
        """
        self._stop_flag.set()
        self._send_motion_command(0.0, 0.0)

    def rotate_to_heading(
        self,
        target_yaw_rad: float,
        *,
        update_pose: Callable[[], Optional[PoseSolution]] = get_robot_pose,
        dt: float = 0.02,
        timeout_s: float = 6.0,
        yaw_tolerance_rad: float = math.radians(3.0),
        max_w_radps: float = 1.2,
        min_w_radps: float = 0.18,
    ) -> bool:
        """
        原地转向对齐到目标航向（弧度）。

        说明：
        - 该方法用于 UI 侧“切段前对齐航向”，避免调用不存在接口导致线程崩溃。
        - 仅下发 (v=0,w!=0)；若无位姿则直接返回 False。
        """
        try:
            target_yaw = float(target_yaw_rad)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(target_yaw):
            return False

        pose0 = update_pose()
        if pose0 is None:
            return False

        started = time.time()
        self._stop_flag.clear()
        ok = False
        stable_count = 0
        stable_needed = max(1, int(round(0.25 / max(1e-3, float(dt)))))  # ~0.25s
        dt_eff = float(dt) if dt and float(dt) > 0 else 0.02

        while not self._stop_flag.is_set():
            now = time.time()
            if timeout_s is not None and float(timeout_s) > 0 and (now - started) >= float(timeout_s):
                break

            pose = update_pose()
            if pose is None:
                break

            err = _wrap_angle(float(target_yaw) - float(pose.yaw))
            if abs(err) <= float(yaw_tolerance_rad):
                stable_count += 1
                self._send_motion_command(0.0, 0.0)
                if stable_count >= stable_needed:
                    ok = True
                    break
                time.sleep(dt_eff)
                continue

            stable_count = 0
            # 简单 P 控制：角速度与误差成正比，同时限幅与最小输出（克服静摩擦）
            w = 1.6 * float(err)
            w = _sat(w, -abs(float(max_w_radps)), abs(float(max_w_radps)))
            if abs(w) < abs(float(min_w_radps)):
                w = math.copysign(abs(float(min_w_radps)), w)
            self._send_motion_command(0.0, float(w))
            time.sleep(dt_eff)

        self._send_motion_command(0.0, 0.0)
        return bool(ok)

    def close(self) -> None:
        """
        结束接收线程并释放 CAN 资源。
        """
        if not self._running:
            return
        self._running = False
        self._stop_flag.set()
        
        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=1.0)
            
        if self.bus is not None and not self._virtual_mode:
            try:
                self.bus.shutdown()
            except AttributeError:
                # 某些 python-can 版本没有 shutdown 接口
                pass
            except Exception as e:
                print(f"[ScoutMiniCAN] 关闭CAN总线时出错: {e}")
            self.bus = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ---------- 轨迹控制：PID 直线 / 圆周 ----------

    def move_straight(
        self,
        distance_m: float,
        speed_mps: float,
        update_pose: Callable[[], Optional[PoseSolution]] = get_robot_pose,
        dt: float = 0.02,
    ) -> None:
        """
        直线运动控制 (PID 路径跟踪 + 基于 IMU/GNSS 的位置修正).

        轨迹定义:
          - 起点: 调用时当前位姿 (x0, y0, yaw0)
          - 理想路径: 以 yaw0 方向为切线的直线
          - 终止条件: 到达路径末端

        :param distance_m: 期望行驶距离 (m), >0 前进, <0 倒车
        :param speed_mps:  标称线速度 (m/s)
        :param update_pose: 位姿获取函数, 默认使用 imu_gnss_pose.get_robot_pose
        :param dt:         控制周期 (s)
        """
        self._stop_flag.clear()
        speed_abs = abs(speed_mps)
        if abs(distance_m) < 1e-3 or speed_abs <= 0.0:
            return

        pose0 = update_pose()
        if pose0 is None:
            print("[ScoutMiniCAN] move_straight: 无位姿信息, 退回简单时间控制.")
            # 回退：按时间估算
            direction = 1.0 if distance_m >= 0 else -1.0
            t_total = abs(distance_m) / max(0.05, speed_abs)
            t_start = time.time()
            while not self._stop_flag.is_set():
                if time.time() - t_start >= t_total:
                    break
                self._send_motion_command(direction * speed_abs, 0.0)
                time.sleep(dt)
            self._send_motion_command(0.0, 0.0)
            return

        x0, y0, yaw0 = pose0.x, pose0.y, pose0.yaw
        path_len = abs(distance_m)
        step = max(0.1, min(0.5, path_len / 50.0))
        n = max(2, min(200, int(path_len / step)))
        waypoints: List[Tuple[float, float]] = []
        for i in range(n + 1):
            s = distance_m * i / n
            waypoints.append((x0 + s * math.cos(yaw0), y0 + s * math.sin(yaw0)))

        speed_cmd = speed_abs if distance_m >= 0 else -speed_abs

        print(
            f"[ScoutMiniCAN] move_straight(PID): start=({x0:.2f},{y0:.2f}), "
            f"yaw={yaw0:.3f} rad, dist={distance_m:.2f} m"
        )

        # 直线段：略增大到达阈值，减少“判到达瞬间仍在动”导致的过冲；
        # 同时启用 slow_down_dist，在终点前按距离线性压低期望速度。
        straight_arrival_dist_m = 0.20
        slow_down_dist = min(
            path_len * 0.55,
            max(0.75, 0.85 * speed_abs + 0.45),
        )
        slow_down_dist = max(slow_down_dist, 2.0 * straight_arrival_dist_m)
        slow_down_dist = min(slow_down_dist, path_len * 0.65)

        self.follow_path_with_pid(
            waypoints=waypoints,
            speed_mps=speed_cmd,
            dt=dt,
            update_pose=update_pose,
            arrival_dist=straight_arrival_dist_m,
            slow_down_dist=slow_down_dist,
            stanley_softening_speed_mps=1.0,
        )

    def move_circle(
        self,
        radius_m: float,
        angle_deg: float = 360.0,
        speed_mps: float = 0.5,
        clockwise: bool = False,
        dt: float = 0.02,
        update_pose: Callable[[], Optional[PoseSolution]] = get_robot_pose,
        metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        sample_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        run_label: Optional[str] = None,
        record_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        圆周运动 (PID 路径跟踪)，按轨迹角度和半径给定。

        轨迹定义:
          - 起点: 调用时当前位姿 (x0, y0, yaw0)
          - 理想圆: 半径 R = radius_m, 圆心根据 yaw0 和顺/逆时针方向确定
          - 终止条件: 到达圆弧末端

        :param radius_m: 圆弧半径 (m), >0
        :param angle_deg: 需要转过的角度 (deg), 正值
        :param speed_mps: 切向线速度 (m/s)
        :param clockwise: True 顺时针, False 逆时针
        :param dt:        控制周期 (s)
        :param update_pose: 位姿获取函数
        """
        self._stop_flag.clear()
        radius_m = max(0.1, abs(radius_m))
        angle_deg = abs(angle_deg)
        speed_abs = abs(speed_mps)
        if speed_abs <= 0.0 or angle_deg <= 0.0:
            return

        pose0 = update_pose()
        if pose0 is None:
            print("[ScoutMiniCAN] move_circle: 无位姿信息, 退回简单几何控制.")
            total_angle_rad = math.radians(angle_deg)
            w = speed_abs / radius_m
            w_cmd = -w if clockwise else w
            v_cmd = speed_abs
            t_total = total_angle_rad / max(1e-3, abs(w))
            t_start = time.time()
            while not self._stop_flag.is_set():
                if time.time() - t_start >= t_total:
                    break
                self._send_motion_command(v_cmd, w_cmd)
                time.sleep(dt)
            self._send_motion_command(0.0, 0.0)
            return

        x0, y0, yaw0 = pose0.x, pose0.y, pose0.yaw

        # 圆心位置
        if clockwise:
            cx = x0 + radius_m * math.sin(yaw0)
            cy = y0 - radius_m * math.cos(yaw0)
            sign = -1.0
        else:
            cx = x0 - radius_m * math.sin(yaw0)
            cy = y0 + radius_m * math.cos(yaw0)
            sign = 1.0

        phi0 = math.atan2(y0 - cy, x0 - cx)
        total_angle_rad = math.radians(angle_deg)
        arc_len = radius_m * total_angle_rad
        # 圆弧离散更密：沿弧长约每 0.06m 一个点（上限防止点数过多）
        target_spacing_m = 0.06
        n = max(20, min(1000, int(math.ceil(arc_len / max(1e-6, target_spacing_m)))))
        waypoints: List[Tuple[float, float]] = []
        for i in range(n + 1):
            dphi = sign * total_angle_rad * i / n
            phi = phi0 + dphi
            waypoints.append((cx + radius_m * math.cos(phi), cy + radius_m * math.sin(phi)))

        print(
            f"[ScoutMiniCAN] move_circle(PID): center=({cx:.2f},{cy:.2f}), "
            f"R={radius_m:.2f}m, angle={angle_deg:.1f}deg, cw={clockwise}"
        )
        stanley_gain_use = self._adaptive_circle_stanley_gain(radius_m)

        self.follow_path_with_pid(
            waypoints=waypoints,
            speed_mps=speed_abs,
            dt=dt,
            update_pose=update_pose,
            lookahead_distance=0.4,
            stanley_gain=stanley_gain_use,
            smoothing_strength=0.8,
            smoothing_strength_curve=0.88,
            w_bias_tau=0.9,
            w_bias_hf_gain=0.25,
            max_w_rate=3.0,
            metrics_callback=metrics_callback,
            sample_callback=sample_callback,
            run_label=run_label,
            record_context=record_context,
        )

    def move_circle_orbit(
        self,
        radius_m: float,
        angle_deg: float = 360.0,
        speed_mps: float = 0.5,
        clockwise: bool = False,
        dt: float = 0.02,
        update_pose: Callable[[], Optional[PoseSolution]] = get_robot_pose,
        *,
        k_heading: float = 0.1,
        k_radius: float = 0.25,
        k_radius_d: float = 0.5,
        enable_speed_gain_scheduling: bool = True,
        speed_gain_reference_mps: float = 1.6,
        k_heading_high_speed_scale: float = 0.65,
        k_radius_high_speed_scale: float = 1.15,
        k_radius_d_high_speed_scale: float = 1.45,
        w_output_tau_high_speed_scale: float = 0.70,
        max_w_step_high_speed_scale: float = 1.60,
        enable_speed_scheduling: bool = True,
        min_speed_scale: float = 0.35,
        v_output_tau: float = 1.2,
        w_output_tau: float = 0.15,
        max_w_rate: float = 3.0,
        max_w_step: Optional[float] = 0.04,
        w_bias_tau: float = 0.9,
        w_bias_hf_gain: float = 0.22,
        metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        sample_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        run_label: Optional[str] = None,
        record_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        圆周/圆弧运动（极坐标轨道控制）。

        控制变量:
          - 半径误差 e_r = current_r - R
          - 航向误差 e_psi = wrap(psi_tangent - psi_motion)

        控制律（简化）:
          w_cmd = sign * (v/R + k_radius * e_r) + k_heading * e_psi
          其中 sign: CCW=+1, CW=-1
        """
        self._stop_flag.clear()
        started_wall_ts = time.time()
        run_key = f"{started_wall_ts:.6f}|{str(run_label or '圆周').strip() or '圆周'}"

        radius_m = max(0.1, abs(float(radius_m)))
        angle_deg = abs(float(angle_deg))
        speed_abs = abs(float(speed_mps))
        if speed_abs <= 0.0 or angle_deg <= 0.0:
            return

        pose0 = update_pose()
        if pose0 is None:
            # 与 move_circle 保持一致：无位姿时退回“定 v,w 跑够时间”
            total_angle_rad = math.radians(angle_deg)
            w0 = speed_abs / radius_m
            w_cmd = -w0 if clockwise else w0
            v_cmd = speed_abs
            t_total = total_angle_rad / max(1e-3, abs(w0))
            t_start = time.time()
            while not self._stop_flag.is_set():
                if time.time() - t_start >= t_total:
                    break
                self._send_motion_command(v_cmd, w_cmd)
                time.sleep(max(0.0, float(dt) if dt else 0.02))
            self._send_motion_command(0.0, 0.0)
            return

        x0, y0, yaw0 = float(pose0.x), float(pose0.y), float(pose0.yaw)
        if clockwise:
            cx = x0 + radius_m * math.sin(yaw0)
            cy = y0 - radius_m * math.cos(yaw0)
            sign = -1.0
        else:
            cx = x0 - radius_m * math.sin(yaw0)
            cy = y0 + radius_m * math.cos(yaw0)
            sign = 1.0

        total_angle_rad = math.radians(angle_deg)
        prev_phi: Optional[float] = None
        accum_angle = 0.0
        dt_eff = float(dt) if dt and float(dt) > 0 else 0.02
        k_radius_d = float(k_radius_d)
        enable_speed_gain_scheduling = bool(enable_speed_gain_scheduling)
        speed_gain_reference_mps = max(0.1, float(speed_gain_reference_mps))
        k_heading_high_speed_scale = max(0.0, float(k_heading_high_speed_scale))
        k_radius_high_speed_scale = max(0.0, float(k_radius_high_speed_scale))
        k_radius_d_high_speed_scale = max(0.0, float(k_radius_d_high_speed_scale))
        w_output_tau_high_speed_scale = max(0.15, float(w_output_tau_high_speed_scale))
        max_w_step_high_speed_scale = max(0.2, float(max_w_step_high_speed_scale))
        enable_speed_scheduling = bool(enable_speed_scheduling)
        min_speed_scale = max(0.05, min(1.0, float(min_speed_scale)))
        v_output_tau = max(0.0, float(v_output_tau))
        w_output_tau = max(0.0, float(w_output_tau))
        max_w_rate = max(0.1, float(max_w_rate))
        max_w_step_val: Optional[float] = None
        if max_w_step is not None:
            try:
                max_w_step_val = abs(float(max_w_step))
            except (TypeError, ValueError):
                max_w_step_val = None
        w_bias_tau = max(0.05, float(w_bias_tau))
        w_bias_hf_gain = max(0.0, min(1.0, float(w_bias_hf_gain)))

        prev_cmd_w: Optional[float] = None
        prev_output_w: Optional[float] = None
        prev_output_v: Optional[float] = 0.0
        w_bias: Optional[float] = None
        prev_radius_error: Optional[float] = None
        traveled = 0.0
        last_pose: Optional[PoseSolution] = None
        peak_abs_radius_error = 0.0
        peak_abs_heading_error_deg = 0.0
        peak_abs_cmd_w = 0.0
        sample_count = 0
        exit_reason = "loop_exit"
        completed = False

        print(
            f"[ScoutMiniCAN] move_circle_orbit: center=({cx:.2f},{cy:.2f}), "
            f"R={radius_m:.2f}m, angle={angle_deg:.1f}deg, cw={bool(clockwise)}"
        )

        set_cluster_csv_straight_path_remaining_m(None)
        while not self._stop_flag.is_set():
            pose = update_pose()
            if pose is None:
                exit_reason = "pose_lost"
                break
            x, y, yaw = float(pose.x), float(pose.y), float(pose.yaw)
            if last_pose is not None:
                traveled += math.hypot(x - float(last_pose.x), y - float(last_pose.y))
            last_pose = pose

            dx = x - cx
            dy = y - cy
            r = math.hypot(dx, dy)
            if r <= 1e-6:
                exit_reason = "radius_invalid"
                break
            phi = math.atan2(dy, dx)

            if prev_phi is None:
                prev_phi = phi
            else:
                dphi = _wrap_angle(phi - prev_phi)
                prev_phi = phi
                accum_angle += sign * float(dphi)

            if accum_angle >= total_angle_rad:
                exit_reason = "goal_arrived"
                completed = True
                break

            # 运动方向航向（倒车时使用 yaw+pi）
            motion_yaw = yaw
            if float(speed_mps) < 0.0:
                motion_yaw = _wrap_angle(yaw + math.pi)

            # 目标切向航向：CCW -> phi+pi/2, CW -> phi-pi/2
            target_heading = _wrap_angle(phi + sign * (math.pi / 2.0))
            heading_error = _wrap_angle(target_heading - motion_yaw)
            radius_error = float(r - radius_m)

            # 半径误差阻尼：用 e_r 的变化率抑制“越修越冲”的振荡
            if prev_radius_error is None:
                radius_error_rate = 0.0
            else:
                radius_error_rate = (radius_error - prev_radius_error) / max(1e-6, dt_eff)
            prev_radius_error = float(radius_error)

            # 误差大时自动降速：减少所需 w 与相位滞后引起的发散
            speed_scale = 1.0
            if enable_speed_scheduling:
                # 航向误差与半径误差共同影响降速（半径误差按 0.25R 归一化，避免大半径过于敏感）
                r_norm = max(0.15, 0.25 * radius_m)
                e = abs(float(heading_error)) + abs(float(radius_error)) / r_norm
                speed_scale = 1.0 / (1.0 + 1.6 * e)
                speed_scale = max(min_speed_scale, min(1.0, float(speed_scale)))

            desired_v = float(speed_abs) * float(speed_scale)

            # 线速度输出软启动/平滑过渡（避免起步瞬间“冲”）
            if prev_output_v is None or v_output_tau <= 1e-6:
                out_v = float(desired_v)
            else:
                alpha_v = dt_eff / (v_output_tau + dt_eff)
                out_v = (1.0 - alpha_v) * float(prev_output_v) + alpha_v * float(desired_v)
            prev_output_v = float(out_v)

            # 速度绑定参数（gain scheduling）
            # speed_ratio≈0: 低速；≈1: 达到参考速度；>1: 视为 1
            speed_ratio = min(1.0, abs(float(out_v)) / speed_gain_reference_mps)
            if enable_speed_gain_scheduling:
                # 高速时更容易出现“呼吸式”摆动：增强半径阻尼/适度增强半径P，
                # 同时降低航向增益避免把航向噪声放大成 w 抖动；
                # 输出更跟手：降低 w_output_tau，且放宽 max_w_step（但仍保留 max_w_rate 限制）。
                k_heading_eff = float(k_heading) * (
                    1.0 - (1.0 - k_heading_high_speed_scale) * speed_ratio
                )
                k_radius_eff = float(k_radius) * (
                    1.0 + (k_radius_high_speed_scale - 1.0) * speed_ratio
                )
                k_radius_d_eff = float(k_radius_d) * (
                    1.0 + (k_radius_d_high_speed_scale - 1.0) * speed_ratio
                )
                w_output_tau_eff = max(
                    0.0,
                    float(w_output_tau)
                    * (1.0 - (1.0 - w_output_tau_high_speed_scale) * speed_ratio),
                )
                max_w_step_eff: Optional[float] = (
                    float(max_w_step_val) * (1.0 + (max_w_step_high_speed_scale - 1.0) * speed_ratio)
                    if max_w_step_val is not None
                    else None
                )
            else:
                k_heading_eff = float(k_heading)
                k_radius_eff = float(k_radius)
                k_radius_d_eff = float(k_radius_d)
                w_output_tau_eff = float(w_output_tau)
                max_w_step_eff = float(max_w_step_val) if max_w_step_val is not None else None

            # 前馈角速度用“实际下发速度”计算，避免 v 软启动但 w 仍按标称 v/R 直接转入导致内切/误差放大
            w_ff = sign * (abs(float(out_v)) / radius_m)
            w_cmd = (
                float(w_ff)
                + float(k_radius_eff) * float(radius_error)
                + float(k_radius_d_eff) * float(radius_error_rate)
                + float(k_heading_eff) * float(heading_error)
            )
            w_cmd = _sat(w_cmd, -self.MAX_ANGULAR_RADPS, self.MAX_ANGULAR_RADPS)

            # 稳态偏置 + 高频抑制（与 follow_path_with_pid 同型）
            if w_bias is None:
                w_bias = w_cmd
            else:
                beta = dt_eff / (w_bias_tau + dt_eff)
                w_bias = (1.0 - beta) * w_bias + beta * w_cmd
            desired_w = w_bias + (w_cmd - w_bias) * w_bias_hf_gain

            # 限角加速度/步进
            if prev_cmd_w is None:
                cmd_w = desired_w
            else:
                dw_max = max_w_rate * dt_eff
                if max_w_step_eff is not None:
                    dw_max = min(dw_max, float(max_w_step_eff))
                dw = desired_w - prev_cmd_w
                if abs(dw) > dw_max:
                    desired_w = prev_cmd_w + math.copysign(dw_max, dw)
                cmd_w = desired_w
            prev_cmd_w = cmd_w

            # 角速度输出再做一次平滑过渡（降低跳变/抖动）
            if prev_output_w is None or w_output_tau_eff <= 1e-6:
                out_w = float(cmd_w)
            else:
                alpha = dt_eff / (w_output_tau_eff + dt_eff)
                out_w = (1.0 - alpha) * float(prev_output_w) + alpha * float(cmd_w)
            prev_output_w = float(out_w)

            self._send_motion_command(out_v, out_w)

            if callable(sample_callback):
                try:
                    now_ts = time.time()
                    fb_v = 0.0
                    fb_w = 0.0
                    try:
                        st = self.get_status()
                        fb_v = float(getattr(st, "linear_speed", 0.0) or 0.0)
                        fb_w = float(getattr(st, "angular_speed", 0.0) or 0.0)
                    except Exception:
                        fb_v = 0.0
                        fb_w = 0.0
                    payload: Dict[str, Any] = {
                        "timestamp": float(now_ts),
                        "relative_time_s": float(max(0.0, now_ts - started_wall_ts)),
                        "run_key": str(run_key),
                        "run_label": str(run_label or ""),
                        "tracking_mode": "circle_orbit",
                        "speed_sign": 1.0,
                        "speed_mps": float(speed_abs),
                        "nominal_speed_abs_mps": float(speed_abs),
                        "cmd_v_mps": float(out_v),
                        "cmd_w_radps": float(out_w),
                        "desired_v_mps": float(desired_v),
                        "desired_w_radps": float(desired_w),
                        # 对齐 UI 字段：圆周用 lateral_error_m 表示半径误差
                        "lateral_error_m": float(radius_error),
                        "heading_error_rad": float(heading_error),
                        "yaw_rate_error_radps": 0.0,
                        "feedback_v_mps": float(fb_v),
                        "feedback_w_radps": float(fb_w),
                        "pose_age_s": 0.0,
                        "current_x_m": float(x),
                        "current_y_m": float(y),
                        "dist_to_goal_m": float(max(0.0, total_angle_rad - accum_angle) * radius_m),
                        "path_s_m": float(accum_angle * radius_m),
                        "path_curvature_inv_m": float(abs(out_w) / max(abs(out_v), 0.1)),
                        "path_ff_w_radps": float(w_ff),
                        "profile_speed_mps": float(speed_abs),
                        "motion_distance_total_m": float(traveled),
                        "motion_distance_total_signed_m": float(traveled),
                    }
                    if isinstance(record_context, dict) and record_context:
                        payload.update({str(k): v for k, v in record_context.items()})
                    sample_callback(payload)
                except Exception:
                    pass

            sample_count += 1
            peak_abs_radius_error = max(peak_abs_radius_error, abs(float(radius_error)))
            peak_abs_heading_error_deg = max(
                peak_abs_heading_error_deg, abs(math.degrees(float(heading_error)))
            )
            peak_abs_cmd_w = max(peak_abs_cmd_w, abs(float(out_w)))

            time.sleep(dt_eff)

        self._send_motion_command(0.0, 0.0)
        set_cluster_csv_straight_path_remaining_m(None)
        if self._stop_flag.is_set() and exit_reason == "loop_exit":
            exit_reason = "stop_flag"
        duration_s = max(0.0, time.time() - started_wall_ts)
        if callable(metrics_callback):
            try:
                record: Dict[str, Any] = {
                    "timestamp": float(started_wall_ts),
                    "run_key": str(run_key),
                    "run_label": str(run_label or ""),
                    "tracking_mode": "circle_orbit",
                    "speed_mps": float(speed_abs),
                    "speed_sign": 1.0,
                    "nominal_speed_abs_mps": float(speed_abs),
                    "duration_s": float(duration_s),
                    "samples": int(sample_count),
                    "peak_abs_lateral_error_m": float(peak_abs_radius_error),
                    "peak_abs_heading_error_deg": float(peak_abs_heading_error_deg),
                    "peak_abs_cmd_w_radps": float(peak_abs_cmd_w),
                    "completed": bool(completed),
                    "exit_reason": str(exit_reason),
                    "motion_distance_total_m": float(traveled),
                    "motion_distance_total_signed_m": float(traveled),
                }
                if isinstance(record_context, dict) and record_context:
                    record.update({str(k): v for k, v in record_context.items()})
                metrics_callback(record)
            except Exception:
                pass

    def follow_path(
        self,
        waypoints: List[Tuple[float, float]],
        speed_mps: float = 0.5,
        dt: float = 0.02,
        update_pose: Callable[[], Optional[PoseSolution]] = get_robot_pose,
        k_lat: float = 1.0,
        k_steer: float = 1.8,
        slow_down_dist: float = 0.8,
    ) -> None:
        """
        依据给定路径点 (ENU 米坐标) 进行 PID 跟踪。
        :param waypoints: [(x, y), ...] 按行驶顺序给出的全局 ENU 米坐标，至少 2 个
        :param speed_mps: 标称线速度 (m/s)
        :param dt: 控制周期 (s)
        :param k_lat: 兼容旧接口保留参数（不再使用）
        :param k_steer: 兼容旧接口保留参数（不再使用）
        """
        self.follow_path_with_pid(
            waypoints=waypoints,
            speed_mps=speed_mps,
            dt=dt,
            update_pose=update_pose,
            slow_down_dist=slow_down_dist,
        )

    # ================ PID 路径跟踪 ================

    def get_directional_straight_tracking_kwargs(self, speed_sign: float) -> Dict[str, Any]:
        """
        与 UI 兼容：返回直线段前进/倒车的专用参数字典。

        注意：当前 `follow_path_with_pid` 是 PID 直线/圆弧跟踪实现，部分 Stanley/PID 扩展参数会被忽略。
        """
        if float(speed_sign) < 0.0:
            # 倒车段：保守一点的角速度变化、稍大前瞻，小积分项消除稳态偏移
            return {
                "lookahead_distance": 4.0,
                "tracking_mode": "stanley",
                "stanley_gain": 0.4,
                "stanley_softening_speed_mps": 1.0,
                "stanley_lateral_pd_kp": 0.0,
                "stanley_lateral_pd_ki": 0.02,
                "stanley_lateral_pd_kd": 0.0,
                "stanley_lateral_pd_output_limit_radps": 1.15,
                "max_w_rate": 3.0,
                "max_w_step": 0.020,
            }
        kwargs = dict(FORWARD_STRAIGHT_TRACKING_KWARGS)
        kwargs.update(
            {
                "tracking_mode": "stanley",
                "stanley_gain": float(kwargs.get("stanley_gain", 0.4)),
                "stanley_softening_speed_mps": float(
                    kwargs.get("stanley_softening_speed_mps", 1.0)
                ),
                "stanley_lateral_pd_kp": 0.0,
                "stanley_lateral_pd_kd": 0.0,
                "stanley_lateral_pd_output_limit_radps": 1.15,
            }
        )
        return kwargs

    def follow_path_with_pid(
        self,
        waypoints: List[Tuple[float, float]],
        speed_mps: float = 0.5,
        dt: float = 0.02,
        update_pose: Callable[[], Optional[PoseSolution]] = get_robot_pose,
        lookahead_distance: float = 4.0,
        slow_down_dist: Optional[float] = None,
        arrival_dist: float = 0.05,
        smoothing_strength: float = 0.6,
        smoothing_strength_curve: float = 0.75,
        w_bias_tau: float = 0.6,
        w_bias_hf_gain: float = 0.35,
        max_v_rate: float = 1.5,
        max_w_rate: float = 4.0,
        max_w_step: Optional[float] = None,
        speed_profile: Optional[List[Tuple[float, float]]] = None,
        metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        sample_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        run_label: Optional[str] = None,
        record_context: Optional[Dict[str, Any]] = None,
        tracking_mode: str = "stanley",
        stanley_gain: float = 0.4,
        stanley_softening_speed_mps: float = 0.55,
        stanley_lateral_pd_kp: float = 0.0,
        stanley_lateral_pd_ki: float = 0.0,
        stanley_lateral_pd_kd: float = 0.0,
        stanley_lateral_pd_output_limit_radps: float = 1.15,
        enable_stanley_w_pid: bool = False,
        **_extra: Any,
    ) -> None:
        """
        使用PID控制的路径跟踪方法
        
        Args:
            waypoints: 路径点列表
            speed_mps: 线速度
            dt: 控制周期
            update_pose: 位姿更新函数
            lookahead_distance: 前瞻距离
            slow_down_dist: 接近终点时的减速距离(米)，None 则不额外减速
            arrival_dist: 判定到达终点的距离阈值(米)
            smoothing_strength: 平滑强度(0~1，越大越平滑)
            smoothing_strength_curve: 高曲率段平滑强度(0~1)
            w_bias_tau: 角速度稳态偏置滤波时间常数(秒)
            w_bias_hf_gain: 角速度高频保留比例(0~1)
            max_v_rate: 线速度变化率上限(m/s^2)
            max_w_rate: 角速度变化率上限(rad/s^2)

        Stanley 模式（与速度符号区分前进/倒退，atan2 形式相同）：
          令 ψ 为车体航向 yaw，k=stanley_gain，v_s=stanley_softening_speed_mps，
          V = max(0.05, |v_nom| + v_s)，v_nom 为当前标称速度幅值。

          前进 (speed_mps≥0): 运动航向 ψ_m = ψ
          倒退 (speed_mps<0): 运动航向 ψ_m = wrap(ψ + π)（与车尾运动方向一致）

          最近路径点 (x_n,y_n)，横向误差 e_y 为 (x_n,y_n) 在「以 ψ_m 为前向」的车体坐标系中的侧向分量
          （与 imu_gnss_pose 约定一致：路径在右侧为正）。

          路径切向 ψ_p = atan2(Δy, Δx)（由最近点→前瞻点段），
          e_ψ = wrap(ψ_p - ψ_m)。

          δ_stanley = atan2(k * e_y, V)
          δ = wrap(e_ψ + δ_stanley)  （若启用横向 PD 则再叠加 PD 项）
          期望角速度 ω = δ（随后经限幅/滤波；横向 PD 当前默认关闭则 kp=kd=0）。
        """
        self._stop_flag.clear()
        started_wall_ts = time.time()
        run_key = f"{started_wall_ts:.6f}|{str(run_label or '轨迹').strip() or '轨迹'}"
        speed_sign = 1.0 if speed_mps >= 0 else -1.0
        speed_abs = abs(speed_mps)
        if speed_abs <= 0.0 or len(waypoints) < 2:
            return

        # 预计算路径长度与累计长度
        s_cum: List[float] = [0.0]
        total_len = 0.0
        for i in range(len(waypoints) - 1):
            dx = waypoints[i + 1][0] - waypoints[i][0]
            dy = waypoints[i + 1][1] - waypoints[i][1]
            seg = math.hypot(dx, dy)
            total_len += seg
            s_cum.append(total_len)

        avg_step = total_len / max(1, len(waypoints) - 1)
        # ===== 停车预判（提前减速段 & 到点阈值下限）=====
        # 目标：速度越高，越早开始减速；到点阈值也略抬高，减少“冲过终点再回拉”的情况。
        # 基于简化停车距离：d ~= v^2/(2a) + k*v + margin，a 取经验值。
        a_assume = 0.70  # m/s^2，经验减速度（越小则预判越保守/更早减速）
        d_stop = (speed_abs * speed_abs) / (2.0 * max(0.25, a_assume))
        d_extra = 0.55 * speed_abs + 0.25
        slow_down_auto = max(0.55, d_stop + d_extra)
        # 不让减速段超过路径太多（短路径避免全程都在减速）
        slow_down_auto = min(slow_down_auto, max(0.6, total_len * 0.65))
        if slow_down_dist is None:
            slow_down_dist = slow_down_auto
        else:
            try:
                slow_down_dist = max(float(slow_down_dist), slow_down_auto)
            except (TypeError, ValueError):
                slow_down_dist = slow_down_auto

        # 到点阈值下限：速度越高略增大，让停车更“提前”
        try:
            arrival_dist = max(float(arrival_dist), min(0.38, 0.16 + 0.12 * speed_abs))
        except (TypeError, ValueError):
            arrival_dist = min(0.38, 0.16 + 0.12 * speed_abs)

        loop_close_dist = max(0.1, arrival_dist * 2.0)
        is_loop = math.hypot(
            waypoints[-1][0] - waypoints[0][0],
            waypoints[-1][1] - waypoints[0][1],
        ) <= loop_close_dist
        backtrack = 5
        forward_window = max(20, int((2.0 * max(0.1, lookahead_distance)) / max(1e-3, avg_step)) + 10)
        if is_loop:
            forward_window = min(forward_window, max(5, int(0.5 * (len(waypoints) - 1))))
        forward_window = min(forward_window, len(waypoints) - 1)
        nearest_hint = 0
        last_pose: Optional[PoseSolution] = None
        traveled = 0.0
        loop_finish_min = 0.0
        if is_loop:
            loop_finish_min = max(total_len * 0.7, total_len - 2.0 * max(0.1, lookahead_distance))
        near_goal_deadband = max(arrival_dist * 1.5, 0.15)
        near_goal_hold_s = 0.3
        near_goal_since: Optional[float] = None
        prev_cmd_v: Optional[float] = None
        prev_cmd_w: Optional[float] = None
        w_bias: Optional[float] = None
        max_v_rate = max(0.1, float(max_v_rate))  # m/s^2, limit accel to reduce jitter
        max_w_rate = max(0.1, float(max_w_rate))  # rad/s^2, limit yaw accel to reduce shake
        smooth_strength = max(0.0, min(0.95, float(smoothing_strength)))
        smooth_strength_curve = max(0.0, min(0.95, float(smoothing_strength_curve)))
        alpha_base = max(0.05, 1.0 - smooth_strength)
        alpha_curve = max(0.05, 1.0 - smooth_strength_curve)
        w_bias_tau = max(0.05, float(w_bias_tau))
        w_bias_hf_gain = max(0.0, min(1.0, float(w_bias_hf_gain)))

        # 重置控制器（仅用于可选的 Stanley-w PID 抑制）
        try:
            self.stanley_w_pid.reset()
        except Exception:
            pass

        mode = str(tracking_mode or "").strip().lower()
        # 移除纯 PID：无论外部传 pid/其他值，都强制走 Stanley 逻辑
        if mode not in {"stanley", "stanley_pid"}:
            mode = "stanley"
        tracking_mode = mode
        use_stanley = True

        seg_kind = ""
        if isinstance(record_context, dict):
            seg_kind = str(record_context.get("segment_kind") or "").strip().lower()

        print(
            f"[ScoutMiniCAN] 路径跟踪开始: {len(waypoints)}个点, 速度={speed_mps}m/s, "
            f"mode={tracking_mode}"
        )

        sample_count = 0
        yaw_fb_samples = 0
        peak_abs_lateral_error = 0.0
        peak_abs_heading_error = 0.0
        peak_abs_cmd_w = 0.0
        peak_abs_yaw_rate_err = 0.0
        peak_abs_fb_w = 0.0
        exit_reason = "loop_exit"
        completed = False

        try:
            while not self._stop_flag.is_set():
                pose = update_pose()
                if pose is None:
                    print("[ScoutMiniCAN] PID路径跟踪: 位姿丢失, 主动停车.")
                    exit_reason = "pose_lost"
                    break
    
                current_x, current_y, current_yaw = pose.x, pose.y, pose.yaw
                if last_pose is not None:
                    traveled += math.hypot(current_x - last_pose.x, current_y - last_pose.y)
                last_pose = pose
    
                # 到终点的距离（用于提前终止与减速）
                dx_goal = waypoints[-1][0] - current_x
                dy_goal = waypoints[-1][1] - current_y
                dist_to_goal = math.hypot(dx_goal, dy_goal)
                if (not is_loop) and dist_to_goal <= arrival_dist:
                    exit_reason = "goal_arrived"
                    completed = True
                    break
    
                # 寻找最近路径点和前瞻点
                search_start = max(0, nearest_hint - backtrack)
                search_end = min(len(waypoints) - 1, nearest_hint + forward_window)
                nearest_idx, lookahead_idx = self._find_lookahead_point(
                    current_x, current_y, waypoints, lookahead_distance, search_start, search_end
                )
                if nearest_idx > nearest_hint:
                    nearest_hint = nearest_idx

                # Cluster CSV：前进直线段将 DRI 第 2 列 R 写为沿路径到终点的剩余距离（m）
                idx_nm = min(nearest_idx, len(s_cum) - 1)
                path_remaining_m = max(0.0, total_len - float(s_cum[idx_nm]))
                if speed_sign > 0.0 and seg_kind == "line":
                    set_cluster_csv_straight_path_remaining_m(path_remaining_m)
                else:
                    set_cluster_csv_straight_path_remaining_m(None)
    
                # 反向倒车时，使用“运动方向”的虚拟航向（yaw+pi）
                motion_yaw = current_yaw
                if speed_sign < 0.0:
                    motion_yaw = _wrap_angle(current_yaw + math.pi)
    
                # 横向误差：用最近路径点计算，保证贴近轨迹
                nearest_x, nearest_y = waypoints[nearest_idx]
                lateral_error = self._calculate_lateral_error(
                    current_x, current_y, motion_yaw, nearest_x, nearest_y
                )
    
                # 航向误差：优先用前瞻方向，避免末端点重合导致角度抖动
                dx_heading = waypoints[lookahead_idx][0] - waypoints[nearest_idx][0]
                dy_heading = waypoints[lookahead_idx][1] - waypoints[nearest_idx][1]
                if dx_heading * dx_heading + dy_heading * dy_heading < 1e-6:
                    if nearest_idx < len(waypoints) - 1:
                        dx_heading = waypoints[nearest_idx + 1][0] - waypoints[nearest_idx][0]
                        dy_heading = waypoints[nearest_idx + 1][1] - waypoints[nearest_idx][1]
                    elif nearest_idx > 0:
                        dx_heading = waypoints[nearest_idx][0] - waypoints[nearest_idx - 1][0]
                        dy_heading = waypoints[nearest_idx][1] - waypoints[nearest_idx - 1][1]
                if dx_heading * dx_heading + dy_heading * dy_heading < 1e-6:
                    target_heading = motion_yaw
                else:
                    target_heading = math.atan2(dy_heading, dx_heading)
                heading_error = _wrap_angle(target_heading - motion_yaw)
    
                stanley_term = 0.0
                stanley_lateral_pd_out = 0.0
                stanley_w_pid_out = 0.0
                if use_stanley:
                    speed_term = max(
                        0.05,
                        abs(float(speed_abs)) + abs(float(stanley_softening_speed_mps)),
                    )
                    stanley_term = math.atan2(float(stanley_gain) * float(lateral_error), speed_term)
                    stanley_output = _wrap_angle(float(heading_error) + float(stanley_term))
                    try:
                        dt_eff_stanley_pid = float(dt) if dt and float(dt) > 0 else 0.02
                    except Exception:
                        dt_eff_stanley_pid = 0.02
                    if (abs(float(stanley_lateral_pd_kp)) > 1e-9
                            or abs(float(stanley_lateral_pd_ki)) > 1e-9
                            or abs(float(stanley_lateral_pd_kd)) > 1e-9):
                        try:
                            dt_eff_pd = float(dt) if dt and float(dt) > 0 else 0.02
                        except Exception:
                            dt_eff_pd = 0.02
                        if not hasattr(self, "_stanley_lat_pd_prev_err"):
                            self._stanley_lat_pd_prev_err = float(lateral_error)
                        prev_err = float(getattr(self, "_stanley_lat_pd_prev_err"))
                        derr = (float(lateral_error) - prev_err) / max(1e-6, dt_eff_pd)
                        self._stanley_lat_pd_prev_err = float(lateral_error)
                        lim = max(0.05, abs(float(stanley_lateral_pd_output_limit_radps)))
                        # 积分项（带抗饱和）：仅在小误差时累积，用于消除稳态偏置
                        if not hasattr(self, "_stanley_lat_pd_integral"):
                            self._stanley_lat_pd_integral = 0.0
                        if abs(float(lateral_error)) < 0.15:
                            self._stanley_lat_pd_integral += float(lateral_error) * dt_eff_pd
                            max_i_contrib = lim * 0.10
                            max_integral = max_i_contrib / max(
                                1e-6, abs(float(stanley_lateral_pd_ki))
                            )
                            self._stanley_lat_pd_integral = _sat(
                                self._stanley_lat_pd_integral, -max_integral, max_integral
                            )
                        else:
                            self._stanley_lat_pd_integral = 0.0
                        stanley_lateral_pd_out = (
                            float(stanley_lateral_pd_kp) * float(lateral_error)
                            + float(stanley_lateral_pd_ki)
                            * float(self._stanley_lat_pd_integral)
                            + float(stanley_lateral_pd_kd) * float(derr)
                        )
                        stanley_lateral_pd_out = _sat(stanley_lateral_pd_out, -lim, lim)
    
                    # 先得到 Stanley 的角速度输出（用于满足横向/航向误差）
                    w_stanley = float(stanley_output) + float(stanley_lateral_pd_out)
    
                    if enable_stanley_w_pid:
                        # 在横向/航向误差“足够小”时，再对 w_stanley 做 PID 抑制，使 w 趋于 0
                        # 误差越小，抑制越强；误差较大时不介入，避免削弱转向纠偏能力
                        lat_thresh_m = 0.20
                        head_thresh_rad = math.radians(10.0)
                        lat_ratio = abs(float(lateral_error)) / max(1e-6, lat_thresh_m)
                        head_ratio = abs(float(heading_error)) / max(1e-6, head_thresh_rad)
                        scale = 1.0 - max(lat_ratio, head_ratio)
                        scale = max(0.0, min(1.0, float(scale)))
                        if scale <= 1e-6:
                            # 误差大：不做 w->0 抑制，且避免积分累积影响后续转向
                            try:
                                self.stanley_w_pid.reset()
                            except Exception:
                                pass
                            stanley_w_pid_out = 0.0
                        else:
                            # 让 w 朝 0 收敛：error = 0 - w_stanley
                            stanley_w_pid_out = float(
                                self.stanley_w_pid.update(float(-w_stanley), dt=dt_eff_stanley_pid)
                            ) * float(scale)
                    else:
                        stanley_w_pid_out = 0.0
    
                    angular_speed = float(w_stanley) + float(stanley_w_pid_out)
                    lateral_correction = float(stanley_output)
                    heading_correction = 0.0
                else:
                    # 理论上不会走到这里：纯 PID 已移除
                    angular_speed = 0.0
                    lateral_correction = 0.0
                    heading_correction = 0.0
                angular_speed = _sat(angular_speed, -self.MAX_ANGULAR_RADPS, self.MAX_ANGULAR_RADPS)
    
                # 自适应速度控制：根据曲率和误差调整速度
                curvature = abs(angular_speed) / max(speed_abs, 0.1)
                speed_factor = 1.0 / (1.0 + 2.0 * curvature + 3.0 * abs(lateral_error))
                speed_scale = max(0.3, speed_factor)
                # 速度剖面：若提供 (s[m], speed_abs[m/s]) 列表，则在路径弧长上做线性插值替换标称速度
                if speed_profile:
                    s_now = float(s_cum[min(nearest_idx, len(s_cum) - 1)])
                    sp = sorted(
                        ((float(s), abs(float(v))) for s, v in speed_profile),
                        key=lambda x: x[0],
                    )
                    if sp:
                        if s_now <= sp[0][0]:
                            speed_abs_profile = sp[0][1]
                        elif s_now >= sp[-1][0]:
                            speed_abs_profile = sp[-1][1]
                        else:
                            speed_abs_profile = speed_abs
                            for (s0, v0), (s1, v1) in zip(sp, sp[1:]):
                                if s0 <= s_now <= s1 and s1 > s0 + 1e-9:
                                    t = (s_now - s0) / (s1 - s0)
                                    speed_abs_profile = (1.0 - t) * v0 + t * v1
                                    break
                        speed_abs = max(0.0, float(speed_abs_profile))
                if slow_down_dist is not None and slow_down_dist > 0.0:
                    if dist_to_goal < slow_down_dist:
                        speed_scale *= max(0.15, dist_to_goal / slow_down_dist)
                adjusted_speed = speed_sign * speed_abs * speed_scale
    
                # 速度/角速度平滑与限幅，减少圆弧抖动
                if dt is None or dt <= 0:
                    dt_eff = 0.02
                else:
                    dt_eff = float(dt)
    
                desired_v = adjusted_speed
                desired_w = angular_speed
    
                # 角速度稳态偏置（低通）+ 高频抑制
                if w_bias is None:
                    w_bias = desired_w
                else:
                    beta = dt_eff / (w_bias_tau + dt_eff)
                    w_bias = (1.0 - beta) * w_bias + beta * desired_w
                desired_w = w_bias + (desired_w - w_bias) * w_bias_hf_gain
    
                if prev_cmd_v is None:
                    cmd_v = desired_v
                    cmd_w = desired_w
                else:
                    dv_max = max_v_rate * dt_eff
                    dw_max = max_w_rate * dt_eff
                    if max_w_step is not None:
                        try:
                            dw_max = min(dw_max, abs(float(max_w_step)))
                        except (TypeError, ValueError):
                            pass
                    dv = desired_v - prev_cmd_v
                    dw = desired_w - prev_cmd_w
                    if abs(dv) > dv_max:
                        desired_v = prev_cmd_v + math.copysign(dv_max, dv)
                    if abs(dw) > dw_max:
                        desired_w = prev_cmd_w + math.copysign(dw_max, dw)
    
                    # 高曲率段采用更强的低通平滑
                    curvature_now = abs(desired_w) / max(abs(desired_v), 0.1)
                    alpha = alpha_curve if curvature_now > 0.6 else alpha_base
                    cmd_v = alpha * desired_v + (1.0 - alpha) * prev_cmd_v
                    cmd_w = alpha * desired_w + (1.0 - alpha) * prev_cmd_w
    
                prev_cmd_v = cmd_v
                prev_cmd_w = cmd_w
    
                self._send_motion_command(cmd_v, cmd_w)
                if callable(sample_callback):
                    try:
                        now_ts = time.time()
                        fb_v = 0.0
                        fb_w = 0.0
                        try:
                            st = self.get_status()
                            fb_v = float(getattr(st, "linear_speed", 0.0) or 0.0)
                            fb_w = float(getattr(st, "angular_speed", 0.0) or 0.0)
                        except Exception:
                            fb_v = 0.0
                            fb_w = 0.0
                        curvature_inv_m = abs(float(cmd_w)) / max(abs(float(cmd_v)), 0.1)
                        payload: Dict[str, Any] = {
                            "timestamp": float(now_ts),
                            "relative_time_s": float(max(0.0, now_ts - started_wall_ts)),
                            "run_key": str(run_key),
                            "run_label": str(run_label or ""),
                            "tracking_mode": str(tracking_mode or "pid"),
                            "speed_sign": float(speed_sign),
                            "speed_mps": float(speed_mps),
                            "nominal_speed_abs_mps": float(speed_abs),
                            "lookahead_base_m": float(lookahead_distance),
                            "arrival_dist_m": float(arrival_dist),
                            "slow_down_dist_m": float(slow_down_dist or 0.0),
                            "stanley_gain": float(stanley_gain),
                            # UI 字段名沿用历史：softening_distance_m（本实现为速度软化项，数值仍可用于对比）
                            "stanley_softening_distance_m": float(stanley_softening_speed_mps),
                            "stanley_term_rad": float(stanley_term),
                            "stanley_w_pid_output_radps": float(stanley_w_pid_out),
                            "lateral_pid_kp": 0.0,
                            "lateral_pid_ki": 0.0,
                            "lateral_pid_kd": 0.0,
                            "heading_pid_kp": 0.0,
                            "heading_pid_ki": 0.0,
                            "heading_pid_kd": 0.0,
                            "yaw_rate_pid_kp": float(getattr(self.stanley_w_pid, "kp", 0.0) or 0.0)
                            if enable_stanley_w_pid
                            else 0.0,
                            "yaw_rate_pid_ki": float(getattr(self.stanley_w_pid, "ki", 0.0) or 0.0)
                            if enable_stanley_w_pid
                            else 0.0,
                            "yaw_rate_pid_kd": float(getattr(self.stanley_w_pid, "kd", 0.0) or 0.0)
                            if enable_stanley_w_pid
                            else 0.0,
                            "cmd_v_mps": float(cmd_v),
                            "cmd_w_radps": float(cmd_w),
                            "desired_v_mps": float(desired_v),
                            "desired_w_radps": float(desired_w),
                            "lateral_error_m": float(lateral_error),
                            "stanley_lateral_pd_output_radps": float(stanley_lateral_pd_out),
                            "heading_error_rad": float(heading_error),
                            "yaw_rate_error_radps": 0.0,
                            "feedback_v_mps": float(fb_v),
                            "feedback_w_radps": float(fb_w),
                            "yaw_rate_feedback_valid": 0,
                            "pose_age_s": 0.0,
                            "current_x_m": float(current_x),
                            "current_y_m": float(current_y),
                            "nearest_x_m": float(nearest_x),
                            "nearest_y_m": float(nearest_y),
                            "lookahead_x_m": float(waypoints[lookahead_idx][0]),
                            "lookahead_y_m": float(waypoints[lookahead_idx][1]),
                            "dist_to_goal_m": float(dist_to_goal),
                            "path_s_m": float(s_cum[min(nearest_idx, len(s_cum) - 1)]),
                            "motion_distance_m": float(traveled),
                            "motion_distance_signed_m": float(speed_sign * traveled),
                            "motion_distance_total_m": float(traveled),
                            "motion_distance_total_signed_m": float(speed_sign * traveled),
                            "lateral_pid_output_radps": float(lateral_correction),
                            "heading_pid_output_radps": float(heading_correction),
                            "yaw_rate_pid_output_radps": 0.0,
                            "path_curvature_inv_m": float(curvature_inv_m),
                            "path_ff_w_radps": 0.0,
                            "profile_speed_mps": float(speed_abs),
                        }
                        if isinstance(record_context, dict) and record_context:
                            payload.update({str(k): v for k, v in record_context.items()})
                        sample_callback(payload)
                    except Exception:
                        pass
    
                sample_count += 1
                peak_abs_lateral_error = max(peak_abs_lateral_error, abs(float(lateral_error)))
                peak_abs_heading_error = max(peak_abs_heading_error, abs(math.degrees(float(heading_error))))
                peak_abs_cmd_w = max(peak_abs_cmd_w, abs(float(cmd_w)))
                peak_abs_yaw_rate_err = max(peak_abs_yaw_rate_err, abs(0.0))
                try:
                    fb_w = float(getattr(self.get_status(), "angular_speed", 0.0) or 0.0)
                except Exception:
                    fb_w = 0.0
                peak_abs_fb_w = max(peak_abs_fb_w, abs(fb_w))
    
                # 检查是否到达终点
                remaining_len = max(0.0, total_len - s_cum[min(nearest_idx, len(s_cum) - 1)])
                if remaining_len <= arrival_dist:
                    if dist_to_goal <= arrival_dist:
                        if (not is_loop) or (traveled >= loop_finish_min):
                            exit_reason = "goal_arrived"
                            completed = True
                            break
                    if (not is_loop) and abs(lateral_error) <= near_goal_deadband:
                        exit_reason = "goal_arrived"
                        completed = True
                        break
                if traveled >= total_len and dist_to_goal <= arrival_dist:
                    if (not is_loop) or (traveled >= loop_finish_min):
                        exit_reason = "goal_arrived"
                        completed = True
                        break
                if not is_loop:
                    if dist_to_goal <= near_goal_deadband:
                        if near_goal_since is None:
                            near_goal_since = time.time()
                        elif time.time() - near_goal_since >= near_goal_hold_s:
                            exit_reason = "goal_arrived"
                            completed = True
                            break
                    else:
                        near_goal_since = None
    
                time.sleep(dt)
        finally:
            set_cluster_csv_straight_path_remaining_m(None)

        self._send_motion_command(0.0, 0.0)
        if self._stop_flag.is_set() and exit_reason == "loop_exit":
            exit_reason = "stop_flag"
        duration_s = max(0.0, time.time() - started_wall_ts)
        if callable(metrics_callback):
            try:
                record: Dict[str, Any] = {
                    "timestamp": float(started_wall_ts),
                    "run_key": str(run_key),
                    "run_label": str(run_label or ""),
                    "tracking_mode": str(tracking_mode or "pid"),
                    "speed_mps": float(speed_mps),
                    "speed_sign": float(speed_sign),
                    "nominal_speed_abs_mps": float(speed_abs),
                    "lookahead_base_m": float(lookahead_distance),
                    "arrival_dist_m": float(arrival_dist),
                    "slow_down_dist_m": float(slow_down_dist or 0.0),
                    "waypoints_count": int(len(waypoints)),
                    "duration_s": float(duration_s),
                    "samples": int(sample_count),
                    "feedback_samples": int(yaw_fb_samples),
                    "peak_abs_lateral_error_m": float(peak_abs_lateral_error),
                    "peak_abs_heading_error_deg": float(peak_abs_heading_error),
                    "peak_abs_yaw_rate_error_radps": float(peak_abs_yaw_rate_err),
                    "peak_abs_cmd_w_radps": float(peak_abs_cmd_w),
                    "peak_abs_feedback_w_radps": float(peak_abs_fb_w),
                    "completed": bool(completed),
                    "exit_reason": str(exit_reason),
                    "motion_distance_total_m": float(traveled),
                    "motion_distance_total_signed_m": float(speed_sign * traveled),
                }
                if isinstance(record_context, dict) and record_context:
                    record.update({str(k): v for k, v in record_context.items()})
                metrics_callback(record)
            except Exception:
                pass
        print("[ScoutMiniCAN] PID路径跟踪完成")

    def _find_lookahead_point(
        self, 
        current_x: float, 
        current_y: float, 
        waypoints: List[Tuple[float, float]], 
        lookahead_distance: float,
        search_start: int = 0,
        search_end: Optional[int] = None,
    ) -> Tuple[int, int]:
        """寻找最近点和前瞻点"""
        if not waypoints:
            return 0, 0
        if search_end is None:
            search_end = len(waypoints) - 1
        search_start = max(0, min(search_start, len(waypoints) - 1))
        search_end = max(search_start, min(search_end, len(waypoints) - 1))

        # 寻找最近点
        min_dist = float('inf')
        nearest_idx = search_start

        for i in range(search_start, search_end + 1):
            wx, wy = waypoints[i]
            dist = math.hypot(wx - current_x, wy - current_y)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        # 寻找前瞻点
        lookahead_idx = nearest_idx
        accumulated_dist = 0.0
        
        for i in range(nearest_idx, len(waypoints) - 1):
            dx = waypoints[i+1][0] - waypoints[i][0]
            dy = waypoints[i+1][1] - waypoints[i][1]
            segment_length = math.hypot(dx, dy)
            accumulated_dist += segment_length
            
            if accumulated_dist >= lookahead_distance:
                lookahead_idx = i + 1
                break
        else:
            lookahead_idx = len(waypoints) - 1
            
        return nearest_idx, lookahead_idx

    def _calculate_lateral_error(
        self, 
        current_x: float, 
        current_y: float, 
        current_yaw: float, 
        target_x: float, 
        target_y: float
    ) -> float:
        """计算横向误差"""
        # 将目标点转换到车辆坐标系
        dx = target_x - current_x
        dy = target_y - current_y
        
        # 旋转到车辆坐标系
        cos_yaw = math.cos(current_yaw)
        sin_yaw = math.sin(current_yaw)
        
        vehicle_x = dx * cos_yaw + dy * sin_yaw
        vehicle_y = -dx * sin_yaw + dy * cos_yaw
        
        # 横向误差（车辆坐标系中的y坐标）
        return vehicle_y

    def set_motion_control(self, linear_mps: float, angular_radps: float) -> None:
        """
        设置运动控制的便捷方法，与 _send_motion_command 功能相同
        """
        self._send_motion_command(linear_mps, angular_radps)

    def emergency_stop(self) -> None:
        """
        紧急停止的别名方法，与 stop() 功能相同
        """
        self.stop()

    # ================= 内部: CAN 打开 / 发送 / 接收 =================

    def _open_bus(self) -> None:
        """
        尝试打开 CAN 总线。失败时只记录错误，不抛异常，避免上层直接崩溃。
        """
        try:
            self.bus = can.interface.Bus(
                channel=self.channel,
                interface=self.interface,
                bitrate=self.bitrate,
            )
            print(
                f"[ScoutMiniCAN] CAN bus 打开成功: {self.channel}, "
                f"{self.interface}@{self.bitrate}"
            )
        except Exception as e:
            print(f"[ScoutMiniCAN] CAN bus 打开失败: {e}")
            self.bus = None

    def _ensure_bus(self) -> bool:
        """
        确保 bus 可用；若为 None 则尝试重新打开。
        避免出现 file descriptor 为 -1 的情况。
        """
        if self.bus is None and not self._virtual_mode:
            self._open_bus()
        return self.bus is not None or self._virtual_mode

    def _send_motion_command(
        self,
        linear_mps: float,
        angular_radps: float,
    ) -> None:
        """
        Send one 0x111 motion control frame (Motorola big-endian).

          - Data[0..1]: linear velocity (int16, mm/s)
          - Data[2..3]: angular velocity (int16, 0.001 rad/s)
          - Data[4..5]: lateral velocity vy (int16, non-holonomic: 0)
          - Data[6..7]: reserved (0)
        """
        if not self._ensure_bus():
            return

        # 饱和
        v = _sat(linear_mps, -self.MAX_LINEAR_MPS, self.MAX_LINEAR_MPS)
        w = _sat(angular_radps, -self.MAX_ANGULAR_RADPS, self.MAX_ANGULAR_RADPS)

        # 转成 mm/s + 0.001rad/s 的 int16
        v_mm_s = int(round(v * 1000.0))     # m/s -> mm/s
        w_mrad = int(round(w * 1000.0))     # rad/s -> mrad/s

        v_mm_s = max(-32768, min(32767, v_mm_s))
        w_mrad = max(-32768, min(32767, w_mrad))

        data = bytearray(8)
        data[0:2] = int(v_mm_s).to_bytes(2, byteorder=self._byteorder, signed=True)
        data[2:4] = int(w_mrad).to_bytes(2, byteorder=self._byteorder, signed=True)
        # 非全向底盘：横向速度无效，置 0
        data[4:8] = b"\x00\x00\x00\x00"

        if self._debug_tx:
            now = time.time()
            if self._debug_tx_interval_s <= 0.0 or (now - self._last_debug_tx_ts) >= self._debug_tx_interval_s:
                self._last_debug_tx_ts = now
                data_hex = data.hex()
                pretty = " ".join(data_hex[i:i + 2] for i in range(0, len(data_hex), 2))
                print(
                    f"[ScoutMiniCAN] TX 0x{self.ID_MOTION_CMD:03X} "
                    f"v={v:.3f} m/s w={w:.3f} rad/s bytes={pretty} ({self._byteorder})"
                )

        # 虚拟模式下只打印不发送
        if self._virtual_mode:
            return

        msg = can.Message(
            arbitration_id=self.ID_MOTION_CMD,
            is_extended_id=False,
            data=bytes(data),
        )
        try:
            self.bus.send(msg)
        except (can.CanError, OSError) as e:
            print(f"[ScoutMiniCAN] CAN send error: {e}")
            # 出现底层句柄错误时，关闭并等待下次重新打开
            self.bus = None

    def _recv_loop(self) -> None:
        """
        后台接收线程：解析 0x211 系统状态 + 0x221 运动反馈。
        """
        while self._running:
            if not self._ensure_bus():
                time.sleep(0.5)
                continue

            try:
                msg = self.bus.recv(timeout=0.1)
            except can.CanError as e:
                print(f"[ScoutMiniCAN] CAN recv error: {e}")
                self.bus = None
                continue

            if msg is None:
                continue

            self._handle_rx_message(msg)

            # Drain any backlog quickly to avoid kernel buffer buildup
            for _ in range(max(0, self._rx_drain_max)):
                try:
                    extra = self.bus.recv(timeout=0)
                except can.CanError as e:
                    print(f"[ScoutMiniCAN] CAN recv error: {e}")
                    self.bus = None
                    break
                if extra is None:
                    break
                self._handle_rx_message(extra)

    def _handle_rx_message(self, msg: can.Message) -> None:
        if msg.arbitration_id == self.ID_MOTION_FB:
            self._parse_motion_feedback(msg.data)
        elif msg.arbitration_id == self.ID_SYS_STATUS:
            self._parse_sys_status(msg.data)

    def _parse_motion_feedback(self, data: bytes) -> None:
        """
        解析 0x221 运动状态反馈：
          - Data[0..1]: 线速度 (int16, mm/s)
          - Data[2..3]: 角速度 (int16, 0.001 rad/s)
        """
        if len(data) < 4:
            return

        v_raw = int.from_bytes(data[0:2], byteorder=self._byteorder, signed=True)
        w_raw = int.from_bytes(data[2:4], byteorder=self._byteorder, signed=True)

        v = v_raw / 1000.0        # -> m/s
        w = w_raw / 1000.0        # -> rad/s

        with self._status_lock:
            self._status.linear_speed = v
            self._status.angular_speed = w
            self._status.last_update = time.time()

        if self._debug_motion and not self._motion_logged:
            self._motion_logged = True
            data_hex = data.hex()
            pretty = " ".join(data_hex[i:i + 2].upper() for i in range(0, len(data_hex), 2))
            print(
                f"[ScoutMiniCAN] RX 0x{self.ID_MOTION_FB:03X} "
                f"v={v:.3f}m/s w={w:.3f}rad/s bytes={pretty}"
            )

    def _parse_sys_status(self, data: bytes) -> None:
        """
        解析 0x211 系统状态反馈：
          - Data[0]: 系统状态
          - Data[1]: 控制模式
          - Data[2..3]: 电池电压 (uint16, 0.1V)
        """
        if len(data) < 4:
            return

        sys_status = data[0]
        mode = data[1]
        raw_volt = int.from_bytes(data[2:4], byteorder=self._byteorder, signed=False)
        voltage = raw_volt / 10.0  # -> V

        with self._status_lock:
            self._status.sys_status = sys_status
            self._status.control_mode = mode
            self._status.battery_voltage = voltage
            self._status.last_update = time.time()

        if self._debug_status and not self._status_logged:
            self._status_logged = True
            data_hex = data.hex()
            pretty = " ".join(data_hex[i:i + 2].upper() for i in range(0, len(data_hex), 2))
            print(
                f"[ScoutMiniCAN] RX 0x{self.ID_SYS_STATUS:03X} "
                f"sys={sys_status} mode={mode} batt={voltage:.1f}V bytes={pretty}"
            )
