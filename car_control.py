import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple

import can

from imu_gnss_pose import get_robot_pose, PoseSolution


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

        # 新增PID控制器（提高积分累积上限与整体控制力度）
        self.heading_pid = PIDController(kp=4.0, ki=0.4, kd=0.8, i_output_limit=6.0)  # 航向角PID
        self.lateral_pid = PIDController(kp=3.8, ki=0.35, kd=0.6, i_output_limit=6.0)  # 横向误差PID

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

        self.follow_path_with_pid(
            waypoints=waypoints,
            speed_mps=speed_cmd,
            dt=dt,
            update_pose=update_pose,
        )

    def move_circle(
        self,
        radius_m: float,
        angle_deg: float = 360.0,
        speed_mps: float = 0.5,
        clockwise: bool = False,
        dt: float = 0.02,
        update_pose: Callable[[], Optional[PoseSolution]] = get_robot_pose,
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
        step = max(0.1, min(0.5, arc_len / 80.0))
        n = max(10, min(400, int(arc_len / step)))
        waypoints: List[Tuple[float, float]] = []
        for i in range(n + 1):
            dphi = sign * total_angle_rad * i / n
            phi = phi0 + dphi
            waypoints.append((cx + radius_m * math.cos(phi), cy + radius_m * math.sin(phi)))

        print(
            f"[ScoutMiniCAN] move_circle(PID): center=({cx:.2f},{cy:.2f}), "
            f"R={radius_m:.2f}m, angle={angle_deg:.1f}deg, cw={clockwise}"
        )

        self.follow_path_with_pid(
            waypoints=waypoints,
            speed_mps=speed_abs,
            dt=dt,
            update_pose=update_pose,
            lookahead_distance=0.4,
            smoothing_strength=0.8,
            smoothing_strength_curve=0.88,
            w_bias_tau=0.9,
            w_bias_hf_gain=0.25,
            max_w_rate=3.0,
        )

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

    def follow_path_with_pid(
        self,
        waypoints: List[Tuple[float, float]],
        speed_mps: float = 0.5,
        dt: float = 0.02,
        update_pose: Callable[[], Optional[PoseSolution]] = get_robot_pose,
        lookahead_distance: float = 0.6,
        slow_down_dist: Optional[float] = None,
        arrival_dist: float = 0.05,
        smoothing_strength: float = 0.6,
        smoothing_strength_curve: float = 0.75,
        w_bias_tau: float = 0.6,
        w_bias_hf_gain: float = 0.35,
        max_v_rate: float = 1.5,
        max_w_rate: float = 4.0,
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
        """
        self._stop_flag.clear()
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
        near_goal_deadband = max(arrival_dist * 3.0, 0.15)
        near_goal_hold_s = 0.4
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

        # 重置PID控制器
        self.heading_pid.reset()
        self.lateral_pid.reset()

        print(f"[ScoutMiniCAN] PID路径跟踪开始: {len(waypoints)}个点, 速度={speed_mps}m/s")

        while not self._stop_flag.is_set():
            pose = update_pose()
            if pose is None:
                print("[ScoutMiniCAN] PID路径跟踪: 位姿丢失, 主动停车.")
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
                break

            # 寻找最近路径点和前瞻点
            search_start = max(0, nearest_hint - backtrack)
            search_end = min(len(waypoints) - 1, nearest_hint + forward_window)
            nearest_idx, lookahead_idx = self._find_lookahead_point(
                current_x, current_y, waypoints, lookahead_distance, search_start, search_end
            )
            if nearest_idx > nearest_hint:
                nearest_hint = nearest_idx

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

            # PID控制
            lateral_correction = self.lateral_pid.update(lateral_error, dt)
            heading_correction = self.heading_pid.update(heading_error, dt)

            # 合成角速度命令
            angular_speed = lateral_correction + heading_correction
            angular_speed = _sat(angular_speed, -self.MAX_ANGULAR_RADPS, self.MAX_ANGULAR_RADPS)

            # 自适应速度控制：根据曲率和误差调整速度
            curvature = abs(angular_speed) / max(speed_abs, 0.1)
            speed_factor = 1.0 / (1.0 + 2.0 * curvature + 3.0 * abs(lateral_error))
            speed_scale = max(0.3, speed_factor)
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

            # 检查是否到达终点
            remaining_len = max(0.0, total_len - s_cum[min(nearest_idx, len(s_cum) - 1)])
            if remaining_len <= arrival_dist:
                if dist_to_goal <= arrival_dist:
                    if (not is_loop) or (traveled >= loop_finish_min):
                        break
                if (not is_loop) and abs(lateral_error) <= near_goal_deadband:
                    break
            if traveled >= total_len and dist_to_goal <= arrival_dist:
                if (not is_loop) or (traveled >= loop_finish_min):
                    break
            if not is_loop:
                if dist_to_goal <= near_goal_deadband:
                    if near_goal_since is None:
                        near_goal_since = time.time()
                    elif time.time() - near_goal_since >= near_goal_hold_s:
                        break
                else:
                    near_goal_since = None

            time.sleep(dt)

        self._send_motion_command(0.0, 0.0)
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
