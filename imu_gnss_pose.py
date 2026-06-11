"""
imu_gnss_pose.py

北云 X1 系列 INSPVAXA UDP ASCII 报文解析 + 位姿统一输出 + 状态监控 + UDP 握手。

解析以下 ASCII 报文：
  - INSPVAXA
对外主要接口：
  get_robot_pose()            -> PoseSolution 或 None
  calibrate_pose_to_current() -> bool
  get_status_summary()        -> ImuStatusSummary
  shutdown_imu_client()       -> None
"""

import math
import os
import socket
import threading
import time
from dataclasses import dataclass, replace
from typing import Optional, Dict, Any, Tuple, Sequence


# WGS84 椭球参数（用于小范围局部平面近似）
WGS84_A_M = 6378137.0               # 半长轴 (m)
WGS84_E2 = 6.69437999014e-3         # 第一偏心率平方


# ======================== 基础工具函数 ========================

def _wrap_angle_rad(angle: float) -> float:
    """将角度规范到 (-pi, pi]。"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def _deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _geodetic_to_enu_raw(
    lat_deg: float,
    lon_deg: float,
    h_m: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_h_m: float,
) -> Tuple[float, float, float]:
    """
    将经纬高转换到局部 ENU 坐标（以 ref 为原点），输出单位：米：
      x: 东向 (m)
      y: 北向 (m)
      z: Up   (m)

    小范围二维应用下，统一将所有点投影到参考点的局部水平面：
      - x/y 仅由经纬度差决定，且使用参考点高度作为固定尺度；
      - z 固定为 0，不再保留逐点高程差。
    """
    lat = _deg2rad(lat_deg)
    lon = _deg2rad(lon_deg)
    ref_lat = _deg2rad(ref_lat_deg)
    ref_lon = _deg2rad(ref_lon_deg)

    dlat = lat - ref_lat
    dlon = lon - ref_lon

    # 参考点纬度对应的曲率半径，确保 x/y 坐标比例与实际米制一致
    sin_ref = math.sin(ref_lat)
    cos_ref = math.cos(ref_lat)
    denom = 1.0 - WGS84_E2 * sin_ref * sin_ref
    rn = WGS84_A_M / math.sqrt(denom)             # 卯酉曲率半径
    rm = rn * (1.0 - WGS84_E2) / denom            # 子午曲率半径

    plane_h_m = ref_h_m
    x = dlon * cos_ref * (rn + plane_h_m)
    y = dlat * (rm + plane_h_m)
    z = 0.0
    return x, y, z


def _apply_enu_calibration_xy(x_m: float, y_m: float) -> Tuple[float, float]:
    if not _enu_calib_enabled:
        return x_m, y_m

    cos_rot = math.cos(_enu_calib_rot_rad)
    sin_rot = math.sin(_enu_calib_rot_rad)
    x_rot = x_m * cos_rot - y_m * sin_rot
    y_rot = x_m * sin_rot + y_m * cos_rot
    return x_rot + _enu_calib_tx, y_rot + _enu_calib_ty


def _geodetic_to_enu(
    lat_deg: float,
    lon_deg: float,
    h_m: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_h_m: float,
) -> Tuple[float, float, float]:
    x_m, y_m, z_m = _geodetic_to_enu_raw(
        lat_deg,
        lon_deg,
        h_m,
        ref_lat_deg,
        ref_lon_deg,
        ref_h_m,
    )
    x_m, y_m = _apply_enu_calibration_xy(x_m, y_m)
    return x_m, y_m, z_m


# ======================== 统一位姿/状态结构体 ========================

@dataclass
class PoseSolution:
    """
    小车统一位姿输出（ENU 局部坐标）：

    source:
      "INS"   - INSPVAXA 融合解
    yaw:
      以东向为 0，逆时针为正（rad），与 main_ui 轨迹坐标系一致。
      与 car_control 一致：校准平面内 +Y 为前进方向、+X 为右侧时，横向误差“路径右侧为正”，w>0 为 CCW 左转。

    当前实现面向小范围二维平面应用，z 固定为 0。
    """
    source: str
    gps_week: Optional[int]
    gps_sec: Optional[float]

    lat: float
    lon: float
    height: float

    x: float
    y: float
    z: float

    yaw: float
    pitch: float
    roll: float

    ins_status: Optional[str]
    ins_pos_type: Optional[str]


@dataclass
class ImuStatusSummary:
    """
    IMU 当前状态概要，用于 UI 状态栏显示。
    """
    mode: str  # "INS", "NONE"
    ins_status: Optional[str]
    ins_pos_type: Optional[str]
    lat_sigma_m: Optional[float]
    lon_sigma_m: Optional[float]
    hgt_sigma_m: Optional[float]

    has_inspvax: bool
    age_inspvax: Optional[float]     # 距离上一次 INSPVAXA 的时间（秒）
    freq_inspvax: Optional[float]    # 估算 INSPVAXA 频率（Hz）


# ======================== 核心客户端 ========================

@dataclass
class EnuCalibrationSummary:
    enabled: bool
    point_count: int
    rotation_deg: float
    translation_x_m: float
    translation_y_m: float
    rms_error_m: Optional[float]


class ImuGnssClient:
    """
    监听 X1 ICOM2 的 UDP ASCII 报文，仅解析 INSPVAXA。

    特点：
      - 启动时自动向惯导 IP 发送若干次 "ok" 做 UDP 握手（仿 Recive.cpp）；
      - 终端周期性打印报文解析概要，方便调试；
      - 统计最近收包间隔与估算频率。
    """

    def __init__(
        self,
        listen_ip: str = "0.0.0.0",
        listen_port: int = 3002,
        ins_ip: str = "192.168.1.110",
        do_udp_handshake: bool = True,
    ) -> None:
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.ins_ip = ins_ip
        self.ins_port = listen_port  # 惯导端口与本地监听端口一致（3002）

        # ENU 原点（第一次获得有效经纬度时锁定）
        self._ref_lat: Optional[float] = None
        self._ref_lon: Optional[float] = None
        self._ref_h: Optional[float] = None

        # 最新解算缓存
        self._ins: Dict[str, Any] = {}

        # 最近收包时间戳
        self._t_inspvax: Optional[float] = None

        # 频率估算：由相邻 INSPVAXA 的 TOW 差或收包墙钟间隔 EMA 得到（如 50Hz 输出）
        self._last_tow_inspvax: Optional[float] = None
        self._freq_inspvax: Optional[float] = None

        # 调试输出限流
        self._last_dbg_print_ins = 0.0
        self._printed_inspvax_once = False

        self._lock = threading.Lock()
        self._stop_flag = False

        # 建立 UDP socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self.listen_ip, self.listen_port))
        self._sock.settimeout(1.0)

        print(
            f"[ImuGnssClient] Listen UDP {self.listen_ip}:{self.listen_port}, "
            f"INS IP = {self.ins_ip}:{self.ins_port}"
        )

        # UDP 握手（仿照 Recive.cpp：连续发送多个 "ok" 激活输出）
        if do_udp_handshake:
            self._do_udp_handshake()

        # 启动后台接收线程
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    # ----------- UDP 握手 -----------

    def _do_udp_handshake(self) -> None:
        msg = b"ok"
        try:
            for _ in range(5):
                self._sock.sendto(msg, (self.ins_ip, self.ins_port))
                time.sleep(0.05)
            print(
                f"[ImuGnssClient] UDP handshake sent ('ok' x5) to "
                f"{self.ins_ip}:{self.ins_port}"
            )
        except OSError as e:
            print(f"[ImuGnssClient] UDP handshake failed: {e}")

    # ----------- 对外接口 -----------

    def get_pose(self) -> Optional[PoseSolution]:
        """
        统一位姿输出：
          - 仅使用 INSPVAXA 融合解；
          - 若不可用，则返回 None。
        """
        with self._lock:
            if not self._has_inspvax():
                return None
            pose = self._build_ins_solution()
            if pose is None:
                return None
            return replace(pose, source="INS")

    def get_status_summary(self) -> ImuStatusSummary:
        """
        UI 状态栏用：返回当前模式、INS 状态、报文收包间隔与估算频率。
        """
        now = time.time()
        with self._lock:
            mode = "INS" if self._has_inspvax() else "NONE"
            age_insp = now - self._t_inspvax if self._t_inspvax is not None else None

            return ImuStatusSummary(
                mode=mode,
                ins_status=(self._ins.get("ins_status") if self._ins else None),
                ins_pos_type=(self._ins.get("pos_type") if self._ins else None),
                lat_sigma_m=(self._ins.get("lat_std") if self._ins else None),
                lon_sigma_m=(self._ins.get("lon_std") if self._ins else None),
                hgt_sigma_m=(self._ins.get("hgt_std") if self._ins else None),
                has_inspvax=self._t_inspvax is not None,
                age_inspvax=age_insp,
                freq_inspvax=self._freq_inspvax,
            )

    def get_reference(self) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            if self._ref_lat is None or self._ref_lon is None or self._ref_h is None:
                return None
            return self._ref_lat, self._ref_lon, self._ref_h

    def stop(self) -> None:
        self._stop_flag = True
        try:
            self._sock.close()
        except OSError:
            pass

    # ----------- 有效性判断 -----------

    def _has_inspvax(self) -> bool:
        if not self._ins:
            return False
        lat = self._ins.get("lat")
        lon = self._ins.get("lon")
        if lat is None or lon is None:
            return False
        if abs(lat) < 1e-10 and abs(lon) < 1e-10:
            return False
        return True

    # ----------- 构造位姿 -----------

    def _ensure_ref(self, lat: float, lon: float, h: float) -> None:
        if self._ref_lat is None:
            self._ref_lat = lat
            self._ref_lon = lon
            self._ref_h = h
            print(
                f"[ImuGnssClient] ENU reference set to "
                f"lat={lat:.8f}, lon={lon:.8f}, h={h:.3f}"
            )

    def _build_ins_solution(self) -> Optional[PoseSolution]:
        ins = self._ins
        if not ins:
            return None

        lat = ins["lat"]
        lon = ins["lon"]
        h = ins["hgt"]
        undulation = ins.get("undulation")

        self._ensure_ref(lat, lon, h)
        if self._ref_lat is None:
            return None

        x, y, z = _geodetic_to_enu(lat, lon, h, self._ref_lat, self._ref_lon, self._ref_h)

        az_deg = ins["azimuth_deg"]
        # 北向顺时针为正 -> ENU 中以东为 0，逆时针为正
        yaw_raw = _wrap_angle_rad(_deg2rad(90.0 - az_deg))
        pitch = _deg2rad(ins["pitch_deg"])
        roll = _deg2rad(ins["roll_deg"])

        yaw = yaw_raw
        if _enu_calib_enabled:
            yaw = _wrap_angle_rad(yaw + _enu_calib_rot_rad)

        gps_week = ins.get("week")
        gps_sec = ins.get("tow")

        return PoseSolution(
            source="INS",
            gps_week=gps_week,
            gps_sec=gps_sec,
            lat=lat,
            lon=lon,
            height=h,
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            ins_status=ins.get("ins_status"),
            ins_pos_type=ins.get("pos_type"),
        )

    # ======================== UDP 接收线程 ========================

    def _recv_loop(self) -> None:
        while not self._stop_flag:
            try:
                data, _addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                continue

            # 支持一次收多行
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                self._handle_line(line)

    def _handle_line(self, line: str) -> None:
        """
        处理一整行 ASCII 报文；自动去掉前面日志前缀，只保留从 '#' 开始部分。
        """
        if "#" in line:
            line = line[line.index("#") :]

        if "INSPVAXA" in line:
            self._parse_inspvaxa(line)

    # ======================== 报文解析 ========================

    @staticmethod
    def _split_header_body(line: str):
        if ";" not in line:
            return None, None
        header_part, rest = line.split(";", 1)
        if "*" in rest:
            body_part, _crc = rest.split("*", 1)
        else:
            body_part = rest
        header_fields = [f.strip() for f in header_part.split(",")]
        body_fields = [f.strip() for f in body_part.split(",")]
        return header_fields, body_fields

    def _parse_inspvaxa(self, line: str) -> None:
        header, body = self._split_header_body(line)
        if not header or not body:
            return

        if len(body) < 11:
            return

        try:
            week = int(header[5])
            tow = float(header[6])

            ins_status = body[0]
            pos_type = body[1]

            lat = float(body[2])
            lon = float(body[3])
            hgt = float(body[4])

            # INSPVAXA 标准字段：height 后带 undulation（当前实测数据包含该字段）
            offset = 1 if len(body) >= 23 else 0
            undulation = float(body[5]) if offset and body[5] != "" else None

            v_n = float(body[5 + offset])
            v_e = float(body[6 + offset])
            v_u = float(body[7 + offset])

            roll_deg = float(body[8 + offset])
            pitch_deg = float(body[9 + offset])
            az_deg = float(body[10 + offset])

            # 标准差等可以按需扩展，这里只做简单保护
            lat_std = float(body[11 + offset]) if len(body) > 11 + offset and body[11 + offset] != "" else None
            lon_std = float(body[12 + offset]) if len(body) > 12 + offset and body[12 + offset] != "" else None
            hgt_std = float(body[13 + offset]) if len(body) > 13 + offset and body[13 + offset] != "" else None

            ext_sol_stat = body[20 + offset] if len(body) > 20 + offset else ""
            time_since_update = (
                float(body[21 + offset]) if len(body) > 21 + offset and body[21 + offset] != "" else None
            )
        except (ValueError, IndexError):
            return

        now = time.time()
        with self._lock:
            prev_tow = self._last_tow_inspvax
            prev_wall = self._t_inspvax
            dt_msg: Optional[float] = None
            if prev_tow is not None:
                d_tow = tow - prev_tow
                if 1e-5 < d_tow < 2.0:
                    dt_msg = d_tow
            if dt_msg is None and prev_wall is not None:
                d_wall = now - prev_wall
                if 1e-4 < d_wall < 0.5:
                    dt_msg = d_wall
            if dt_msg is not None and dt_msg > 1e-6:
                inst_hz = min(200.0, max(1.0, 1.0 / dt_msg))
                if self._freq_inspvax is None:
                    self._freq_inspvax = inst_hz
                else:
                    self._freq_inspvax = 0.88 * self._freq_inspvax + 0.12 * inst_hz

            self._last_tow_inspvax = tow

            self._ins = {
                "week": week,
                "tow": tow,
                "ins_status": ins_status,
                "pos_type": pos_type,
                "lat": lat,
                "lon": lon,
                "hgt": hgt,
                "undulation": undulation,
                "v_n": v_n,
                "v_e": v_e,
                "v_u": v_u,
                "roll_deg": roll_deg,
                "pitch_deg": pitch_deg,
                "azimuth_deg": az_deg,
                "lat_std": lat_std,
                "lon_std": lon_std,
                "hgt_std": hgt_std,
                "ext_sol_stat": ext_sol_stat,
                "time_since_update": time_since_update,
            }
            self._t_inspvax = now

        # 仅打印一次位置信息，避免终端刷屏
        if not self._printed_inspvax_once:
            self._printed_inspvax_once = True
            print(
                f"[INSPVAXA] week={week} tow={tow:.3f} "
                f"status={ins_status} pos={pos_type} "
                f"lat={lat:.8f} lon={lon:.8f} h={hgt:.2f} "
                f"vn={v_n:.3f} ve={v_e:.3f} vu={v_u:.3f} "
                f"az={az_deg:.3f} | freq≈{self._freq_inspvax or 0:.1f}Hz"
            )


_default_client: Optional[ImuGnssClient] = None

_calib_enabled = False
_calib_x = 0.0
_calib_y = 0.0
_calib_yaw = 0.0
_calib_zero_yaw_enabled = True
_yaw_offset_rad = 0.0
_enu_calib_enabled = False
_enu_calib_rot_rad = 0.0
_enu_calib_tx = 0.0
_enu_calib_ty = 0.0
_enu_calib_point_count = 0
_enu_calib_rms_error_m: Optional[float] = None

_yaw_offset_env = os.getenv("IMU_YAW_OFFSET_DEG", "").strip()
if _yaw_offset_env:
    try:
        _yaw_offset_rad = _deg2rad(float(_yaw_offset_env))
        print(f"[imu_gnss_pose] IMU yaw offset set from env: {float(_yaw_offset_env):.3f} deg")
    except ValueError:
        _yaw_offset_rad = 0.0


def _get_client() -> ImuGnssClient:
    global _default_client
    if _default_client is None:
        _default_client = ImuGnssClient(
            listen_ip="0.0.0.0",
            listen_port=3002,
            ins_ip="192.168.1.110",
            do_udp_handshake=True,
        )
    return _default_client


def get_robot_pose() -> Optional[PoseSolution]:
    """
    对外统一接口：
      - INS 解可用 → 返回 source='INS'；
      - 全不可用 → 返回 None。

    若调用过 calibrate_pose_to_current()，则输出坐标/航向已平移到“起点为 (0,0,0)”。
    """
    client = _get_client()
    pose = client.get_pose()
    if pose is None:
        return None

    if _calib_enabled:
        yaw = pose.yaw
        if _calib_zero_yaw_enabled:
            yaw = _wrap_angle_rad(yaw - _calib_yaw)
        if _yaw_offset_rad != 0.0:
            yaw = _wrap_angle_rad(yaw + _yaw_offset_rad)
        return replace(
            pose,
            x=pose.x - _calib_x,
            y=pose.y - _calib_y,
            yaw=yaw,
        )
    if _yaw_offset_rad != 0.0:
        return replace(pose, yaw=_wrap_angle_rad(pose.yaw + _yaw_offset_rad))
    return pose


def get_absolute_robot_pose() -> Optional[PoseSolution]:
    """
    返回未经 calibrate_pose_to_current() 平移/归零的绝对 ENU 位姿。
    仍会应用固定的 yaw offset，用于车体安装偏角修正。
    """
    client = _get_client()
    pose = client.get_pose()
    if pose is None:
        return None

    if _yaw_offset_rad != 0.0:
        return replace(pose, yaw=_wrap_angle_rad(pose.yaw + _yaw_offset_rad))
    return pose


def calibrate_pose_to_current() -> bool:
    """
    将当前位姿设置为原点 (0,0,0)，用于试验起点校准。
    """
    global _calib_enabled, _calib_x, _calib_y, _calib_yaw, _calib_zero_yaw_enabled
    client = _get_client()
    pose = client.get_pose()
    if pose is None:
        return False

    _calib_x = pose.x
    _calib_y = pose.y
    _calib_yaw = pose.yaw
    _calib_enabled = True
    _calib_zero_yaw_enabled = True
    print(
        f"[imu_gnss_pose] Calibration set at "
        f"x0={_calib_x:.3f}, y0={_calib_y:.3f}, yaw0={_calib_yaw:.3f} rad"
    )
    return True


def set_position_origin_to_current() -> bool:
    """
    仅将当前位置设为坐标原点，保留当前校准后的航向定义。
    """
    global _calib_enabled, _calib_x, _calib_y, _calib_yaw, _calib_zero_yaw_enabled
    client = _get_client()
    pose = client.get_pose()
    if pose is None:
        return False

    _calib_x = pose.x
    _calib_y = pose.y
    _calib_yaw = pose.yaw
    _calib_enabled = True
    _calib_zero_yaw_enabled = False
    print(
        f"[imu_gnss_pose] Position origin set at "
        f"x0={_calib_x:.3f}, y0={_calib_y:.3f}, keep_yaw={_calib_yaw:.3f} rad"
    )
    return True


def reset_navigation_position_origin() -> None:
    """
    清除「位置平移原点」(_calib_x/_calib_y)，停止对 get_robot_pose 做平面平移。
    在重新经纬度定系、旋转 ENU 平面期间应先调用，避免旧原点与新旋转混用导致
    原点错误地像「当前车位置」。
    """
    global _calib_enabled, _calib_x, _calib_y, _calib_zero_yaw_enabled
    _calib_enabled = False
    _calib_x = 0.0
    _calib_y = 0.0
    _calib_zero_yaw_enabled = False


def set_position_origin_to_xy(x_m: float, y_m: float) -> bool:
    """
    将指定平面坐标 (x,y) 设为位置原点（与 get_absolute_robot_pose 同一套 ENU+校准平面坐标），
    不修改航向零点（与 set_position_origin_to_current 一致）。
    用于“以某一已知经纬度点为原点”而与当前车位无关。
    """
    global _calib_enabled, _calib_x, _calib_y, _calib_yaw, _calib_zero_yaw_enabled
    client = _get_client()
    pose = client.get_pose()
    _calib_x = float(x_m)
    _calib_y = float(y_m)
    if pose is not None:
        _calib_yaw = pose.yaw
    else:
        _calib_yaw = 0.0
    _calib_enabled = True
    _calib_zero_yaw_enabled = False
    print(
        f"[imu_gnss_pose] Position origin set to fixed xy "
        f"x0={_calib_x:.3f}, y0={_calib_y:.3f}, keep_yaw={_calib_yaw:.3f} rad"
    )
    return True


def absolute_planar_xy_from_geodetic(
    lat_deg: float,
    lon_deg: float,
    height_m: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    将经纬度映射到当前 ENU 参考 + ENU 校准后的平面 (x,y,z)（米），
    不含 calibrate_pose / set_position_origin 的平移。
    """
    client = _get_client()
    ref = client.get_reference()
    if ref is None:
        raise RuntimeError("ENU reference is not initialized yet")
    ref_lat, ref_lon, ref_h = ref
    h_m = float(height_m) if height_m is not None else float(ref_h)
    return _geodetic_to_enu(float(lat_deg), float(lon_deg), h_m, ref_lat, ref_lon, ref_h)


def _as_sys_xy(point: Sequence[float]) -> Tuple[float, float]:
    if len(point) < 2:
        raise ValueError("system ENU point requires at least x and y")
    return float(point[0]), float(point[1])


def _as_lla(point: Sequence[float], default_h_m: float) -> Tuple[float, float, float]:
    if len(point) < 2:
        raise ValueError("LLA point requires at least lat and lon")
    lat = float(point[0])
    lon = float(point[1])
    if len(point) >= 3 and point[2] is not None:
        h_m = float(point[2])
    else:
        h_m = float(default_h_m)
    return lat, lon, h_m


def _solve_rigid_transform_2d(
    source_points: Sequence[Tuple[float, float]],
    target_points: Sequence[Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    count = len(source_points)
    if count != len(target_points):
        raise ValueError("point count mismatch")
    if count < 2:
        raise ValueError("at least two point pairs are required")

    src_cx = sum(point[0] for point in source_points) / count
    src_cy = sum(point[1] for point in source_points) / count
    dst_cx = sum(point[0] for point in target_points) / count
    dst_cy = sum(point[1] for point in target_points) / count

    dot_sum = 0.0
    cross_sum = 0.0
    src_spread = 0.0
    dst_spread = 0.0
    for (src_x, src_y), (dst_x, dst_y) in zip(source_points, target_points):
        src_x_c = src_x - src_cx
        src_y_c = src_y - src_cy
        dst_x_c = dst_x - dst_cx
        dst_y_c = dst_y - dst_cy
        dot_sum += src_x_c * dst_x_c + src_y_c * dst_y_c
        cross_sum += src_x_c * dst_y_c - src_y_c * dst_x_c
        src_spread += src_x_c * src_x_c + src_y_c * src_y_c
        dst_spread += dst_x_c * dst_x_c + dst_y_c * dst_y_c

    if src_spread <= 1e-9 or dst_spread <= 1e-9:
        raise ValueError("control points are degenerate")

    rot_rad = math.atan2(cross_sum, dot_sum)
    cos_rot = math.cos(rot_rad)
    sin_rot = math.sin(rot_rad)
    tx_m = dst_cx - (src_cx * cos_rot - src_cy * sin_rot)
    ty_m = dst_cy - (src_cx * sin_rot + src_cy * cos_rot)

    err_sq_sum = 0.0
    for (src_x, src_y), (dst_x, dst_y) in zip(source_points, target_points):
        fit_x = src_x * cos_rot - src_y * sin_rot + tx_m
        fit_y = src_x * sin_rot + src_y * cos_rot + ty_m
        err_x = fit_x - dst_x
        err_y = fit_y - dst_y
        err_sq_sum += err_x * err_x + err_y * err_y

    rms_error_m = math.sqrt(err_sq_sum / count)
    return rot_rad, tx_m, ty_m, rms_error_m


def calibrate_enu_from_points(
    points_sys: Sequence[Sequence[float]],
    points_lla: Sequence[Sequence[float]],
) -> EnuCalibrationSummary:
    global _enu_calib_enabled
    global _enu_calib_rot_rad
    global _enu_calib_tx
    global _enu_calib_ty
    global _enu_calib_point_count
    global _enu_calib_rms_error_m

    if len(points_sys) != len(points_lla):
        raise ValueError("points_sys and points_lla must contain the same number of points")
    if len(points_sys) < 2:
        raise ValueError("at least two control points are required")

    client = _get_client()
    ref = client.get_reference()
    if ref is None:
        raise RuntimeError("ENU reference is not initialized yet")
    ref_lat, ref_lon, ref_h = ref

    system_points_xy = []
    raw_points_xy = []
    for point_sys, point_lla in zip(points_sys, points_lla):
        sys_x, sys_y = _as_sys_xy(point_sys)
        lat, lon, h_m = _as_lla(point_lla, ref_h)
        raw_x, raw_y, _ = _geodetic_to_enu_raw(lat, lon, h_m, ref_lat, ref_lon, ref_h)
        system_points_xy.append((sys_x, sys_y))
        raw_points_xy.append((raw_x, raw_y))

    rot_rad, tx_m, ty_m, rms_error_m = _solve_rigid_transform_2d(
        raw_points_xy,
        system_points_xy,
    )

    _enu_calib_enabled = True
    _enu_calib_rot_rad = rot_rad
    _enu_calib_tx = tx_m
    _enu_calib_ty = ty_m
    _enu_calib_point_count = len(points_sys)
    _enu_calib_rms_error_m = rms_error_m

    summary = EnuCalibrationSummary(
        enabled=True,
        point_count=_enu_calib_point_count,
        rotation_deg=math.degrees(_enu_calib_rot_rad),
        translation_x_m=_enu_calib_tx,
        translation_y_m=_enu_calib_ty,
        rms_error_m=_enu_calib_rms_error_m,
    )
    print(
        "[imu_gnss_pose] ENU calibration updated: "
        f"points={summary.point_count}, "
        f"rot={summary.rotation_deg:.6f} deg, "
        f"tx={summary.translation_x_m:.3f} m, "
        f"ty={summary.translation_y_m:.3f} m, "
        f"rms={(summary.rms_error_m or 0.0):.3f} m"
    )
    return summary


def align_enu_y_axis_with_points(points_sys: Sequence[Sequence[float]]) -> EnuCalibrationSummary:
    global _enu_calib_enabled
    global _enu_calib_rot_rad
    global _enu_calib_tx
    global _enu_calib_ty
    global _enu_calib_point_count
    global _enu_calib_rms_error_m

    if len(points_sys) != 2:
        raise ValueError("exactly two control points are required")

    point_a_x, point_a_y = _as_sys_xy(points_sys[0])
    point_b_x, point_b_y = _as_sys_xy(points_sys[1])
    dx = point_b_x - point_a_x
    dy = point_b_y - point_a_y
    baseline_m = math.hypot(dx, dy)
    if baseline_m < 1e-6:
        raise ValueError("control points are too close to define a Y-axis direction")

    line_yaw_rad = math.atan2(dy, dx)
    delta_rot_rad = _wrap_angle_rad((math.pi * 0.5) - line_yaw_rad)

    cos_delta = math.cos(delta_rot_rad)
    sin_delta = math.sin(delta_rot_rad)
    old_tx = _enu_calib_tx
    old_ty = _enu_calib_ty

    _enu_calib_enabled = True
    _enu_calib_rot_rad = _wrap_angle_rad(_enu_calib_rot_rad + delta_rot_rad)
    _enu_calib_tx = old_tx * cos_delta - old_ty * sin_delta
    _enu_calib_ty = old_tx * sin_delta + old_ty * cos_delta
    _enu_calib_point_count = 2
    _enu_calib_rms_error_m = 0.0

    summary = EnuCalibrationSummary(
        enabled=True,
        point_count=_enu_calib_point_count,
        rotation_deg=math.degrees(_enu_calib_rot_rad),
        translation_x_m=_enu_calib_tx,
        translation_y_m=_enu_calib_ty,
        rms_error_m=_enu_calib_rms_error_m,
    )
    print(
        "[imu_gnss_pose] ENU Y-axis alignment updated: "
        f"baseline={baseline_m:.3f} m, "
        f"line_yaw={math.degrees(line_yaw_rad):.6f} deg, "
        f"delta_rot={math.degrees(delta_rot_rad):.6f} deg, "
        f"total_rot={summary.rotation_deg:.6f} deg"
    )
    return summary


def define_local_frame_from_two_geodetic_points(
    origin_lat_deg: float,
    origin_lon_deg: float,
    axis_lat_deg: float,
    axis_lon_deg: float,
    origin_height_m: Optional[float] = None,
    axis_height_m: Optional[float] = None,
) -> EnuCalibrationSummary:
    """
    用两个经纬度点建立当前跟踪用平面坐标系（小范围平面近似）：

    - 清除既有 ENU 旋转平移校准后重算；
    - 原点对应 origin 点（在位置平移后，该点在 get_robot_pose 下为 (0,0)）；
    - 从 origin 指向 axis 的水平方向对齐为 +Y 轴（与 align_enu_y_axis_with_points 一致）；
    - INS 航向会随平面旋转叠加同一旋转角，与路径坐标一致。

    高度未给定时采用 ENU 参考高程，仅影响尺度修正项，二维路径仍落在水平面。
    """
    clear_enu_calibration()

    client = _get_client()
    ref = client.get_reference()
    if ref is None:
        raise RuntimeError("ENU reference is not initialized yet; wait for GNSS fix")
    ref_lat, ref_lon, ref_h = ref

    h0 = float(origin_height_m) if origin_height_m is not None else float(ref_h)
    h1 = float(axis_height_m) if axis_height_m is not None else float(ref_h)

    ox, oy, _ = _geodetic_to_enu_raw(
        float(origin_lat_deg),
        float(origin_lon_deg),
        h0,
        ref_lat,
        ref_lon,
        ref_h,
    )
    ax, ay, _ = _geodetic_to_enu_raw(
        float(axis_lat_deg),
        float(axis_lon_deg),
        h1,
        ref_lat,
        ref_lon,
        ref_h,
    )

    summary = align_enu_y_axis_with_points([(ox, oy), (ax, ay)])

    ox_c, oy_c, _ = _geodetic_to_enu(
        float(origin_lat_deg),
        float(origin_lon_deg),
        h0,
        ref_lat,
        ref_lon,
        ref_h,
    )
    set_position_origin_to_xy(ox_c, oy_c)
    print(
        "[imu_gnss_pose] Geodetic frame: origin LLA="
        f"({float(origin_lat_deg):.8f},{float(origin_lon_deg):.8f}) -> "
        f"plane (x0,y0)=({ox_c:.3f},{oy_c:.3f}) m after Y-align; "
        "get_robot_pose now relative to this landmark, not the vehicle snap from before."
    )
    return summary


def clear_enu_calibration() -> None:
    global _enu_calib_enabled
    global _enu_calib_rot_rad
    global _enu_calib_tx
    global _enu_calib_ty
    global _enu_calib_point_count
    global _enu_calib_rms_error_m

    _enu_calib_enabled = False
    _enu_calib_rot_rad = 0.0
    _enu_calib_tx = 0.0
    _enu_calib_ty = 0.0
    _enu_calib_point_count = 0
    _enu_calib_rms_error_m = None
    print("[imu_gnss_pose] ENU calibration cleared.")
    reset_navigation_position_origin()


def get_enu_calibration_summary() -> EnuCalibrationSummary:
    return EnuCalibrationSummary(
        enabled=_enu_calib_enabled,
        point_count=_enu_calib_point_count,
        rotation_deg=math.degrees(_enu_calib_rot_rad),
        translation_x_m=_enu_calib_tx,
        translation_y_m=_enu_calib_ty,
        rms_error_m=_enu_calib_rms_error_m,
    )


def set_yaw_offset_deg(offset_deg: float) -> None:
    """
    设置航向角固定偏置（度），用于修正 IMU 与车体轴线的小角度偏差。
    """
    global _yaw_offset_rad
    _yaw_offset_rad = _deg2rad(float(offset_deg))
    print(f"[imu_gnss_pose] Yaw offset set: {float(offset_deg):.3f} deg")


def get_status_summary() -> ImuStatusSummary:
    client = _get_client()
    return client.get_status_summary()


def get_ins_odometry_for_cluster_csv() -> Optional[Dict[str, float]]:
    """
    供 ARS40X Cluster CSV 采集脚本填充惯导字段（雷达 CAN 无法给出的量）。
    航向与平面坐标与 get_robot_pose() 使用同一套校准 / yaw offset；
    速度为 INSPVAXA 东北天水平速度模长 √(v_e²+v_n²)（m/s）。
    无有效 INS 时返回 None。
    """
    pose = get_robot_pose()
    if pose is None:
        return None
    client = _get_client()
    with client._lock:
        ins = client._ins
    if not ins:
        return None
    try:
        vn = float(ins["v_n"])
        ve = float(ins["v_e"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(vn) and math.isfinite(ve)):
        return None
    speed = math.hypot(ve, vn)
    return {
        "heading_deg": math.degrees(pose.yaw),
        "speed_mps": speed,
        "enu_x_m": pose.x,
        "enu_y_m": pose.y,
        "pitch_deg": float(ins.get("pitch_deg", float("nan"))),
        "roll_deg": float(ins.get("roll_deg", float("nan"))),
    }


# Cluster CSV（DRI Raw）：前进直线段落盘时可将第 2 列 R 写为「沿当前路径到终点的剩余距离」
# （由路径跟踪线程写入；无覆盖时仍用雷达几何斜距 hypot(DX,DY)）。
_cluster_csv_path_rem_lock = threading.Lock()
_cluster_csv_straight_path_remaining_m: Optional[float] = None


def set_cluster_csv_straight_path_remaining_m(dist_m: Optional[float]) -> None:
    """
    由 car_control.follow_path_with_pid 在控制周期内更新。
    dist_m 为 None 或非有限值时清除覆盖，Cluster 行 R 恢复为雷达几何距离。
    """
    global _cluster_csv_straight_path_remaining_m
    with _cluster_csv_path_rem_lock:
        if dist_m is None:
            _cluster_csv_straight_path_remaining_m = None
            return
        try:
            v = float(dist_m)
        except (TypeError, ValueError):
            _cluster_csv_straight_path_remaining_m = None
            return
        if not math.isfinite(v):
            _cluster_csv_straight_path_remaining_m = None
            return
        _cluster_csv_straight_path_remaining_m = max(0.0, v)


def get_cluster_csv_straight_path_remaining_m() -> Optional[float]:
    with _cluster_csv_path_rem_lock:
        if _cluster_csv_straight_path_remaining_m is None:
            return None
        return float(_cluster_csv_straight_path_remaining_m)


# 与「雷达帧」强制行对齐：仅在一次雷达 CSV 行写入前调用；行频=雷达帧频，不单独按 INS 插行。
_ins_csv_hold: Optional[Dict[str, float]] = None
_ins_csv_hold_wall_t: float = 0.0

_ins_hold_max_s_env = os.getenv("INS_CLUSTER_CSV_HOLD_MAX_S", "").strip()
try:
    INS_CLUSTER_CSV_HOLD_MAX_S: Optional[float] = (
        float(_ins_hold_max_s_env) if _ins_hold_max_s_env else None
    )
except ValueError:
    INS_CLUSTER_CSV_HOLD_MAX_S = None


def reset_ins_cluster_csv_row_hold() -> None:
    """新开一段 CSV 录制时清空惯导 hold，避免与上一段串值。"""
    global _ins_csv_hold, _ins_csv_hold_wall_t
    _ins_csv_hold = None
    _ins_csv_hold_wall_t = 0.0
    set_cluster_csv_straight_path_remaining_m(None)


def sample_ins_for_radar_csv_row() -> Optional[Dict[str, float]]:
    """
    仅在每一「雷达帧」写 CSV 行前调用一次：行频=雷达帧频，不按惯导频率单独增行。

    - 有新鲜 INS：更新缓存并返回。
    - 短时无 INS：在 hold 窗口内返回上一帧惯导，保证与雷达行数 1:1。
      环境变量 INS_CLUSTER_CSV_HOLD_MAX_S（秒）非空则超时后该帧惯导填 NaN；
      未设置则一直沿用上一次有效惯导直至再次出现 INS。
    """
    global _ins_csv_hold, _ins_csv_hold_wall_t
    fresh = get_ins_odometry_for_cluster_csv()
    now = time.time()
    if fresh is not None:
        _ins_csv_hold = dict(fresh)
        _ins_csv_hold_wall_t = now
        return dict(fresh)

    if _ins_csv_hold is None:
        return None

    if INS_CLUSTER_CSV_HOLD_MAX_S is not None:
        if (now - _ins_csv_hold_wall_t) > float(INS_CLUSTER_CSV_HOLD_MAX_S):
            return None

    return dict(_ins_csv_hold)


def shutdown_imu_client() -> None:
    """兼容旧代码：关闭 IMU 客户端及其接收线程。"""
    global _default_client
    if _default_client is not None:
        try:
            _default_client.stop()
            print("[imu_gnss_pose] IMU client stopped.")
        except Exception as e:
            print(f"[imu_gnss_pose] stop() 异常: {e}")
        finally:
            _default_client = None
