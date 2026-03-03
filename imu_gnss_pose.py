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
from typing import Optional, Dict, Any, Tuple


# WGS84 椭球参数（用于将经纬度差换算成米）
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


def _geodetic_to_enu(
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

    x = dlon * cos_ref * (rn + h_m)
    y = dlat * (rm + h_m)
    z = h_m - ref_h_m
    return x, y, z


# ======================== 统一位姿/状态结构体 ========================

@dataclass
class PoseSolution:
    """
    小车统一位姿输出（ENU 局部坐标）：

    source:
      "INS"   - INSPVAXA 融合解
    yaw:
      以东向为 0，逆时针为正（rad），与 main_ui 轨迹坐标系一致。
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

    has_inspvax: bool
    age_inspvax: Optional[float]     # 距离上一次 INSPVAXA 的时间（秒）
    freq_inspvax: Optional[float]    # 估算 INSPVAXA 频率（Hz）


# ======================== 核心客户端 ========================

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

        # 频率估算（固定 20 Hz）
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
                has_inspvax=self._t_inspvax is not None,
                age_inspvax=age_insp,
                freq_inspvax=self._freq_inspvax,
            )

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
            self._freq_inspvax = 20.0
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
_yaw_offset_rad = 0.0

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
        return replace(
            pose,
            x=pose.x - _calib_x,
            y=pose.y - _calib_y,
            yaw=_wrap_angle_rad(pose.yaw - _calib_yaw + _yaw_offset_rad),
        )
    if _yaw_offset_rad != 0.0:
        return replace(pose, yaw=_wrap_angle_rad(pose.yaw + _yaw_offset_rad))
    return pose


def calibrate_pose_to_current() -> bool:
    """
    将当前位姿设置为原点 (0,0,0)，用于试验起点校准。
    """
    global _calib_enabled, _calib_x, _calib_y, _calib_yaw
    client = _get_client()
    pose = client.get_pose()
    if pose is None:
        return False

    _calib_x = pose.x
    _calib_y = pose.y
    _calib_yaw = pose.yaw
    _calib_enabled = True
    print(
        f"[imu_gnss_pose] Calibration set at "
        f"x0={_calib_x:.3f}, y0={_calib_y:.3f}, yaw0={_calib_yaw:.3f} rad"
    )
    return True


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
