# can_init.py
"""
CAN 接口初始化工具 (用于 Linux + socketcan)

典型用法:
    from can_init import init_can

    result = init_can(interface="can0", bitrate=500000)
    if result.ok:
        print("CAN 初始化成功:", result.message)
    else:
        print("CAN 初始化失败:", result.message)

注意:
    - 需要在 Linux 上运行 (树莓派 OK)
    - 如需在非 root 账号下自动初始化 CAN, 建议配置 /etc/sudoers 对 ip 命令免密码,
      并将 USE_SUDO=True.
"""

import platform
import subprocess
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

# 如果为 True, 所有 ip 命令前面自动加 "sudo"
# 使用前请确保 /etc/sudoers 已允许当前用户免密码执行 ip 命令
USE_SUDO = True


@dataclass
class CanInitResult:
    ok: bool            # True: 初始化成功; False: 失败或跳过
    interface: str      # 如 "can0"
    bitrate: int        # 标称波特率 (bit/s)
    message: str        # 状态 / 错误描述


def _run_cmd(args: List[str], use_sudo: bool = False) -> Tuple[int, str, str]:
    """
    内部工具: 运行子进程, 返回 (retcode, stdout, stderr)

    :param args: 不含 sudo 的命令参数列表, 例如 ["ip", "link", "show", "can0"]
    :param use_sudo: 为 True 时在前面自动加上 "sudo"
    """
    cmd = ["sudo"] + args if use_sudo else args
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def init_can(interface: str = "can0", bitrate: int = 500000) -> CanInitResult:
    """
    初始化 socketcan 接口:

      1) ip link show <interface>                  (# 检查接口存在)
      2) ip link set <interface> down
      3) ip link set <interface> type can bitrate <bitrate>
      4) ip link set <interface> up

    返回:
      - CanInitResult.ok = True 表示所有命令执行成功且接口处于 UP 状态
      - message 字段包含简要说明或错误信息

    注意:
      - 如 USE_SUDO=True, 请在 /etc/sudoers 配置免密码执行 ip, 否则可能出现
        "no tty present and no askpass program specified" 或卡住等待密码.
    """
    # 仅在 Linux 下尝试配置 socketcan
    if platform.system().lower() != "linux":
        return CanInitResult(
            ok=False,
            interface=interface,
            bitrate=bitrate,
            message="非 Linux 系统，跳过 socketcan 初始化 (无需使用 ip link)",
        )

    # 检查 ip 命令是否存在
    try:
        rc_ip, _, err_ip = _run_cmd(["ip", "-V"], use_sudo=False)
        if rc_ip != 0:
            return CanInitResult(
                ok=False,
                interface=interface,
                bitrate=bitrate,
                message=f"执行 ip 命令失败，请确认已安装 iproute2: {err_ip}",
            )
    except FileNotFoundError:
        return CanInitResult(
            ok=False,
            interface=interface,
            bitrate=bitrate,
            message="未找到 ip 命令，请安装 iproute2 后重试",
        )

    # 检查接口是否存在
    rc, out, err = _run_cmd(["ip", "link", "show", interface], use_sudo=False)
    if rc != 0:
        return CanInitResult(
            ok=False,
            interface=interface,
            bitrate=bitrate,
            message=f"接口 {interface} 不存在或未被系统识别: {err or out}",
        )

    # 先尝试 down (忽略失败)
    _run_cmd(["ip", "link", "set", interface, "down"], use_sudo=USE_SUDO)

    # 配置为 CAN 类型 + 波特率
    rc2, out2, err2 = _run_cmd(
        ["ip", "link", "set", interface, "type", "can", "bitrate", str(bitrate)],
        use_sudo=USE_SUDO,
    )
    if rc2 != 0:
        return CanInitResult(
            ok=False,
            interface=interface,
            bitrate=bitrate,
            message=f"设置 CAN 参数失败 (bitrate={bitrate}): {err2 or out2}",
        )

    # 拉起接口
    rc3, out3, err3 = _run_cmd(
        ["ip", "link", "set", interface, "up"],
        use_sudo=USE_SUDO,
    )
    if rc3 != 0:
        return CanInitResult(
            ok=False,
            interface=interface,
            bitrate=bitrate,
            message=f"接口 {interface} up 失败: {err3 or out3}",
        )

    # 再确认一下状态 (这里不用 sudo 也行, 只是读)
    rc4, out4, err4 = _run_cmd(["ip", "link", "show", interface], use_sudo=False)
    if rc4 != 0:
        return CanInitResult(
            ok=True,
            interface=interface,
            bitrate=bitrate,
            message="CAN 初始化命令已执行，但无法确认接口状态，请手动检查 ip link",
        )

    if "state UP" in out4:
        msg = f"接口 {interface} 已成功初始化为 CAN, bitrate={bitrate} bit/s, state=UP"
    else:
        msg = f"接口 {interface} 已配置为 CAN, 但当前状态非 UP: {out4}"

    return CanInitResult(
        ok=True,
        interface=interface,
        bitrate=bitrate,
        message=msg,
    )


# -------------------- ARS408 object decode helpers --------------------

def extract_motorola_u(data: bytes, start: int, length: int) -> int:
    """Motorola(big-endian) bit extract, DBC @0."""
    if length <= 0:
        return 0
    if len(data) != 8:
        data = data.ljust(8, b"\x00")[:8]

    byte = start // 8
    bit = start % 8  # 0=LSB, 7=MSB
    val = 0
    for _ in range(length):
        if byte < 0 or byte >= 8:
            break
        b = (data[byte] >> bit) & 0x1
        val = (val << 1) | int(b)
        if bit == 0:
            byte += 1
            bit = 7
        else:
            bit -= 1
    return val


def _phys(data: bytes, start: int, length: int, offset: float, res: float) -> float:
    return float(extract_motorola_u(data, start, length) * res + offset)


@dataclass
class ObjMeas:
    oid: int
    x: float
    y: float
    vx: float
    vy: float
    dyn: int
    rcs_db: float
    t: float


def decode_ars408_60A(data: bytes) -> Tuple[int, int]:
    """0x60A: Object_NofObjects + Object_MeasCounter."""
    b = data.ljust(4, b"\x00")[:4]
    nof = int(b[0])
    meas_counter = int.from_bytes(b[2:4], byteorder="big", signed=False)
    return nof, meas_counter


def decode_ars408_60B(
    data: bytes,
    roi_x: Tuple[float, float] = (0.0, 120.0),
    roi_y_abs: float = 30.0,
) -> Optional[ObjMeas]:
    """0x60B: Object_1_General -> ObjMeas."""
    oid = extract_motorola_u(data, 7, 8)
    x = _phys(data, 15, 13, -500.0, 0.2)
    y = _phys(data, 18, 11, -204.6, 0.2)
    vx = _phys(data, 39, 10, -128.0, 0.25)
    vy = _phys(data, 45, 9, -64.0, 0.25)
    dyn = extract_motorola_u(data, 50, 3)
    rcs = _phys(data, 63, 8, -64.0, 0.5)

    if roi_x is not None and not (roi_x[0] <= x <= roi_x[1]):
        return None
    if roi_y_abs is not None and abs(y) > roi_y_abs:
        return None

    return ObjMeas(
        oid=int(oid),
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        dyn=int(dyn),
        rcs_db=rcs,
        t=time.time(),
    )


class RadarObjectConverter:
    """Collects 0x60B objects and returns a list when 0x60A ends a cycle."""

    def __init__(
        self,
        roi_x: Tuple[float, float] = (0.0, 120.0),
        roi_y_abs: float = 30.0,
    ) -> None:
        self._roi_x = roi_x
        self._roi_y_abs = roi_y_abs
        self._cycle_objects: Dict[int, ObjMeas] = {}
        self.last_object_count: Optional[int] = None
        self.last_meas_counter: Optional[int] = None

    def reset(self) -> None:
        self._cycle_objects = {}
        self.last_object_count = None
        self.last_meas_counter = None

    def feed_frame(self, arbitration_id: int, data: bytes) -> List[ObjMeas]:
        if arbitration_id == 0x60B and len(data) == 8:
            m = decode_ars408_60B(data, roi_x=self._roi_x, roi_y_abs=self._roi_y_abs)
            if m is not None:
                self._cycle_objects[m.oid] = m
            return []

        if arbitration_id == 0x60A and len(data) >= 4:
            nof, meas_counter = decode_ars408_60A(data)
            self.last_object_count = nof
            self.last_meas_counter = meas_counter
            objects = list(self._cycle_objects.values())
            self._cycle_objects = {}
            return objects

        return []


def init_radar_object_converter(
    roi_x: Tuple[float, float] = (0.0, 120.0),
    roi_y_abs: float = 30.0,
) -> RadarObjectConverter:
    """Initialize a radar object converter (0x60A/0x60B -> ObjMeas)."""
    return RadarObjectConverter(roi_x=roi_x, roi_y_abs=roi_y_abs)
