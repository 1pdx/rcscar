# pi_power.py
"""
树莓派供电影响监测 (基于 INA219 / I2C)

- 在树莓派上使用 I2C 总线 + INA219 采集电压、电流和估算电量百分比
- 在非树莓派平台 (如 Windows 开发机) 上不会报错，只会标记为不可用
"""
from dataclasses import dataclass
from typing import Optional

try:
    import smbus  # 仅在树莓派上存在
except ImportError:
    smbus = None

# 寄存器定义
_REG_CONFIG = 0x00
_REG_SHUNTVOLTAGE = 0x01
_REG_BUSVOLTAGE = 0x02
_REG_POWER = 0x03
_REG_CURRENT = 0x04
_REG_CALIBRATION = 0x05


class BusVoltageRange:
    RANGE_16V = 0x00  # 16 V
    RANGE_32V = 0x01  # 32 V


class Gain:
    DIV_1_40MV = 0x00
    DIV_2_80MV = 0x01
    DIV_4_160MV = 0x02
    DIV_8_320MV = 0x03


class ADCResolution:
    ADCRES_9BIT_1S = 0x00
    ADCRES_10BIT_1S = 0x01
    ADCRES_11BIT_1S = 0x02
    ADCRES_12BIT_1S = 0x03
    ADCRES_12BIT_2S = 0x09
    ADCRES_12BIT_4S = 0x0A
    ADCRES_12BIT_8S = 0x0B
    ADCRES_12BIT_16S = 0x0C
    ADCRES_12BIT_32S = 0x0D
    ADCRES_12BIT_64S = 0x0E
    ADCRES_12BIT_128S = 0x0F


class Mode:
    POWERDOW = 0x00
    SVOLT_TRIGGERED = 0x01
    BVOLT_TRIGGERED = 0x02
    SANDBVOLT_TRIGGERED = 0x03
    ADCOFF = 0x04
    SVOLT_CONTINUOUS = 0x05
    BVOLT_CONTINUOUS = 0x06
    SANDBVOLT_CONTINUOUS = 0x07


class INA219:
    """
    你的原始 INA219 驱动略做整理，保留原有校准参数。
    """

    def __init__(self, i2c_bus: int = 1, addr: int = 0x40):
        if smbus is None:
            raise RuntimeError("smbus 未找到，当前环境不支持 I2C (可能不是树莓派)。")
        self.bus = smbus.SMBus(i2c_bus)
        self.addr = addr

        self._cal_value = 0
        self._current_lsb = 0.0
        self._power_lsb = 0.0

        # 根据你原脚本默认使用 16V/5A 的校准
        self.set_calibration_16V_5A()

    def read(self, address: int) -> int:
        data = self.bus.read_i2c_block_data(self.addr, address, 2)
        return (data[0] << 8) + data[1]

    def write(self, address: int, data: int) -> None:
        temp = [0, 0]
        temp[1] = data & 0xFF
        temp[0] = (data >> 8) & 0xFF
        self.bus.write_i2c_block_data(self.addr, address, temp)

    def set_calibration_16V_5A(self) -> None:
        """
        16V / 5A 范围，参考你原脚本中的参数。
        """
        self._current_lsb = 0.1524
        self._cal_value = 26868
        self._power_lsb = 0.003048

        self.write(_REG_CALIBRATION, self._cal_value)

        self.bus_voltage_range = BusVoltageRange.RANGE_16V
        self.gain = Gain.DIV_2_80MV
        self.bus_adc_resolution = ADCResolution.ADCRES_12BIT_32S
        self.shunt_adc_resolution = ADCResolution.ADCRES_12BIT_32S
        self.mode = Mode.SANDBVOLT_CONTINUOUS

        config = (
            (self.bus_voltage_range << 13)
            | (self.gain << 11)
            | (self.bus_adc_resolution << 7)
            | (self.shunt_adc_resolution << 3)
            | self.mode
        )
        self.write(_REG_CONFIG, config)

    def get_shunt_voltage_mV(self) -> float:
        self.write(_REG_CALIBRATION, self._cal_value)
        value = self.read(_REG_SHUNTVOLTAGE)
        if value > 32767:
            value -= 65535
        return value * 0.01

    def get_bus_voltage_V(self) -> float:
        self.write(_REG_CALIBRATION, self._cal_value)
        # 先读一次丢弃，符合芯片要求
        self.read(_REG_BUSVOLTAGE)
        return (self.read(_REG_BUSVOLTAGE) >> 3) * 0.004

    def get_current_mA(self) -> float:
        value = self.read(_REG_CURRENT)
        if value > 32767:
            value -= 65535
        return value * self._current_lsb

    def get_power_W(self) -> float:
        self.write(_REG_CALIBRATION, self._cal_value)
        value = self.read(_REG_POWER)
        if value > 32767:
            value -= 65535
        return value * self._power_lsb


@dataclass
class PiPowerStatus:
    voltage: float      # V
    current: float      # A
    power: float        # W
    percent: float      # 0~100


class PiPowerMonitor:
    """
    树莓派供电电量监控封装：

    - 正常：每次 read_status() 读一次 INA219，返回电压/电流/功率/百分比
    - 非树莓派或 I2C 不可用时：available=False，read_status() 返回 None
    """

    def __init__(
        self,
        i2c_bus: int = 1,
        addr: int = 0x41,
        v_min: float = 9.0,
        v_max: float = 12.6,
    ) -> None:
        """
        :param v_min: 电压百分比 0% 时对应的电压 (例如 9.0 V)
        :param v_max: 电压百分比 100% 时对应的电压 (例如 12.6 V)
        """
        self.v_min = v_min
        self.v_max = v_max
        self.available = False
        self._ina: Optional[INA219] = None

        if smbus is None:
            print("[PiPowerMonitor] smbus 未找到，树莓派电量监控不可用。")
            return

        try:
            self._ina = INA219(i2c_bus=i2c_bus, addr=addr)
            self.available = True
        except Exception as e:
            print(f"[PiPowerMonitor] 初始化 INA219 失败: {e}")
            self._ina = None
            self.available = False

    def read_status(self) -> Optional[PiPowerStatus]:
        if not self.available or self._ina is None:
            return None
        try:
            bus_voltage = self._ina.get_bus_voltage_V()
            shunt_voltage = self._ina.get_shunt_voltage_mV() / 1000.0
            current_mA = self._ina.get_current_mA()
            power_W = self._ina.get_power_W()

            # 你原脚本的百分比计算方法
            p = (bus_voltage - self.v_min) / (self.v_max - self.v_min) * 100.0
            if p > 100.0:
                p = 100.0
            if p < 0.0:
                p = 0.0

            return PiPowerStatus(
                voltage=bus_voltage,
                current=current_mA / 1000.0,
                power=power_W,
                percent=p,
            )
        except Exception as e:
            print(f"[PiPowerMonitor] 读取 INA219 失败: {e}")
            return None
