import random
from typing import List, Tuple
from .vehicle import Vehicle
import math

class BaseStation:
    # 物理常量
    BOLTZMANN = 1.38e-23    # 玻尔兹曼常数 J/K
    TEMP = 290              # 室温 290K
    LIGHT_SPEED = 3e8       # 光速 m/s
    CARRIER_FREQ = 2.4e9    # 2.4GHz载波频率

    def __init__(self, bs_id: str, position: Tuple[float, float], coverage_radius: float = 2.0):
        self.bs_id = bs_id
        self.position = position
        self.coverage_radius = coverage_radius  # 基站覆盖半径(km/m，仿真自定义单位)
        self.bandwidth = random.uniform(500, 1500)
        self.wireless_bandwidth = 1e5           # 无线信道带宽 Hz
        self.snr_max = 40                       # 最大SNR上限 dB
        self.transmit_power = 0.1               # 基站发射功率 W
        self.connected_vehicles = []
        self.connected_edge_servers = []
        self.energy_consumption = 0.0
        self.power_consumption = 1.0
        self.communication_power = 0.5

    def can_connect(self, vehicle: Vehicle) -> bool:
        distance = math.sqrt(
            (vehicle.position[0] - self.position[0]) ** 2 +
            (vehicle.position[1] - self.position[1]) ** 2
        )
        return distance <= self.coverage_radius

    def add_vehicle(self, vehicle: Vehicle):
        if vehicle not in self.connected_vehicles and self.can_connect(vehicle):
            self.connected_vehicles.append(vehicle)
            vehicle.base_station_id = self.bs_id

    def remove_vehicle(self, vehicle: Vehicle):
        if vehicle in self.connected_vehicles:
            self.connected_vehicles.remove(vehicle)

    def calculate_channel_quality(self, vehicle: Vehicle) -> float:
        distance = math.sqrt(
            (vehicle.position[0] - self.position[0]) ** 2 +
            (vehicle.position[1] - self.position[1]) ** 2
        )
        if distance > self.coverage_radius:
            return 0.0
        distance_ratio = 1 - (distance / self.coverage_radius)
        noise = random.uniform(-0.1, 0.1)
        # 10%概率突发强干扰，90%轻微干扰
        if random.random() < 0.1:
            interference = random.uniform(0.2, 0.5)
        else:
            interference = random.uniform(0, 0.1)
        channel_quality = distance_ratio + noise - interference
        return max(0.0, min(1.0, channel_quality))

    def calculate_received_power(self, vehicle: Vehicle) -> float:
        distance = math.sqrt(
            (vehicle.position[0] - self.position[0]) ** 2 +
            (vehicle.position[1] - self.position[1]) ** 2
        )
        if distance == 0:
            return vehicle.communication_power
        # 自由空间路径损耗 FSPL
        path_loss = (4 * math.pi * distance * self.CARRIER_FREQ / self.LIGHT_SPEED) ** 2
        received_power = vehicle.communication_power / path_loss
        # 兜底最小接收功率，防止数值过小
        return max(received_power, 1e-15)

    def calculate_noise_power(self) -> float:
        # 热噪声功率 Pn = k*T*B
        thermal_noise = self.BOLTZMANN * self.TEMP * self.wireless_bandwidth
        return thermal_noise

    def calculate_snr(self, vehicle: Vehicle) -> float:
        # 1. 获取接收信号功率、热噪声功率
        p_r = self.calculate_received_power(vehicle)
        p_n = self.calculate_noise_power()

        # 2. 随机同频干扰功率，干扰为噪声功率倍数
        if random.random() < 0.1:
            interference_power = random.uniform(0.2, 0.5) * p_n
        else:
            interference_power = random.uniform(0, 0.1) * p_n

        # 3. 总基底功率 = 热噪声 + 外部干扰
        total_noise = p_n + interference_power
        snr_linear = p_r / total_noise

        # 4. 转换dB，负数值兜底-90dB(工程合理下限)
        if snr_linear > 0:
            snr_dB = 10 * math.log10(snr_linear)
        else:
            snr_dB = -90

        # 限制SNR不超过硬件最大上限
        snr_dB = min(snr_dB, self.snr_max)
        # 转回线性值供香农公式使用
        snr_linear_clamped = 10 ** (snr_dB / 10)
        return snr_linear_clamped

    def update_energy_consumption(self, time_step: float, data_throughput: float = 0):
        communication_energy = data_throughput * self.communication_power * (time_step / 3600)
        base_energy = self.power_consumption * (time_step / 3600)
        time_step_energy = base_energy + communication_energy
        self.energy_consumption += time_step_energy
        return time_step_energy