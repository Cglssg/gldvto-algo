import random
from typing import List, Dict, Optional, Tuple
from .vehicle import Vehicle
import math

class BaseStation:
    def __init__(self, bs_id: str, position: Tuple[float, float], coverage_radius: float = 2.0):
        self.bs_id = bs_id
        self.position = position
        self.coverage_radius = coverage_radius
        self.bandwidth = random.uniform(500, 1500)

        # 模拟信道设置
        self.wireless_bandwidth = 1e5
        self.snr_max = 40
        self.transmit_power = 0.1

        self.connected_vehicles = []
        self.connected_edge_servers = []

        self.energy_consumption = 0.0
        self.power_consumption = 1.0

        self.communication_power = 0.5

    def can_connect(self, vehicle: Vehicle) -> bool:
        distance = math.sqrt(
            (vehicle.position[0] - self.position[0]) **2 +
            (vehicle.position[1] - self.position[1])** 2
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

        if random.random() < 0.1:
            interference = random.uniform(0.2, 0.5)
        else:
            interference = random.uniform(0, 0.1)

        channel_quality = distance_ratio + noise - interference
        return max(0.0, min(1.0, channel_quality))

    def calculate_received_power(self, vehicle: Vehicle) -> float:
        carrier_frequency = 2.4e9
        c = 3e8
        distance = math.sqrt(
            (vehicle.position[0] - self.position[0]) ** 2 +
            (vehicle.position[1] - self.position[1]) ** 2
        )
        if distance == 0:
            return self.transmit_power
        path_loss = (4 * math.pi * distance * carrier_frequency / c) ** 2
        received_power = self.transmit_power / path_loss
        # return self.transmit_power
        return received_power

    # 计算噪声功率
    def calculate_noise_power(self) -> float:
        return 1

    def calculate_snr(self, vehicle: Vehicle) -> float:
        p_r = self.calculate_received_power(vehicle)
        p_n = self.calculate_noise_power()

        if random.random() < 0.1:  # 10%概率突发干扰
            interference_power = random.uniform(0.2, 0.5) * p_n
        else:
            interference_power = random.uniform(0, 0.1) * p_n

        total_noise = p_n + interference_power
        snr_linear = p_r / total_noise
        snr_dB = 10 * math.log10(snr_linear) if snr_linear > 0 else -100
        snr_dB = min(snr_dB, self.snr_max)
        snr_linear_clamped = 10 ** (snr_dB / 10)
        return snr_linear_clamped

    def update_energy_consumption(self, time_step: float, data_throughput: float = 0):
        communication_energy = data_throughput * self.communication_power * (time_step / 3600)
        base_energy = self.power_consumption * (time_step / 3600)
        time_step_energy = base_energy + communication_energy
        self.energy_consumption += time_step_energy
        return time_step_energy