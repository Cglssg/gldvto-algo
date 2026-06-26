from .utils_class import LinkStatus, CommunicationType
import random
from .base_station import BaseStation
from .edge_server import EdgeServer
from typing import Tuple

class CommunicationLink:
    def __init__(self, link_id: str, base_station: BaseStation, edge_server: EdgeServer):
        self.link_id = link_id
        self.base_station = base_station
        self.edge_server = edge_server
        self.bandwidth = 1000
        self.latency = 0.01
        self.packet_loss_rate = random.uniform(0.0001, 0.005)
        self.jitter = random.uniform(0.001, 0.01)
        self.status = LinkStatus.CONNECTED
        self.technology = random.choice(list(CommunicationType))
        self.energy_consumption = 0.0
        self.power_consumption = 0.1
        self.idle_power = 0.05

    def establish(self):
        self.status = LinkStatus.CONNECTED
        self.base_station.connected_edge_servers.append(self.edge_server)

    def transmit(self, data_size: float) -> Tuple[float, float]:
        if self.status != LinkStatus.CONNECTED:
            return float('inf'), 0.0

        base_time = data_size * 8 / self.bandwidth
        channel_quality = self.base_station.calculate_channel_quality(self.edge_server)
        interference = (1 - channel_quality) * random.uniform(0.001, 0.01)
        actual_time = base_time + self.latency + random.uniform(0, self.jitter) + interference

        transmission_energy = self.power_consumption * (actual_time / 3600)
        self.energy_consumption += transmission_energy

        if random.random() < self.packet_loss_rate:
            actual_time *= 2
            transmission_energy *= 2

        return actual_time, transmission_energy

    def update_status(self, current_time: float):
        if random.random() < 0.01:
            if self.status == LinkStatus.CONNECTED:
                self.status = random.choice([LinkStatus.DISCONNECTED, LinkStatus.RECONNECTING])
            elif self.status == LinkStatus.DISCONNECTED:
                self.status = LinkStatus.RECONNECTING
            else:
                self.status = LinkStatus.CONNECTED

    def update_idle_energy(self, time_step: float):
        if self.status == LinkStatus.CONNECTED:
            idle_energy = self.idle_power * (time_step / 3600)
            self.energy_consumption += idle_energy

    def update_energy_consumption(self, time_step: float):
        self.update_idle_energy(time_step)