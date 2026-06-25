from .utils_class import TaskType, Position, CommunicationType, TaskStatus, ComputeLocation
import random
import heapq
from typing import List, Optional, Tuple
from .task import Task
import math
from .utils_class import *


class Vehicle:
    def __init__(self, vehicle_id: str, position: Tuple[float, float],
                 speed: float = 0, direction: float = 0, resource_management_algorithm=None):
        self.vehicle_id = vehicle_id
        self.position = position
        self.speed = speed
        self.direction = direction
        self.base_station_id = None

        self.energy_consumption = 0.0
        self.computation_energy = 0.0
        self.communication_energy = 0.0

        self.computation_power = 0.1
        self.communication_power = 0.05

        self.cpu_capacity = 4
        self.memory_capacity = 4

        self.pending_tasks = []
        self.completed_tasks = []
        self.task_queue = []
        self.processing_tasks = []
        self.remaining_resources = {
            'cpu': self.cpu_capacity,
            'memory': self.memory_capacity
        }

        self.communication_type = random.choice(list(CommunicationType))

        self.is_moving = False
        self.network_status = False
        self.resource_utilization = 0.0

        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.idle_power = 0.1

    def can_handle_local_task(self, task: Task) -> bool:
        return (self.remaining_resources['cpu'] >= task.compute_demand and
                self.remaining_resources['memory'] >= task.memory_demand)

    def schedule_local_task(self, task: Task, current_time: float,processing_time) -> bool:
        if task.status != TaskStatus.PENDING:
            return False

        if not self.can_handle_local_task(task):
            return False

        task.status = TaskStatus.PROCESSING
        task.start_time = current_time
        task.assigned_vehicle = self.vehicle_id
        task.compute_location = ComputeLocation.LOCAL
        task.transmission_delay = 0

        processing_time = self.estimate_local_processing_time(task)
        processing_time_by_step = processing_time // 1
        completion_time = current_time + processing_time_by_step

        priority = task.priority if hasattr(task, 'priority') else 1

        if completion_time > (task.deadline // 1):
            task.status = TaskStatus.FAILED
            return False

        heapq.heappush(self.task_queue, (priority, completion_time, task, processing_time))

        self.processing_tasks.append(task)
        self.remaining_resources['cpu'] -= task.compute_demand
        self.remaining_resources['memory'] -= task.memory_demand

        self.update_resource_utilization()

        return True

    def process_local_tasks(self, current_time: float) -> List[Task]:
        completed_tasks = []

        while self.task_queue and self.task_queue[0][1] <= current_time:
            priority, completion_time, task, processing_time = heapq.heappop(self.task_queue)

            task.completion_time = current_time
            task.status = TaskStatus.COMPLETED
            task.processing_delay = processing_time
            task.total_delay = task.start_time + processing_time - task.submit_time

            task_energy = self.estimate_local_energy_consumption(task)
            task.energy_consumed = task_energy

            self.computation_energy += task_energy
            self.energy_consumption += task_energy

            self.completed_tasks.append(task)
            completed_tasks.append(task)
            self.processing_tasks.remove(task)

            self.remaining_resources['cpu'] += task.compute_demand
            self.remaining_resources['memory'] += task.memory_demand

        self.update_resource_utilization()

        return completed_tasks

    def update_resource_utilization(self):
        self.cpu_usage = 1 - (self.remaining_resources['cpu'] / self.cpu_capacity)
        self.memory_usage = 1 - (self.remaining_resources['memory'] / self.memory_capacity)
        self.resource_utilization = (self.cpu_usage + self.memory_usage) / 2

    def generate_task(self, current_time: float) -> 'Task':
        task_type = random.choice(list(TaskType))

        if task_type == TaskType.PERCEPTION:
            compute_demand = 1
            memory_demand = 1
            data_size = 50
            delay_constraint = random.uniform(0.2, 2.0)
        elif task_type == TaskType.NAVIGATION:
            compute_demand = 2
            memory_demand = 2
            data_size = 512
            delay_constraint = random.uniform(0.2, 2.0)
        else:
            compute_demand = 3
            memory_demand = 2
            data_size = 128
            delay_constraint = random.uniform(0.2, 2.0)

        task = Task(
            task_id=f"task_{len(self.pending_tasks) + len(self.completed_tasks) + 1}",
            vehicle_id=self.vehicle_id,
            submit_time=current_time,
            delay_constraint=delay_constraint,
            task_type=task_type,
            compute_demand=compute_demand,
            memory_demand=memory_demand,
            data_size=data_size,
            priority=random.randint(1, 5)
        )
        delay_constraint = 0.8 * (100/(100-current_time))
        task.delay_constraint = delay_constraint

        self.pending_tasks.append(task)
        return task

    def _calculate_transmission_time_total(self, data_size: float, base_stations: List['BaseStation']) -> float:
        nearest_bs = min(base_stations, key=lambda bs: self.calculate_distance(self.position, bs.position))
        transmission_time_up = self.calculate_transmission_time(data_size, nearest_bs)
        transmission_time_down = self.calculate_transmission_time(data_size * 0.1, nearest_bs)
        return transmission_time_up + transmission_time_down

    def offload_decision(self, task: 'Task', edge_servers: List['EdgeServer'],
                         base_stations: List['BaseStation'], cloud_server: 'CloudServer') -> Tuple[
        ComputeLocation, Optional['EdgeServer']]:
        return self.resource_management_algorithm(vehicle=self, task=task, edge_servers=edge_servers,
                                                  base_stations=base_stations, cloud_server=cloud_server)

    def estimate_local_processing_time(self, task: 'Task') -> float:
        available_compute = self.cpu_capacity * (1 - self.resource_utilization)
        if available_compute <= 0:
            return float('inf')
        return task.compute_demand / available_compute

    def estimate_local_energy_consumption(self, task: 'Task') -> float:
        processing_time = self.estimate_local_processing_time(task)
        if processing_time == float('inf'):
            return float('inf')
        energy = processing_time * self.computation_power * (1 / 1)
        return energy

    def estimate_edge_processing_time(self, task: 'Task', edge_server: 'EdgeServer',
                                      base_stations: List['BaseStation']) -> float:
        total_transmission_time = self._calculate_transmission_time_total(task.data_size, base_stations)
        processing_time = edge_server.estimate_processing_time(task)
        return total_transmission_time + processing_time

    def estimate_cloud_processing_time(self, task: 'Task', cloud_server: 'CloudServer',
                                       base_stations: List['BaseStation']) -> float:
        nearest_bs = min(base_stations,
                         key=lambda bs: self.calculate_distance(self.position, bs.position))

        transmission_time_up = self.calculate_transmission_time(task.data_size, nearest_bs)
        transmission_time_down = self.calculate_transmission_time(task.data_size * 0.1, nearest_bs)

        processing_time = task.compute_demand / cloud_server.compute_capacity

        return transmission_time_up + processing_time + transmission_time_down + cloud_server.communication_delay

    def estimate_transmission_energy(self, data_size: float, base_stations: List['BaseStation']) -> float:
        nearest_bs = min(base_stations,
                         key=lambda bs: self.calculate_distance(self.position, bs.position))
        transmission_time = self.calculate_transmission_time(data_size, nearest_bs)
        energy = transmission_time * self.communication_power * (data_size / 100)
        return energy

    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def calculate_transmission_time(self, data_size: float, base_station: 'BaseStation') -> float:
        channel_quality = base_station.calculate_channel_quality(self)
        snr_max = base_station.snr_max
        snr = snr_max * channel_quality
        snr = max(1e-5, snr)

        B = base_station.wireless_bandwidth
        channel_capacity = B * math.log2(1 + snr)
        base_delay = data_size / channel_capacity

        beta = random.uniform(0.1, 0.5)
        channel_interference = (1 - channel_quality) * beta * base_delay

        total_delay = base_delay + channel_interference
        return total_delay

    def update_energy_consumption(self, time_step: float):
        if not self.pending_tasks and not self.completed_tasks:
            computation_energy = 0.0
        else:
            computation_energy = self.resource_utilization * self.cpu_capacity * self.computation_power * (
                    time_step / 3600)
        self.computation_energy += computation_energy
        self.energy_consumption += computation_energy

    def add_communication_energy(self, energy: float):
        self.communication_energy += energy
        self.energy_consumption += energy

    def move(self, time_step: float = 1.0):
        if self.speed > 0:
            dx = self.speed * time_step / 3600 * math.cos(math.radians(self.direction))
            dy = self.speed * time_step / 3600 * math.sin(math.radians(self.direction))

            self.position = (self.position[0] + dx, self.position[1] + dy)
            self.is_moving = True
        else:
            self.is_moving = False