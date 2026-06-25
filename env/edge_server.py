import random
import heapq
from typing import Tuple
from .task import Task
from .utils_class import *

class EdgeServer:
    def __init__(self, server_id: str, position: Tuple[float, float]):
        self.server_id = server_id
        self.position = position
        self.coverage_range = 2.5

        self.cpu_capacity = 16
        self.memory_capacity = 64
        self.storage_capacity = 4096
        self.bandwidth = 4000

        self.energy_consumption = 0.0
        self.idle_power = 0.5
        self.computation_power = 0.2
        self.communication_power = 0.01

        self.current_load = 0.0
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.connected_vehicles_count = 0
        self.remaining_resources = {
            'cpu': self.cpu_capacity,
            'memory': self.memory_capacity
        }

        self.task_queue = []
        self.processing_tasks = []
        self.completed_tasks = []

        self.communication_delay = random.uniform(0.01, 0.05)
        self.packet_loss_rate = random.uniform(0.001, 0.01)
        self.supported_protocols = [CommunicationType.V2X, CommunicationType.NR_V2X]

    def can_handle_task(self, task: Task) -> bool:
        return (self.remaining_resources['cpu'] >= task.compute_demand and
                self.remaining_resources['memory'] >= task.memory_demand)

    def schedule_task(self, task: Task, current_time: float, transmission_delay: float = 0):
        if task.status != TaskStatus.PENDING:
            return False
        if self.can_handle_task(task):
            priority = task.priority if hasattr(task, 'priority') else 1
            task.status = TaskStatus.PROCESSING
            task.start_time = current_time
            task.assigned_server = self.server_id
            task.compute_location = ComputeLocation.EDGE
            task.transmission_delay = transmission_delay

            processing_time = self.estimate_processing_time(task)
            processing_time_by_step = processing_time // 1
            transmission_delay_by_step = transmission_delay // 1
            completion_time = current_time + processing_time_by_step + transmission_delay_by_step

            if completion_time > (task.deadline // 1):
                task.status = TaskStatus.FAILED
                return False

            heapq.heappush(self.task_queue, (priority, completion_time, task, processing_time))

            self.processing_tasks.append(task)
            self.remaining_resources['cpu'] -= task.compute_demand
            self.remaining_resources['memory'] -= task.memory_demand

            self.update_resource_usage()
            return True
        return False

    def process_task(self, current_time: float):
        completed_tasks = []

        while self.task_queue and self.task_queue[0][1] <= current_time:

            priority, completion_time, task, processing_time = heapq.heappop(self.task_queue)
            task.completion_time = completion_time
            task.status = TaskStatus.COMPLETED
            task.processing_delay = processing_time
            task.total_delay = task.start_time + processing_time - task.submit_time

            task_energy = self.estimate_task_energy(task, processing_time,task.start_time - task.submit_time)
            task.energy_consumed = task_energy
            self.energy_consumption += task_energy

            self.completed_tasks.append(task)
            completed_tasks.append(task)

            self.remaining_resources['cpu'] += task.compute_demand
            self.remaining_resources['memory'] += task.memory_demand

            if task in self.processing_tasks:
                self.processing_tasks.remove(task)

        self.update_resource_usage()
        return completed_tasks

    def estimate_processing_time(self, task: Task) -> float:
        available_compute = self.cpu_capacity * (1 - self.cpu_usage)
        if available_compute <= 0:
            return float('inf')

        base_time = task.compute_demand / available_compute
        load_penalty = self.current_load * 0.5

        return base_time + load_penalty

    def estimate_task_energy(self, task: Task, processing_time: float,transmission_delay) -> float:
        computation_energy = self.computation_power * (processing_time / 1)
        communication_energy = self.communication_power * (transmission_delay / 1)

        return computation_energy + communication_energy

    def update_resource_usage(self):
        if self.remaining_resources['cpu'] / self.cpu_capacity > 1 :
            self.cpu_usage = 0
        else:
            self.cpu_usage = 1 - (self.remaining_resources['cpu'] / self.cpu_capacity)
        self.memory_usage = 1 - (self.remaining_resources['memory'] / self.memory_capacity)
        self.current_load = (self.cpu_usage + self.memory_usage) / 2

        self.connected_vehicles_count = len([t for t in self.task_queue if t[2].status == TaskStatus.PROCESSING])

    def update_idle_energy(self, time_step: float):
        idle_energy = self.idle_power * (time_step / 3600)
        self.energy_consumption += idle_energy
        return idle_energy