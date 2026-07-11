import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from .edge_server import EdgeServer
from .task import Task
from .utils_class import *
import heapq

class CloudServer:
    def __init__(self):
        self.compute_capacity = 1000.0
        self.storage_capacity = 10000
        self.communication_delay = 0.05
        self.energy_consumption = 0.0
        self.power_consumption = 5.0
        self.power_communication = 1
        self.cloud_tasks = []

    def schedule_task(self, task: Task, current_time: float, transmission_delay: float = 0):
        if not task.validate(current_time):
            return False

        task.status = TaskStatus.PROCESSING
        task.start_time = current_time
        task.compute_location = ComputeLocation.CLOUD
        task.transmission_delay = transmission_delay
        priority = task.priority

        processing_time = task.compute_demand / self.compute_capacity
        processing_time_by_step = processing_time // 1
        transmission_delay_by_step = transmission_delay // 1
        communication_delay_by_step = self.communication_delay // 1
        completion_time = current_time + processing_time_by_step + transmission_delay_by_step + communication_delay_by_step

        if completion_time > task.deadline:
            task.status = TaskStatus.FAILED
            return False

        heapq.heappush(self.cloud_tasks, (priority,completion_time, task, processing_time,transmission_delay))
        return True

    def process_task(self, current_time: float):
        completed_tasks = []

        while self.cloud_tasks and self.cloud_tasks[0][0] <= current_time:
            priority,completion_time, task, processing_time,transmission_delay = heapq.heappop(self.cloud_tasks)
            task.completion_time = completion_time
            task.status = TaskStatus.COMPLETED
            task.processing_delay = processing_time
            task.total_delay = task.start_time + processing_time - task.submit_time

            task_energy = self.estimate_task_energy(task, processing_time,transmission_delay)
            task.energy_consumed = task_energy
            self.energy_consumption += task_energy

            completed_tasks.append(task)

        return completed_tasks

    def estimate_processing_time(self, task: Task) -> float:
        return task.compute_demand / self.compute_capacity

    def estimate_task_energy(self, task: Task, processing_time: float,transmission_time: float = 0) -> float:
        computation_energy = task.compute_demand * processing_time
        communication_energy = self.power_communication * transmission_time
        return computation_energy + communication_energy

    def train_model(self, data_batch: List[Dict]) -> Dict:
        training_energy = self.power_consumption * 0.1
        self.energy_consumption += training_energy
        return {'model_updated': True, 'timestamp': np.random.random(), 'energy_used': training_energy}

    def push_model(self, edge_servers: List[EdgeServer]):
        for edge_server in edge_servers:
            push_energy = self.power_consumption * 0.01
            self.energy_consumption += push_energy

    def update_idle_energy(self, time_step: float):
        idle_energy = self.power_consumption * 0.1 * time_step
        self.energy_consumption += idle_energy
        return idle_energy