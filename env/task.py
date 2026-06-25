from .utils_class import TaskType, TaskStatus
from typing import Dict

class Task:
    def __init__(self, task_id: str, vehicle_id: str, submit_time: float,
                 delay_constraint: float, task_type: TaskType, compute_demand: float,
                 memory_demand: float, data_size: float,priority = 1):
        self.task_id = task_id
        self.vehicle_id = vehicle_id
        self.submit_time = submit_time
        self.delay_constraint = delay_constraint
        self.deadline = submit_time + delay_constraint
        self.task_type = task_type
        self.compute_demand = compute_demand
        self.memory_demand = memory_demand
        self.data_size = data_size
        self.status = TaskStatus.PENDING
        self.start_time = None
        self.completion_time = None
        self.assigned_server = None
        self.compute_location = None
        self.energy_consumed = 0.0
        self.processing_delay = 0.0
        self.transmission_delay = 0.0
        self.total_delay = 0.0
        self.priority = priority

    def validate(self, current_time: float) -> bool:
        if current_time > self.deadline:
            self.status = TaskStatus.TIMEOUT
            return False
        return True

    def pack_data(self) -> Dict:
        return {
            'task_id': self.task_id,
            'vehicle_id': self.vehicle_id,
            'task_type': self.task_type.value,
            'compute_demand': self.compute_demand,
            'memory_demand': self.memory_demand,
            'data_size': self.data_size,
            'deadline': self.deadline,
            'status': self.status.value,
            'submit_time': self.submit_time,
            'total_delay': self.total_delay
        }

    def parse_result(self, result_data: Dict):
        self.completion_time = result_data.get('completion_time')
        self.energy_consumed = result_data.get('energy_consumed', 0.0)
        self.processing_delay = result_data.get('processing_delay', 0.0)
        self.transmission_delay = result_data.get('transmission_delay', 0.0)
        self.total_delay = result_data.get('total_delay', 0.0)

        if self.completion_time is None:
            self.status = TaskStatus.FAILED
            return
        if self.completion_time <= self.deadline:
            self.status = TaskStatus.COMPLETED
        else:
            self.status = TaskStatus.TIMEOUT

    def __lt__(self, other):
        if not isinstance(other, Task):
            return NotImplemented
        return self.task_id < other.task_id

    def __gt__(self, other):
        if not isinstance(other, Task):
            return NotImplemented
        return self.task_id > other.task_id