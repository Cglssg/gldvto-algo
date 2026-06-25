# 从当前包的utils_class模块中导入TaskType和TaskStatus两个类
from .utils_class import TaskType, TaskStatus
# 从typing模块导入Dict类型，用于类型注解
from typing import Dict

# 定义Task类，用于表示一个任务的相关信息和操作
class Task:
    # 类的初始化方法，用于创建Task实例时初始化属性
    def __init__(self, task_id: str, vehicle_id: str, submit_time: float,
                 delay_constraint: float, task_type: TaskType, compute_demand: float,
                 memory_demand: float, data_size: float,priority = 1):
        self.task_id = task_id  # 任务唯一标识
        self.vehicle_id = vehicle_id  # 提交任务的车辆标识
        self.submit_time = submit_time  # 任务提交时间
        self.delay_constraint = delay_constraint  # 任务延迟约束（最大允许延迟时间）
        self.deadline = submit_time + delay_constraint  # 任务截止时间（提交时间+延迟约束）
        self.task_type = task_type  # 任务类型（TaskType枚举类型）
        self.compute_demand = compute_demand  # 计算需求（计算资源消耗量）
        self.memory_demand = memory_demand  # 内存需求（内存资源消耗量）
        self.data_size = data_size  # 数据大小（任务相关数据的大小 bit）
        self.status = TaskStatus.PENDING  # 任务状态，初始为待处理（PENDING）
        self.start_time = None  # 任务开始处理时间，初始为None
        self.completion_time = None  # 任务完成时间，初始为None
        self.assigned_server = None  # 分配处理该任务的服务器，初始为None
        self.compute_location = None  # 计算位置（在哪里处理该任务），初始为None
        self.energy_consumed = 0.0  # 消耗的能量，初始为0.0
        self.processing_delay = 0.0  # 处理延迟，初始为0.0
        self.transmission_delay = 0.0  # 传输延迟，初始为0.0
        self.total_delay = 0.0  # 总延迟（处理延迟+传输延迟），初始为0.0
        self.priority = priority       # 任务优先权重（越大越优先）

    # 验证任务是否在截止时间内，返回布尔值表示是否有效
    def validate(self, current_time: float) -> bool:
        if current_time > self.deadline:  # 如果当前时间超过截止时间
            self.status = TaskStatus.TIMEOUT  # 将任务状态设为超时（TIMEOUT）
            return False  # 返回无效
        return True  # 未超时则返回有效

    # 将任务的关键信息打包成字典返回
    def pack_data(self) -> Dict:
        return {
            'task_id': self.task_id,
            'vehicle_id': self.vehicle_id,
            'task_type': self.task_type.value,  # 存储TaskType的枚举值
            'compute_demand': self.compute_demand,
            'memory_demand': self.memory_demand,
            'data_size': self.data_size,
            'deadline': self.deadline,
            'status': self.status.value,  # 状态
            'submit_time': self.submit_time,  # 提交时间
            'total_delay': self.total_delay  # 总延迟
        }

    # 解析任务处理结果数据，更新任务属性
    def parse_result(self, result_data: Dict):
        self.completion_time = result_data.get('completion_time')  # 从结果中获取完成时间
        self.energy_consumed = result_data.get('energy_consumed', 0.0)  # 获取消耗的能量，默认0.0
        self.processing_delay = result_data.get('processing_delay', 0.0)  # 获取处理延迟，默认0.0
        self.transmission_delay = result_data.get('transmission_delay', 0.0)  # 获取传输延迟，默认0.0
        self.total_delay = result_data.get('total_delay', 0.0)  # 获取总延迟，默认0.0

        # 根据完成时间和截止时间更新任务状态
        # 防御None值
        if self.completion_time is None:
            self.status = TaskStatus.FAILED  # 新增FAILED状态
            return
        if self.completion_time <= self.deadline:
            self.status = TaskStatus.COMPLETED  # 在截止时间前完成，状态设为已完成（COMPLETED）
        else:
            self.status = TaskStatus.TIMEOUT    # 未按时完成，状态设为超时（TIMEOUT）

    # 定义小于（<）比较规则，基于task_id（也可选择其他字段，如submit_time）
    def __lt__(self, other):
        if not isinstance(other, Task):
            return NotImplemented
        return self.task_id < other.task_id

    # 可选：补充其他比较方法（如__gt__），确保比较逻辑完整
    def __gt__(self, other):
        if not isinstance(other, Task):
            return NotImplemented
        return self.task_id > other.task_id

