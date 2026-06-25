from .utils_class import TaskType, Position, CommunicationType, TaskStatus, ComputeLocation
# 导入随机数生成模块
import random
# 导入堆队列模块（用于实现优先队列）
import heapq
# 导入类型提示相关模块，用于指定列表、可选值、元组等类型
from typing import List, Optional, Tuple
# 从当前包的task模块导入Task类
from .task import Task
# 导入数学运算模块
import math
# 从当前包的utils_class模块导入所有内容（补充导入）
from .utils_class import *


# 定义Vehicle（车辆）类，用于模拟车辆的各种属性和行为
class Vehicle:
    # 类的初始化方法，用于创建Vehicle实例时初始化属性
    def __init__(self, vehicle_id: str, position: Tuple[float, float],
                 speed: float = 0, direction: float = 0, resource_management_algorithm=None):
        # 车辆基本标识与状态
        self.vehicle_id = vehicle_id  # 车辆唯一ID
        self.position = position  # 车辆当前位置 (x, y) 坐标
        self.speed = speed  # 车辆速度（默认0）
        self.direction = direction  # 车辆行驶方向（角度，默认0）
        self.base_station_id = None  # 当前连接的基站ID（初始未连接）

        # 能耗相关属性 - 移除移动能耗
        self.energy_consumption = 0.0  # 总能耗
        self.computation_energy = 0.0  # 计算能耗
        self.communication_energy = 0.0  # 通信能耗

        # 能耗系数（单位能耗率）
        self.computation_power = 0.1  # 计算功率（kW per TFLOPS）
        self.communication_power = 0.05  # 通信功率（kW per Mbps）

        # 计算资源（随机初始化范围）
        # self.cpu_capacity = random.randint(1, 4)  # CPU容量（0到4之间随机）
        self.cpu_capacity = 4
        # self.memory_capacity = random.randint(4, 16)  # 内存容量（4到16之间随机）
        self.memory_capacity = 4

        # 任务相关属性（仿照边缘服务器扩展）
        self.pending_tasks = []  # 待处理任务列表
        self.completed_tasks = []  # 已完成任务列表
        self.task_queue = []  # 本地任务优先队列（堆实现）
        self.processing_tasks = []  # 正在处理的任务列表
        # 剩余资源字典（仿照边缘服务器）
        self.remaining_resources = {
            'cpu': self.cpu_capacity,
            'memory': self.memory_capacity
        }

        # 随机选择通信类型（如V2X、5G等，从CommunicationType枚举中选择）
        self.communication_type = random.choice(list(CommunicationType))


        # 状态标识
        self.is_moving = False  # 是否正在移动（初始为否）
        self.network_status = False  # 是否处于基站联网范围内（初始为否）
        self.resource_utilization = 0.0  # 资源利用率（0-1之间，初始为0）

        # 使用资源
        self.cpu_usage = 0.0  # 当前被占用的CPU资源（核心：记录任务占用）
        self.memory_usage = 0.0  # 当前被占用的内存资源
        self.idle_power = 0.1  # 本地空闲功率（仿照边缘服务器）


    # 判断车辆本地是否能处理指定任务
    def can_handle_local_task(self, task: Task) -> bool:
        """检查本地剩余CPU和内存是否满足任务需求"""
        return (self.remaining_resources['cpu'] >= task.compute_demand and
                self.remaining_resources['memory'] >= task.memory_demand)

    # 【新增】调度任务到本地处理
    def schedule_local_task(self, task: Task, current_time: float,processing_time) -> bool:
        """
        将任务调度到车辆本地处理
        :param task: 待处理任务
        :param current_time: 当前时间
        :return: 是否调度成功
        """
        # 校验任务状态
        if task.status != TaskStatus.PENDING:
            return False

        # 检查本地资源是否足够
        if not self.can_handle_local_task(task):
            return False

        # 更新任务状态和属性
        task.status = TaskStatus.PROCESSING
        task.start_time = current_time
        task.assigned_vehicle = self.vehicle_id  # 记录分配的车辆ID
        task.compute_location = ComputeLocation.LOCAL  # 标记为本地计算
        task.transmission_delay = 0  # 本地处理无传输延迟

        # 计算本地处理时间
        processing_time = self.estimate_local_processing_time(task)
        # 计算任务完成时间
        # 需要将 processing_time 转换为对应的几个 step
        processing_time_by_step = processing_time // 1
        completion_time = current_time + processing_time_by_step

        # 任务优先级（1-5，数字越小优先级越高）
        priority = task.priority if hasattr(task, 'priority') else 1

        # 校验完成时间是否超过截止时间
        if completion_time > (task.deadline // 1):
            task.status = TaskStatus.FAILED
            return False

        # 加入本地优先队列：(优先级, 完成时间, 任务, 处理时间)
        heapq.heappush(self.task_queue, (priority, completion_time, task, processing_time))

        # 更新本地资源占用
        self.processing_tasks.append(task)
        self.remaining_resources['cpu'] -= task.compute_demand
        self.remaining_resources['memory'] -= task.memory_demand

        # 更新资源利用率
        self.update_resource_utilization()

        return True

    # 【新增】处理本地任务（检查并完成已到完成时间的任务）
    def process_local_tasks(self, current_time: float) -> List[Task]:
        """
        处理本地任务队列，完成已到时间的任务
        :param current_time: 当前时间
        :return: 本次完成的任务列表
        """
        completed_tasks = []

        # 循环检查任务队列，处理已到完成时间的任务
        while self.task_queue and self.task_queue[0][1] <= current_time:
            # 弹出队列中最早完成的任务
            priority, completion_time, task, processing_time = heapq.heappop(self.task_queue)

            # 更新任务状态和属性
            task.completion_time = current_time
            task.status = TaskStatus.COMPLETED
            task.processing_delay = processing_time
            task.total_delay = task.start_time + processing_time - task.submit_time

            # 计算任务消耗的能量
            task_energy = self.estimate_local_energy_consumption(task)
            task.energy_consumed = task_energy

            # 更新车辆能耗
            self.computation_energy += task_energy
            self.energy_consumption += task_energy

            # 将任务加入已完成列表
            self.completed_tasks.append(task)
            completed_tasks.append(task)
            self.processing_tasks.remove(task)

            # 释放本地资源
            self.remaining_resources['cpu'] += task.compute_demand
            self.remaining_resources['memory'] += task.memory_demand

        # 更新资源利用率
        self.update_resource_utilization()

        return completed_tasks

    # 【新增】更新本地资源利用率（仿照边缘服务器的update_resource_usage）
    def update_resource_utilization(self):
        """更新本地资源利用率和负载状态"""
        # 计算CPU和内存使用率
        self.cpu_usage = 1 - (self.remaining_resources['cpu'] / self.cpu_capacity)
        self.memory_usage = 1 - (self.remaining_resources['memory'] / self.memory_capacity)
        # 资源利用率（CPU和内存使用率的平均值）
        self.resource_utilization = (self.cpu_usage + self.memory_usage) / 2

    # 生成任务的方法，根据当前时间创建一个任务并添加到待处理列表
    def generate_task(self, current_time: float) -> 'Task':
        task_type = random.choice(list(TaskType))  # 随机选择任务类型

        # 根据任务类型设置不同的任务参数范围
        # if task_type == TaskType.PERCEPTION:
        #     compute_demand = random.randint(1, 3)  # 计算需求
        #     memory_demand = random.randint(1, 2)  # 内存需求
        #     data_size = random.randint(50, 1024)  # 数据大小（bit）
        #     delay_constraint = random.uniform(0.8, 2.0)  # 延迟约束
        # elif task_type == TaskType.NAVIGATION:
        #     compute_demand = random.randint(1, 2)
        #     memory_demand = random.randint(1, 2)
        #     data_size = random.randint(10, 1024)
        #     delay_constraint = random.uniform(0.8, 2.0)
        # else:  # 其他任务类型（如默认）
        #     compute_demand = random.randint(1, 3)
        #     memory_demand = random.randint(2, 8)
        #     data_size = random.randint(100, 1024)
        #     delay_constraint = random.uniform(0.2, 2.0)
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
        else:  # 其他任务类型（如默认）
            compute_demand = 3
            memory_demand = 2
            data_size = 128
            delay_constraint = random.uniform(0.2, 2.0)

        # 创建Task实例
        task = Task(
            task_id=f"task_{len(self.pending_tasks) + len(self.completed_tasks) + 1}",  # 任务ID（基于已有任务数生成）
            vehicle_id=self.vehicle_id,  # 所属车辆ID
            submit_time=current_time,  # 提交时间
            delay_constraint=delay_constraint,  # 延迟约束
            task_type=task_type,  # 任务类型
            compute_demand=compute_demand,  # 计算需求
            memory_demand=memory_demand,  # 内存需求
            data_size=data_size,  # 数据大小
            priority=random.randint(1, 5)  # 添加优先级属性
        )
        delay_constraint = 0.8 * (100/(100-current_time))
        task.delay_constraint = delay_constraint

        self.pending_tasks.append(task)  # 添加到待处理任务列表
        return task  # 返回生成的任务

    def _calculate_transmission_time_total(self, data_size: float, base_stations: List['BaseStation']) -> float:
        """通用方法：计算上下行总传输时间"""
        # 找到最近的基站
        nearest_bs = min(base_stations, key=lambda bs: self.calculate_distance(self.position, bs.position))
        # 传输时间（上行+下行）
        transmission_time_up = self.calculate_transmission_time(data_size, nearest_bs)
        # 假设结果数据为原始数据的10%，计算下行传输时间
        transmission_time_down = self.calculate_transmission_time(data_size * 0.1, nearest_bs)
        return transmission_time_up + transmission_time_down

    # 任务卸载决策方法，调用资源管理算法决定任务处理位置
    def offload_decision(self, task: 'Task', edge_servers: List['EdgeServer'],
                         base_stations: List['BaseStation'], cloud_server: 'CloudServer') -> Tuple[
        ComputeLocation, Optional['EdgeServer']]:
        """任务卸载方法"""

        # 调用资源管理算法获取决策结果
        return self.resource_management_algorithm(vehicle=self, task=task, edge_servers=edge_servers,
                                                  base_stations=base_stations, cloud_server=cloud_server)

    # 估算本地处理时间
    def estimate_local_processing_time(self, task: 'Task') -> float:
        available_compute = self.cpu_capacity * (1 - self.resource_utilization)  # 可用计算资源（考虑利用率）
        if available_compute <= 0:  # 无可用资源时，时间为无穷大
            return float('inf')
        return task.compute_demand / available_compute  # 处理时间 = 计算需求 / 可用计算资源

    # 本地能耗计算增加防御
    def estimate_local_energy_consumption(self, task: 'Task') -> float:
        processing_time = self.estimate_local_processing_time(task)
        if processing_time == float('inf'):
            return float('inf')
        # 能耗 = 处理时间 * 计算需求 * 计算功率
        energy = processing_time * self.computation_power * (1 / 1)
        return energy

    # 估算边缘处理时间
    def estimate_edge_processing_time(self, task: 'Task', edge_server: 'EdgeServer',
                                      base_stations: List['BaseStation']) -> float:
        total_transmission_time = self._calculate_transmission_time_total(task.data_size, base_stations)
        processing_time = edge_server.estimate_processing_time(task)
        return total_transmission_time + processing_time  # 传输上下行时间 + 处理时间

    # 估算云端处理时间
    def estimate_cloud_processing_time(self, task: 'Task', cloud_server: 'CloudServer',
                                       base_stations: List['BaseStation']) -> float:
        # 找到最近的基站
        nearest_bs = min(base_stations,
                         key=lambda bs: self.calculate_distance(self.position, bs.position))

        # 传输时间（上行+下行）
        transmission_time_up = self.calculate_transmission_time(task.data_size, nearest_bs)  # 上行传输时间
        # 假设结果数据为原始数据的10%，计算下行传输时间
        transmission_time_down = self.calculate_transmission_time(task.data_size * 0.1, nearest_bs)

        # 云端处理时间（云端处理更快，用云端计算容量计算）
        processing_time = task.compute_demand / cloud_server.compute_capacity

        # 总时间 = 上行 + 处理 + 下行 + 云端通信延迟
        return transmission_time_up + processing_time + transmission_time_down + cloud_server.communication_delay

    # 估算传输能耗
    def estimate_transmission_energy(self, data_size: float, base_stations: List['BaseStation']) -> float:
        # 找到最近的基站
        nearest_bs = min(base_stations,
                         key=lambda bs: self.calculate_distance(self.position, bs.position))
        transmission_time = self.calculate_transmission_time(data_size, nearest_bs)  # 传输时间
        # 能耗 = 传输时间 * 通信功率 *（数据大小/100）（数据大小归一化）
        energy = transmission_time * self.communication_power * (data_size / 100)
        return energy

    # 计算两个位置之间的距离（欧氏距离）
    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    # 计算传输时间
    def calculate_transmission_time(self, data_size: float, base_station: 'BaseStation') -> float:

        # 获取信道质量 & 计算信噪比 SNR
        channel_quality = base_station.calculate_channel_quality(self)
        snr_max = base_station.snr_max  # 基站最大信噪比
        snr = snr_max * channel_quality
        # snr = base_station.calculate_snr(self)
        snr = max(1e-5, snr)

        # 香农公式计算信道容量 & 基础时延
        B = base_station.wireless_bandwidth  # 模拟信道带宽 Hz
        channel_capacity = B * math.log2(1 + snr)
        base_delay = data_size / channel_capacity

        # 信道干扰项（保留你原有逻辑）
        beta = random.uniform(0.1, 0.5)
        channel_interference = (1 - channel_quality) * beta * base_delay

        # 总传输时延
        total_delay = base_delay + channel_interference
        return total_delay

    # 更新能耗（基于时间步长）
    def update_energy_consumption(self, time_step: float):
        # 只保留计算能耗，移除移动能耗
        # 计算能耗 = 资源利用率 * CPU容量 * 计算功率 *（时间步长/3600）（时间单位转换为小时）
        if not self.pending_tasks and not self.completed_tasks:
            computation_energy = 0.0  # 无任务时无计算能耗
        else:
            computation_energy = self.resource_utilization * self.cpu_capacity * self.computation_power * (
                    time_step / 3600)
        self.computation_energy += computation_energy
        self.energy_consumption += computation_energy

    # 添加通信能耗
    def add_communication_energy(self, energy: float):
        self.communication_energy += energy  # 更新通信能耗
        self.energy_consumption += energy  # 更新总能耗

    # 车辆移动方法（根据速度和方向更新位置）
    def move(self, time_step: float = 1.0):
        if self.speed > 0:  # 速度大于0时才移动
            # 计算x方向位移：速度（km/h）* 时间步长（秒）/3600（转为小时）* cos(方向角弧度)
            dx = self.speed * time_step / 3600 * math.cos(math.radians(self.direction))
            # 计算y方向位移：速度（km/h）* 时间步长（秒）/3600（转为小时）* sin(方向角弧度)
            dy = self.speed * time_step / 3600 * math.sin(math.radians(self.direction))

            # 更新位置
            self.position = (self.position[0] + dx, self.position[1] + dy)
            self.is_moving = True  # 标记为正在移动
        else:
            self.is_moving = False  # 速度为0时，标记为未移动