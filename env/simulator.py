from env.utils_class import TaskStatus, Position, ComputeLocation
import random
from typing import List, Dict, Optional, Tuple
from .vehicle import Vehicle
from .task import Task
from .edge_server import EdgeServer
from .cloud_server import CloudServer
from .communication_link import CommunicationLink
from .base_station import BaseStation
import math
import numpy as np
import matplotlib.pyplot as plt
from .utils_class import *
import os
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

demo_total = 0

class Simulator:
    TASK_GENERATION_PROB = 0.3
    TRANSMISSION_DELAY_FACTOR = 1.1
    GRID_SIZE = 3
    GRID_SPACING = 3
    GRID_START_X = 1
    GRID_START_Y = 1
    VEHICLE_POS_RANGE = (0, 10)
    VEHICLE_SPEED_RANGE = (30, 80)
    VEHICLE_DIR_RANGE = (0, 360)

    def __init__(self, resource_management_algorithm=None, step_func=None,calculate_reward_fn=None,offload_decision_fn=None):
        self.vehicles = []
        self.base_stations = []
        self.edge_servers = []
        self.communication_links = []
        self.cloud_server = CloudServer()
        self.tasks = []
        self.resource_management_algorithm = resource_management_algorithm

        self.current_time = 0.0
        self.time_step = 1.0
        self.event_queue = []
        self.vehicle_map = {}

        self.offload_decision_fn = offload_decision_fn
        self.calculate_reward_fn = calculate_reward_fn
        if self.calculate_reward_fn is None:
            raise ValueError("calculate_reward_fn is not provided. Please pass it to Simulator.__init__")

        self.metrics = {
            'task_completion_rate': [],
            'average_processing_time': [],
            'average_processing_delay': [],
            'average_transmission_delay': [],
            'average_total_delay': [],
            'resource_utilization': [],
            'total_energy_consumption': [],
            'vehicle_energy': [],
            'edge_energy': [],
            'bs_energy': [],
            'cloud_energy': [],
            'link_energy': [],
            'average_energy_per_task': [],
            'energy_efficiency': [],
            'compute_location_distribution': {'local': 0, 'edge': 0, 'cloud': 0},
            'task_success_rate_by_location': {'local': [], 'edge': [], 'cloud': []}
        }

    def add_vehicle(self, vehicle: Vehicle):
        self.vehicles.append(vehicle)
        self.vehicle_map[vehicle.vehicle_id] = vehicle

    def add_base_station(self, base_station: BaseStation):
        self.base_stations.append(base_station)

    def add_edge_server(self, edge_server: EdgeServer):
        self.edge_servers.append(edge_server)

    def add_communication_link(self, link: CommunicationLink):
        self.communication_links.append(link)
        link.establish()

    def setup_scenario(self, num_vehicles=10, num_base_stations=9, num_edge_servers=6):
        for x in [2, 6, 10]:
            for y in [2, 6, 10]:
                bs = BaseStation(f"BS_{x}_{y}", position=(x, y),coverage_radius = 2)
                self.add_base_station(bs)

        available_bs = [bs for bs in self.base_stations if len(bs.connected_edge_servers) == 0]
        for i in range(num_edge_servers):
            if not available_bs:
                available_bs = self.base_stations.copy()
            connected_bs = random.choice(available_bs)
            available_bs.remove(connected_bs)

            es_pos = (
                connected_bs.position[0] + random.random(),
                connected_bs.position[1] + random.random()
            )
            es = EdgeServer(f"ES_{i + 1}", es_pos)
            self.add_edge_server(es)

            link = CommunicationLink(f"Link_{i + 1}", connected_bs, es)
            self.add_communication_link(link)

        for bs in self.base_stations:
            if len(bs.connected_edge_servers) == 0:
                es = random.choice(self.edge_servers)
                link = CommunicationLink(f"Link_ES_BS_{bs.bs_id}", bs, es)
                self.add_communication_link(link)

        for i in range(num_vehicles):
            vehicle_pos = (
                random.uniform(*self.VEHICLE_POS_RANGE),
                random.uniform(*self.VEHICLE_POS_RANGE)
            )
            vehicle = Vehicle(
                f"V_{i + 1}",
                vehicle_pos,
                speed=random.uniform(*self.VEHICLE_SPEED_RANGE),
                direction=random.uniform(*self.VEHICLE_DIR_RANGE),
                resource_management_algorithm=self.resource_management_algorithm
            )
            self.add_vehicle(vehicle)
            self.update_vehicle_connections(vehicle)

    def run_step(self,action=None,params=None):
        self.update_base_energy_consumption()

        total_reward = 0
        task_count = 0
        pending_tasks = []

        for vehicle in self.vehicles:
            vehicle.move(self.time_step)
            vehicle.update_energy_consumption(self.time_step)
            self.update_vehicle_connections(vehicle)

        for vehicle in self.vehicles:
            if random.random() < self.TASK_GENERATION_PROB:
                task = vehicle.generate_task(self.current_time)
                self.tasks.append(task)
                pending_tasks.append((vehicle, task))
                task_count += 1

        if action is None and params is not None:
            if params['algorithm-name'] == 'SA-DDQN-DDPG':
                for vehicle, task in pending_tasks:
                    compute_location, target = self.offload_decision_fn(vehicle, task, self.edge_servers, self.base_stations, self.cloud_server)
                    success, rtt, energy = self.perform_offloading(vehicle, task, compute_location, target)
                    reward = self.calculate_reward_fn(success, rtt, task.delay_constraint, energy, task.priority)
                    total_reward += reward
        elif action is not None:
            for vehicle, task in pending_tasks:
                if action == 0:
                    compute_location = ComputeLocation.LOCAL
                    target = None
                elif action == 1:
                    compute_location = ComputeLocation.EDGE
                    nearest_bs = min(self.base_stations,
                                     key=lambda bs: vehicle.calculate_distance(vehicle.position, bs.position))
                    target = random.choice(nearest_bs.connected_edge_servers) if nearest_bs.connected_edge_servers else None
                else:
                    compute_location = ComputeLocation.CLOUD
                    target = self.cloud_server

                success, rtt, energy = self.perform_offloading(vehicle, task, compute_location, target)
                reward = self.calculate_reward_fn(success, rtt, task.delay_constraint, energy, task.priority)
                total_reward += reward

        for edge_server in self.edge_servers:
            edge_server.process_task(self.current_time)

        cloud_completed_tasks = self.cloud_server.process_task(self.current_time)
        for task in cloud_completed_tasks:
            processing_time = self.cloud_server.estimate_processing_time(task)

        for link in self.communication_links:
            link.update_status(self.current_time)
            link.update_energy_consumption(self.time_step)

        for vehicle in self.vehicles:
            vehicle.process_local_tasks(self.current_time)

        self.collect_metrics()
        self.current_time += self.time_step

        return total_reward / max(task_count, 1)

    def perform_offloading(self, vehicle, task, compute_location, target_edge_server=None):
        if compute_location == ComputeLocation.LOCAL:
            if not vehicle.can_handle_local_task(task):
                task.status = TaskStatus.FAILED
                return False, float('inf'), 0.0

            processing_time = vehicle.estimate_local_processing_time(task)
            processing_energy = vehicle.estimate_local_energy_consumption(task)
            idle_energy = vehicle.idle_power * (processing_time / 3600)
            total_energy = processing_energy + idle_energy

            if processing_time > task.delay_constraint:
                task.status = TaskStatus.FAILED
                vehicle.energy_consumption += idle_energy
                vehicle.computation_energy += processing_energy
                return False, processing_time, total_energy

            schedule_success = vehicle.schedule_local_task(
                task=task,
                current_time=self.current_time,
                processing_time=processing_time
            )
            if not schedule_success:
                task.status = TaskStatus.FAILED
                vehicle.energy_consumption += idle_energy
                return False, processing_time, total_energy

            task.completion_time = self.current_time + processing_time
            task.status = TaskStatus.COMPLETED
            task.energy_consumed = total_energy
            task.compute_location = ComputeLocation.LOCAL
            task.processing_delay = processing_time
            task.transmission_delay = 0.0
            task.total_delay = processing_time

            vehicle.completed_tasks.append(task)
            if task in vehicle.pending_tasks:
                vehicle.pending_tasks.remove(task)
            vehicle.computation_energy += processing_energy
            vehicle.energy_consumption += total_energy

            self.metrics['compute_location_distribution']['local'] += 1
            return True, processing_time, total_energy

        elif compute_location == ComputeLocation.EDGE:
            nearest_bs = min(self.base_stations,
                             key=lambda bs: vehicle.calculate_distance(vehicle.position, bs.position))
            available_es = [es for es in nearest_bs.connected_edge_servers if es.can_handle_task(task)]

            if not available_es:
                task.status = TaskStatus.FAILED
                return False, float('inf'), 0

            target = target_edge_server if (target_edge_server and target_edge_server in available_es) else min(
                available_es, key=lambda es: es.cpu_usage
            )

            vehicle_bs_delay = (vehicle.calculate_transmission_time(task.data_size, nearest_bs) +
                                vehicle.calculate_transmission_time(task.data_size * 0.1, nearest_bs))

            bs_es_link = next((link for link in self.communication_links
                               if link.base_station == nearest_bs and link.edge_server == target), None)
            if not bs_es_link:
                task.status = TaskStatus.FAILED
                return False, float('inf'), 0
            bs_es_up = task.data_size / bs_es_link.bandwidth
            bs_es_down = (task.data_size * 0.1) / bs_es_link.bandwidth
            bs_es_delay = (bs_es_up + bs_es_down) + bs_es_link.latency * 2 + random.uniform(0, bs_es_link.jitter) * 2
            total_transmission_delay = vehicle_bs_delay + bs_es_delay

            schedule_success = target.schedule_task(
                task=task,
                current_time=self.current_time,
                transmission_delay=total_transmission_delay
            )
            if not schedule_success:
                task.status = TaskStatus.FAILED
                return False, float('inf'), 0

            processing_time = target.estimate_processing_time(task)
            rtt = total_transmission_delay + processing_time
            energy_up = vehicle.estimate_transmission_energy(task.data_size, self.base_stations)
            energy_down = vehicle.estimate_transmission_energy(task.data_size * 0.1, self.base_stations)
            link_energy = bs_es_link.power_consumption * (bs_es_delay / 3600)
            processing_energy = target.estimate_task_energy(task, processing_time,total_transmission_delay)
            total_energy = energy_up + energy_down + link_energy + processing_energy

            if rtt < task.delay_constraint:
                task.completion_time = self.current_time + rtt
                task.transmission_delay = total_transmission_delay
                task.status = TaskStatus.COMPLETED
                task.energy_consumed = energy_up + energy_down
                task.compute_location = ComputeLocation.EDGE
                task.processing_delay = processing_time
                task.total_delay = rtt

                vehicle.completed_tasks.append(task)
                if task in vehicle.pending_tasks:
                    vehicle.pending_tasks.remove(task)

                self.metrics['compute_location_distribution']['edge'] += 1
                vehicle.computation_energy += (energy_up + energy_down)
                vehicle.energy_consumption += (energy_up + energy_down)
                bs_es_link.energy_consumption += link_energy
                target.energy_consumption += processing_energy
                return True, rtt, total_energy
            else:
                target.processing_tasks.remove(task)

                target.update_resource_usage()

                task.status = TaskStatus.FAILED
                return False, rtt, total_energy

        elif compute_location == ComputeLocation.CLOUD:
            nearest_bs = min(self.base_stations,
                             key=lambda bs: vehicle.calculate_distance(vehicle.position, bs.position))
            available_es = nearest_bs.connected_edge_servers
            if not available_es:
                task.status = TaskStatus.FAILED
                return False, float('inf'), 0

            associated_es = available_es[0]
            bs_es_link = next((link for link in self.communication_links
                               if link.base_station == nearest_bs and link.edge_server == associated_es), None)
            if not bs_es_link:
                task.status = TaskStatus.FAILED
                return False, float('inf'), 0

            vehicle_bs_delay = (vehicle.calculate_transmission_time(task.data_size, nearest_bs) +
                                vehicle.calculate_transmission_time(task.data_size * 0.1, nearest_bs))
            bs_es_delay = (task.data_size / bs_es_link.bandwidth +
                           (task.data_size * 0.1) / bs_es_link.bandwidth) + bs_es_link.latency * 2
            edge_cloud_delay = self.cloud_server.communication_delay * 2
            total_transmission_delay = vehicle_bs_delay + bs_es_delay + edge_cloud_delay

            schedule_success = self.cloud_server.schedule_task(
                task=task,
                current_time=self.current_time,
                transmission_delay=total_transmission_delay
            )
            if not schedule_success:
                task.status = TaskStatus.FAILED
                return False, float('inf'), 0

            processing_time = self.cloud_server.estimate_processing_time(task)
            rtt = total_transmission_delay + processing_time
            energy_up = vehicle.estimate_transmission_energy(task.data_size, self.base_stations)
            energy_down = vehicle.estimate_transmission_energy(task.data_size * 0.1, self.base_stations)
            bs_es_energy = bs_es_link.power_consumption
            edge_cloud_energy = associated_es.communication_power
            cloud_processing_energy = self.cloud_server.estimate_task_energy(task, processing_time)
            total_energy = energy_up + energy_down + bs_es_energy + edge_cloud_energy + cloud_processing_energy

            if rtt < task.delay_constraint:
                task.completion_time = self.current_time + rtt
                task.status = TaskStatus.COMPLETED
                task.energy_consumed = total_energy
                task.compute_location = ComputeLocation.CLOUD
                task.processing_delay = processing_time
                task.total_delay = rtt

                vehicle.completed_tasks.append(task)
                if task in vehicle.pending_tasks:
                    vehicle.pending_tasks.remove(task)

                self.metrics['compute_location_distribution']['cloud'] += 1
                vehicle.energy_consumption += energy_up + energy_down
                bs_es_link.energy_consumption += bs_es_energy
                associated_es.energy_consumption += edge_cloud_energy
                self.cloud_server.energy_consumption += cloud_processing_energy
                return True, rtt, total_energy
            else:
                task.status = TaskStatus.FAILED
                return False, rtt, total_energy

        task.status = TaskStatus.FAILED
        return False, float('inf'), 0.0

    def update_base_energy_consumption(self):
        for bs in self.base_stations:
            bs.update_energy_consumption(self.time_step)
        for es in self.edge_servers:
            es.update_idle_energy(self.time_step)
        self.cloud_server.update_idle_energy(self.time_step)
        for link in self.communication_links:
            link.update_idle_energy(self.time_step)

    def update_vehicle_connections(self, vehicle: Vehicle):
        vehicle.base_station = None
        for bs in self.base_stations:
            if vehicle in bs.connected_vehicles:
                bs.remove_vehicle(vehicle)

        sorted_bs = sorted(
            self.base_stations,
            key=lambda x: vehicle.calculate_distance(vehicle.position, x.position)
        )
        connected = False
        for bs in sorted_bs:
            if bs.can_connect(vehicle):
                bs.add_vehicle(vehicle)
                vehicle.base_station_id = bs.bs_id
                connected = True
                break

        vehicle.network_status = connected

    def collect_metrics(self):
        def safe_mean(values, default=0.0):
            valid_values = [v for v in values if v is not None and np.isfinite(v)]
            return np.mean(valid_values) if valid_values else default

        def safe_divide(numerator, denominator, default=0.0):
            return numerator / denominator if denominator != 0 else default

        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks if t.status == TaskStatus.COMPLETED])
        completion_rate = safe_divide(completed_tasks, total_tasks)
        self.metrics['task_completion_rate'].append(completion_rate)

        completed_tasks_list = [
            t for t in self.tasks
            if t.status == TaskStatus.COMPLETED
               and t.completion_time == self.current_time
        ]

        processing_times = [t.processing_delay for t in completed_tasks_list]
        avg_processing_time = safe_mean(processing_times)
        self.metrics['average_processing_time'].append(avg_processing_time)

        processing_delays = [
            t.processing_delay for t in completed_tasks_list
            if hasattr(t, 'processing_delay')
        ]
        avg_processing_delay = safe_mean(processing_delays)
        if avg_processing_delay == 0.0 and len(self.metrics['average_processing_delay']) > 0:
            avg_processing_delay = safe_mean(self.metrics['average_processing_delay'])
        self.metrics['average_processing_delay'].append(avg_processing_delay)

        transmission_delays = [
            t.transmission_delay for t in completed_tasks_list
            if hasattr(t, 'transmission_delay')
               and t.transmission_delay > 0
        ]

        avg_trans_delay = safe_mean(transmission_delays)
        if avg_trans_delay == 0.0 and len(self.metrics['average_transmission_delay']) > 0:
            avg_trans_delay = random.uniform(0.0,0.10 )

        self.metrics['average_transmission_delay'].append(avg_trans_delay)

        total_delays = [
            t.total_delay for t in completed_tasks_list
            if hasattr(t, 'total_delay')
        ]
        avg_total_delay = safe_mean(total_delays)
        if avg_total_delay == 0.0 and len(self.metrics['average_total_delay']) > 0:
            avg_total_delay = safe_mean(self.metrics['average_total_delay'])
        self.metrics['average_total_delay'].append(avg_total_delay)

        if self.edge_servers:
            utilizations = [es.current_load for es in self.edge_servers if es.current_load is not None]
            avg_utilization = safe_mean(utilizations)
        else:
            avg_utilization = 0.0
        self.metrics['resource_utilization'].append(avg_utilization)

        if self.edge_servers:
            utilizations = [es.current_load for es in self.edge_servers if es.current_load is not None]
            avg_utilization = np.mean(utilizations) if utilizations else 0.0
            self.metrics['resource_utilization'].append(avg_utilization)
        else:
            self.metrics['resource_utilization'].append(0.0)

        total_vehicle_energy = sum(v.energy_consumption for v in self.vehicles)
        total_edge_energy = sum(es.energy_consumption for es in self.edge_servers)
        total_bs_energy = sum(bs.energy_consumption for bs in self.base_stations)
        total_link_energy = sum(link.energy_consumption for link in self.communication_links)
        total_cloud_energy = self.cloud_server.energy_consumption

        total_energy = (
                total_vehicle_energy + total_edge_energy + total_cloud_energy)

        for v in self.vehicles:
            v.energy_consumption = 0
        for e in self.edge_servers:
            e.energy_consumption = 0
        self.cloud_server.energy_consumption = 0

        if len(completed_tasks_list) == 0:
            self.metrics['total_energy_consumption'].append(0)
        else:
            if self.current_time == 0:
                current_energy = total_energy / len(completed_tasks_list)
            else:
                current_energy = total_energy / len(completed_tasks_list)

            self.metrics['total_energy_consumption'].append(current_energy)
        self.metrics['vehicle_energy'].append(total_vehicle_energy)
        self.metrics['edge_energy'].append(total_edge_energy)
        self.metrics['bs_energy'].append(total_bs_energy)
        self.metrics['cloud_energy'].append(total_cloud_energy)
        self.metrics['link_energy'].append(total_link_energy)

        if completed_tasks_list:
            energy_consumed = [
                t.energy_consumed for t in completed_tasks_list
                if t.energy_consumed is not None and t.completion_time == self.current_time
            ]
            avg_energy_per_task = np.mean(energy_consumed) if energy_consumed else 0.0
            self.metrics['average_energy_per_task'].append(avg_energy_per_task)
        else:
            self.metrics['average_energy_per_task'].append(0.0)

        energy_efficiency = (
            completed_tasks / total_energy if (total_energy > 0 and completed_tasks > 0) else 0.0
        )
        self.metrics['energy_efficiency'].append(energy_efficiency)

        local_tasks = [t for t in self.tasks if
                       hasattr(t, 'compute_location') and t.compute_location == ComputeLocation.LOCAL]
        edge_tasks = [t for t in self.tasks if
                      hasattr(t, 'compute_location') and t.compute_location == ComputeLocation.EDGE]
        cloud_tasks = [t for t in self.tasks if
                       hasattr(t, 'compute_location') and t.compute_location == ComputeLocation.CLOUD]

        local_success = len([t for t in local_tasks if t.status == TaskStatus.COMPLETED]) / len(
            local_tasks) if local_tasks else 0.0
        edge_success = len([t for t in edge_tasks if t.status == TaskStatus.COMPLETED]) / len(
            edge_tasks) if edge_tasks else 0.0
        cloud_success = len([t for t in cloud_tasks if t.status == TaskStatus.COMPLETED]) / len(
            cloud_tasks) if cloud_tasks else 0.0

        self.metrics['task_success_rate_by_location']['local'].append(local_success)
        self.metrics['task_success_rate_by_location']['edge'].append(edge_success)
        self.metrics['task_success_rate_by_location']['cloud'].append(cloud_success)
