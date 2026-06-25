import math
from enum import Enum

class TaskStatus(Enum):
    PENDING = "待处理"
    PROCESSING = "处理中"
    COMPLETED = "已完成"
    FAILED = "超时失败"


class CommunicationType(Enum):
    V2X = "V2X"
    NR_V2X = "NR-V2X"
    WIFI_6 = "WiFi-6"
    C5G = "5G"


class LinkStatus(Enum):
    CONNECTING = "连接中"
    CONNECTED = "已连接"
    DISCONNECTED = "断开"
    RECONNECTING = "重连中"


class TaskType(Enum):
    PERCEPTION = "感知计算"
    NAVIGATION = "导航规划"
    MULTIMEDIA = "多媒体处理"


class ComputeLocation(Enum):
    LOCAL = "本地计算"
    EDGE = "边缘计算"
    CLOUD = "云端计算"




class Position:
    """位置类，处理经纬度和道路坐标"""

    def __init__(self, latitude: float = 0.0, longitude: float = 0.0, road_id: str = ""):
        self.latitude = latitude
        self.longitude = longitude
        self.road_id = road_id

    def distance_to(self, other: 'Position') -> float:
        """计算两个位置之间的欧氏距离（简化版）"""
        lat_diff = self.latitude - other.latitude
        lon_diff = self.longitude - other.longitude
        return math.sqrt(lat_diff ** 2 + lon_diff ** 2)

    def __str__(self):
        return f"({self.latitude:.6f}, {self.longitude:.6f}, Road:{self.road_id})"