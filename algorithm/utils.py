import numpy as np
from algorithm.algo_config import EnvConfig

def smooth_data_ema(data, alpha=0.9):
    if EnvConfig.IS_SMOOTH_DATA:
        if len(data) == 0:
            return data
        data = np.asarray(data)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * ema[i - 1] + (1 - alpha) * data[i]

        return ema
    else:
        return data

def convert_to_json_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif obj is None:
        return None
    else:
        return obj

