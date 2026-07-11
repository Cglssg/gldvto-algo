# GLDVTO

## Scenario Description

This simulator models a four-layer urban Internet of Vehicles (IoV) architecture, consisting of mobile vehicle terminals, base stations, edge servers, and cloud servers. Tasks support three offloading destinations: local device, edge server, and cloud server. The framework integrates embedded communication channel models, computation energy consumption models, vehicle mobility models, and stochastic task generation models to construct a full-stack vehicular edge computing simulation environment.

```
algorithm_gather_comparison/ 
├── algo_config.py             # Global simulation configuration file
├── run_gather.py              # Unified entry script for batch execution of all algorithms
├── run_DDPG.py
├── run_GAT_LSTM_D3QN.py
├── run_GAT_LSTM_DDQN.py
├── run_GCN_D3QN.py
├── run_GCN_DDQN.py
├── run_PPO.py
├── run_TD3.py
├── svg_to_pdf.py              # Tool for converting SVG vector plots to PDF files
└── utils.py                   # General utility functions
env/                           # Core module of simulation environment
├── __init__.py
├── base_station.py            # Base station entity class, implements wireless channel & SNR calculation
├── cloud_server.py            # Cloud server entity class for processing computation-intensive tasks
├── communication_link.py      # Communication link model between base stations and edge servers
├── edge_server.py             # Edge server module for resource scheduling and task computation
├── simulator.py               # Global simulation scheduler, controls overall simulation workflow
├── task.py                    # Data structure definition for task entities
├── utils_class.py             # Global enumerations and fundamental utility classes
└── vehicle.py                 # Vehicle terminal class: mobility, local computation & wireless transmission
README.md
```

## Experiment Settings

**Baseline Experiment:** 

DDPG, PPO, TD3, GCN-DDQN, GLDVTO

**Ablation Experiments:** 

GCN-D3QN, GCN-DDQN, GAT-LSTM-DDQN, GLDVTO

**Quick Start**

```
python algorithm/run_gather.py
```

| parameter                   | function                                                                                             |
| -------------------- |------------------------------------------------------------------------------------------------------|
| `--use-cache`        | Load the local plot cache directly and skip training.                                                |
| `--cache-path xxx.pkl` | Manually specify the path to the cache `.pkl` file instead of using the auto-generated default path. |
| `--save-dir ./xxx`   | Change the output directory for generated images and cache files.                                    |
| `--no-save-cache`    | Do not export the `.pkl` cache file after training is completed.                                     |

This script sequentially executes all algorithms stored in the directory, automatically collects all simulation metrics, and generates comparative visualization curves. Toggle the `IS_BASE` parameter in `algorithm_gather_comparison/algo_config.py` to switch between baseline experiments and ablation experiments.

## Related References

This simulation platform draws inspiration from EdgeSimPy, a mainstream general edge computing simulation framework. The corresponding citation is provided below:

```latex
@article{souza2023edgesimpy,    
    author={Paulo S. Souza and Tiago Ferreto and Rodrigo N. Calheiros},    
    title={EdgeSimPy: Python-Based Modeling and Simulation of Edge Computing Resource Management Policies},    
    journal={Future Generation Computer Systems},    
    year={2023},    
    issn={0167-739X},    
    volume={148},    
    pages={446-459},    
    doi={https://doi.org/10.1016/j.future.2023.06.013},    
    publisher={Elsevier} 
}
```

