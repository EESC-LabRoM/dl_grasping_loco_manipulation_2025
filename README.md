# dl_grasping_loco_manipulation_2025

Deep learning pipeline for grasping with the Boston Dynamics Spot robot. Developed at the Mobile Robotics Group, EESC (São Carlos Engineering School).

The project covers simulation-based dataset generation, neural network training for grasp selection, and real robot deployment using Spot's arm and gripper camera.

- Genesis docs: [genesis-world.readthedocs.io](https://genesis-world.readthedocs.io/en/latest/)
- Genesis repo: [Genesis GitHub](https://github.com/Genesis-Embodied-AI/Genesis)

## Repository Structure

```
├── urdf/             # Robot description files (URDF, meshes)
├── scripts/          # Genesis simulation scripts
│   └── spot_gripper/ # Gripper grasping experiments, including normal-aligned grasping and D2NT validation
├── dataset/          # Dataset extraction notebooks and 3D bottle models
├── grasp_selection/  # Grasp selection neural network training
└── spot_deploy/      # Real robot deployment pipeline (YOLO + D2NT + GraspNN)
    ├── evaluation/   # Full integration pipeline
    ├── images/       # Camera data from Spot's gripper
    └── tuning/       # Parameter tuning scripts
```

## Setup

Simulations run on [Genesis](https://genesis-world.readthedocs.io/en/latest/). See the [installation guide](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html).

Using conda:

```bash
conda create -n genesis_env python=3.12
conda activate genesis_env
pip install genesis-world
```

Using pyenv:

```bash
pyenv virtualenv 3.12 genesis_env
pyenv activate genesis_env
pip install genesis-world
```

An NVIDIA GPU is recommended. Install [CUDA drivers](https://developer.nvidia.com/cuda-downloads) for better performance.

## Real Robot

Deployment uses the Boston Dynamics Spot SDK:

- [Python examples](https://github.com/boston-dynamics/spot-sdk/tree/7569b7998d486109f80de31dd5f86470016bb141/python/examples)
- [Protobuf API](https://github.com/boston-dynamics/spot-sdk/tree/7569b7998d486109f80de31dd5f86470016bb141/protos/bosdyn/api)

Normal maps on the real robot are generated using [D2NT](https://mias.group/D2NT/). In simulation, Genesis provides native tools for this.

## Genesis Documentation

- [User Guide](https://genesis-world.readthedocs.io/en/latest/user_guide/index.html)
- [API Reference](https://genesis-world.readthedocs.io/en/latest/api_reference/index.html)

## Contributing

Open an issue or pull request if you have suggestions or improvements.
