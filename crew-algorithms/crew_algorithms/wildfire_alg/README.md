# CREW-WILDFIRE

[![docs badge](https://img.shields.io/badge/docs-reference-blue.svg)](https://generalroboticslab.github.io/crew-docs/)

[Project Website](http://www.generalroboticslab.com/CREW) | [Video](https://www.youtube.com/watch?v=RINSo3uI0dI) | [Paper](https://arxiv.org/abs/2408.00170)

CREW-WILDFIRE is an open-source benchmark designed to evaluate LLM-based multi-agent Agentic AI systems in complex, dynamic, real-world tasks. Built on the human-AI teaming CREW simulation platform, it provides procedurally generated wildfire response scenarios that test the limits of multi-agent coordination, communication, and planning under uncertainty.

# Authors
[Jonathan Hyun](https://github.com/jphyun2019), [Nicholas Waytowich](https://nicholaswaytowich.com/), [Boyuan Chen](http://boyuanchen.com/).

Duke University, [General Robotics Lab](http://generalroboticslab.com/)

## Overview

CREW-WILDFIRE addresses the gap in existing multi-agent benchmarks by offering:

- Large-scale environments supporting 2000+ agents
- Heterogeneous agent types (drones, helicopters, bulldozers, firefighters)
- Partial observability and stochastic dynamics
- Long-horizon planning objectives
- Both low-level control and high-level natural language interactions


Through these features, researchers can evaluate and develop next-generation multi-agent Agentic AI frameworks in high scale and complex scenarios, bridging the gap between conceptual toy enviornments and real world problems.

![crew teaser](assets/wildfire-teaser.png)


UPDATE WHEN PUBLISHED
# Citation
```
@inproceedings{zhang2024crew,
  title={CREW: Facilitating Human-AI Teaming Research},
  author={Zhang, Lingyu and Ji, Zhengran and Chen, Boyuan},
  booktitle={Preprint},
  year={2024}
}  
```

# Documentation

For quick examples to get started, API references, tutorials on how to run and develop algorithms and environments, please refer to our [documentation website](https://generalroboticslab.github.io/crew-docs/).



## Implemented Algorithms

CREW-WILDFIRE includes implementations of four state-of-the-art LLM-based multi-agent frameworks:

* **CAMON**: A hybrid coordination system featuring dynamic leadership where task assignments and global updates are managed by a changing leader agent. Leadership, along with global data is passed through each agnet by agent-to-agent interactions.

* **COELA**: A decentralized structure where each agent independently proposes information, evaluates the necessity of communication, and performs actions based on its generated plans.

* **Embodied**: A decentralized system without leader agents, featuring alternating communication rounds that allow both broadcast and targeted message exchanges, followed by independent action planning.

* **HMAS-2**: A hybrid framework combining centralized planning with distributed agent feedback. The system proposes actions through a central planner and refines them through agent consensus.

These implementations showcase different approaches to multi-agent coordination, from fully decentralized to hybrid architectures, enabling comprehensive evaluation of various coordination strategies.

## Key Features

* **Procedural Environment Generation**: Dynamic creation of environments using Perlin noise textures for vegetation, elevation, moisture, settlement, and wind vector maps
* **Heterogeneous Agent Support**: Multiple agent types with diverse capabilities
* **Scalable Architecture**: Support for large-scale agent deployment (2000+ agents)
* **Flexible Interaction Modes**: Both low-level control primitives and natural language commands
* **Comprehensive Evaluation**: Built-in metrics for assessing coordination, spatial reasoning, and plan adaptation
* **LLM Integration**: Native support for implementing and testing LLM-based multi-agent frameworks
* **Real-world Complexity**: Partial observability, stochastic dynamics, and complex objectives

## Project Structure

```
.
├── algorithms/           # Main algorithm implementations
│   ├── coela/           # COELA algorithm implementation
│   ├── embodied_agents/ # Embodied agents implementation
│   ├── hmas_2/         # HMAS 2.0 implementation
│   ├── manual/         # Manual control implementation
│   └── template/       # Template for new algorithms
│
├── core/               # Core functionality
│   ├── alg_utils.py   # Common algorithm utilities
│   └── utils.py       # General utility functions
│
├── config/            # Configuration management
│   ├── build_config.py # Configuration builder
│   └── configs.py     # Configuration definitions
│
├── data/             # Data scripts
|
├── libraries/           # Action and option libraries
│   ├── *_action_library.py  # Action definitions for different agents
│   └── *_option_library.py  # Option definitions for different agents
│
└── results/          # All output files
    ├── logs/         # All chat logs, scores, and screenshots
    ├── plots/        # Generated plots
    └── data/         # Output data files
```

## Contributing

CREW-WILDFIRE is designed to accelerate progress in large-scale Agentic intelligence. We welcome contributions that:

1. Implement new LLM-based multi-agent frameworks
2. Add novel evaluation metrics or scenarios
3. Improve environment generation or simulation features
4. Fix bugs or improve performance

Please follow these steps to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/Feature`)
3. Commit your changes (`git commit -m 'Add some Feature'`)
4. Push to the branch (`git push origin feature/Feature`)
5. Open a Pull Request

### Coding Standards

- Follow PEP 8 guidelines
- Include docstrings for all functions and classes
- Add unit tests for new features
- Update documentation as needed


