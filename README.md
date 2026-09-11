# ORCH: Organizational principles enable collective intelligence in embodied AI

![ORCH Demo](assets/ORCH%20Website%20Long%20Gif.gif)

[![license badge](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

[Project Website](http://www.generalroboticslab.com/ORCH) | [Video](https://www.youtube.com/watch?v=RINSo3uI0dI) | [Paper](https://arxiv.org/abs/2408.00170)

Collective intelligence depends not only on the capabilities of individual members, but also on how those members are organized. Yet artificial multi-agent systems are typically assembled using fixed organizational structures, even when the physical tasks they perform impose fundamentally different coordination requirements. Here we show that principles from human organization theory can be operationalized to organize large, heterogeneous collectives of embodied artificial agents. We introduce \projectname (Organizing Roles and Coordination Hierarchies), which constructs task-specific hierarchical organizations by combining pooled interdependence for work that can proceed concurrently with sequential interdependence for work governed by prerequisite relationships. Across 25 wildfire-response missions spanning reconnaissance, rescue, transportation, resource management, containment and suppression, we evaluated teams of up to 50 heterogeneous agents using eight large language models. Organizations constructed using these principles consistently outperformed four representative embodied multi-agent approaches across mission outcome, execution efficiency, exploration and computational resource use. Human-designed ORCH organizations improved final score by 63.97\% and execution efficiency by 74.29\% on average relative to the four prior frameworks. Organizations generated automatically by language models improved these measures by 43.63\% and 52.53\%, respectively. These advantages persisted across missions and underlying language models. Notably, collective performance was not monotonically determined by model scale. Analysis of long-horizon missions showed that hierarchical organization enabled teams to preserve concurrent activity within specialized groups while coordinating ordered transitions between mission phases. These results establish organizational design as a fundamental dimension of artificial collective intelligence and suggest that principles developed to understand human organizations can guide the construction of scalable embodied AI collectives.

# Authors
[Zhengran Ji](https://jzr01.github.io/), [Jonathan Hyun](https://www.linkedin.com/in/jonathan-hyun-21617b294/), [Boyuan Chen](http://boyuanchen.com/).

Duke University, [General Robotics Lab](http://generalroboticslab.com/)

# Result
![ORCH_Result](assets/Aggregated%20Result%20by%20Algorithm.png)

# Acknowledgments

This work is supported by the ARL STRONG program under awards W911NF2320182, W911NF2220113, and W911NF242021; the DARPA TIAMAT program under award HR00112490419; and ARO under award W911NF2410405.


# Citation
```
@misc{ji2026orchorganizationalprinciplesenable,
      title={ORCH: Organizational Principles Enable Collective Intelligence in Embodied AI},
      author={Zhengran Ji and Jonathan Hyun and Boyuan Chen},
      year={2026},
      eprint={2609.11737},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2609.11737},
}
```



