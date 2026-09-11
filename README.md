# ORCH: Organizational principles enable collective intelligence in embodied AI

![ORCH Demo](assets/ORCH%20Website%20Long%20Gif.gif)

[![license badge](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

[Project Website](http://www.generalroboticslab.com/ORCH) | [Video](https://www.youtube.com/watch?v=RINSo3uI0dI) | [Paper](https://arxiv.org/abs/2408.00170)

# Authors
[Zhengran Ji](https://jzr01.github.io/), [Jonathan Hyun](https://www.linkedin.com/in/jonathan-hyun-21617b294/), [Boyuan Chen](http://boyuanchen.com/).

Duke University, [General Robotics Lab](http://generalroboticslab.com/)

# Get Started

## 1. Install the CREW-WildFire Environment 

Install the CREW-WildFire environment  following [this instruction](https://generalroboticslab.github.io/wildfire-docs/getting-started/quick-installation/).

## 2. Download the game build

Download the game build at [here](https://drive.google.com/file/d/1TPkYg6dqn1-BVt_Jr2bpITzqNsL_gxGO/view?usp=sharing).

## 3. Configure the LLM you are going to use 

Make your chosen model available through an API or a running local model server. Configure `envs.llm_model` and `envs.llm_url` for the provider and endpoint, and supply the required API key. For the `gpt` provider, set `OPENAI_API_KEY`. Review the model names in [the wildfire configuration](crew-algorithms/crew_algorithms/wildfire_alg/config/configs.py) to ensure they match the models available to you.

## 4. Run the experiment 

From `crew-algorithms`, configure [run_ORCH.sh](crew-algorithms/run_ORCH.sh) with your model (`MODEL`), endpoint (`URL`), available GPUs (`GPU_IDS`), and desired parallelism (`MAX_JOBS`). Select the missions and seeds in `PRESETS` and `SEEDS`, then run:

```
conda activate crew
bash run_ORCH.sh
```

## 5. Check the result

When you run `bash run_ORCH.sh` from `crew-algorithms`, outputs are saved in the following locations (paths below are relative to the repository root).

**Experiment console logs** are saved separately for each model, mission, and seed:

```text
crew-algorithms/experiment_logs/<MODEL>/WILDFIRE/<LEVEL>/seed<SEED>.log
```

These files capture standard output and errors, including the run configuration and completion or failure status. Start here when checking progress or troubleshooting a run. Running the same model, mission, and seed again overwrites its console log.

**Results and detailed agent logs** are saved in a timestamped directory for each run:

```text
crew-algorithms/crew_algorithms/wildfire_alg/results/logs/WILDFIRE/<MODEL>/<TEAM_GENERATION_TYPE>/<LEVEL>/<SEED>/<TIMESTAMP>/
```

For `run_ORCH.sh`, `<TEAM_GENERATION_TYPE>` is `preset` because the script supplies a team configuration. `<TIMESTAMP>` uses the format `YYYY-MM-DD-HH-MM-SS`.

| File within the run directory | Contents |
| --- | --- |
| `data.csv` | Per-timestep mission metrics, including exploration, rescues, fire suppression, agent losses, cumulative API calls, token usage, cost, and time. |
| `master_logs/master_log_*.txt` | Human-readable log of agent events and coordination. |
| `master_logs/master_log_*.json` | Structured version of the master event log for analysis. |
| `Agent_<ID>/chats.txt` | Individual agent conversation logs, written as messages are recorded. |

For example, a `gpt` run of `Scout_Fire_small` with seed `4651` writes its console log to `crew-algorithms/experiment_logs/gpt/WILDFIRE/Scout_Fire_small/seed4651.log` and its results under `crew-algorithms/crew_algorithms/wildfire_alg/results/logs/WILDFIRE/gpt/preset/Scout_Fire_small/4651/<TIMESTAMP>/`.


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



