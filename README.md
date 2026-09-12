# ORCH: Organizational principles enable collective intelligence in embodied AI

![ORCH Demo](assets/ORCH%20Website%20Long%20Gif.gif)

[![license badge](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

[Project Website](http://www.generalroboticslab.com/ORCH) | [Video](https://www.youtube.com/watch?v=M7MvJTFMfOA&t=57s) | [Paper](https://arxiv.org/abs/2609.11737v1)

# Authors
[Zhengran Ji](https://jzr01.github.io/), [Jonathan Hyun](https://www.linkedin.com/in/jonathan-hyun-21617b294/), [Boyuan Chen](http://boyuanchen.com/).

Duke University, [General Robotics Lab](http://generalroboticslab.com/)

# Get Started

## 1. Clone this repo

```
git clone https://github.com/generalroboticslab/ORCH.git
```

## 2. Install the crew conda environment and launch the Docker image

Install the CREW-WildFire environment  following [this instruction](https://generalroboticslab.github.io/wildfire-docs/getting-started/quick-installation/).

## 3. Download the game build

Download the game build at [here](https://drive.google.com/file/d/1TPkYg6dqn1-BVt_Jr2bpITzqNsL_gxGO/view?usp=sharing). After downloading the zip file, unzip it and put the Wildfire-StandaloneLinux64-Server folder into the path:

```
ORCH/crew-dojo/Builds/
```


## 4. Run the experiment

### LLM Usage

You can use an API service such as the OpenAI API or host a model locally for inference.

If you use the same LLMs as we do, simply set `MODEL` in [run_ORCH.sh](crew-algorithms/run_ORCH.sh) to the corresponding supported name: `gpt`, `qwen`, `deepseek`, `gemma`, `glm`, `llama`, `ernie`, or `nemotron`. For example:

```bash
MODEL="qwen"
```

These names select the model configurations already implemented in the code. No code changes are needed; configure your endpoint (`URL`) and API key as described below. Use the same model name in [run_algorithm_baseline.sh](crew-algorithms/run_algorithm_baseline.sh) when running baselines.

If you choose a different LLM, you need to modify the code before running experiments. Update the model configuration in [config/configs.py](crew-algorithms/crew_algorithms/wildfire_alg/config/configs.py) and the model-name mappings in [ORCH/agent.py](crew-algorithms/crew_algorithms/wildfire_alg/algorithms/ORCH/agent.py) and [ORCH/utils.py](crew-algorithms/crew_algorithms/wildfire_alg/algorithms/ORCH/utils.py). If you introduce a new model name, also update model validation and configuration loading in the relevant algorithm's `__main__.py`. For baselines, update the corresponding algorithm's model handling as needed to support your LLM.

#### Using the OpenAI API

To use GPT models through the OpenAI API, configure the following:

- **Model:** `gpt`
- **API base URL:** `https://api.openai.com/v1`
- **API key:** Set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

#### Using a Locally Hosted Model

Host your model with an OpenAI-compatible API server and configure the API base URL to point to your server, for example:

```text
http://localhost:8000/v1
```

If you use one of the models tested in our paper, enable API-key authentication when starting the server. Set the corresponding environment variable using the model name in uppercase, followed by `_API_KEY`:

```bash
export <MODEL_NAME>_API_KEY="your-local-api-key"
```

For example, for a Qwen model:

```bash
export QWEN_API_KEY="your-local-api-key"
```

The key must match the API key configured on your model server.

### Run Baselines

Configure [run_algorithm_baseline.sh](crew-algorithms/run_algorithm_baseline.sh) with your model (`MODEL`), endpoint (`URL`), available GPUs (`GPU_IDS`), and desired parallelism (`MAX_JOBS`). Select the baseline algorithms in `ALGOS` (`CAMON`, `COELA`, `HMAS_2`, and/or `Embodied`), and choose the missions and seeds in `PRESETS` and `SEEDS`.

From the repository root, run:

```bash
cd crew-algorithms
conda activate crew
bash run_algorithm_baseline.sh
```

### Run ORCH

Configure [run_ORCH.sh](crew-algorithms/run_ORCH.sh) with your model (`MODEL`), endpoint (`URL`), available GPUs (`GPU_IDS`), and desired parallelism (`MAX_JOBS`). Select the missions and seeds in `PRESETS` and `SEEDS`, then run from `crew-algorithms`:

```bash
conda activate crew
bash run_ORCH.sh
```

## 5. Check the result

### ORCH Results

When you run `bash run_ORCH.sh` from `crew-algorithms`, outputs are saved in the following locations (paths below are relative to the repository root).

**Experiment console logs** are saved separately for each model, mission, and seed:

```text
crew-algorithms/experiment_logs/<MODEL>/ORCH/<LEVEL>/seed<SEED>.log
```

These files capture standard output and errors, including the run configuration and completion or failure status. Start here when checking progress or troubleshooting a run. Running the same model, mission, and seed again overwrites its console log.

**Results and detailed agent logs** are saved in a timestamped directory for each run:

```text
crew-algorithms/crew_algorithms/wildfire_alg/results/logs/ORCH/<MODEL>/<TEAM_GENERATION_TYPE>/<LEVEL>/<SEED>/<TIMESTAMP>/
```

For `run_ORCH.sh`, `<TEAM_GENERATION_TYPE>` is `preset` because the script supplies a team configuration. `<TIMESTAMP>` uses the format `YYYY-MM-DD-HH-MM-SS`.

| File within the run directory | Contents |
| --- | --- |
| `data.csv` | Per-timestep mission metrics, including exploration, rescues, fire suppression, agent losses, cumulative API calls, token usage, cost, and time. |
| `master_logs/master_log_*.txt` | Human-readable log of agent events and coordination. |
| `master_logs/master_log_*.json` | Structured version of the master event log for analysis. |
| `Agent_<ID>/chats.txt` | Individual agent conversation logs, written as messages are recorded. |

For example, a `gpt` run of `Scout_Fire_small` with seed `4651` writes its console log to `crew-algorithms/experiment_logs/gpt/ORCH/Scout_Fire_small/seed4651.log` and its results under `crew-algorithms/crew_algorithms/wildfire_alg/results/logs/ORCH/gpt/preset/Scout_Fire_small/4651/<TIMESTAMP>/`.


### Baseline Results

When you run `bash run_algorithm_baseline.sh` from `crew-algorithms`, outputs are saved separately for each baseline. In the paths below, `<ALGO>` is `CAMON`, `COELA`, `HMAS_2`, or `Embodied`, as selected in `ALGOS`. All paths are relative to the repository root.

**Experiment console logs:**

```text
crew-algorithms/experiment_logs/<MODEL>/<ALGO>/<LEVEL>/seed<SEED>.log
```

Check these files for progress, errors, and completion or failure status. Running the same algorithm, model, mission, and seed again overwrites its console log.

**Results:**

```text
crew-algorithms/crew_algorithms/wildfire_alg/results/logs/<ALGO>/<MODEL>/<LEVEL>/<SEED>/<TIMESTAMP>/
```

Each run directory contains a `data.csv` file with per-timestep mission metrics. `<TIMESTAMP>` uses the format `YYYY-MM-DD-HH-MM-SS`. Baseline result paths do not include a `<TEAM_GENERATION_TYPE>` directory.

For example, a `CAMON` baseline run using `gpt` on `Scout_Fire_small` with seed `4651` writes its console log to `crew-algorithms/experiment_logs/gpt/CAMON/Scout_Fire_small/seed4651.log` and its metrics to `crew-algorithms/crew_algorithms/wildfire_alg/results/logs/CAMON/gpt/Scout_Fire_small/4651/<TIMESTAMP>/data.csv`.

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
