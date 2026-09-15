# ORCH: Organizational principles enable collective intelligence in embodied AI

![ORCH Demo](assets/ORCH%20Website%20Long%20Gif.gif)

[![license badge](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

[Project Website](http://www.generalroboticslab.com/ORCH) | [Video](https://www.youtube.com/watch?v=M7MvJTFMfOA&t=57s) | [Paper](https://arxiv.org/abs/2609.11737v1)

# Authors
[Zhengran Ji](https://jzr01.github.io/), [Jonathan Hyun](https://www.linkedin.com/in/jonathan-hyun-21617b294/), [Boyuan Chen](http://boyuanchen.com/).

Duke University, [General Robotics Lab](http://generalroboticslab.com/)

# Get Started

We currently only support Linux machine to run the experiment.

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

Set `MODEL`, `URL`, and `API_KEY` in [run_ORCH.sh](crew-algorithms/run_ORCH.sh) to the exact model ID exposed by the API server, its OpenAI-compatible base URL, and its optional API key. For example:

```bash
MODEL="Inferact/Qwen3.8-Flash-Next-NVFP4"
URL="http://localhost:8000/v1"
API_KEY=""
```

The exact `MODEL` value is sent to the API; adding a new model no longer requires changes to Python configuration files. `MODEL_FAMILY` is selected automatically: an official OpenAI URL uses `gpt`, while every other OpenAI-compatible endpoint uses `custom`. Use the same three settings in [run_algorithm_baseline.sh](crew-algorithms/run_algorithm_baseline.sh) when running baselines.

#### Using the OpenAI API

To use a GPT model through the official OpenAI API, set the exact model name, URL, and key directly in the runner:

```bash
MODEL="<exact-openai-model-name>"
URL="https://api.openai.com/v1"
API_KEY="your-openai-api-key"
```

If `API_KEY` is blank for the official OpenAI URL, the code falls back to the `OPENAI_API_KEY` environment variable. It raises an error if neither is available.

#### Using a Locally Hosted Model

Host the model with an OpenAI-compatible API server and enter the exact served model ID. For example, for a local vLLM server:

```bash
MODEL="Inferact/Qwen3.8-Flash-Next-NVFP4"
URL="http://localhost:8000/v1"
API_KEY=""
```

Leave `API_KEY` blank when the local server does not require authentication. The code supplies a harmless `EMPTY` placeholder because the OpenAI client requires a key argument. If the server was launched with API-key authentication, enter the matching value instead:

```bash
API_KEY="your-local-api-key"
```

### Run Baselines

Configure [run_algorithm_baseline.sh](crew-algorithms/run_algorithm_baseline.sh) with the exact model ID (`MODEL`), endpoint (`URL`), optional key (`API_KEY`), and desired parallelism (`MAX_JOBS`). Select the baseline algorithms in `ALGOS` (`CAMON`, `COELA`, `HMAS_2`, and/or `Embodied`), and choose the missions and seeds in `PRESETS` and `SEEDS`.

From the repository root, run:

```bash
cd crew-algorithms
conda activate crew
bash run_algorithm_baseline.sh
```

### Run ORCH

Configure [run_ORCH.sh](crew-algorithms/run_ORCH.sh) with the exact model ID (`MODEL`), endpoint (`URL`), optional key (`API_KEY`), and desired parallelism (`MAX_JOBS`). Select the missions and seeds in `PRESETS` and `SEEDS`, then run from `crew-algorithms`:

```bash
cd crew-algorithms
conda activate crew
bash run_ORCH.sh
```

## 5. Check the result

### ORCH Results

When you run `bash run_ORCH.sh` from `crew-algorithms`, outputs are saved in the following locations (paths below are relative to the repository root).

**Experiment console logs** are saved separately for each model, mission, and seed:

```text
crew-algorithms/experiment_logs/<MODEL_BASENAME>/ORCH/<LEVEL>/seed<SEED>.log
```

These files capture standard output and errors, including the run configuration and completion or failure status. Start here when checking progress or troubleshooting a run. Running the same model, mission, and seed again overwrites its console log.

**Results and detailed agent logs** are saved in a timestamped directory for each run:

```text
crew-algorithms/crew_algorithms/wildfire_alg/results/logs/ORCH/<MODEL_BASENAME>/<TEAM_GENERATION_TYPE>/<LEVEL>/<SEED>/<TIMESTAMP>/
```

`<MODEL_BASENAME>` is the portion after the final `/` in the exact model ID. For example, `Qwen/Qwen3.8-Flash-Next-NVFP4` uses `Qwen3.8-Flash-Next-NVFP4`. For `run_ORCH.sh`, `<TEAM_GENERATION_TYPE>` is `preset` because the script supplies a team configuration. `<TIMESTAMP>` uses the format `YYYY-MM-DD-HH-MM-SS`.

| File within the run directory | Contents |
| --- | --- |
| `data.csv` | Per-timestep mission metrics, including exploration, rescues, fire suppression, agent losses, cumulative API calls, token usage, cost, and time. |
| `master_logs/master_log_*.txt` | Human-readable log of agent events and coordination. |
| `master_logs/master_log_*.json` | Structured version of the master event log for analysis. |
| `Agent_<ID>/chats.txt` | Individual agent conversation logs, written as messages are recorded. |

For example, an `Inferact/Qwen3.8-Flash-Next-NVFP4` run of `Scout_Fire_small` with seed `4651` writes its console log to `crew-algorithms/experiment_logs/Qwen3.8-Flash-Next-NVFP4/ORCH/Scout_Fire_small/seed4651.log` and its results under `crew-algorithms/crew_algorithms/wildfire_alg/results/logs/ORCH/Qwen3.8-Flash-Next-NVFP4/preset/Scout_Fire_small/4651/<TIMESTAMP>/`.


### Baseline Results

When you run `bash run_algorithm_baseline.sh` from `crew-algorithms`, outputs are saved separately for each baseline. In the paths below, `<ALGO>` is `CAMON`, `COELA`, `HMAS_2`, or `Embodied`, as selected in `ALGOS`. All paths are relative to the repository root.

**Experiment console logs:**

```text
crew-algorithms/experiment_logs/<MODEL_BASENAME>/<ALGO>/<LEVEL>/seed<SEED>.log
```

Check these files for progress, errors, and completion or failure status. Running the same algorithm, model, mission, and seed again overwrites its console log.

**Results:**

```text
crew-algorithms/crew_algorithms/wildfire_alg/results/logs/<ALGO>/<MODEL_BASENAME>/<LEVEL>/<SEED>/<TIMESTAMP>/
```

Each run directory contains a `data.csv` file with per-timestep mission metrics. `<MODEL_BASENAME>` follows the same final-component rule described above, and `<TIMESTAMP>` uses the format `YYYY-MM-DD-HH-MM-SS`. Baseline result paths do not include a `<TEAM_GENERATION_TYPE>` directory.

For example, a `CAMON` run using `Inferact/Qwen3.8-Flash-Next-NVFP4` on `Scout_Fire_small` with seed `4651` writes its console log to `crew-algorithms/experiment_logs/Qwen3.8-Flash-Next-NVFP4/CAMON/Scout_Fire_small/seed4651.log` and its metrics to `crew-algorithms/crew_algorithms/wildfire_alg/results/logs/CAMON/Qwen3.8-Flash-Next-NVFP4/Scout_Fire_small/4651/<TIMESTAMP>/data.csv`.

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
