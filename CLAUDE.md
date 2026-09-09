# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CREW-Wildfire** is a real-time wildfire simulation for Human-AI teaming research. Human players and AI agents collaborate to fight fires, rescue civilians, and manage resources across a grid-based map. The platform records gameplay, agent decisions, and human feedback for research purposes.

## System Architecture

Five containers, started in dependency order:

```
postgres → nakama → algorithm → backend → frontend
                                              ↕
                                           caddy (reverse proxy)
```

- **postgres** — Nakama's database
- **nakama** — Real-time game server (Go plugin compiled from `crew-dojo/Nakama/`)
- **algorithm** — Runs the Unity game executable + Python AI agents (`crew-algorithms/crew_algorithms/wildfire_alg/`)
- **backend** — FastAPI lobby manager (`wildfire-human-interface/scripts/fastapi_backend.py`)
- **frontend** — Next.js player interface (`wildfire-human-interface/`)
- **caddy** — HTTPS termination; routes `/api/*` → backend, `/nakama/*` → nakama, `/` → frontend

The **backend** and **algorithm** containers share a `/tmp` volume for file-based IPC (action files, observation files, feedback files keyed by `{lobby_id}`).

## Key Components

### Unity Game Environment (`crew-dojo/Unity/Assets/Examples/Wildfire/`)
The 2D grid-based game world. Agents (Firefighter, Bulldozer, Drone, Helicopter) each have their own C# controller (`Scripts/`). `GameManager.cs` runs the main loop and computes rewards. `MapManager.cs` handles fire spread simulation and entity tracking. The game is compiled to a Linux binary baked into the algorithm Docker image.

**Unity ↔ Python connection:** Unity ML-Agents framework. `AIAgent.cs` (extends `Unity.MLAgents.Agent`) collects observations (61×61 grid + position) and receives continuous actions `[action_type, x, y]`. The Python algorithm side uses **TorchRL** + the ML-Agents env wrapper (`core/utils.py: make_env()`) to step the environment.

### Wildfire Algorithm Service (`crew-algorithms/crew_algorithms/wildfire_alg/`)
FastAPI service (port 8001) that spawns and manages game sessions.

- `algorithm_service.py` — HTTP API; spawns `algorithms/WILDFIRE/` as a subprocess per lobby
- `algorithms/WILDFIRE/` — The main AI algorithm: hierarchical LLM-powered multi-agent system
  - `__main__.py` — Entry point; async game loop with bottom-up status phase and top-down action phase
  - `agent.py` — Manager agent (phase-based planning, assigns tasks to children)
  - `worker_agent.py` — Worker agents (Firefighter/Bulldozer/Drone/Helicopter); execute LLM-chosen options
- `core/gpt.py` — LLM integration (OpenAI API); structured outputs for `Action`, `Option`, `Critique`
- `libraries/` — Per-agent action libraries mapping high-level options → Unity action vectors
- `config/` — Hydra configs; `build_config.py` defines 20+ named level presets

### Backend (`wildfire-human-interface/scripts/fastapi_backend.py`)
Manages lobbies and coordinates between the frontend and algorithm service. Batches human player actions and forwards them once all players have submitted. Runs a background polling thread per game to cache observations and detect game end or player disconnection (120s heartbeat timeout).

### Frontend (`wildfire-human-interface/`)
Next.js 15 + TypeScript + Tailwind. Two pages: `/` (lobby join/create) and `/lobby/[id]` (game view). `components/game-view.tsx` is the main game UI — handles observation rendering, action submission, chat, feedback, and team hierarchy display.

## Development Commands

### Full stack (Docker)
```bash
docker compose -f docker-compose.dev.yml up -d
docker compose -f docker-compose.dev.yml logs -f
```

### Algorithm service (Python 3.10, conda env `crew`)
```bash
cd crew-algorithms/
bash install.sh          # one-time setup
conda activate crew
poetry install

poetry run black crew_algorithms/
poetry run isort crew_algorithms/
poetry run flake8 crew_algorithms/

python crew_algorithms/wildfire_alg/algorithm_service.py
```

### Frontend
```bash
cd wildfire-human-interface/
npm install
npm run dev    # localhost:3000
npm run lint
```

### Backend
```bash
cd wildfire-human-interface/
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python scripts/fastapi_backend.py    # localhost:8000
```

### Tests
```bash
cd crew-algorithms/ && bash test.sh
```

### Build & deploy
```bash
./build-dev.sh       # build + push :dev images
./build-release.sh   # build + push :latest, pins digests in docker-compose.ec2.yml
```
See `DEPLOY_WALKTHROUGH.md` for full AWS EC2 setup (g4dn.xlarge, NVIDIA drivers, Caddy TLS).

## Environment Variables

| Variable | Service | Notes |
|---|---|---|
| `OPENAI_API_KEY` | algorithm, backend | Required |
| `ALGORITHM_SERVICE_URL` | backend | Default: `http://algorithm:8001` |
| `NEXT_PUBLIC_API_URL` | frontend | Baked at build time (`/api` for prod) |

## Code Quality
- Python: Black (line length 88), isort (black profile), Flake8 (Google docstrings)
- Frontend: Next.js ESLint (`npm run lint`)
- Python version for algorithms: **3.10.7–3.10.11** (strict; use the `crew` conda env)
