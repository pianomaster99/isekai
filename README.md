# Isekai NPC Engine API

Small HTTP API for game levels where a player talks to an NPC and passes the level when an internal reward score reaches a threshold.

The current implementation uses the local `./alice-in-the-dark-1b` model for NPC text generation. The reward model is intentionally a random placeholder with the same API boundary a trained scorer can use later.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for the interactive OpenAPI UI.

## API Flow

Create a level/session:

```bash
curl -X POST http://localhost:8000/levels \
  -H "Content-Type: application/json" \
  -d '{
    "npc_system_prompt": "You are Mira, a careful archivist in a flooded library. Keep replies brief and stay in character.",
    "reward_system_prompt": "Score higher when the player respectfully learns where the silver key is hidden.",
    "objective": "Convince Mira to reveal the silver key location.",
    "pass_score": 0.8
  }'
```

Send player dialogue:

```bash
curl -X POST http://localhost:8000/sessions/YOUR_SESSION_ID/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I need your help finding a silver key before the water rises."}'
```

Inspect state:

```bash
curl http://localhost:8000/sessions/YOUR_SESSION_ID
```

Reset a session:

```bash
curl -X POST http://localhost:8000/sessions/YOUR_SESSION_ID/reset \
  -H "Content-Type: application/json" \
  -d '{"score": 0}'
```

## Notes For The Game Developer

- `npc_system_prompt` owns the NPC persona, setting, scene, rules, and style.
- `reward_system_prompt` owns the scoring criteria. It is stored and returned in placeholder scoring reasons now, then can drive a trained reward model later.
- `score` is normalized from `0.0` to `1.0`.
- `passed` becomes true when `score >= pass_score`.
- Sessions are in memory for now. Restarting the server clears active sessions.
