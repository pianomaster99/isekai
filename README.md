# Isekai NPC Engine API

Function-call API for game levels where a player talks to an NPC and passes when an internal reward score reaches a threshold.

The current implementation uses the local `./alice-in-the-dark-1b` model for NPC text generation. The reward model is intentionally random for now, but it already has a clean boundary for swapping in a trained scorer later.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Use From Game Code

```python
from game_api import GameNpcEngine

engine = GameNpcEngine(model_path="./alice-in-the-dark-1b")

level = engine.create_level(
    npc_system_prompt=(
        "You are Mira, a careful archivist in a flooded library. "
        "Keep replies brief and stay in character."
    ),
    reward_system_prompt=(
        "Score higher when the player respectfully learns where the silver key is hidden."
    ),
    objective="Convince Mira to reveal the silver key location.",
    pass_score=0.8,
)

session_id = level["session_id"]
turn = engine.send_player_message(
    session_id,
    "I need your help finding a silver key before the water rises.",
)

print(turn["npc_reply"])
print(turn["score"], turn["passed"])
```

## Public Functions

- `create_level(...)`: starts a conversation session from an NPC prompt, reward prompt, objective, and pass score.
- `send_player_message(session_id, message, ...)`: gets the NPC reply, updates the random score, and returns pass/fail state.
- `get_state(session_id)`: returns current score, pass state, turn count, and conversation history.
- `reset_session(session_id, score=0.0)`: clears history and resets score.
- `delete_session(session_id)`: removes a session from memory.

## Return Shape

`send_player_message` returns a dictionary with:

```python
{
    "session_id": "...",
    "objective": "...",
    "score": 0.42,
    "pass_score": 0.8,
    "passed": False,
    "turn_count": 1,
    "history": [...],
    "npc_reply": "...",
    "score_delta": 0.12,
    "reward_reason": "random placeholder scorer...",
}
```

## Notes For The Game Developer

- `npc_system_prompt` owns the NPC persona, setting, scene, rules, and style.
- `reward_system_prompt` owns the scoring criteria. It is stored now and can drive a trained reward model later.
- `score` is normalized from `0.0` to `1.0`.
- `passed` becomes true when `score >= pass_score`.
- Sessions are in memory for now. Recreating `GameNpcEngine` clears active sessions.

