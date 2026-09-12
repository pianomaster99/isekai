# Isekai NPC Engine API

Function-call API for game levels where a player talks to an NPC and passes when an internal reward score reaches a threshold.

The current implementation defaults to the local `./qwen3-1.7b` model for NPC text generation. The reward model can still run as a random placeholder, or load a trained LoRA reward adapter after `train_reward_model.py` finishes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Use From Game Code

```python
from game_api import GameNpcEngine

engine = GameNpcEngine(model_path="./qwen3-1.7b")

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


## Rowan Demo Training

The Rowan scene is organized in `scene.md`. Two demo datasets were generated from it:

- `datasets/rowan_ashford_sft_all.jsonl`: merged chat SFT examples for Rowan text generation.
- `datasets/rowan_ashford_reward_all.jsonl`: merged scalar reward examples with normalized scores.

Train the text-generation adapter:

```bash
python train_text_model.py
```

Train the reward/score adapter:

```bash
python train_reward_model.py
```

Use the trained adapters from game code. The game API can choose text and scoring models independently:

```python
from game_api import GameNpcEngine

engine = GameNpcEngine(
    text_model_path="./qwen3-1.7b",
    text_adapter_path="./models/rowan-qwen3-1.7b-sft",
    scorer="trained",
    reward_base_model_path="./qwen3-1.7b",
    reward_model_path="./models/rowan-qwen3-1.7b-reward",
)
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

