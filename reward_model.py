import random
from dataclasses import dataclass
from typing import List, Optional

from llm_engine import ChatMessage, ConversationSession


@dataclass
class RewardResult:
    score: float
    delta: float
    passed: bool
    reason: str


class RandomRewardModel:
    """Placeholder reward model with the same API shape as a trained scorer."""

    def __init__(self, seed: Optional[int] = None):
        self._random = random.Random(seed)

    def score_turn(
        self,
        session: ConversationSession,
        player_message: str,
        npc_reply: str,
        history: List[ChatMessage],
    ) -> RewardResult:
        del player_message, npc_reply, history

        delta = self._random.uniform(-0.1, 0.25)
        score = max(0.0, min(1.0, session.score + delta))
        passed = score >= session.pass_score

        return RewardResult(
            score=score,
            delta=delta,
            passed=passed,
            reason=(
                "random placeholder scorer; prompt retained for future reward model: "
                f"{session.reward_system_prompt[:160]}"
            ),
        )
