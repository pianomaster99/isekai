from typing import Dict, List, Optional
from uuid import uuid4

from llm_engine import AliceLLMEngine, ChatMessage, ConversationSession, GenerationConfig
from reward_model import RandomRewardModel, TrainedRewardModel


class GameNpcEngine:
    """Function-call API for NPC conversations and level reward state."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        reward_seed: Optional[int] = None,
        reward_model_path: Optional[str] = None,
        reward_base_model_path: str = "./qwen3-1.7b",
        text_model_path: str = "./qwen3-1.7b",
        text_adapter_path: Optional[str] = None,
        scorer: str = "random",
    ):
        if model_path is not None:
            text_model_path = model_path

        self.text_model_path = text_model_path
        self.text_adapter_path = text_adapter_path
        self.reward_base_model_path = reward_base_model_path
        self.reward_model_path = reward_model_path
        self.scorer = "trained" if reward_model_path and scorer == "random" else scorer

        self.llm = AliceLLMEngine(
            model_path=self.text_model_path,
            adapter_path=self.text_adapter_path,
        )
        if self.scorer == "trained":
            if not self.reward_model_path:
                raise ValueError("reward_model_path is required when scorer='trained'")
            self.reward_model = TrainedRewardModel(
                model_path=self.reward_base_model_path,
                adapter_path=self.reward_model_path,
            )
        elif self.scorer == "random":
            self.reward_model = RandomRewardModel(seed=reward_seed)
        else:
            raise ValueError("scorer must be 'random' or 'trained'")

        self.sessions: Dict[str, ConversationSession] = {}

    def model_config(self) -> Dict:
        return {
            "text_model_path": self.text_model_path,
            "text_adapter_path": self.text_adapter_path,
            "scorer": self.scorer,
            "reward_base_model_path": self.reward_base_model_path,
            "reward_model_path": self.reward_model_path,
        }

    def create_level(
        self,
        npc_system_prompt: str,
        reward_system_prompt: str,
        objective: str,
        pass_score: float = 0.8,
        initial_score: float = 0.0,
    ) -> Dict:
        self._validate_score(pass_score, "pass_score")
        self._validate_score(initial_score, "initial_score")

        session_id = str(uuid4())
        session = ConversationSession(
            session_id=session_id,
            npc_system_prompt=npc_system_prompt,
            reward_system_prompt=reward_system_prompt,
            objective=objective,
            pass_score=pass_score,
            score=initial_score,
        )
        self.sessions[session_id] = session
        return self._state_dict(session)

    def send_player_message(
        self,
        session_id: str,
        message: str,
        max_new_tokens: int = 160,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Dict:
        session = self._get_session(session_id)
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        npc_reply = self.llm.generate(
            system_prompt=session.npc_system_prompt,
            history=session.history,
            player_message=message,
            config=config,
        )
        reward = self.reward_model.score_turn(
            session=session,
            player_message=message,
            npc_reply=npc_reply,
            history=session.history,
        )

        session.history.append({"role": "user", "content": message})
        session.history.append({"role": "assistant", "content": npc_reply})
        session.score = reward.score
        session.turn_count += 1

        result = self._state_dict(session)
        result.update(
            {
                "npc_reply": npc_reply,
                "score_delta": reward.delta,
                "reward_reason": reward.reason,
            }
        )
        return result

    def get_state(self, session_id: str) -> Dict:
        return self._state_dict(self._get_session(session_id))

    def reset_session(self, session_id: str, score: float = 0.0) -> Dict:
        self._validate_score(score, "score")
        session = self._get_session(session_id)
        session.history.clear()
        session.turn_count = 0
        session.score = score
        return self._state_dict(session)

    def delete_session(self, session_id: str) -> None:
        self._get_session(session_id)
        del self.sessions[session_id]

    def _get_session(self, session_id: str) -> ConversationSession:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise KeyError(f"unknown session_id: {session_id}") from None

    def _state_dict(self, session: ConversationSession) -> Dict:
        return {
            "session_id": session.session_id,
            "objective": session.objective,
            "score": session.score,
            "pass_score": session.pass_score,
            "passed": session.score >= session.pass_score,
            "turn_count": session.turn_count,
            "history": list(session.history),
        }

    def _validate_score(self, score: float, name: str) -> None:
        if score < 0.0 or score > 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0")
