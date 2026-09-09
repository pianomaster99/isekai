from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm_engine import AliceLLMEngine, ChatMessage, ConversationSession, GenerationConfig
from reward_model import RandomRewardModel


app = FastAPI(
    title="Isekai NPC Engine API",
    version="0.1.0",
    description="HTTP API for LLM-driven NPC dialogue and reward-model level progress.",
)

llm = AliceLLMEngine()
reward_model = RandomRewardModel()
sessions: Dict[str, ConversationSession] = {}


class LevelCreateRequest(BaseModel):
    npc_system_prompt: str = Field(
        ...,
        description="Complete prompt describing NPC persona, setting, scene, rules, and style.",
    )
    reward_system_prompt: str = Field(
        ...,
        description="Complete prompt describing how the reward model should score progress.",
    )
    objective: str = Field(..., description="Game-facing level objective.")
    pass_score: float = Field(0.8, ge=0.0, le=1.0)
    initial_score: float = Field(0.0, ge=0.0, le=1.0)


class LevelCreateResponse(BaseModel):
    session_id: str
    score: float
    pass_score: float
    passed: bool


class ChatRequest(BaseModel):
    message: str
    max_new_tokens: int = Field(160, ge=1, le=512)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0, le=200)


class ChatResponse(BaseModel):
    session_id: str
    npc_reply: str
    score: float
    score_delta: float
    pass_score: float
    passed: bool
    turn_count: int
    reward_reason: str
    history: List[ChatMessage]


class SessionStateResponse(BaseModel):
    session_id: str
    objective: str
    score: float
    pass_score: float
    passed: bool
    turn_count: int
    history: List[ChatMessage]


class ResetScoreRequest(BaseModel):
    score: Optional[float] = Field(None, ge=0.0, le=1.0)


@app.get("/health")
def health():
    return {"status": "ok", "model": llm.model_path}


@app.post("/levels", response_model=LevelCreateResponse)
def create_level(request: LevelCreateRequest):
    session_id = str(uuid4())
    session = ConversationSession(
        session_id=session_id,
        npc_system_prompt=request.npc_system_prompt,
        reward_system_prompt=request.reward_system_prompt,
        objective=request.objective,
        pass_score=request.pass_score,
        score=request.initial_score,
    )
    sessions[session_id] = session

    return LevelCreateResponse(
        session_id=session_id,
        score=session.score,
        pass_score=session.pass_score,
        passed=session.score >= session.pass_score,
    )


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: str, request: ChatRequest):
    session = _get_session(session_id)
    config = GenerationConfig(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
    )

    npc_reply = llm.generate(
        system_prompt=session.npc_system_prompt,
        history=session.history,
        player_message=request.message,
        config=config,
    )

    reward = reward_model.score_turn(
        session=session,
        player_message=request.message,
        npc_reply=npc_reply,
        history=session.history,
    )

    session.history.append({"role": "user", "content": request.message})
    session.history.append({"role": "assistant", "content": npc_reply})
    session.score = reward.score
    session.turn_count += 1

    return ChatResponse(
        session_id=session.session_id,
        npc_reply=npc_reply,
        score=session.score,
        score_delta=reward.delta,
        pass_score=session.pass_score,
        passed=reward.passed,
        turn_count=session.turn_count,
        reward_reason=reward.reason,
        history=session.history,
    )


@app.get("/sessions/{session_id}", response_model=SessionStateResponse)
def get_session(session_id: str):
    session = _get_session(session_id)
    return _session_state(session)


@app.post("/sessions/{session_id}/reset", response_model=SessionStateResponse)
def reset_session(session_id: str, request: ResetScoreRequest):
    session = _get_session(session_id)
    session.history.clear()
    session.turn_count = 0
    session.score = 0.0 if request.score is None else request.score
    return _session_state(session)


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str):
    _get_session(session_id)
    del sessions[session_id]
    return None


def _get_session(session_id: str) -> ConversationSession:
    try:
        return sessions[session_id]
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")


def _session_state(session: ConversationSession) -> SessionStateResponse:
    return SessionStateResponse(
        session_id=session.session_id,
        objective=session.objective,
        score=session.score,
        pass_score=session.pass_score,
        passed=session.score >= session.pass_score,
        turn_count=session.turn_count,
        history=session.history,
    )
