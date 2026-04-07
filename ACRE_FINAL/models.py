from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field


class ObservationModel(BaseModel):
    code_length: float
    complexity_score: float
    runtime_s: float
    error_flag: bool

    @classmethod
    def from_vector(cls, values: Sequence[float]) -> "ObservationModel":
        vector = list(values)
        if len(vector) != 4:
            raise ValueError(f"observation vector must have length 4, got {len(vector)}")
        return cls(
            code_length=float(vector[0]),
            complexity_score=float(vector[1]),
            runtime_s=float(vector[2]),
            error_flag=bool(vector[3]),
        )

    def to_vector(self) -> List[float]:
        return [
            float(self.code_length),
            float(self.complexity_score),
            float(self.runtime_s),
            float(int(self.error_flag)),
        ]


class ActionModel(BaseModel):
    action: int = Field(ge=0, le=4)
    action_name: Optional[str] = None


class RewardModel(BaseModel):
    raw: float
    normalized: float = Field(ge=0.0, le=1.0)
    components: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    env: str
    version: str


class CompatibilityHealthResponse(BaseModel):
    status: str
    service: str


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None
    code: Optional[str] = None


class StepRequest(BaseModel):
    action: int = Field(ge=0, le=4)


class GradeRequest(BaseModel):
    code: str


class TaskInfo(BaseModel):
    id: str
    name: str
    description: str
    difficulty: str
    initial_code: str


class TasksResponse(BaseModel):
    tasks: List[TaskInfo]


class GradeResponse(BaseModel):
    task_id: str
    score: float
    passed: bool


class StateResponse(BaseModel):
    current_code: str
    episode_steps: int
    max_steps: int
    complexity: float
    last_runtime: float
    last_error: bool
    sample_id: Optional[str]
    language: Optional[str]
    task_id: Optional[str]
    observation: ObservationModel
    observation_vector: List[float]
    action_meanings: Dict[int, str]


class ResetResponse(BaseModel):
    observation: ObservationModel
    observation_vector: List[float]
    info: Dict[str, Any]
    task_id: Optional[str]
    state: StateResponse


class StepResponse(BaseModel):
    action: ActionModel
    observation: ObservationModel
    observation_vector: List[float]
    reward: RewardModel
    done: bool
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    state: StateResponse


class OptimizeRequest(BaseModel):
    code: str
    task_id: Optional[str] = None
    max_steps: int = Field(default=5, ge=1, le=5)
    use_rl: bool = True
    use_llm: bool = False
    fallback_to_llm: bool = True
    rl_model_path: Optional[str] = None
    api_base_url: Optional[str] = None
    model_name: Optional[str] = None
    api_token: Optional[str] = None


class OptimizationStep(BaseModel):
    step: int
    action: int
    action_name: str
    reason: str
    source: str
    reward: float
    normalized_reward: float
    changed: bool
    complexity: float


class OptimizeResponse(BaseModel):
    original_code: str
    optimized_code: str
    diff: str
    steps: List[OptimizationStep]
    cumulative_reward: float
    task_id: Optional[str]
    task_score: Optional[float]
