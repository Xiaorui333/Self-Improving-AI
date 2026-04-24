"""Structured output schemas used by the Planner, Verifier, and Executor."""

from pydantic import BaseModel


class QueryAnalysis(BaseModel):
    concise_summary: str
    required_skills: str
    relevant_tools: str
    additional_considerations: str

    def __str__(self) -> str:
        return (
            f"Concise Summary: {self.concise_summary}\n\n"
            f"Required Skills:\n{self.required_skills}\n\n"
            f"Relevant Tools:\n{self.relevant_tools}\n\n"
            f"Additional Considerations:\n{self.additional_considerations}"
        )


class NextStep(BaseModel):
    justification: str
    context: str
    sub_goal: str
    tool_name: str


class MemoryVerification(BaseModel):
    analysis: str
    stop_signal: bool


class ToolCommand(BaseModel):
    analysis: str
    explanation: str
    command: str
