"""Configuration module for Shannon Bench."""

from pydantic import BaseModel, Field


class Config(BaseModel):
  """Configuration for the benchmark.

  Attributes:
    name: The name of the benchmark run.
    iterations: The number of iterations to run.
  """

  name: str = Field(..., description="The name of the benchmark run.")
  iterations: int = Field(10, description="The number of iterations to run.", gt=0)
