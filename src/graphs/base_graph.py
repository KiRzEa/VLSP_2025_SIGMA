from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Union, Dict, TypedDict

from langgraph.graph import StateGraph

class BaseGraph(ABC):
    @abstractmethod
    def build(self) -> StateGraph:
        """
        Build the LangGraph subgraph.
        Should be implemented by all child classes.
        """
        pass

    @abstractmethod
    def run(self, state: Union[BaseModel, Dict]) -> Union[BaseModel, Dict]:
        """
        Run the graph with given state.
        """
        pass
