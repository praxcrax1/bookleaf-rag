from abc import ABC, abstractmethod
from typing import Any

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Extend this class to implement custom agent logic and interfaces.
    """
    def __init__(self, config: Any, retriever: Any = None, vector_store: Any = None):
        self.config = config
        self.retriever = retriever
        self.vector_store = vector_store

    @abstractmethod
    def invoke(self, query: str) -> Any:
        """
        Main entrypoint for agent queries. Must be implemented by subclasses.
        """
        pass

    # Add more shared methods or properties here as needed
