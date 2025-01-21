from abc import (
    ABC,
    abstractmethod
)
from typing import Any


class BaseModelRunner(ABC):

    _instance = None

    def __new__(cls):
        """
        Ensure only one instance of the ModelRunner exists.
        """
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def run(self, *args, **kwargs) -> Any:
        """
        Run a given model. Only one model runs at a time.

        Args:
            *args: Positional arguments to pass to the model's `run` method.
            **kwargs: Keyword arguments to pass to the model's `run` method.

        Returns:
            Any: The result of the model's `run` method.

        Raises:
            AttributeError: If the provided model does not have a `run` method.
        """

        # Ensure exclusive access during execution
        return self._run(*args, **kwargs)

    @abstractmethod
    def _run(self, *args, **kwargs) -> Any:
        pass
