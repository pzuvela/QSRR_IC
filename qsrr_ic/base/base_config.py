from abc import (
    abstractmethod,
    ABC
)
from typing import (
    Any,
    Dict
)
from qsrr_ic.io import JsonOps


class BaseConfig(ABC):

    @abstractmethod
    def to_dict(self, *args, **kwargs):
        pass

    @abstractmethod
    def from_dict(self, *args, **kwargs) -> "BaseConfig":
        pass

    @classmethod
    def from_json(cls, filename: str) -> "BaseConfig":
        """
        Load a BaseConfig type of object from a JSON file.

        :param filename: Path to the JSON file.
        :return: BaseConfig object.
        """
        dict_: Dict[str, Any] = JsonOps.from_json(filename)
        return cls.from_dict(dict_)

    def to_json(self, filename: str):
        """
        Save the Config object to a JSON file.

        :param filename: Path to the JSON file.
        """
        JsonOps.to_json(filename, self.to_dict())
