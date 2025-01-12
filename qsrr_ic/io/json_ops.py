import json
from typing import (
    Any,
    Dict
)


class JsonOps:

    @staticmethod
    def to_json(filename: str, obj: Dict[str, Any]) -> None:
        """
        Save the Config object to a JSON file.

        :param filename: Path to the JSON file.
        :param obj:
        """
        with open(filename, "w") as f:
            json.dump(obj, f, indent=4)

    @staticmethod
    def from_json(filename: str) -> Dict[str, Any]:
        """
        Load a Config object from a JSON file.

        :param filename: Path to the JSON file.
        :return: dictionary object.
        """
        with open(filename, "r") as f:
            dict_ = json.load(f)
        return dict_
