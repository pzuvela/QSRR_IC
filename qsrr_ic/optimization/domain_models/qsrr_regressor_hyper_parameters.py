from typing import (
    Dict,
    Optional,
    Union
)


class HyperParameter:

    def __init__(self, value: Union[float, int]):
        if not isinstance(value, (float, int)):
            raise ValueError("HyperParameter value must be a float or int.")
        self.__value = value

    def __eq__(self, other) -> bool:
        return self.value == other.value

    @property
    def value(self) -> Union[float, int]:
        return self.__value

    @value.setter
    def value(self, value: Union[float, int]):
        """
        Prevents modification of the value after initialization.
        """
        raise ValueError("HyperParameter value cannot be changed after instantiation!")


class HyperParameterRange:
    """
    Class to represent a range of values for a hyperparameter.
    """

    def __init__(self, lower: HyperParameter, upper: HyperParameter):
        self.__lower = lower
        self.__upper = upper

    def get_static_value(self) -> Optional[HyperParameter]:
        """
        Returns the lower or upper bound if they are equal, indicating a static value.
        """
        if self.upper == self.lower:
            return self.lower
        return None

    @property
    def lower(self) -> HyperParameter:
        return self.__lower

    @lower.setter
    def lower(self, value: HyperParameter):
        raise ValueError("Lower bound cannot be changed after instantiation!")

    @property
    def upper(self) -> HyperParameter:
        return self.__upper

    @upper.setter
    def upper(self, value: HyperParameter):
        raise ValueError("Upper bound cannot be changed after instantiation!")


class HyperParameterRegistry:
    """
    A collection class for storing immutable HyperParameter and HyperParameterRange objects.
    """

    def __init__(self):
        """
        Initializes an empty collection for HyperParameters and HyperParameterRanges.
        """
        self.__parameters: Dict[str, Union[HyperParameter, HyperParameterRange]] = {}

    def add(self, name: str, value: Union[HyperParameter, HyperParameterRange]) -> None:
        """
        Add a new HyperParameter or HyperParameterRange to the collection.
        If the name exists, raises an error.

        Parameters
        ----------
        name : str
            The name of the hyperparameter.
        value : Union[HyperParameter, HyperParameterRange]
            The value of the hyperparameter (can be either a HyperParameter or a HyperParameterRange).
        """
        if name in self.__parameters:
            raise ValueError(f"HyperParameter or HyperParameterRange '{name}' already exists in the collection.")
        if not isinstance(value, (HyperParameter, HyperParameterRange)):
            raise ValueError("Value must be an instance of HyperParameter or HyperParameterRange.")
        self.__parameters[name] = value

    def names(self):
        return list(self.__parameters)

    def to_dict(self):
        return self.__parameters

    def get(self, name: str) -> Union[HyperParameter, HyperParameterRange]:
        """
        Retrieve a HyperParameter or HyperParameterRange by its name.

        Parameters
        ----------
        name : str
            The name of the hyperparameter to retrieve.

        Returns
        -------
        Union[HyperParameter, HyperParameterRange]
            The HyperParameter or HyperParameterRange object corresponding to the given name.

        Raises
        ------
        KeyError
            If the given name does not exist in the collection.
        """
        if name not in self.__parameters:
            raise KeyError(f"HyperParameter or HyperParameterRange '{name}' not found in the collection.")
        return self.__parameters[name]

    def remove(self, name: str) -> None:
        """
        Remove a HyperParameter or HyperParameterRange from the collection. Once removed, it cannot be recovered.

        Parameters
        ----------
        name : str
            The name of the hyperparameter to remove.

        Raises
        ------
        KeyError
            If the hyperparameter does not exist in the collection.
        """
        if name not in self.__parameters:
            raise KeyError(f"HyperParameter or HyperParameterRange '{name}' not found in the collection.")
        del self.__parameters[name]

    def __repr__(self) -> str:
        """
        String representation of the HyperParameterRegistry.
        """
        return f"HyperParameterRegistry({self.__parameters})"
