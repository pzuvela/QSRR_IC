from typing import (
    Dict,
    List,
    Optional,
    Union,
    Type
)

from qsrr_ic.optimization.enums import (
    HyperParameterName,
    HYPER_PARAMETER_TYPE_MAPPING
)


class HyperParameter:

    def __init__(self, value: Union[float, int]):
        if not isinstance(value, (float, int)):
            raise ValueError("HyperParameter value must be a float or int.")
        self.__value = value

    def type(self) -> Type[Union[float, int]]:
        return type(self.value)

    def __eq__(self, other: 'HyperParameter') -> bool:
        if not isinstance(other, HyperParameter):
            return NotImplemented
        return (self.value == other.value) and (self.type() == other.type())

    def __repr__(self) -> str:
        return f"HyperParameter(value={self.value})"

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

        if lower.type() != upper.type():
            raise ValueError("Both lower and upper bounds must be of the same type!")
        if lower.value > upper.value:
            raise ValueError("Lower bound cannot be greater than upper bound!")

        self.__lower = lower
        self.__upper = upper

    def __repr__(self) -> str:
        return f"HyperParameterRange(lower={self.lower}, upper={self.upper})"


    def type(self) -> Type[Union[float, int]]:
        return self.lower.type()

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

    @property
    def values(self) -> List[Union[float, int]]:
        return [self.lower.value, self.upper.value]

    @values.setter
    def values(self, value: List[Union[float, int]]):
        raise ValueError("Values cannot be changed after instantiation!")

class HyperParameterRegistry:
    """
    A collection class for storing immutable HyperParameter and HyperParameterRange objects.
    """

    def __init__(self):
        """
        Initializes an empty collection for HyperParameters and HyperParameterRanges.
        """
        self.__parameters: Dict[HyperParameterName, Union[HyperParameter, HyperParameterRange]] = {}

    def __repr__(self) -> str:
        """String representation of the HyperParameterRegistry."""
        params = {name: str(value) for name, value in self.__parameters.items()}
        return f"HyperParameterRegistry({params})"

    def __iter__(self):
        for name, hp in self.__parameters.items():
            yield name, hp

    def __len__(self):
        return len(self.__parameters)

    def add(self, name: HyperParameterName, value: Union[HyperParameter, HyperParameterRange]) -> None:
        """
        Add a new HyperParameter or HyperParameterRange to the collection.
        If the name exists, raises an error.

        Parameters
        ----------
        name : HyperParameterName
            The name of the hyperparameter.
        value : Union[HyperParameter, HyperParameterRange]
            The value of the hyperparameter (can be either a HyperParameter or a HyperParameterRange).
        """

        if not isinstance(value, (HyperParameter, HyperParameterRange)):
            raise ValueError("Value must be an instance of HyperParameter or HyperParameterRange.")

        parameters = [value]

        if isinstance(value, HyperParameterRange):
            parameters = [value.lower, value.upper]

        for parameter in parameters:
            self.__validate_parameter(name, parameter)
        self.__parameters[name] = value

    def __validate_parameter(self, name: HyperParameterName, value: HyperParameter) -> None:
        if name in self.__parameters:
            raise ValueError(f"HyperParameter or HyperParameterRange '{name}' already exists in the collection.")
        if value.type() != HYPER_PARAMETER_TYPE_MAPPING[name]:
            expected_type = HYPER_PARAMETER_TYPE_MAPPING[name].__name__
            actual_type = value.type().__name__
            raise ValueError(
                f"HyperParameter '{name}' has incorrect type. "
                f"Expected {expected_type}, got {actual_type}"
            )

    def get(self, name: HyperParameterName) -> Union[HyperParameter, HyperParameterRange]:
        """
        Retrieve a HyperParameter or HyperParameterRange by its name.

        Parameters
        ----------
        name : HyperParameterName
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

    def remove(self, name: HyperParameterName) -> None:
        """
        Remove a HyperParameter or HyperParameterRange from the collection. Once removed, it cannot be recovered.

        Parameters
        ----------
        name : HyperParameterName
            The name of the hyperparameter to remove.

        Raises
        ------
        KeyError
            If the hyperparameter does not exist in the collection.
        """
        if name not in self.__parameters:
            raise KeyError(f"HyperParameter or HyperParameterRange '{name}' not found in the collection.")
        del self.__parameters[name]

    def names(self) -> List[HyperParameterName]:
        return list(self.__parameters)

    def to_dict(self) -> Dict[str, Union[Union[List[float], float], Union[List[int], int]]]:
        """
        Convert the registry to a dictionary of primitive values.

        Returns
        -------
        Dict[str, Union[Union[List[float], float], Union[List[int], int]]]
            Dictionary with parameter names as keys and either single values or lists of two values as values.
        """
        def get_val(hp: Union[HyperParameter, HyperParameterRange]):
            return hp.value if isinstance(hp, HyperParameter) else hp.values
        return {
            name.value: get_val(value) for name, value in self.__parameters.items()
        }

    @classmethod
    def from_dict(
        cls,
        hyper_parameter_dict: Dict[Union[str, HyperParameterName], Union[Union[List[float], float], Union[List[int], int]]]
    ) -> 'HyperParameterRegistry':
        registry = cls()
        """
        Create a registry from a dictionary of primitive values.
        
        Parameters
        ----------
        hyper_parameter_dict : Dict[Union[str, HyperParameterName], Union[Union[List[float], float], Union[List[int], int]]]
            Dictionary with parameter names as keys and either single values or lists of two values as values.
        
        Returns
        -------
        HyperParameterRegistry
            New registry populated with the provided parameters.
        
        Raises
        ------
        ValueError
            If any parameter name is unknown or if list values don't have exactly 2 elements.
        """
        for name, value in hyper_parameter_dict.items():

            if isinstance(name, str):
                name = HyperParameterName(name)

            if name not in HYPER_PARAMETER_TYPE_MAPPING:
                raise ValueError(f"Unknown hyperparameter name: {name}")

            hp: Union[HyperParameter, HyperParameterRange]

            if isinstance(value, list):
                if len(value) != 2:
                    raise ValueError(f"Range values must have exactly 2 elements, got {len(value)}")
                lower = HyperParameter(HYPER_PARAMETER_TYPE_MAPPING[name](value[0]))
                upper = HyperParameter(HYPER_PARAMETER_TYPE_MAPPING[name](value[1]))
                hp = HyperParameterRange(lower, upper)
            else:
                hp = HyperParameter(HYPER_PARAMETER_TYPE_MAPPING[name](value))

            registry.add(name=name, value=hp)

        return registry
