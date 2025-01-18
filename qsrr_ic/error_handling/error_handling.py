class ErrorHandling:

    @staticmethod
    def get_property_value_error(property_name: str) -> "ValueError":
        """
        Constructs a ValueError for read-only properties.

        Args:
            property_name (str): The name of the property.

        Returns:
            ValueError: Error indicating the property is read-only.
        """
        return ValueError(f"{property_name} is read-only and cannot be set directly.")
