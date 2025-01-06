import numpy as np


class ProcessCurveData:

    @staticmethod
    def knee(x: np.ndarray, y: np.ndarray, threshold: float = 1e-6) -> int:
        """
        Finds the knee point in a curve defined by x and y values.

        Parameters
        ----------
        x : np.ndarray
            Array of x-values (independent variable).
        y : np.ndarray
            Array of y-values (dependent variable).
        threshold : float, optional
            A threshold for detecting a flat curve based on second-order differences.
            Default is 1e-6.

        Returns
        -------
        int
            The x-value corresponding to the knee point.

        Raises
        ------
        ValueError
            If the input arrays are too short or no clear knee point is found.
        """
        if len(x) < 3 or len(y) < 3:
            raise ValueError("Input arrays must have at least 3 elements.")

        # Compute first-order differences
        first_differences = np.diff(y)

        # Compute second-order differences
        second_differences = np.diff(first_differences)

        # Check for flat curve by comparing second-order differences to the threshold
        if np.all(np.abs(second_differences) < threshold):
            raise ValueError("No clear knee point found due to a flat curve.")

        # Find the index of the maximum second-order difference
        max_index = np.argmax(second_differences)

        # Map the second-order difference index to the corresponding x-value
        knee_point = x[max_index + 1]  # Correct offset

        return int(knee_point)
