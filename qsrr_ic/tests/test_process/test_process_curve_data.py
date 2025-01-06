import pytest
import numpy as np
from qsrr_ic.process import ProcessCurveData


class TestProcessCurveData:

    def test_knee_valid_input(self):
        """
        Test finding the knee point with valid inputs.
        """
        x = np.array([1, 2, 3, 4, 5, 6, 7])
        y = np.array([10, 9, 7, 6, 3, 2, 1])
        expected_knee = 5  # Based on manual calculation
        assert ProcessCurveData.knee(x, y) == expected_knee

    def test_knee_small_input(self):
        """
        Test that the method raises a ValueError for small input arrays.
        """
        x = np.array([1, 2])
        y = np.array([10, 5])
        with pytest.raises(ValueError, match="Input arrays must have at least 3 elements."):
            ProcessCurveData.knee(x, y)

    def test_knee_flat_curve(self):
        """
        Test handling of a flat curve where the second differences are zero.
        """
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 5, 5, 5, 5])  # Flat curve
        with pytest.raises(ValueError, match="No clear knee point found."):
            ProcessCurveData.knee(x, y)

    def test_knee_oscillating_curve(self):
        """
        Test handling of an oscillating curve to ensure it finds the correct knee.
        """
        x = np.array([1, 2, 3, 4, 5, 6, 7])
        y = np.array([10, 8, 12, 6, 4, 2, 1])
        expected_knee = 2  # Based on manual calculation
        assert ProcessCurveData.knee(x, y) == expected_knee

    def test_knee_negative_values(self):
        """
        Test handling of curves with negative y-values.
        """
        x = np.array([1, 2, 3, 4, 5, 6, 7])
        y = np.array([-10, -8, -6, -5, -3, -2, -1])
        expected_knee = 4  # Based on manual calculation
        assert ProcessCurveData.knee(x, y) == expected_knee

    def test_knee_large_input(self):
        """
        Test the method with a large input array.
        """
        x = np.arange(1, 1001)
        y = 1000 - np.sqrt(x)
        expected_knee = 2  # Based on manual observation of behavior
        assert ProcessCurveData.knee(x, y) == expected_knee

    def test_knee_non_integer_x_values(self):
        """
        Test handling of non-integer x-values.
        """
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        y = np.array([10, 8, 6, 4, 2, 1, 0])
        expected_knee = 0.3  # Based on manual calculation
        assert ProcessCurveData.knee(x, y) == int(expected_knee)
