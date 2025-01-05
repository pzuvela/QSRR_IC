import os

class TestConstants:
    RESULTS = "results"
    ISO2GRAD_TEST_RESULTS = "2025-01-05-iso2grad_test_results.csv"

class TestPaths:
    TESTS_PATH = os.path.dirname(os.path.dirname(__file__))
    RESULTS_PATH = os.path.join(TESTS_PATH, TestConstants.RESULTS)
    ISO2GRAD_TEST_RESULTS_PATH = os.path.join(RESULTS_PATH, TestConstants.ISO2GRAD_TEST_RESULTS)
