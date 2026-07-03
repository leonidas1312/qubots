from pathlib import Path

from qubots.detect.problems import ProblemSpecBackedProblem


class ImportedDataProblem(ProblemSpecBackedProblem):
    def __init__(self):
        super().__init__(Path(__file__).with_name('problem_spec.yaml'))
