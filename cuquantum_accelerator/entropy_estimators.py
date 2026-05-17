from abc import ABC, abstractmethod
import numpy as np

class EntropyEstimator(ABC):
    @abstractmethod
    def estimate_entropy(self, samples: np.ndarray) -> float:
        pass

    @abstractmethod
    def compute_confidence_intervals(self, samples: np.ndarray) -> tuple[float, float]:
        pass

    def get_estimator_info(self) -> dict:
        return {
            "name": self.__class__.__name__,
            "bias_correction": {
                "kozachenko": self.use_kozachenko,
                "miller_madow": self.use_miller_madow
            }
        }

class EntropyBase(ABC):
    def __init__(self):
        self.use_kozachenko = False
        self.use_miller_madow = False

    @property
    def use_kozachenko(self) -> bool:
        return self._use_kozachenko

    @use_kozachenko.setter
    def use_kozachenko(self, value: bool):
        self._use_kozachenko = value

    @property
    def use_miller_madow(self) -> bool:
        return self._use_miller_madow

    @use_miller_madow.setter
    def use_miller_madow(self, value: bool):
        self._use_miller_madow = value

    def validate_samples(self, samples: np.ndarray) -> bool:
        if not samples.size or samples.ndim != 1:
            raise ValueError("Invalid sample input")
        return True

    @abstractmethod
    def estimate_entropy(self, samples: np.ndarray) -> float:
        pass

    @abstractmethod
    def compute_confidence_intervals(self, samples: np.ndarray) -> tuple[float, float]:
        pass

class KozachenkoLeonenko(EntropyBase):
    def estimate_entropy(self, samples: np.ndarray) -> float:
        if not self.validate_samples(samples):
            return 0.0
        # Placeholder for actual implementation
        return 0.5

    def compute_confidence_intervals(self, samples: np.ndarray) -> tuple[float, float]:
        # Placeholder for confidence interval calculation
        return (0.0, 1.0)

class MillerMadow(EntropyBase):
    def estimate_entropy(self, samples: np.ndarray) -> float:
        if not self.validate_samples(samples):
            return 0.0
        # Placeholder for actual implementation
        return 0.5

# Example usage
if __name__ == "__main__":
    estimator = KozachenkoLeonenko()
    print(estimator.get_estimator_info())
    print(estimator.estimate_entropy(np.array([1, 2, 3])))
