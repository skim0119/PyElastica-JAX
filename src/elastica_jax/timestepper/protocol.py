__doc__ = "Time stepper interface"

from typing import Protocol

from elastica_jax.protocol import JAXSystems

import numpy as np


class StepperProtocol(Protocol):
    """Protocol for all time-steppers"""

    def integrate(
        self,
        SystemCollection: JAXSystems,
        time: float,
        final_time: float,
        dt: float,
    ) -> np.float64: ...
