from __future__ import annotations

import os

from resq.qaoa import QAOAResult
from resq.qubo import DispatchProblem


def crash_worker(problem: DispatchProblem, **kwargs) -> QAOAResult:
    """Terminate the disposable worker to exercise crash recovery."""
    os._exit(1)
