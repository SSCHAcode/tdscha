"""The q-space Lanczos must refuse to run under NumPy 1.x.

NumPy 1.26.4 on Python 3.14 aliases the masked metric products and silently
corrupts the Krylov vectors.  The recursion still completes with finite
coefficients and with ``b == c`` to machine precision, so no invariant in the
algorithm catches it -- only the NumPy version does.  See
``numpy1_python314_qspace_issue.md``.
"""

import numpy as np
import pytest

import tdscha.QSpaceLanczos as QS


def test_guard_passes_on_the_installed_numpy():
    """Whatever CI runs on must be a NumPy the Lanczos is correct under."""
    QS.check_numpy_version()
    assert int(np.__version__.split(".")[0]) >= 2


@pytest.mark.parametrize("version", ["1.26.4", "1.20.0", "0.9"])
def test_guard_rejects_numpy_1(monkeypatch, version):
    monkeypatch.setattr(QS.np, "__version__", version)
    with pytest.raises(RuntimeError) as excinfo:
        QS.check_numpy_version()
    message = str(excinfo.value)
    assert version in message
    # The message has to name the escape route, not just the problem.
    assert "PYTHONPATH" in message
    assert "numpy1_python314_qspace_issue.md" in message


@pytest.mark.parametrize("version", ["2.0.0", "2.4.6", "3.1.0"])
def test_guard_accepts_numpy_2_and_above(monkeypatch, version):
    monkeypatch.setattr(QS.np, "__version__", version)
    QS.check_numpy_version()


def test_run_FT_calls_the_guard(monkeypatch):
    """The guard must fire from run_FT itself, before any linear algebra.

    QSpaceAtomFourierLanczos inherits run_FT, so guarding it here covers the
    interpolated path too.
    """
    monkeypatch.setattr(QS.np, "__version__", "1.26.4")
    lanczos = QS.QSpaceLanczos.__new__(QS.QSpaceLanczos)
    # psi is deliberately left unset: the version check has to happen first,
    # otherwise a corrupted environment could still start a recursion.
    with pytest.raises(RuntimeError, match="NumPy 1.26.4"):
        lanczos.run_FT(1)
