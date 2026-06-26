# Bug Report: `np.ndarray` subscripted type annotation blocks all imports

## Severity
**CRITICAL** — The package is completely unimportable.

## Environment
- **Python**: 3.9.7
- **NumPy**: 1.20.3
- **tdscha**: 1.6.2 (`main` branch)
- **OS**: Linux

## Description

`np.ndarray` does not support subscript notation (`[...]`) — it is not a generic type and lacks `__class_getitem__`. On line 5032 of `Modules/DynamicalLanczos.py`, the annotation `np.ndarray[np.float64]` is used as a parameter type hint:

```python
def get_green_function_continued_fraction(self, w_array : np.ndarray[np.float64], ...):
```

Because Python evaluates type annotations at definition time (unless `from __future__ import annotations` is active), this raises:

```
TypeError: 'type' object is not subscriptable
```

The error occurs at class definition time, meaning **any import of `tdscha` fails immediately**.

## Steps to Reproduce

```bash
python -c "import tdscha"
```

Observed output:
```
TypeError: 'type' object is not subscriptable
  File ".../tdscha/DynamicalLanczos.py", line 5032, in Lanczos
    def get_green_function_continued_fraction(self, w_array : np.ndarray[np.float64], ...):
```

## Root Cause

`np.ndarray` is a concrete type, not a generic. NumPy provides `numpy.typing.NDArray` for subscriptable array annotations, but the code uses `np.ndarray` directly in subscript form, which is syntactically invalid.

## Affected Code

**File**: `Modules/DynamicalLanczos.py` (installed as `tdscha/DynamicalLanczos.py`)
**Line**: 5032

```python
def get_green_function_continued_fraction(self, w_array : np.ndarray[np.float64], ...):
```

Additionally, lines 5388 and 5424 use `np.float64` as parameter/return type annotations, which are syntactically valid but unnecessarily restrictive (a plain `float` would be more appropriate).

## Suggested Fixes

### Option A (simplest — single line)
Add `from __future__ import annotations` to the top of `Modules/DynamicalLanczos.py`. This makes all annotations lazy strings (PEP 563), bypassing the runtime evaluation entirely:

```python
from __future__ import print_function, division, annotations
```

This is backward-compatible with Python 3.8+ and requires no other code changes.

### Option B (best practice)
Replace the invalid subscript annotation with the proper NumPy typing idiom:

```python
from numpy.typing import NDArray

def get_green_function_continued_fraction(self, w_array: NDArray[np.float64], ...):
```

And optionally change `np.float64` annotations on lines 5388 and 5424 to `float` for broader compatibility.

## Workaround

None — the package cannot be imported at all until this is fixed.
