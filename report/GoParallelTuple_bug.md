# Bug in `cellconstructor.Settings.GoParallelTuple` with mpi4py (1 process)

## Summary

`GoParallelTuple` silently drops all but the first element of the returned tuple when running with `mpi4py` and a single MPI process (the most common local dev scenario).

## Reproduction

```python
import cellconstructor.Settings as Parallel
import numpy as np

def func(x):
    return [np.array([1.0, 2.0]), np.array([[3.0, 4.0], [5.0, 6.0]])]

result = Parallel.GoParallelTuple(func, [[1, 2]], '+')
print(len(result))  # Expected: 2, Actual: 1
print(result)       # [array([1., 2.])]  -- second element lost
```

## Root Cause

File: `cellconstructor/Settings.py`, around line 417 in `GoParallelTuple`.

The mpi4py reduction path does:

```python
# After per-process reduction, result = [f_array, d2v_array]  (len 2)

if __PARALLEL_TYPE__ == "mpi4py":
    comm = mpi4py.MPI.COMM_WORLD
    results = []                          # shadows outer `results`
    for i in range(len(result)):          # i = 0, 1
        results.append(comm.allgather(result[i]))
    # With 1 proc: results = [[f_array], [d2v_array]]

# BUG: overwrites `result` with first allgathered list
result = results[0]          # result = [f_array]  (len 1!)
for j in range(len(results)):
    for i in range(1, len(results[j])):
        result[j] += results[j][i]       # never executes with 1 proc

return result   # returns [f_array] -- d2v_array is lost
```

The line `result = results[0]` is wrong. `results` here is a list of allgathered lists, one per tuple element: `[[f_from_rank0, f_from_rank1, ...], [d2v_from_rank0, d2v_from_rank1, ...]]`. Setting `result = results[0]` takes only the first tuple element's gathered list and discards the rest.

## Fix

Replace the final reduction block with:

```python
result = []
for j in range(len(results)):
    reduced = results[j][0]
    for i in range(1, len(results[j])):
        if reduce_op == "+":
            reduced += results[j][i]
        elif reduce_op == "*":
            reduced *= results[j][i]
    result.append(reduced)
return result
```

## Impact

- Any code using `GoParallelTuple` with `mpi4py` (even 1 process) and a function returning 2+ elements will silently get wrong results.
- The serial path (`__PARALLEL_TYPE__ == "serial"`) works correctly.
- Since `mpi4py` is the default when installed, most users hit this bug even in non-parallel runs.

## Workaround (used in tdscha)

Pack all return values into a single flat numpy array, use `GoParallel` (which works correctly), then unpack:

```python
def combined(start_end):
    f, d2v = julia_function(...)
    return np.concatenate([f, d2v.ravel()])

result = Parallel.GoParallel(combined, indices, "+")
f = result[:n_modes]
d2v = result[n_modes:].reshape(n_modes, n_modes)
```
