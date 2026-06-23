

import _technical_analysis_cpp
import time

N = 10_000_000

start = time.perf_counter()
j = 0
for i in range(N):
    result = i + j
    j += 1
print("Python:", time.perf_counter() - start)

start = time.perf_counter()
j = 0
for i in range(N):
    result = _technical_analysis_cpp.add(i, j)
    j += 1
print("C++ per-call:", time.perf_counter() - start)