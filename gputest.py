import cupy as cp

# Check if CuPy is using the GPU
print("Is CuPy using the GPU?", cp.cuda.is_available())

# Perform a simple calculation on the GPU
a = cp.array([1, 2, 3])
b = cp.array([4, 5, 6])
c = a + b

print("Result of GPU calculation:", c)