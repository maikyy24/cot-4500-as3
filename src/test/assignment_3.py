import numpy as np
np.set_printoptions(precision=5, suppress=True, linewidth=100)


#Question 1
def f(t, y):
    return t - y**2

a = 0
b = 2
n = 10
h = (b - a) / n

t = np.zeros(n+1)
y = np.zeros(n+1)
y[0] = 1

for i in range(n):
    y[i+1] = y[i] + h * f(t[i], y[i])
    t[i+1] = t[i] + h
    
print("{:.5f}".format(y[10]))
print()


#Question 2


def f(t, y):
    return t - y**2

a = 0
b = 2
n = 10
h = (b - a) / n

t = np.zeros(n+1)
y = np.zeros(n+1)
y[0] = 1

for i in range(n):
    k1 = h * f(t[i], y[i])
    k2 = h * f(t[i] + h/2, y[i] + k1/2)
    k3 = h * f(t[i] + h/2, y[i] + k2/2)
    k4 = h * f(t[i] + h, y[i] + k3)
    
    y[i+1] = y[i] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    t[i+1] = t[i] + h

print("{:.5f}".format(y[10]))
print()

#Question 3

# Define the augmented matrix
A = np.array([[2, -1, 1, 6],
              [1, 3, 1, 0],
              [-1, 5, 4, -3]], dtype= float)

# Perform Gaussian elimination
n = len(A)
for i in range(n):
    # Divide the ith row by the diagonal element
    pivot = A[i, i]
    A[i, :] /= pivot
    
    # Subtract the ith row from the lower rows
    for j in range(i+1, n):
        factor = A[j, i]
        A[j, :] -= factor * A[i, :]

# Perform backward substitution
x = np.zeros(n)
for i in range(n-1, -1, -1):
    x[i] = A[i, -1]
    for j in range(i+1, n):
        x[i] -= A[i, j] * x[j]

# Print the solution
print(x)
print()

#Question 4


# Define the matrix
A = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]] , dtype= float)

# Initialize L and U matrices with zeros
L = np.zeros_like(A)
U = np.zeros_like(A)

# Perform LU Factorization
n = A.shape[0]

for k in range(n):
    L[k,k] = 1
    U[k,k:] = A[k,k:] - L[k,:k] @ U[:k,k:]
    L[k+1:,k] = (A[k+1:,k] - L[k+1:,:k] @ U[:k,k]) / U[k,k]

# Calculate the matrix determinant
det = np.prod(np.diag(U)) * np.prod(np.diag(L))
print("{:.5f}".format(det))

print()

# Print out the L matrix
print(L)
print()

# Print out the U matrix
print(U)
print()



#Question 5

# Define the matrix
A = np.array([[9, 0, 5, 2, 1],
              [3, 9, 1, 2, 1],
              [0, 1, 7, 2, 3],
              [4, 2, 3, 12, 2],
              [3, 2, 4, 0, 8]])

# Calculate the absolute diagonal values
abs_diag = np.abs(np.diag(A))

# Calculate the sum of the absolute values of the off-diagonal elements in each row
row_sum = np.sum(np.abs(A), axis=1) - abs_diag

# Check if the matrix is diagonally dominant
is_diagonally_dominant = np.all(abs_diag >= row_sum)

# Print the result
if is_diagonally_dominant:
    print("True")
    print()
else:
    print("False")
    print()

#Question 6 

# Define the matrix
A = np.array([[2, 2, 1],
              [2, 3, 0],
              [1, 0, 2]])


# Check if the matrix is positive definite
try:
    np.linalg.cholesky(A)
    print("True")
except np.linalg.LinAlgError:
    print("False")
