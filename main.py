import math
import time
import matplotlib.pyplot as plt


def Matrix(size, a1, a2, a3):
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        matrix[i][i] = a1
        if i < size - 1:
            matrix[i + 1][i] = a2
            matrix[i][i + 1] = a2
        if i < size - 2:
            matrix[i + 2][i] = a3
            matrix[i][i + 2] = a3
    return matrix


def Vector(size, f):
    return [math.sin(i * (f + 1)) for i in range(size)]


"""def matvecmul(matrix, vector):
    if len(vector) != len(matrix[0]):
        raise "Matrix and vector sizes wrong!"
    result = [0 for _ in range(len(matrix))]
    for i in range(len(matrix)):
        res = 0
        for j in range(len(vector)):
            res += matrix[i][j] * vector[j]
        result[i] = res
    return result"""


def matvecmul(matrix, vector):
    if len(vector) != len(matrix[0]):
        raise ValueError("Matrix and vector sizes wrong!")

    result = [0] * len(matrix)
    for i, row in enumerate(matrix):
        res = sum(row[j] * vector[j] for j in range(len(vector)))
        result[i] = res

    return result


"""def vecvecsub(vector1, vector2):
    if len(vector1) != len(vector2):
        raise "Vectors sizes wrong!"
    result = [0 for _ in range(len(vector1))]
    for i in range(len(vector1)):
        result[i] = vector1[i] - vector2[i]
    return result"""


def vecvecsub(vector1, vector2):
    if len(vector1) != len(vector2):
        raise "Vectors sizes wrong!"
    return [x - y for x, y in zip(vector1, vector2)]


def ones(N):
    return [1] * N


def residuum(A, x, b):
    return vecvecsub(matvecmul(A, x), b)


def norm(vector):
    return math.sqrt(sum(x ** 2 for x in vector))


def Jacobi(A, b, treshold):
    iters = 0
    n = len(b)
    x = ones(n)
    res_norm = float('inf')
    start = time.time()
    while res_norm >= treshold or math.isnan(res_norm):
        iters += 1
        x_previous = x.copy()
        for i in range(n):
            S = sum(A[i][j] * x_previous[j] for j in range(n) if j != i)
            x[i] = (b[i] - S) / A[i][i]
        res_norm = norm(residuum(A, x, b))
    end = time.time()
    duration = end - start
    return x, iters, duration, res_norm


def Gauss_Seidl(A, b, treshold):
    iters = 0
    n = len(b)
    x = ones(n)
    res_norm = float('inf')
    start = time.time()
    while res_norm >= treshold or math.isnan(res_norm):
        iters += 1
        x_previous = x.copy()
        for i in range(n):
            S = sum(A[i][j] * x[j] for j in range(i)) + sum(A[i][j] * x_previous[j] for j in range(i + 1, n))
            x[i] = (b[i] - S) / A[i][i]
        res_norm = norm(residuum(A, x, b))
    end = time.time()
    duration = end - start
    return x, iters, duration, res_norm


def LU_Factorization(A, b):
    n = len(b)
    U = [row[:] for row in A]
    L = Matrix(n, 1, 0, 0)
    y = [0] * n
    x = [0] * n
    start = time.time()
    for i in range(n - 1):
        for j in range(i + 1, n):
            L[j][i] = U[j][i] / U[i][i]
            for k in range(i, n):
                U[j][k] -= L[j][i] * U[i][k]
    for i in range(n):
        S = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - S) / L[i][i]
    for i in range(n - 1, -1, -1):
        S = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - S) / U[i][i]
    end = time.time()
    duration = end - start
    return x, duration, norm(residuum(A, x, b))


def main():
    # A
    e = 5
    N = 986
    f = 8
    """A = Matrix(N, 5 + e, -1, -1)
    b = Vector(N, f)"""
    treshold = 1e-9

    # B
    """jacobi_result = Jacobi(A, b, treshold)
    print("Jacobi method:")
    print("Iterations: ", jacobi_result[1])
    print("Duration [s]: ", jacobi_result[2])
    print("Norm of residual vector: ", jacobi_result[3])
    gauss_seidl_result = Gauss_Seidl(A, b, treshold)
    print("Gauss-Seidl method:")
    print("Iterations: ", gauss_seidl_result[1])
    print("Duration [s]: ", gauss_seidl_result[2])
    print("Norm of residual vector: ", gauss_seidl_result[3])"""

    # C
    """A = Matrix(N, 3, -1, -1)
    jacobi_result = Jacobi(A, b, treshold)
    print("Jacobi method:")
    print("Iterations: ", jacobi_result[1])
    print("Duration [s]: ", jacobi_result[2])
    print("Norm of residual vector: ", jacobi_result[3])
    gauss_seidl_result = Gauss_Seidl(A, b, treshold)
    print("Gauss-Seidl method:")
    print("Iterations: ", gauss_seidl_result[1])
    print("Duration [s]: ", gauss_seidl_result[2])
    print("Norm of residual vector: ", gauss_seidl_result[3])"""

    # D
    """A = Matrix(N, 3, -1, -1)
    lu_factorization_result = LU_Factorization(A, b)
    print("LU Factorization method:")
    print("Duration [s]: ", lu_factorization_result[1])
    print("Norm of residual vector: ", lu_factorization_result[2])"""

    # E
    N = [100, 500, 1000, 2000, 3000]
    jacobi_time = []
    gauss_seidl_time = []
    lu_time = []
    for n in N:
        print("N: ", n)
        A = Matrix(n, 5 + e, -1, -1)
        b = Vector(n, f)
        jacobi_result = Jacobi(A, b, treshold)
        print("Jacobi method:")
        print("Duration: ", jacobi_result[2])
        jacobi_time.append(jacobi_result[2])
        gauss_seidl_result = Gauss_Seidl(A, b, treshold)
        print("Gauss-Seidl method:")
        print("Duration: ", gauss_seidl_result[2])
        gauss_seidl_time.append(gauss_seidl_result[2])
        lu_factorization_result = LU_Factorization(A, b)
        print("LU Factorization method:")
        print("Duration: ", lu_factorization_result[1])
        lu_time.append(lu_factorization_result[1])
        print()
    plt.plot(N, jacobi_time)
    plt.plot(N, gauss_seidl_time)
    plt.plot(N, lu_time)
    plt.xlabel("liczba niewiadomych")
    plt.ylabel("czas [s]")
    plt.legend(["Jacobi method", "Gauss-Seidl method", "LU Factorization"])
    plt.title("The dependance of the duration algorithms on the number of variables")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
