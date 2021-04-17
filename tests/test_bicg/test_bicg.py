import numpy as np
import scipy.sparse.linalg
import sscha, sscha.Tools


def test_bicg():
    # Generate a random matrix
    np.random.seed(1)
    SIZE = 55


    #A = np.random.uniform(size = (SIZE,SIZE))
    #b = np.random.uniform(size = SIZE)
    #x = b.copy()

    A = np.double(np.diag(np.arange(SIZE)**4 + 1))
    A += np.random.uniform(size = (SIZE,SIZE))*0.01
    A = (A + A.T) / 2
    b = np.ones(SIZE)
    x = b.copy()

    # Get the exact solution
    x_real = np.linalg.inv(A).dot(b)

    def mv(x):
        return A.dot(x)

    def pmv(x):
        x1 = x / np.sqrt(np.diag(A))
        return x1
    A_new = scipy.sparse.linalg.LinearOperator(A.shape, matvec = mv)
    Prec = scipy.sparse.linalg.LinearOperator(A.shape, matvec = pmv)

    x_mine = sscha.Tools.minimum_residual_algorithm_precond(A_new, b, precond_half = Prec, verbose = True)


    print("Real solution is:")
    print(x_real)

    print("")
    print("My solution is:")
    print(x_mine)

    print()
    print("Eigenvalues:")
    print(np.linalg.eigvals(A))

    assert np.max(x_real - x_mine) < 1e-5


if __name__ == "__main__":
    
    test_bicg()
