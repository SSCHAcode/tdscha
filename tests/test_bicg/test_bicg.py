import numpy as np
import scipy.sparse.linalg
import sscha, sscha.Tools

import matplotlib.pyplot as plt


def get_matrix():
    
    np.random.seed(1)
    SIZE = 6


    A = np.random.uniform(size = (SIZE,SIZE))
    b = np.random.uniform(size = SIZE)

    #A = np.double(np.diag(np.arange(SIZE)**4 + 1))
    #A += np.random.uniform(size = (SIZE,SIZE))*0.01
    #b = np.ones(SIZE)
    
    A = (A + A.T) / 2


    return A, b


def test_fom(plot = False):
    print("Getting matrix...")
    A, b = get_matrix()
    
    x = b.copy()

    # Get the exact solution
    x_real = np.linalg.inv(A).dot(b)

    def mv(x):
        return A.dot(x)
    A_new = scipy.sparse.linalg.LinearOperator(A.shape, matvec = mv)

    distance = []
    def store(x, iters):
        r = x_real - x
        distance.append(np.sqrt(r.dot(r)))

    krylov_dim = 4

    x_mine = sscha.Tools.restarted_full_orthogonalization_method(A_new, b, x, max_iters = 3, krylov_dimension = krylov_dim, callback = store)
    

    if plot:
        plt.plot(distance, marker  = "o")
        plt.show()

def test_bicg():
    # Generate a random matrix
    A, b = get_matrix()
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
    test_fom(True)
    #test_bicg()
