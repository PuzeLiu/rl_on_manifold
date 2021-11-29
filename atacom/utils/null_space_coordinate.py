import numpy as np
import scipy.sparse
from scipy import linalg
import sympy
import matplotlib.pyplot as plt


def pinv_null(a, rcond=None, return_rank=False):
    u, s, vh = linalg.svd(a, full_matrices=True, check_finite=False)
    M, N = u.shape[0], vh.shape[1]

    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond

    rank = np.sum(s > tol)
    Q = vh[rank:, :].T.conj()

    u = u[:, :rank]
    u /= s[:rank]
    B = np.transpose(np.conjugate(np.dot(u, vh[:rank])))

    if return_rank:
        return B, Q, rank
    else:
        return B, Q


def rref_sympy(V, row_vectors=True):
    if not row_vectors:
        V = V.T
    sym_V = sympy.Matrix(V)
    basis = sympy.matrices.matrix2numpy(sym_V.rref()[0], dtype=float)
    if not row_vectors:
        return basis.T
    else:
        return basis


def rref(A, row_vectors=True, tol=None):
    # Follow the implementation:
    # https://www.mathworks.com/matlabcentral/fileexchange/21583-fast-reduced-row-echelon-form
    V = A.copy()
    if not row_vectors:
        V = V.T
    m, n = V.shape

    if tol is None:
        tol = max(m, n) * np.finfo(V.dtype).eps * linalg.norm(V, np.inf)

    i = 0
    j = 0
    jb = list()
    while i < m and j < n:
        # Find value and index of largest element in the remainder of column j.
        k = np.argmax(np.abs(V[i:m, j]))
        k += i
        p = abs(V[k, j])

        if p <= tol:
            # The column is negligible, zero it out.
            V[i:m, j] = 0
            j += 1
        else:
            # Remember column index
            jb += [j]
            # Swap i-th and k-th rows.
            V[[i, k], j:n] = V[[k, i], j:n]
            # Divide the pivot row by the pivot element.
            Vi = V[i, j:n] / V[i, j]
            # Subtract multiples of the pivot row from all the other rows.
            V[:, j:n] = V[:, j:n] - np.outer(V[:, j], Vi)
            V[i, j:n] = Vi
            i += 1
            j += 1

    if not row_vectors:
        return V.T
    return V


def gram_schmidt(A, row_vectors=True, norm_dim=None):
    V = A.copy()
    if not row_vectors:
        V = V.T
    for i, v in enumerate(V):
        prev_basis = V[0:i]
        coeff_vec = prev_basis @ v
        v -= coeff_vec @ prev_basis
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-10:
            v /= v_norm
        else:
            v[v<1e-10] = 0

    if norm_dim is not None:
        for i, v in enumerate(V):
            bn = np.linalg.norm(v[:norm_dim])
            if bn > 0.01:
                V[i] = v / bn

    if not row_vectors:
        return np.array(V).T
    else:
        return np.array(V)


def orthogonalization_test():
    m = 3
    n = 2
    V_1 = np.random.randn(m, n)
    z_1 = np.cross(V_1[:, 0], V_1[:, 1])
    print("original: \n", V_1)

    V_2 = gram_schmidt(V_1, row_vectors=False)
    z_2 = np.cross(V_2[:, 0], V_2[:, 1])
    print("gs: \n", V_2)

    V_3 = rref(V_1, row_vectors=False)
    z_3 = np.cross(V_3[:, 0], V_3[:, 1])
    print("rref: \n", V_3)

    V_4 = gram_schmidt(V_3, row_vectors=False)
    z_4 = np.cross(V_4[:, 0], V_4[:, 1])
    print("gs + rref:\n", V_4)

    V_q, v_r = np.linalg.qr(z_1[:, np.newaxis], mode='complete')
    V_q = V_q[:, np.where(v_r.flatten() == 0)[0]]
    z_q = np.cross(V_q[:, 0], V_q[:, 1])
    print("qr: \n", V_q)

    V_svd = scipy.linalg.null_space(z_1[np.newaxis, :])
    z_svd = np.cross(V_svd[:, 0], V_svd[:, 1])
    print("svd: \n", V_svd)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.quiver(0, 0, 0, V_1[0, 0], V_1[1, 0], V_1[2, 0], color='tab:blue', label='original')
    ax.quiver(0, 0, 0, V_1[0, 1], V_1[1, 1], V_1[2, 1], color='tab:blue')
    ax.quiver(0, 0, 0, z_1[0], z_1[1], z_1[2], color='tab:blue', linestyle='--')

    ax.quiver(0, 0, 0, V_2[0, 0], V_2[1, 0], V_2[2, 0], color='tab:orange', label='GS')
    ax.quiver(0, 0, 0, V_2[0, 1], V_2[1, 1], V_2[2, 1], color='tab:orange')
    ax.quiver(0, 0, 0, z_2[0], z_2[1], z_2[2], color='tab:orange', linestyle='--')

    ax.quiver(0, 0, 0, V_3[0, 0], V_3[1, 0], V_3[2, 0], color='tab:red', label='RREF')
    ax.quiver(0, 0, 0, V_3[0, 1], V_3[1, 1], V_3[2, 1], color='tab:red')
    ax.quiver(0, 0, 0, z_3[0], z_3[1], z_3[2], color='tab:red', linestyle='--')

    ax.quiver(0, 0, 0, V_4[0, 0], V_4[1, 0], V_4[2, 0], color='tab:pink', label='RREF + GS')
    ax.quiver(0, 0, 0, V_4[0, 1], V_4[1, 1], V_4[2, 1], color='tab:pink')
    ax.quiver(0, 0, 0, z_4[0], z_4[1], z_4[2], color='tab:pink', linestyle='--')

    ax.quiver(0, 0, 0, V_q[0, 0], V_q[1, 0], V_q[2, 0], color='tab:brown', label='QR')
    ax.quiver(0, 0, 0, V_q[0, 1], V_q[1, 1], V_q[2, 1], color='tab:brown')
    ax.quiver(0, 0, 0, z_q[0], z_q[1], z_q[2], color='tab:brown', linestyle='--')

    ax.quiver(0, 0, 0, V_svd[0, 0], V_svd[1, 0], V_svd[2, 0], color='tab:purple', label='SVD')
    ax.quiver(0, 0, 0, V_svd[0, 1], V_svd[1, 1], V_svd[2, 1], color='tab:purple')
    ax.quiver(0, 0, 0, z_svd[0], z_svd[1], z_svd[2], color='tab:purple', linestyle='--')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.show()


def rref_test():
    for i in range(100):
        m, n = np.random.randint(1, 10, 2)
        V = np.random.randn(m, n)

        V_1 = rref(V)
        V_2 = rref_sympy(V)
        print(np.isclose(V_1, V_2).all())


def gram_schmidt_test():
    for i in range(100):
        m = 7
        n = 5

        V = np.random.randn(m, n)
        V_rref = rref(V, row_vectors=False)
        V_orth = gram_schmidt(V_rref, row_vectors=False)
        print(V_orth.max(), np.abs(V_orth).min())



if __name__ == '__main__':
    # rref_test()
    # gram_schmidt_test()
    orthogonalization_test()


