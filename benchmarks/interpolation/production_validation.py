"""Exact Gaussian quadrature through the production Python/Julia operator.

The scalar x is a mass-weighted molecular internal displacement. Its
force-carrying subspace is integrated exactly; independent harmonic spectator
coordinates have zero connected anharmonic contractions and are integrated
out analytically. Signed cubature weights are integration weights, not Monte
Carlo probabilities. All calls use the real production Julia estimator.

Counterterms are fixed from the coarse Gaussian once per fixture. Fine
reference vertices are analytic derivatives of that SAME polynomial (their
third/fourth Gaussian averages do not depend on the covariance). No separate
fine minimization or force counterterm adjustment is performed.
"""
import itertools
import numpy as np
import cellconstructor.Structure
import tdscha.QSpaceAtomFourier as AF
import tdscha.JuliaExt as JE
import polynomial_reference as ref


def quadrature(d):
    """Smolyak difference rule exact for Gaussian polynomials of degree <=7."""
    rules = []
    for level in range(4):
        x, w = np.polynomial.hermite_e.hermegauss(level+1)
        pairs = list(zip(x, w/np.sqrt(2*np.pi)))
        if level:
            x, w = np.polynomial.hermite_e.hermegauss(level)
            pairs += list(zip(x, -w/np.sqrt(2*np.pi)))
        rules.append(pairs)
    nodes = {}
    for total in range(4):
        for active in itertools.combinations_with_replacement(range(d), total):
            levels = np.bincount(active, minlength=d)
            axes = np.flatnonzero(levels)
            for values in itertools.product(*(rules[levels[a]] for a in axes)):
                x = np.zeros(d)
                weight = 1.
                for a, (v, w) in zip(axes, values):
                    x[a] = v
                    weight *= w
                key = tuple(np.round(x, 14))
                nodes[key] = nodes.get(key, 0.) + weight
    x = np.array(list(nodes))
    w = np.array(list(nodes.values()))
    assert abs(w.sum()-1) < 1e-10
    assert abs(w @ x[:, 0]**6-15) < 1e-10
    # Mixed moments test the multidimensional quadrature, not only its axes.
    assert abs(w @ (x[:, 0]+x[:, 1]+x[:, 2])**6-15*3**3) < 1e-9
    return x, w


def make_operator(nc=9, nf=18, Q=0., embedded=False, thermal=0.):
    """Reduced internal-coordinate sector, with two inactive spectator axes.

    No stochastic contraction or interpolation method is replaced. Only the
    ensemble arrays and exactly prescribed harmonic reference are supplied.
    """
    obj = AF.QSpaceAtomFourierLanczos.__new__(AF.QSpaceAtomFourierLanczos)
    nb = obj.n_bands = 6 if embedded else 3
    obj.T = thermal*157887.32400374097
    obj.n_q, obj.cn_q = nf, nc
    obj.fine_mesh, obj.coarse_mesh = np.array([1, 1, nf]), np.array([1, 1, nc])
    obj.uci_structure = cellconstructor.Structure.Structure(nb//3)
    obj.uci_structure.unit_cell = np.eye(3)
    obj.uci_structure.coords[:] = 0.
    if embedded:
        obj.uci_structure.coords[1] = [.13, .21, 1.37]
    obj._fine_idx = np.array([[0, 0, k] for k in range(nf)])
    obj._coarse_idx = np.array([[0, 0, k] for k in range(nc)])
    obj._q_lookup = {tuple(p): i for i, p in enumerate(obj._fine_idx)}
    obj._coarse_lookup = {tuple(p): i for i, p in enumerate(obj._coarse_idx)}
    obj.q_points = obj._fine_idx/obj.fine_mesh
    obj.cq_points = obj._coarse_idx/obj.coarse_mesh
    def modes(n):
        q = np.arange(n)/n
        w = np.array([ref.frequency(q)]+[np.ones(n)*(i+1) for i in range(1, nb)])
        E = np.repeat(np.eye(nb, dtype=complex)[:, :, None], n, axis=2)
        if embedded:
            # Unequal masses 1:2; optical relative and acoustic COM vectors.
            for a in range(3):
                E[a, a], E[a+3, a] = np.sqrt(2/3), -np.sqrt(1/3)
                E[a, a+3], E[a+3, a+3] = np.sqrt(1/3), np.sqrt(2/3)
                w[a+3] = (a+1)*2*np.abs(np.sin(np.pi*q))
            # Rebase atom 1 by one cell and use a nontrivial smooth mode gauge.
            E[3:, :, :] *= np.exp(2j*np.pi*q)
            E[:, 0, :] *= np.exp(1j*phase(q))
        return w, E
    obj.w_q, obj.pols_q = modes(nf)
    obj.cw_q, obj.cpols_q = modes(nc)
    obj.valid_modes_q = obj.w_q > 1e-8
    obj.cvalid_modes_q = obj.cw_q > 1e-8
    obj._distributed = False
    obj.ignore_v3 = obj.ignore_v4 = obj.ignore_harmonic = False
    obj.n_syms_qspace = 1
    obj._build_atom_fourier_kernel()
    obj.build_q_pair_map(int(round(Q*nf)))
    points, obj.rho = quadrature(nc)
    occupation = 0 if thermal == 0 else 1/np.expm1(obj.cw_q[0]/thermal)
    c = np.fft.ifft((1+2*occupation)/(2*obj.cw_q[0])).real
    C = c[(np.arange(nc)[:, None]-np.arange(nc)) % nc]
    u = points @ np.linalg.cholesky(C).T
    residual = np.zeros_like(u)
    for order, terms in ((3, ref.CUBIC), (4, ref.QUARTIC)):
        for g, h in terms:
            s = sum(hr*np.roll(u, -r, axis=1) for r, hr in enumerate(h))
            variance = sum(hi*hj*c[(i-j) % nc]
                           for i, hi in enumerate(h) for j, hj in enumerate(h))
            ds = g/2*(s*s-variance) if order == 3 else g/6*(s**3-3*variance*s)
            for r, hr in enumerate(h):
                residual -= hr*np.roll(ds, r, axis=1)
    obj.N = len(points)
    obj.X_q = np.zeros((nc, obj.N, nb), dtype=complex)
    obj.Y_q = np.zeros_like(obj.X_q)
    obj.X_q[:, :, 0] = np.fft.fft(u, axis=1).T/np.sqrt(nc)
    obj.Y_q[:, :, 0] = np.fft.fft(residual, axis=1).T/np.sqrt(nc)
    if embedded:
        gauge = np.exp(-1j*phase(np.arange(nc)/nc))[:, None]
        obj.X_q[:, :, 0] *= gauge
        obj.Y_q[:, :, 0] *= gauge
    jl = JE.get_main()
    jl.eval('_cached_qspace_symmetries[] = [SparseMatrixCSC{ComplexF64,Int32}(I, %d, %d)]'
            % (nb*nc, nb*nc))
    return obj


def phase(q):
    return .37*np.sin(2*np.pi*np.asarray(q))


def probe(nc=9, nf=18, Q=0., embedded=False, ignore_v3=False,
          ignore_v4=False, thermal=0., check=True, mutation=None, parent=False):
    obj = make_operator(nc, nf, Q, embedded, thermal)
    obj.ignore_v3, obj.ignore_v4 = ignore_v3, ignore_v4
    if mutation == 'missing_fold_scale':
        # Drop the sqrt(N_c/N_f) of the fold: d3 pair->R and d4 are off.
        fold = obj._fold
        obj._fold = lambda blocks: [b*np.sqrt(nf/nc) for b in fold(blocks)]
    elif mutation == 'conjugate_kernel':
        obj._atom_fourier_kernel = obj._atom_fourier_kernel.conj()
    if parent:
        assert nc == nf
        # Exercise the q-space parent's own apply_anharmonic_FT as well.
        obj.__class__ = AF.QL.QSpaceLanczos
    indices = [0] + list(obj._block_offsets_a) + list(obj._block_offsets_b)
    matrix = np.empty((len(indices), len(indices)), dtype=complex)
    for j, index in enumerate(indices):
        v = np.zeros(obj.get_psi_size(), dtype=complex)
        v[index] = 1
        out = obj.apply_full_L(v)
        matrix[:, j] = out[indices]
        excluded = out.copy()
        excluded[indices] = 0
        if check:
            assert np.linalg.norm(excluded) < 1e-10
    v, W = ref.vertices(nf, Q)
    if embedded:
        q = np.arange(nf)/nf
        pairphase = np.exp(-1j*(phase(q)+phase(Q-q)))
        v = pairphase*v*np.exp(1j*phase(Q))
        W = pairphase[:, None]*W*pairphase.conj()[None, :]
    if ignore_v3:
        v[:] = 0
    if ignore_v4:
        W[:] = 0
    pairs = np.array(obj.unique_pairs)
    k = pairs[:, 0]
    multiplicity = np.where(pairs[:, 0] == pairs[:, 1], 1, 2)
    w1, w2 = obj.w_q[0, pairs[:, 0]], obj.w_q[0, pairs[:, 1]]
    n1 = 0 if thermal == 0 else 1/np.expm1(w1/thermal)
    n2 = 0 if thermal == 0 else 1/np.expm1(w2/thermal)
    chia = np.sqrt(np.maximum(-(w1-w2)*(n1-n2)/(4*w1*w2), 0))
    chib = np.sqrt((w1+w2)*(1+n1+n2)/(4*w1*w2))
    chi = np.concatenate([chia, -chib])
    k = np.tile(k, 2)
    multiplicity = np.tile(multiplicity, 2)
    harmonic_pair = np.concatenate([(w1-w2)**2, (w1+w2)**2])
    expected = np.zeros_like(matrix)
    expected[0, 0] = -ref.frequency(Q)**2
    expected[1:, 0] = chi*v[k]
    expected[0, 1:] = chi*v[k].conj()*multiplicity
    expected[1:, 1:] = -np.diag(harmonic_pair) - chi[:, None]*W[np.ix_(k, k)]*chi[None, :]*multiplicity
    errors = {'operator': ref.relative(matrix, expected)}
    errors['D3'] = (np.linalg.norm(matrix[0, 1:])+np.linalg.norm(matrix[1:, 0])
                    if ignore_v3 else max(ref.relative(matrix[0, 1:], expected[0, 1:]),
                                         ref.relative(matrix[1:, 0], expected[1:, 0])))
    d4 = matrix[1:, 1:]+np.diag(harmonic_pair)
    d4exact = expected[1:, 1:]+np.diag(harmonic_pair)
    errors['D4'] = float(np.linalg.norm(d4)) if ignore_v4 else ref.relative(d4, d4exact)
    errors['response'] = errors['self_energy'] = errors['pair_resolvent'] = 0.
    for z in np.linspace(.3, 2.8, 41)+.04j:
        inverse = np.linalg.inv(z*z*np.eye(len(matrix))+matrix)
        exact_inverse = np.linalg.inv(z*z*np.eye(len(matrix))+expected)
        errors['pair_resolvent'] = max(errors['pair_resolvent'],
            ref.relative(inverse[1:, 1:], exact_inverse[1:, 1:]))
        sigma, G = ref.response(nf, Q, z, v, W, thermal=thermal)
        errors['response'] = max(errors['response'], abs(inverse[0, 0]-G)/abs(G))
        if not ignore_v3:
            observed = z*z-ref.frequency(Q)**2-1/inverse[0, 0]
            errors['self_energy'] = max(errors['self_energy'], abs(observed-sigma)/max(abs(sigma), 1e-10))
    if check:
        assert errors['operator'] < 1e-11, errors
        assert max(errors[key] for key in ('D3', 'D4', 'response', 'pair_resolvent')) < 1e-10, errors
        assert errors['self_energy'] < 1e-8, errors
    return matrix, expected, errors


def rotated_action():
    """Dense, nonsymmetric mode blocks at nonzero Q; test all source sectors."""
    nc, nf, Q, thermal = 9, 18, 1/3, .3
    _, exact, _ = probe(nc, nf, Q, True, thermal=thermal)
    obj = make_operator(nc, nf, Q, True, thermal)
    vectors = []
    for n, E, w, fields in ((nc, obj.cpols_q, obj.cw_q, (obj.X_q, obj.Y_q)),
                             (nf, obj.pols_q, obj.w_q, ())):
        amplitudes = []
        for iq in range(n):
            q = iq/n
            # Unitary mixing of three degenerate optical modes, not just phases.
            theta = .71+.27*np.cos(2*np.pi*q)
            phi = .43*np.sin(2*np.pi*q)
            U = np.eye(6, dtype=complex)
            c, s = np.cos(theta), np.sin(theta)
            U[:3, :3] = [[c, s, 0], [-s*c, c*c, s], [s*s, -s*c, c]]
            U[:3, :3] *= np.exp(1j*phi*np.arange(3))[None, :]
            assert np.allclose(U.conj().T @ U, np.eye(6))
            E[:, :, iq] = E[:, :, iq] @ U
            w[:3, iq] = w[0, iq]
            for field in fields:
                field[iq] = field[iq] @ U.conj()
            amplitudes.append(U.conj().T[:, 0])
        vectors.append(np.array(amplitudes))
    t = vectors[1]
    size = obj.get_psi_size()
    S = np.zeros((size, len(exact)), dtype=complex)
    S[:6, 0] = t[obj.iq_pert]
    column = 1
    for sector in ('a', 'b'):
        for p, (k1, k2) in enumerate(obj.unique_pairs):
            obj.set_block_in_psi(p, np.outer(t[k1], t[k2]), sector, S[:, column])
            column += 1
    # Independent packed-pair metric: ordered partners and off-diagonal
    # entries of a symmetric same-q block each occur twice in the full sum.
    metric = np.ones(size)
    weights = [1]
    for sector in ('a', 'b'):
        for p, (k1, k2) in enumerate(obj.unique_pairs):
            offset = obj.get_block_offset(p, sector)
            if k1 != k2:
                metric[offset:offset+36] = 2
                weights.append(2)
            else:
                metric[offset:offset+21] = [1 if a == b else 2
                                           for a in range(6) for b in range(a, 6)]
                weights.append(1)
    projection = (S.conj().T*metric)/np.array(weights)[:, None]
    assert np.allclose(projection @ S, np.eye(len(exact)))
    pairs = np.array(obj.unique_pairs)
    w1, w2 = obj.w_q[0, pairs[:, 0]], obj.w_q[0, pairs[:, 1]]
    harmonic = -np.concatenate([[ref.frequency(Q)**2], (w1-w2)**2, (w1+w2)**2])
    anharm = exact-np.diag(harmonic)
    rng = np.random.default_rng(712)
    worst = 0.
    for _ in range(4):
        x = rng.normal(size=size)+1j*rng.normal(size=size)
        obj.psi = x
        actual = obj.apply_anharmonic_FT()
        expected = S @ (anharm @ (projection @ x))
        error = ref.relative(actual, expected)
        worst = max(worst, error)
        assert error < 1e-10
    return worst


if __name__ == '__main__':
    print('Maximum dense mode action error: {:.3e}'.format(rotated_action()))
