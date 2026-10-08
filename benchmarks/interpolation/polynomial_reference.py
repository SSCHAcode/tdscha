"""Deterministic finite-range optical-chain interpolation oracle (NumPy only).

Run: OPENBLAS_NUM_THREADS=1 python3 benchmarks/interpolation/polynomial_reference.py
Production checks are in test_polynomial_production.py and
production_validation.py. Units: hbar = mass = lattice spacing = 1.
"""
import json
import numpy as np


CUBIC = [(0.08, np.array([1., .35]))]
QUARTIC = [(0.12, np.array([1., .37, -.29, .21])),
           (0.04, np.array([1., -.23]))]


def frequency(q):
    return np.sqrt(1. + .8 * np.cos(2 * np.pi * np.asarray(q)))


def form(h, q):
    return np.exp(2j * np.pi * np.asarray(q)[..., None] * np.arange(len(h))) @ h


def counterterms(master_size=4096):
    """Evaluate once and reuse: these are coefficients of one fixed potential."""
    q = np.arange(master_size)/master_size
    return [float(np.mean(abs(form(h, q))**2/(2*frequency(q))))
            for _, h in CUBIC + QUARTIC]


def energy_force(x, covariances):
    """Actual periodic, finite-range, normal-ordered polynomial and force."""
    Kx = x + .4*(np.roll(x, 1)+np.roll(x, -1))
    energy, force = .5*np.dot(x, Kx), -Kx
    for i, (g, h) in enumerate(CUBIC + QUARTIC):
        s = sum(hr*np.roll(x, -r) for r, hr in enumerate(h))
        C = covariances[i]
        if i < len(CUBIC):
            energy += g/6*np.sum(s**3-3*C*s)
            ds = g/2*(s*s-C)
        else:
            energy += g/24*np.sum(s**4-6*C*s*s+3*C*C)
            ds = g/6*(s**3-3*C*s)
        for r, hr in enumerate(h):
            force -= hr*np.roll(ds, r)
    return float(energy), force


def vertex(terms, *momenta):
    """Analytic derivative of sum_n g/p! (sum_r h_r x_(n+r))**p."""
    return sum(g * np.prod(np.broadcast_arrays(*[form(h, q) for q in momenta]),
                           axis=0) for g, h in terms)


def vertices(n, Q):
    k = np.arange(n) / n
    v = vertex(CUBIC, -k, k-Q, Q) / np.sqrt(n)
    W = vertex(QUARTIC, -k[:, None], k[:, None]-Q,
               k[None, :], Q-k[None, :]) / n
    return v, W


def cardinal(n, points):
    """Independent exact trigonometric reconstruction, odd coarse meshes."""
    assert n % 2 == 1
    r = np.arange(-(n//2), n//2+1)
    return np.exp(2j*np.pi*(np.asarray(points)[:, None, None]
                          - np.arange(n)[None, :, None]/n)*r).mean(axis=2)


def interpolate_vertices(nc, nf, Q):
    v, W = vertices(nc, Q)
    P = cardinal(nc, np.arange(nf)/nf)
    return np.sqrt(nc/nf)*(P @ v), (nc/nf)*(P @ W @ P.conj().T)


def response(n, Q, z, v=None, W=None, quartic=True, thermal=0.):
    """Ordered full-BZ pair convention, Lambda includes the pair factor 1/2.

    Sigma = v^dagger [I - Lambda W]^-1 Lambda v.
    G = [z^2 - omega_Q^2 - Sigma]^-1. No quasiparticle approximation.
    """
    if v is None:
        v, W = vertices(n, Q)
    k = np.arange(n)/n
    w1, w2 = frequency(k), frequency(Q-k)
    n1 = np.zeros_like(w1) if thermal == 0 else 1/np.expm1(w1/thermal)
    n2 = np.zeros_like(w2) if thermal == 0 else 1/np.expm1(w2/thermal)
    lam = -((w1+w2)*(1+n1+n2)/((w1+w2)**2-z*z)
            - (w1-w2)*(n1-n2)/((w1-w2)**2-z*z))/(4*w1*w2)
    dressed = np.linalg.solve(np.eye(n)-lam[:, None]*W, lam*v) if quartic else lam*v
    sigma = np.vdot(v, dressed)
    return sigma, 1/(z*z-frequency(Q)**2-sigma)


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b), 1e-30))


def benchmark(nc=9, nf=36):
    zgrid = np.linspace(.3, 2.8, 81) + .04j
    report = {}
    x = np.random.default_rng(713).normal(size=nc)
    C = counterterms()
    _, force = energy_force(x, C)
    eps = 1e-5
    numerical = np.array([-(energy_force(x+eps*e, C)[0]
                           - energy_force(x-eps*e, C)[0])/(2*eps)
                          for e in np.eye(nc)])
    report["force_finite_difference_error"] = relative(force, numerical)
    assert report["force_finite_difference_error"] < 1e-8
    for Q in (0., 1/3):
        v, W = vertices(nf, Q)
        vi, Wi = interpolate_vertices(nc, nf, Q)
        row = {"D3_error": relative(vi, v), "D4_error": relative(Wi, W)}
        spectra = {}
        for quartic in (False, True):
            direct = np.array([response(nf, Q, z, quartic=quartic) for z in zgrid])
            interp = np.array([response(nf, Q, z, vi, Wi, quartic) for z in zgrid])
            coarse = np.array([response(nc, Q, z, quartic=quartic) for z in zgrid])
            label = "full" if quartic else "bubble"
            row[label+"_response_error"] = relative(interp, direct)
            row[label+"_self_energy_error"] = relative(interp[:, 0], direct[:, 0])
            row[label+"_coarse_quadrature_error"] = relative(coarse, direct)
            spectra[label] = direct
        row["beyond_bubble_self_energy_fraction"] = relative(
            spectra["full"][:, 0], spectra["bubble"][:, 0])
        report[str(Q)] = row
        assert row["D3_error"] < 1e-12
        assert row["D4_error"] < 1e-12
        assert row["full_response_error"] < 1e-11
        assert row["bubble_response_error"] < 1e-11
        assert row["full_self_energy_error"] < 1e-11
        assert row["bubble_self_energy_error"] < 1e-11
        assert row["beyond_bubble_self_energy_fraction"] > .01

    # A response is not finite range. Even with identical internal quadrature,
    # reconstructing G(Q,z) from coarse external Q samples need not be exact.
    Q = 1/18
    P = cardinal(nc, [Q])[0]
    exact = np.array([response(nf, Q, z)[1] for z in zgrid])
    sampled = np.array([[response(nf, q, z)[1] for z in zgrid]
                        for q in np.arange(nc)/nc])
    report["response_first_external_Q_error"] = relative(P @ sampled, exact)
    # Negative control: a mesh smaller than twice the cluster span aliases D4.
    _, bad = interpolate_vertices(3, nf, 0.)
    _, good = vertices(nf, 0.)
    report["underresolved_D4_error"] = relative(bad, good)
    assert report["underresolved_D4_error"] > .01
    return report


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
