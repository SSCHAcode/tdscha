# QSpaceHessian: Fast Free Energy Hessian in Q-Space

The `QSpaceHessian` class computes the free energy Hessian matrix (second derivative of free energy with respect to atomic displacements) by working in the Bloch (q-space) basis, where the problem decomposes into independent blocks at each q-point.

This is the **fastest and most memory-efficient** method available in the SSCHA ecosystem to compute the full anharmonic free energy Hessian, including fourth-order contributions. It supersedes both the direct `get_free_energy_hessian` from `python-sscha` and the real-space `StaticHessian` for systems with periodic boundary conditions.

## Why QSpaceHessian?

The free energy Hessian is defined as ([R. Bianco et al., Phys. Rev. B **96**, 014111 (2017)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.014111)):

$$
\frac{\partial^2 \mathcal{F}}{\partial \mathcal{R} \partial \mathcal{R}} = \overset{(2)}{D} - \overset{(3)}{D}\sqrt{-W}\left(\mathbb{1} + \sqrt{-W}\overset{(4)}{D}\sqrt{-W}\right)^{-1}\sqrt{-W}\,\overset{(3)}{D}
$$

The direct approach (`ensemble.get_free_energy_hessian()`) builds the $\overset{(4)}{D}$ matrix explicitly. This matrix lives in the space of mode *pairs* ($N_\text{modes}^2 \times N_\text{modes}^2$), which makes the method scale as:

| | Direct (`get_free_energy_hessian`) | **QSpaceHessian** |
|---|---|---|
| **Memory** | $O(N_\text{modes}^4)$ | $O(n_\text{bands}^4)$ |
| **Time** | $O(N_\text{modes}^6)$ | $O(n_\text{bands}^3 \times N_{q,\text{irr}})$ |

where $N_\text{modes} = N_q \times n_\text{bands}$ is the total number of modes in the supercell. Since $N_\text{modes}$ grows linearly with the number of unit cells $N_\text{cell}$, the speedup of QSpaceHessian scales as:

$$
\text{Speedup} \sim \frac{N_\text{cell}^3}{N_{q,\text{irr}}}
$$

For a $4\times 4\times 4$ supercell ($N_\text{cell} = 64$, $N_{q,\text{irr}} \approx 4$): the speedup is roughly **65,000x**, and the memory reduction makes calculations feasible that would otherwise require terabytes of RAM.

Even compared to the real-space `StaticHessian` (which already avoids building $\overset{(4)}{D}$ explicitly), QSpaceHessian is faster by a factor of $\sim N_\text{cell}$ because each iterative solve operates on $n_\text{bands}$-sized vectors instead of $N_\text{modes}$-sized ones, and point-group symmetry reduces the number of q-points to solve.

## Basic Usage

### Computing the Full Hessian

```python
import cellconstructor as CC
import cellconstructor.Phonons
import sscha.Ensemble
import tdscha.QSpaceHessian as QH

# Load the SSCHA dynamical matrix and ensemble
dyn = CC.Phonons.Phonons("dyn_pop1_", nqirr=3)
ensemble = sscha.Ensemble.Ensemble(dyn, T=300)
ensemble.load_bin("data/", population_id=1)

# Create and initialize the Hessian solver
hess = QH.QSpaceHessian(ensemble)
hess.init()

# Compute the full Hessian at all q-points
hessian_dyn = hess.compute_full_hessian()

# Save the result
hessian_dyn.save_qe("hessian_dyn_")

# Extract anharmonic phonon frequencies
w_hessian, pols = hessian_dyn.DiagonalizeSupercell()
print("Anharmonic frequencies (cm-1):", w_hessian * CC.Units.RY_TO_CM)
```

### Computing at a Single Q-Point

If you only need the Hessian at a specific q-point (e.g., to check for a soft mode at a high-symmetry point):

```python
hess = QH.QSpaceHessian(ensemble)
hess.init()

# Compute the Hessian at the Gamma point (iq=0)
H_gamma = hess.compute_hessian_at_q(iq=0)

# Eigenvalues give the squared renormalized frequencies
eigvals = np.linalg.eigvalsh(H_gamma)
freqs_cm = np.sqrt(np.abs(eigvals)) * np.sign(eigvals) * CC.Units.RY_TO_CM
print("Frequencies at Gamma (cm-1):", np.sort(freqs_cm))
```

### Comparison with the Direct Method

To verify your results or for small systems where both methods are feasible:

```python
# Direct method (from python-sscha)
hessian_direct = ensemble.get_free_energy_hessian(include_v4=True)

# Q-space method
hess = QH.QSpaceHessian(ensemble)
hess.init()
hessian_qspace = hess.compute_full_hessian()

# Compare eigenvalues at each q-point
for iq in range(len(dyn.q_tot)):
    w_direct = np.linalg.eigvalsh(hessian_direct.dynmats[iq])
    w_qspace = np.linalg.eigvalsh(hessian_qspace.dynmats[iq])
    print(f"iq={iq}: max diff = {np.max(np.abs(w_direct - w_qspace)):.2e}")
```

## Tuning the Solver

### Convergence Parameters

The iterative solver (GMRES with BiCGSTAB fallback) can be tuned:

```python
hessian_dyn = hess.compute_full_hessian(
    tol=1e-6,           # Convergence tolerance (default: 1e-6)
    max_iters=500,       # Maximum iterations per band (default: 500)
    use_preconditioner=True  # Harmonic preconditioner (default: True)
)
```

The harmonic preconditioner uses the inverse of the harmonic Liouvillian, which is diagonal in the mode basis, and dramatically reduces the number of iterations. Disabling it is not recommended.

### Dense Fallback

For problematic q-points where the iterative solver does not converge (e.g., systems near a phase transition where modes are very soft), a dense fallback can be enabled:

```python
hessian_dyn = hess.compute_full_hessian(
    dense_fallback=True  # WARNING: O(psi_size^2) memory
)
```

!!! warning
    The dense fallback builds the full Liouvillian matrix in memory. For large systems this can be extremely expensive. It is **disabled by default** and should only be used for small systems or debugging. If the iterative solver fails, first try increasing `max_iters` or loosening `tol`.

### Symmetries

By default, `QSpaceHessian` uses crystal symmetries (via `spglib`) to reduce the number of q-points to the irreducible set. The Hessian at symmetry-equivalent q-points is obtained by rotation, saving a factor of $|G|/N_{q,\text{irr}}$ in computation time. This can be disabled:

```python
hess.init(use_symmetries=False)  # Solve at every q-point independently
```

## Deep Dive: How It Works

### From Hessian to Linear System

The key insight ([L. Monacelli and F. Mauri, 2021](https://journals.aps.org/prb/abstract/10.1103/8611-5k5v)) is that the free energy Hessian can be reformulated as the inverse of the static susceptibility:

$$
\frac{\partial^2 \mathcal{F}}{\partial \mathcal{R}_a \partial \mathcal{R}_b} = \left[\mathcal{G}(\omega=0)\right]^{-1}_{ab}
$$

where $\mathcal{G}$ is the static Green's function. Instead of building and inverting the $\overset{(4)}{D}$ matrix, we define an auxiliary Liouvillian $\mathcal{L}$ that acts on a vector $(\Upsilon, \bar{\mathcal{R}})$ in the extended space of phonon displacements ($\bar{\mathcal{R}}$, dimension $N_\text{modes}$) and two-phonon amplitudes ($\Upsilon$, dimension $N_\text{modes}^2$):

$$
\mathcal{L} \begin{pmatrix} \Upsilon \\ \bar{\mathcal{R}} \end{pmatrix} = \begin{pmatrix} 0 \\ \boldsymbol{a} \end{pmatrix}
$$

Solving this system for each unit vector $\boldsymbol{a} = \boldsymbol{e}_i$ yields $\bar{\mathcal{R}} = \mathcal{G}\boldsymbol{e}_i$, i.e., column $i$ of the static susceptibility $\mathcal{G}$. The Hessian follows as $\mathcal{H} = \mathcal{G}^{-1}$.

### The Static Liouvillian

The Liouvillian $\mathcal{L}$ splits into a harmonic part (diagonal in the mode basis) and an anharmonic part (computed via ensemble averages):

$$
\mathcal{L}^\text{(har)} = \begin{pmatrix} -W^{-1} & 0 \\ 0 & \overset{(2)}{D} \end{pmatrix}, \qquad
\mathcal{L}^\text{(anh)} = \begin{pmatrix} \overset{(4)}{D} & \overset{(3)}{D} \\ \overset{(3)}{D} & 0 \end{pmatrix}
$$

The harmonic part is trivially invertible ($W^{-1}$ is the inverse of the two-phonon propagator, $\overset{(2)}{D}$ contains the squared SSCHA frequencies $\omega_\mu^2$). The anharmonic part is applied efficiently via importance-sampled ensemble averages, scaling as $O(N^2)$ per application instead of $O(N^4)$ to store $\overset{(4)}{D}$.

### Q-Space Block Diagonality

In the Bloch representation, translational symmetry makes $\mathcal{L}$ block-diagonal:

$$
\mathcal{L}(\mathbf{q}, \mathbf{q}') = \delta_{\mathbf{q}\mathbf{q}'}\, \mathcal{L}(\mathbf{q})
$$

Each block $\mathcal{L}(\mathbf{q})$ has dimension $n_\text{bands} + n_\text{bands}^2 \times n_\text{pairs}$ (where $n_\text{pairs}$ is the number of unique q-point pairs such that $\mathbf{q}_1 + \mathbf{q}_2 = \mathbf{q}$), independent of the supercell size. This is where the massive speedup comes from: instead of one huge linear system of size $N_\text{modes} + N_\text{modes}^2$, we solve $N_{q,\text{irr}}$ small systems of size $\sim n_\text{bands}^2$.

### Iterative Solver and Preconditioner

Each system $\mathcal{L}(\mathbf{q})\, \boldsymbol{x}_i = \boldsymbol{e}_i$ is solved iteratively with GMRES, preconditioned by the inverse of the harmonic Liouvillian:

$$
M = [\mathcal{L}^\text{(har)}]^{-1} = \begin{pmatrix} -W & 0 \\ 0 & [\overset{(2)}{D}]^{-1} \end{pmatrix}
$$

Since $\mathcal{L}$ is Hermitian with respect to a weighted inner product $\langle \boldsymbol{x} | \boldsymbol{y} \rangle = \boldsymbol{x}^\dagger D_\text{mask}\, \boldsymbol{y}$ (where $D_\text{mask}$ accounts for the multiplicity of off-diagonal vs. diagonal mode pairs), a similarity transformation $\tilde{\mathcal{L}} = D_\text{mask}^{1/2}\, \mathcal{L}\, D_\text{mask}^{-1/2}$ is applied to convert the problem to standard Hermitian form. This ensures proper convergence of the Krylov solvers and Hermiticity of the resulting Hessian.

### Handling of Acoustic Modes and Non-Self-Conjugate Q-Points

Special care is needed at q-points $\mathbf{q}$ for which the pair decomposition $\mathbf{q}_1 + \mathbf{q}_2 = \mathbf{q}$ involves the $\Gamma$ point ($\mathbf{q}_1 = \mathbf{0}$). This happens whenever $\mathbf{q}$ is not mapped to $-\mathbf{q}$ by a point-group symmetry (i.e., $\mathbf{q} \neq -\mathbf{q} + \mathbf{G}$ for any reciprocal lattice vector $\mathbf{G}$). In such cases, one of the q-point pairs is $(\mathbf{0}, \mathbf{q})$, and the three acoustic modes at $\Gamma$ (with $\omega = 0$) introduce a null space in the two-phonon propagator $W^{-1}$ (since $1/\Lambda \to 0$ when an acoustic mode is involved).

Anharmonic $\overset{(3)}{D}$ coupling can mix this null space with the displacement sector $\bar{\mathcal{R}}$, making the linear system $\mathcal{L}\boldsymbol{x} = \boldsymbol{e}_i$ formally inconsistent. QSpaceHessian handles this by projecting out acoustic-mode components from the inner product (setting $D_\text{mask} = 0$ for two-phonon entries where either mode is acoustic). This ensures the null space remains confined to the two-phonon sector and does not contaminate the displacement response.

### Hermiticity Enforcement

The static susceptibility $\mathcal{G}$ should be Hermitian by construction: $\mathcal{G}_{ab} = \mathcal{G}_{ba}^*$. However, finite convergence tolerances in the iterative solver can introduce small anti-Hermitian components. QSpaceHessian explicitly symmetrizes both the susceptibility $\mathcal{G} \to (\mathcal{G} + \mathcal{G}^\dagger)/2$ and the final Hessian $\mathcal{H} \to (\mathcal{H} + \mathcal{H}^\dagger)/2$ to guarantee physically meaningful (real) eigenvalues.

### Point-Group Symmetry Reduction

Crystal point-group symmetries $\{R|\mathbf{t}\}$ map q-points into stars: $\mathbf{q}' = R\mathbf{q}$. The Hessian at a rotated q-point is obtained from the irreducible representative:

$$
\mathcal{H}(\mathbf{q}') = D(R)\, \mathcal{H}(\mathbf{q}_\text{irr})\, D(R)^\dagger
$$

where $D(R)$ is the representation matrix of the symmetry operation in the phonon mode basis, accounting for atom permutations, Cartesian rotation, and Bloch phase factors.

## References

1. R. Bianco, I. Errea, L. Paulatto, M. Calandra, and F. Mauri, *Phys. Rev. B* **96**, 014111 (2017). [DOI: 10.1103/PhysRevB.96.014111](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.014111)
2. L. Monacelli, *Phys. Rev. B* **112**, 014109 (2025). [DOI: 10.1103/8611-5k5v](https://journals.aps.org/prb/abstract/10.1103/8611-5k5v)
