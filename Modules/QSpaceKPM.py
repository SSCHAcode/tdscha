"""Q-space KPM built on top of QSpaceLanczos."""

from __future__ import print_function, division

import os
import sys
import time

import numpy as np

from tdscha.QSpaceLanczos import QSpaceLanczos, __EPSILON__
from cellconstructor.Settings import ParallelPrint as print
import cellconstructor.Settings as Parallel
import cellconstructor.Units as Units


class QSpaceKPM(QSpaceLanczos):
    """Kernel Polynomial Method using the q-space Liouvillian."""

    def __init__(self, ensemble, lo_to_split=None, **kwargs):
        # Handle ensemble=None case (for loading KPM files for plotting)
        if ensemble is None:
            # Call grandparent's __init__ with None to get bare initialization
            import tdscha.DynamicalLanczos as DL
            DL.Lanczos.__init__(self, ensemble=None)
            # Set default for use_wigner (needed for spectral function computation)
            self.use_wigner = True
            # Register KPM attributes AND QSpaceLanczos attributes needed for bare init
            self.__total_attributes__.extend([
                # KPM attributes
                "kpm_moments", "kpm_vector_norm", "kpm_n_moments",
                "kpm_lambda_min", "kpm_lambda_max", "kpm_rescale_a",
                "kpm_rescale_b", "kpm_bound_factor",
                "kpm_chebyshev_vm", "kpm_chebyshev_v",
                # QSpaceLanczos attributes needed for from_qspace_lanczos
                "n_q", "n_bands", "q_points", "w_q", "pols_q", "valid_modes_q",
                "m", "X_q", "Y_q", "rho",
                "iq_pert", "q_pair_map", "unique_pairs",
                "_psi_size", "_block_offsets_a", "_block_offsets_b", "_block_sizes",
                "n_syms_qspace", "_qspace_sym_data", "_qspace_sym_q_map",
                "_distributed", "_N_global", "_N_eff_global", "_N_local",
                "lo_to_split",
            ])
            self._init_kpm_attributes()
            return
        
        super().__init__(ensemble, lo_to_split=lo_to_split, **kwargs)
        self.__total_attributes__.extend([
            "kpm_moments", "kpm_vector_norm", "kpm_n_moments",
            "kpm_lambda_min", "kpm_lambda_max", "kpm_rescale_a",
            "kpm_rescale_b", "kpm_bound_factor",
            "kpm_chebyshev_vm", "kpm_chebyshev_v"
        ])
        self._init_kpm_attributes()

    def _init_kpm_attributes(self):
        self.kpm_moments = None
        self.kpm_vector_norm = None
        self.kpm_n_moments = 0
        self.kpm_lambda_min = None
        self.kpm_lambda_max = None
        self.kpm_rescale_a = None
        self.kpm_rescale_b = None
        self.kpm_bound_factor = None
        self.kpm_chebyshev_vm = None
        self.kpm_chebyshev_v = None

    def _invalidate_kpm_cache(self):
        self.kpm_moments = None
        self.kpm_vector_norm = None
        self.kpm_n_moments = 0
        self.kpm_lambda_min = None
        self.kpm_lambda_max = None
        self.kpm_rescale_a = None
        self.kpm_rescale_b = None
        self.kpm_bound_factor = None
        self.kpm_chebyshev_vm = None
        self.kpm_chebyshev_v = None

    def save_status(self, file):
        """Save KPM iteration state to NPZ file for checkpoint/restart.

        Saves all KPM parameters, moments, and Chebyshev recurrence vectors
        so that run_KPM can be continued from where it left off.

        Parameters
        ----------
        file : str
            Path to the output file. '.npz' extension is added if missing.
        """
        if self.kpm_moments is None:
            raise ValueError("Run run_KPM before saving status")
        Parallel.barrier()
        if Parallel.am_i_the_master():
            if ".npz" not in file.lower():
                file += ".npz"
            save_dict = dict(
                kpm_n_moments=self.kpm_n_moments,
                kpm_vector_norm=self.kpm_vector_norm,
                kpm_lambda_min=self.kpm_lambda_min,
                kpm_lambda_max=self.kpm_lambda_max,
                kpm_rescale_a=self.kpm_rescale_a,
                kpm_rescale_b=self.kpm_rescale_b,
                kpm_bound_factor=self.kpm_bound_factor,
                kpm_moments=self.kpm_moments,
            )
            if self.kpm_chebyshev_vm is not None:
                save_dict["kpm_chebyshev_vm"] = self.kpm_chebyshev_vm
            if self.kpm_chebyshev_v is not None:
                save_dict["kpm_chebyshev_v"] = self.kpm_chebyshev_v
            np.savez_compressed(file, **save_dict)

    def load_status(self, file):
        """Load KPM iteration state from NPZ file.

        Restores all KPM parameters, moments, and (if present) the Chebyshev
        recurrence vectors needed for continuation with run_KPM.

        Parameters
        ----------
        file : str
            Path to the NPZ file. '.npz' extension is added if missing.
        """
        Parallel.barrier()
        if ".npz" not in file.lower():
            file += ".npz"
        if Parallel.am_i_the_master():
            if not os.path.exists(file):
                raise IOError("KPM status file not found: {}".format(file))
            data = dict(np.load(file, allow_pickle=True))
        else:
            data = None
        data = Parallel.broadcast(data)

        self.kpm_n_moments = int(data["kpm_n_moments"])
        self.kpm_vector_norm = float(data["kpm_vector_norm"])
        self.kpm_lambda_min = float(data["kpm_lambda_min"])
        self.kpm_lambda_max = float(data["kpm_lambda_max"])
        self.kpm_rescale_a = float(data["kpm_rescale_a"])
        self.kpm_rescale_b = float(data["kpm_rescale_b"])
        self.kpm_bound_factor = float(data["kpm_bound_factor"])
        self.kpm_moments = data["kpm_moments"]
        # Chebyshev vectors for continuation (may be absent in old files)
        self.kpm_chebyshev_vm = data.get("kpm_chebyshev_vm", None)
        self.kpm_chebyshev_v = data.get("kpm_chebyshev_v", None)

    def reset_q(self):
        super().reset_q()
        self._invalidate_kpm_cache()

    def _metric_dot(self, v1, v2, mask=None):
        if mask is None:
            mask = self.mask_dot_wigner(False)
        return np.real(np.conj(v1).dot(v2 * mask))

    def _get_kpm_bounds(self, lambda_min=None, lambda_max=None, bound_factor=2.5, edge_buffer=1e-8):
        if (lambda_min is None) != (lambda_max is None):
            raise ValueError("lambda_min and lambda_max must be both specified or both omitted")
        if lambda_min is None:
            max_w = np.max(np.abs(self.w_q[self.valid_modes_q]))
            spectral_width = (2 * max_w) ** 2
            margin = (bound_factor - 1) * spectral_width
            if self.use_wigner:
                # Wigner eigenvalues in [-(2*w_max)^2, 0]
                lmin = -spectral_width - margin
                lmax = margin
            else:
                # Non-Wigner eigenvalues in [0, (2*w_max)^2]
                lmin = -margin
                lmax = spectral_width + margin
            span = lmax - lmin
            return lmin - edge_buffer * span, lmax + edge_buffer * span
        if lambda_min >= lambda_max:
            raise ValueError("Invalid KPM bounds: lambda_min must be smaller than lambda_max")
        span = lambda_max - lambda_min
        return lambda_min - edge_buffer * span, lambda_max + edge_buffer * span

    def estimate_kpm_steps(self, precision_cm, bound_factor=1.2, regularization="jackson"):
        """Estimate the number of KPM steps needed to achieve desired frequency precision.

        The KPM resolution in eigenvalue space (λ = -ω²) scales as:
            Δλ ≈ π × rescale_a / n_moments (for Jackson kernel)
        
        Converting to frequency precision δω via dλ/dω = -2ω:
            δλ = 2 × ω_min × δω
        
        Therefore, the required number of moments is:
            n_moments ≈ π × rescale_a / (2 × ω_min × δω)
        
        where rescale_a ≈ 0.5 × bound_factor × (2 × ω_max)²

        Parameters
        ----------
        precision_cm : float
            Desired frequency precision in cm⁻¹. The KPM will be able to resolve
            peaks separated by at least this amount at the smallest frequency.
        bound_factor : float, default=1.2
            The bound factor to use for KPM bounds. Must be > 1.0.
        regularization : str, default="jackson"
            The regularization kernel type. Currently only "jackson" is supported.

        Returns
        -------
        int
            Estimated number of KPM steps (moments) required.

        Raises
        ------
        ValueError
            If precision_cm is not positive, bound_factor <= 1.0, or if
            no perturbation has been prepared (iq_pert is None).
        """
        if precision_cm <= 0:
            raise ValueError("precision_cm must be positive")
        if bound_factor <= 1.0:
            raise ValueError("bound_factor must be > 1.0")
        if self.iq_pert is None:
            raise ValueError("Must prepare a perturbation before estimating KPM steps")
        
        # Get frequency range
        max_w = np.max(np.abs(self.w_q[self.valid_modes_q]))
        
        # Get smallest non-zero frequency at the perturbation q-point
        w_at_q = self.w_q[:, self.iq_pert]
        valid_at_q = self.valid_modes_q[:, self.iq_pert]
        w_min = np.min(np.abs(w_at_q[valid_at_q]))
        
        if w_min < __EPSILON__:
            raise ValueError("Smallest frequency at q-point is effectively zero; cannot estimate steps")
        
        # Convert precision from cm⁻¹ to Ry
        delta_omega = precision_cm / Units.RY_TO_CM
        
        # Compute KPM bounds and rescale_a
        lambda_min, lambda_max = self._get_kpm_bounds(
            lambda_min=None, lambda_max=None, 
            bound_factor=bound_factor, edge_buffer=1e-8)
        rescale_a = 0.5 * (lambda_max - lambda_min)
        
        # Resolution in λ-space: δλ = 2 × ω_min × δω
        delta_lambda = 2.0 * w_min * delta_omega
        
        # For Jackson kernel, resolution is approximately π × rescale_a / n_moments
        # So n_moments ≈ π × rescale_a / δλ
        if regularization.lower() == "jackson":
            n_moments = int(np.ceil(np.pi * rescale_a / delta_lambda))
        else:
            # For no regularization, resolution is better but Gibbs oscillations occur
            # Use a conservative estimate (same as Jackson)
            n_moments = int(np.ceil(np.pi * rescale_a / delta_lambda))
        
        # Ensure at least a minimum number of steps
        n_moments = max(n_moments, 8)
        
        return n_moments

    def _apply_rescaled_L(self, vec):
        return (self.apply_full_L(vec) - self.kpm_rescale_b * vec) / self.kpm_rescale_a

    def run_KPM(self, n_moments, lambda_min=None, lambda_max=None, bound_factor=1.2,
                edge_buffer=1e-8, verbose=True):
        """Run the Kernel Polynomial Method to compute Chebyshev moments.

        Uses the two-vector Chebyshev trick to extract two moments per
        L application, halving the number of expensive operator calls.
        The identity  mu_{m+n} + mu_{|m-n|} = 2 <T_m v, T_n v>_M
        (valid because L is self-adjoint under the Wigner metric) gives:
            mu_{2n}   = 2 <t_n, t_n>_M - mu_0
            mu_{2n+1} = 2 <t_{n+1}, t_n>_M - mu_1

        If Chebyshev recurrence vectors are available from a previous run
        (via save_status/load_status), the iteration continues from where
        it left off rather than restarting from scratch.

        Parameters
        ----------
        n_moments : int
            Total number of Chebyshev moments to compute.
        lambda_min, lambda_max : float, optional
            Explicit bounds for the KPM. If not provided, bounds are
            estimated automatically from the maximum phonon frequency.
            Ignored during continuation (previous bounds are reused).
        bound_factor : float, default=1.2
            Factor controlling the width of the KPM bounds relative to
            the estimated spectral width. Smaller values (closer to 1.0)
            give better resolution but require the eigenvalue to be within
            the bounds. Values > 1.5 may cause significant baseline artifacts.
            Ignored during continuation.
        edge_buffer : float, default=1e-8
            Small buffer added to the bounds to avoid edge effects.
            Ignored during continuation.
        verbose : bool, default=True
            Print progress information.
        """
        if n_moments < 1:
            raise ValueError("n_moments must be positive")
        if self.psi is None:
            raise ValueError("Prepare a perturbation before running KPM")

        mask = self.mask_dot_wigner(False)
        n_L_applications = 0

        # Check for continuation from a previous run
        existing = self.kpm_n_moments
        can_continue = (existing > 0 and self.kpm_moments is not None
                        and self.kpm_chebyshev_v is not None)

        if can_continue and n_moments <= existing:
            if verbose:
                print("KPM: already have {} moments, {} requested. Nothing to do.".format(
                    existing, n_moments))
            return

        if can_continue:
            # --- Continuation path ---
            if verbose:
                print("KPM: continuing from {} to {} moments".format(existing, n_moments))

            moments = np.zeros(n_moments, dtype=np.float64)
            moments[:existing] = self.kpm_moments[:existing]
            vm = self.kpm_chebyshev_vm.copy()
            v = self.kpm_chebyshev_v.copy()

            if existing == 1:
                # Only moment[0]=1.0 exists, v=T_0. Need to compute T_1 and moment[1].
                v1 = self._apply_rescaled_L(v)
                n_L_applications += 1
                moments[1] = self._metric_dot(vm, v1, mask)
                # vm was T_{-1} placeholder; now set vm=T_0, v=T_1
                vm, v = v, v1
                n_start = 1
                if verbose:
                    print("KPM: moment 1 = {:.8e}".format(moments[1]))
            elif existing % 2 == 0:
                # Even count: last step completed cleanly.
                # vm = T_{n-1}, v = T_n where n = existing // 2
                n_start = existing // 2
            else:
                # Odd count: broke mid-step after even moment.
                # vm = T_{n-1}, v = T_n where n = existing // 2
                # Need to compute the missing odd moment and advance.
                n_half = existing // 2
                # Compute T_{n+1} to get the odd moment
                vp = 2 * self._apply_rescaled_L(v) - vm
                n_L_applications += 1
                moments[2 * n_half + 1] = (
                    2 * self._metric_dot(vp, v, mask) - moments[1])
                if verbose:
                    print("KPM: moment {} = {:.8e} (completing interrupted step)".format(
                        2 * n_half + 1, moments[2 * n_half + 1]))
                vm, v = v, vp
                n_start = n_half + 1

            self.kpm_n_moments = int(n_moments)

            # Continue the main loop
            n = n_start
            while 2 * n < self.kpm_n_moments:
                if verbose:
                    print("\n ===== KPM STEP {} =====\n".format(n))
                    sys.stdout.flush()

                moments[2 * n] = 2 * self._metric_dot(v, v, mask) - moments[0]
                if verbose:
                    print("KPM: moment {} = {:.8e}".format(2 * n, moments[2 * n]))
                if 2 * n + 1 >= self.kpm_n_moments:
                    break

                t1 = time.time()
                vp = 2 * self._apply_rescaled_L(v) - vm
                t2 = time.time()
                n_L_applications += 1
                moments[2 * n + 1] = (
                    2 * self._metric_dot(vp, v, mask) - moments[1])
                if verbose:
                    print("Time for L application: {:.3f} s".format(t2 - t1))
                    print("KPM: moment {} = {:.8e}".format(
                        2 * n + 1, moments[2 * n + 1]))
                    print("KPM step {} completed.".format(n))

                vm, v = v, vp
                n += 1

        else:
            # --- Fresh start path ---
            psi0 = self.psi.copy()
            norm_sq = self._metric_dot(psi0, psi0, mask)
            if norm_sq <= __EPSILON__ or np.isnan(norm_sq):
                raise ValueError("Prepare a non-zero perturbation before running KPM")

            self.kpm_lambda_min, self.kpm_lambda_max = self._get_kpm_bounds(
                lambda_min=lambda_min, lambda_max=lambda_max,
                bound_factor=bound_factor, edge_buffer=edge_buffer)
            self.kpm_rescale_a = 0.5 * (self.kpm_lambda_max - self.kpm_lambda_min)
            self.kpm_rescale_b = 0.5 * (self.kpm_lambda_max + self.kpm_lambda_min)
            self.kpm_vector_norm = np.sqrt(norm_sq)
            self.kpm_n_moments = int(n_moments)
            self.kpm_bound_factor = bound_factor

            v0 = psi0 / self.kpm_vector_norm
            moments = np.zeros(self.kpm_n_moments, dtype=np.float64)
            moments[0] = 1.0

            if verbose:
                print("KPM: moment 0 = {:.8e}, norm = {:.8e}".format(
                    moments[0], self.kpm_vector_norm))

            vm, v = v0, v0  # placeholder until v1 is computed
            if self.kpm_n_moments > 1:
                v1 = self._apply_rescaled_L(v0)
                n_L_applications += 1
                moments[1] = self._metric_dot(v0, v1, mask)
                if verbose:
                    print("KPM: moment 1 = {:.8e}".format(moments[1]))

                vm, v = v0, v1
                n = 1
                while 2 * n < self.kpm_n_moments:
                    if verbose:
                        print("\n ===== KPM STEP {} =====\n".format(n))
                        sys.stdout.flush()

                    moments[2 * n] = 2 * self._metric_dot(v, v, mask) - moments[0]
                    if verbose:
                        print("KPM: moment {} = {:.8e}".format(
                            2 * n, moments[2 * n]))
                    if 2 * n + 1 >= self.kpm_n_moments:
                        break

                    t1 = time.time()
                    vp = 2 * self._apply_rescaled_L(v) - vm
                    t2 = time.time()
                    n_L_applications += 1
                    moments[2 * n + 1] = (
                        2 * self._metric_dot(vp, v, mask) - moments[1])
                    if verbose:
                        print("Time for L application: {:.3f} s".format(t2 - t1))
                        print("KPM: moment {} = {:.8e}".format(
                            2 * n + 1, moments[2 * n + 1]))
                        print("KPM step {} completed.".format(n))

                    vm, v = v, vp
                    n += 1

            self.psi = psi0

        # Save Chebyshev recurrence vectors for potential continuation
        self.kpm_chebyshev_vm = vm.copy()
        self.kpm_chebyshev_v = v.copy()
        self.kpm_moments = moments

        if verbose:
            print("KPM completed: {} moments, {} L applications".format(
                self.kpm_n_moments, n_L_applications))

    def save_kpm(self, file):
        """
        Save KPM moments and parameters to a single text file.
        The first line is a comment with all parameters, followed by the moments.
        Only master process saves the file.
        """
        if self.kpm_moments is None:
            raise ValueError("Run run_KPM before saving")
        Parallel.barrier()
        if not Parallel.am_i_the_master():
            return
        header = "kpm_n_moments={} kpm_vector_norm={} kpm_lambda_min={} kpm_lambda_max={} kpm_rescale_a={} kpm_rescale_b={}".format(
            self.kpm_n_moments, self.kpm_vector_norm,
            self.kpm_lambda_min, self.kpm_lambda_max,
            self.kpm_rescale_a, self.kpm_rescale_b)
        np.savetxt(file, self.kpm_moments, header=header)

    def load_kpm(self, file):
        """
        Load KPM moments and parameters from file.
        Parses the header to restore state for get_spectral_function_KPM.
        Only master process loads and broadcasts to others.
        """
        Parallel.barrier()
        if Parallel.am_i_the_master():
            with open(file, 'r') as f:
                header_line = f.readline()
            if not header_line.startswith('#'):
                raise ValueError("Invalid KPM save file: missing header")
            parts = header_line[1:].strip().split()
            params = {}
            for part in parts:
                if '=' in part:
                    key, val = part.split('=')
                    try:
                        params[key] = float(val)
                    except ValueError:
                        params[key] = int(val)
            moments = np.loadtxt(file)
            data = {
                'kpm_n_moments': int(params['kpm_n_moments']),
                'kpm_vector_norm': params['kpm_vector_norm'],
                'kpm_lambda_min': params['kpm_lambda_min'],
                'kpm_lambda_max': params['kpm_lambda_max'],
                'kpm_rescale_a': params['kpm_rescale_a'],
                'kpm_rescale_b': params['kpm_rescale_b'],
                'kpm_moments': moments
            }
        else:
            data = None
        data = Parallel.broadcast(data)
        self.kpm_n_moments = data['kpm_n_moments']
        self.kpm_vector_norm = data['kpm_vector_norm']
        self.kpm_lambda_min = data['kpm_lambda_min']
        self.kpm_lambda_max = data['kpm_lambda_max']
        self.kpm_rescale_a = data['kpm_rescale_a']
        self.kpm_rescale_b = data['kpm_rescale_b']
        self.kpm_moments = data['kpm_moments']

    @staticmethod
    def _jackson_kernel(n_moments):
        n = np.arange(n_moments, dtype=np.float64)
        phi = np.pi / (n_moments + 1)
        return ((n_moments - n + 1) * np.cos(n * phi) + np.sin(n * phi) / np.tan(phi)) / (n_moments + 1)

    def _get_kpm_damping(self, regularization="jackson", damping_factors=None):
        if self.kpm_moments is None:
            raise ValueError("Run run_KPM before requesting the spectral function")
        if damping_factors is not None:
            g = np.asarray(damping_factors, dtype=np.float64)
        elif regularization is None or str(regularization).lower() == "none":
            g = np.ones(self.kpm_n_moments, dtype=np.float64)
        elif str(regularization).lower() == "jackson":
            g = self._jackson_kernel(self.kpm_n_moments)
        else:
            raise ValueError("Unknown KPM regularization: {}".format(regularization))
        if len(g) != self.kpm_n_moments:
            raise ValueError("Damping factors must have length {}".format(self.kpm_n_moments))
        return g

    def get_spectral_function_KPM(self, w_array, regularization="jackson", damping_factors=None):
        g = self._get_kpm_damping(regularization=regularization, damping_factors=damping_factors)
        w_array = np.asarray(w_array, dtype=np.float64)
        lambdas = -(w_array ** 2) if self.use_wigner else (w_array ** 2)
        x = (lambdas - self.kpm_rescale_b) / self.kpm_rescale_a
        inside = np.abs(x) < 1
        spectral = np.zeros_like(w_array, dtype=np.float64)
        if not np.any(inside):
            return spectral

        xin = np.clip(x[inside], -1 + 1e-15, 1 - 1e-15)
        t0 = np.ones_like(xin)
        rho = g[0] * self.kpm_moments[0] * t0
        if self.kpm_n_moments > 1:
            t1 = xin.copy()
            rho += 2 * g[1] * self.kpm_moments[1] * t1
            for i in range(2, self.kpm_n_moments):
                t2 = 2 * xin * t1 - t0
                rho += 2 * g[i] * self.kpm_moments[i] * t2
                t0, t1 = t1, t2
        rho /= np.pi * self.kpm_rescale_a * np.sqrt(1 - xin ** 2)
        rho *= self.kpm_vector_norm ** 2
        spectral[inside] = np.pi * rho
        return spectral

    @classmethod
    def from_qspace_lanczos(cls, qlanc):
        """Create a QSpaceKPM from an existing QSpaceLanczos object.

        This method reuses the qlanc object and extends it with KPM attributes.
        The object shares the same data (X_q, Y_q, rho, etc.) with the
        input QSpaceLanczos object.

        Parameters
        ----------
        qlanc : QSpaceLanczos
            An initialized QSpaceLanczos object.

        Returns
        -------
        QSpaceKPM
            A QSpaceKPM object with the same underlying data.
        """
        # Get all attributes from qlanc (including those not in __total_attributes__)
        # that are needed for KPM operation
        attrs_to_copy = [
            # Distributed mode attributes
            '_distributed', '_N_global', '_N_eff_global', '_N_local',
            # Ensemble data
            'ensemble', 'T', 'N', 'N_eff', 'rho',
            # Q-space specific
            'n_q', 'n_bands', 'q_points', 'w_q', 'pols_q', 'valid_modes_q',
            'm', 'X_q', 'Y_q',
            # Lanczos state
            'iq_pert', 'q_pair_map', 'unique_pairs',
            '_psi_size', '_block_offsets_a', '_block_offsets_b', '_block_sizes',
            'n_syms_qspace', '_qspace_sym_data', '_qspace_sym_q_map',
            # Flags
            'ignore_v3', 'ignore_v4', 'ignore_harmonic', 'use_wigner',
            # Dynamical matrix
            'dyn', 'uci_structure', 'super_structure',
            # Psi-related
            'psi', 'initialized', 'verbose',
        ]
        
        # Create a new KPM instance (bare initialization with ensemble=None)
        kpm = cls(None, lo_to_split=qlanc.lo_to_split if hasattr(qlanc, 'lo_to_split') else None)
        
        # Copy all relevant attributes from qlanc to kpm
        for attr in attrs_to_copy:
            if hasattr(qlanc, attr):
                val = getattr(qlanc, attr)
                try:
                    setattr(kpm, attr, val)
                except Exception as e:
                    # Log but continue - some attributes may not be copyable
                    pass
        
        # Also copy any additional attributes that qlanc has
        for attr in qlanc.__total_attributes__:
            if attr not in attrs_to_copy and hasattr(qlanc, attr):
                val = getattr(qlanc, attr)
                try:
                    setattr(kpm, attr, val)
                except Exception as e:
                    pass
        
        # Ensure __total_attributes__ includes all necessary attributes
        for attr in attrs_to_copy:
            if attr not in kpm.__total_attributes__:
                kpm.__total_attributes__.append(attr)

        # Invalidate KPM cache to ensure fresh start
        kpm._invalidate_kpm_cache()

        return kpm
