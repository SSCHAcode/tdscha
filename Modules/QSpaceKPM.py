"""Q-space KPM built on top of QSpaceLanczos."""

from __future__ import print_function, division

import numpy as np

from tdscha.QSpaceLanczos import QSpaceLanczos, __EPSILON__
from cellconstructor.Settings import ParallelPrint as print
import cellconstructor.Settings as Parallel


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
            # Register KPM attributes
            self.__total_attributes__.extend([
                "kpm_moments", "kpm_vector_norm", "kpm_n_moments",
                "kpm_lambda_min", "kpm_lambda_max", "kpm_rescale_a",
                "kpm_rescale_b", "kpm_bound_factor"
            ])
            self._init_kpm_attributes()
            return
        
        super().__init__(ensemble, lo_to_split=lo_to_split, **kwargs)
        self.__total_attributes__.extend([
            "kpm_moments", "kpm_vector_norm", "kpm_n_moments",
            "kpm_lambda_min", "kpm_lambda_max", "kpm_rescale_a",
            "kpm_rescale_b", "kpm_bound_factor"
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
        self.kpm_moments = None
        self.kpm_vector_norm = None
        self.kpm_n_moments = 0
        self.kpm_lambda_min = None
        self.kpm_lambda_max = None
        self.kpm_rescale_a = None
        self.kpm_rescale_b = None
        self.kpm_bound_factor = None

    def _invalidate_kpm_cache(self):
        self.kpm_moments = None
        self.kpm_vector_norm = None
        self.kpm_n_moments = 0
        self.kpm_lambda_min = None
        self.kpm_lambda_max = None
        self.kpm_rescale_a = None
        self.kpm_rescale_b = None
        self.kpm_bound_factor = None

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

    def _apply_rescaled_L(self, vec):
        return (self.apply_full_L(vec) - self.kpm_rescale_b * vec) / self.kpm_rescale_a

    def run_KPM(self, n_moments, lambda_min=None, lambda_max=None, bound_factor=2.5,
                edge_buffer=1e-8, verbose=True):
        """Run the Kernel Polynomial Method to compute Chebyshev moments.

        Uses the two-vector Chebyshev trick to extract two moments per
        L application, halving the number of expensive operator calls.
        The identity  mu_{m+n} + mu_{|m-n|} = 2 <T_m v, T_n v>_M
        (valid because L is self-adjoint under the Wigner metric) gives:
            mu_{2n}   = 2 <t_n, t_n>_M - mu_0
            mu_{2n+1} = 2 <t_{n+1}, t_n>_M - mu_1
        """
        if n_moments < 1:
            raise ValueError("n_moments must be positive")
        if self.psi is None:
            raise ValueError("Prepare a perturbation before running KPM")

        mask = self.mask_dot_wigner(False)
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
        n_L_applications = 0

        if verbose:
            print("KPM: moment 0 = {:.8e}, norm = {:.8e}".format(
                moments[0], self.kpm_vector_norm))

        if self.kpm_n_moments > 1:
            v1 = self._apply_rescaled_L(v0)
            n_L_applications += 1
            moments[1] = self._metric_dot(v0, v1, mask)
            if verbose:
                print("KPM: moment 1 = {:.8e}".format(moments[1]))

            # Two-vector Chebyshev iteration: t_n stored in v, t_{n-1} in vm
            vm, v = v0, v1
            n = 1
            while 2 * n < self.kpm_n_moments:
                # Even moment from <t_n, t_n>_M (no L needed)
                moments[2 * n] = 2 * self._metric_dot(v, v, mask) - moments[0]
                if verbose:
                    print("KPM: moment {} = {:.8e}".format(
                        2 * n, moments[2 * n]))
                if 2 * n + 1 >= self.kpm_n_moments:
                    break
                # Odd moment needs t_{n+1}
                vp = 2 * self._apply_rescaled_L(v) - vm
                n_L_applications += 1
                moments[2 * n + 1] = (
                    2 * self._metric_dot(vp, v, mask) - moments[1])
                if verbose:
                    print("KPM: moment {} = {:.8e}".format(
                        2 * n + 1, moments[2 * n + 1]))
                vm, v = v, vp
                n += 1

        self.kpm_moments = moments
        self.psi = psi0

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
