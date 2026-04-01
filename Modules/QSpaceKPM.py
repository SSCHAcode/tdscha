"""Q-space KPM built on top of QSpaceLanczos."""

from __future__ import print_function, division

import numpy as np

from tdscha.QSpaceLanczos import QSpaceLanczos, __EPSILON__


class QSpaceKPM(QSpaceLanczos):
    """Kernel Polynomial Method using the q-space Liouvillian."""

    def __init__(self, ensemble, lo_to_split=None, **kwargs):
        super().__init__(ensemble, lo_to_split=lo_to_split, **kwargs)
        self.__total_attributes__.extend([
            "kpm_moments", "kpm_vector_norm", "kpm_n_moments",
            "kpm_lambda_min", "kpm_lambda_max", "kpm_rescale_a",
            "kpm_rescale_b", "kpm_bound_factor"
        ])
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
            bound = (2 * bound_factor * max_w) ** 2
            bound *= (1 + edge_buffer)
            return -bound, bound
        if lambda_min >= lambda_max:
            raise ValueError("Invalid KPM bounds: lambda_min must be smaller than lambda_max")
        span = lambda_max - lambda_min
        return lambda_min - edge_buffer * span, lambda_max + edge_buffer * span

    def _apply_rescaled_L(self, vec):
        return (self.apply_full_L(vec) - self.kpm_rescale_b * vec) / self.kpm_rescale_a

    def run_KPM(self, n_moments, lambda_min=None, lambda_max=None, bound_factor=2.5,
                edge_buffer=1e-8, verbose=True):
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

        if self.kpm_n_moments > 1:
            v1 = self._apply_rescaled_L(v0)
            moments[1] = self._metric_dot(v0, v1, mask)
            vm, v = v0, v1
            for i in range(2, self.kpm_n_moments):
                vp = 2 * self._apply_rescaled_L(v) - vm
                moments[i] = self._metric_dot(v0, vp, mask)
                vm, v = v, vp

        self.kpm_moments = moments
        self.psi = psi0

        if verbose:
            print("KPM completed with {} moments".format(self.kpm_n_moments))

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
