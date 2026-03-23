# Q-Space Lanczos Julia Extension
# ================================
#
# This module implements the q-space perturbed average calculations
# in Julia for performance. The q-space version exploits Bloch's
# theorem so that translations are handled via Fourier transform,
# and only point-group symmetries remain.
#
# The q-space ensemble data X_q and Y_q are complex arrays indexed
# as (n_q, n_configs, n_bands).
#
# Three separate functions mirror the real-space tdscha_core.jl:
#   get_d2v_from_R_pert_qspace  — D3 contribution to d2v
#   get_d2v_from_Y_pert_qspace  — D4 contribution to d2v
#   get_f_from_Y_pert_qspace    — D4 contribution to f_pert

using SparseArrays
using LinearAlgebra
using LinearAlgebra.BLAS

if !isdefined(@__MODULE__, :RY_TO_K_Q)
    const RY_TO_K_Q = 157887.32400374097
end

# Global cache for q-space symmetry matrices (complex)
if !isdefined(@__MODULE__, :_cached_qspace_symmetries)
    const _cached_qspace_symmetries = Ref{Union{Nothing,Vector{SparseMatrixCSC{ComplexF64,Int32}}}}(nothing)
end


"""
    get_d2v_from_R_pert_qspace(...)

D3 contribution to d2v from R^(1) perturbation.
Mirrors get_d2v_dR2_from_R_pert_sym_fast in tdscha_core.jl.

For each (config, sym):
  weight_R  = sum_nu f_Y[nu,iq_pert] * x_rot[iq_pert,nu] * conj(R1[nu]) * rho/3
  weight_Rf = sum_nu conj(R1[nu]) * y_rot[iq_pert,nu] * rho/3

  For each pair p = (q1,q2):
    d2v[p] += -weight_R  * (r1_q1 * conj(r2_q2)^T + r2_q1 * conj(r1_q2)^T)
    d2v[p] += -weight_Rf * r1_q1 * conj(r1_q2)^T

where r1 = f_Y * x, r2 = y.
"""
function get_d2v_from_R_pert_qspace(
    X_q::Array{ComplexF64,3},
    Y_q::Array{ComplexF64,3},
    f_Y::Matrix{Float64},
    rho::Vector{Float64},
    R1::Vector{ComplexF64},
    symmetries::Vector{SparseMatrixCSC{ComplexF64,Int32}},
    iq_pert::Int64,
    unique_pairs::Matrix{Int32},
    n_bands::Int64,
    n_q::Int64,
    start_index::Int64,
    end_index::Int64
)
    n_pairs = size(unique_pairs, 1)
    n_syms = length(symmetries)
    n_total = n_q * n_bands
    N_eff = sum(rho)

    # Output
    d2v_blocks = [zeros(ComplexF64, n_bands, n_bands) for _ in 1:n_pairs]

    # Buffers
    x_buf = zeros(ComplexF64, n_total)
    y_buf = zeros(ComplexF64, n_total)

    for bigindex in start_index:end_index
        i_config = div(bigindex - 1, n_syms) + 1
        j_sym = mod(bigindex - 1, n_syms) + 1

        # Build combined vector for this config
        for iq in 1:n_q
            for nu in 1:n_bands
                idx = (iq - 1) * n_bands + nu
                x_buf[idx] = X_q[iq, i_config, nu]
                y_buf[idx] = Y_q[iq, i_config, nu]
            end
        end

        # Apply symmetry
        x_rot = symmetries[j_sym] * x_buf
        y_rot = symmetries[j_sym] * y_buf

        # Views at q_pert
        x_pert = view(x_rot, (iq_pert-1)*n_bands+1:iq_pert*n_bands)
        y_pert = view(y_rot, (iq_pert-1)*n_bands+1:iq_pert*n_bands)

        # weight_R = sum_nu f_Y[nu,iq_pert] * x_pert[nu] * conj(R1[nu]) * rho/3
        weight_R = zero(ComplexF64)
        for nu in 1:n_bands
            weight_R += f_Y[nu, iq_pert] * x_pert[nu] * conj(R1[nu])
        end
        weight_R *= rho[i_config] / 3.0

        # weight_Rf = sum_nu conj(R1[nu]) * y_pert[nu] * rho/3
        weight_Rf = zero(ComplexF64)
        for nu in 1:n_bands
            weight_Rf += conj(R1[nu]) * y_pert[nu]
        end
        weight_Rf *= rho[i_config] / 3.0

        # Accumulate d2v for each pair
        for p in 1:n_pairs
            iq1 = unique_pairs[p, 1]
            iq2 = unique_pairs[p, 2]

            x_q1 = view(x_rot, (iq1-1)*n_bands+1:iq1*n_bands)
            y_q1 = view(y_rot, (iq1-1)*n_bands+1:iq1*n_bands)
            x_q2 = view(x_rot, (iq2-1)*n_bands+1:iq2*n_bands)
            y_q2 = view(y_rot, (iq2-1)*n_bands+1:iq2*n_bands)

            for nu1 in 1:n_bands
                r1_1 = f_Y[nu1, iq1] * x_q1[nu1]  # r1 at q1
                r2_1 = y_q1[nu1]                     # r2 at q1
                for nu2 in 1:n_bands
                    r1_2 = f_Y[nu2, iq2] * x_q2[nu2]  # r1 at q2
                    r2_2 = y_q2[nu2]                     # r2 at q2

                    # -weight_R * (r1_q1 * conj(r2_q2)^T + r2_q1 * conj(r1_q2)^T)
                    contrib = -weight_R * (r1_1 * conj(r2_2) + r2_1 * conj(r1_2))
                    # -weight_Rf * r1_q1 * conj(r1_q2)^T
                    contrib -= weight_Rf * r1_1 * conj(r1_2)

                    d2v_blocks[p][nu1, nu2] += contrib
                end
            end
        end
    end

    # Normalize
    norm_factor = n_syms * N_eff
    for p in 1:n_pairs
        d2v_blocks[p] ./= norm_factor
    end

    return d2v_blocks
end


"""
    get_d2v_from_Y_pert_qspace(...)

D4 contribution to d2v from alpha1 (Y1/Upsilon) perturbation.
Mirrors get_d2v_dR2_from_Y_pert_sym_fast in tdscha_core.jl.

CRITICAL: weights are TOTAL (summed over ALL pairs), not per-pair.

For each (config, sym):
  1. Compute buffer_u(q,nu) = sum_nu2 alpha1(q, q_pert-q)[nu,nu2] * x(q_pert-q, nu2)
  2. Compute total_wD4 = -sum_all_pairs x^H * alpha1 * x, scaled by rho/8
  3. Compute total_wb = -sum_{q,nu} conj(buffer_u) * f_psi * y, scaled by rho/4
  4. Apply TOTAL weights to ALL d2v blocks
"""
function get_d2v_from_Y_pert_qspace(
    X_q::Array{ComplexF64,3},
    Y_q::Array{ComplexF64,3},
    f_Y::Matrix{Float64},
    f_psi::Matrix{Float64},
    rho::Vector{Float64},
    alpha1_blocks::Vector{Matrix{ComplexF64}},
    symmetries::Vector{SparseMatrixCSC{ComplexF64,Int32}},
    iq_pert::Int64,
    unique_pairs::Matrix{Int32},
    n_bands::Int64,
    n_q::Int64,
    start_index::Int64,
    end_index::Int64
)
    n_pairs = size(unique_pairs, 1)
    n_syms = length(symmetries)
    n_total = n_q * n_bands
    N_eff = sum(rho)

    # Output
    d2v_blocks = [zeros(ComplexF64, n_bands, n_bands) for _ in 1:n_pairs]

    # Buffers
    x_buf = zeros(ComplexF64, n_total)
    y_buf = zeros(ComplexF64, n_total)
    buffer_u = zeros(ComplexF64, n_q, n_bands)

    for bigindex in start_index:end_index
        i_config = div(bigindex - 1, n_syms) + 1
        j_sym = mod(bigindex - 1, n_syms) + 1

        # Build combined vector
        for iq in 1:n_q
            for nu in 1:n_bands
                idx = (iq - 1) * n_bands + nu
                x_buf[idx] = X_q[iq, i_config, nu]
                y_buf[idx] = Y_q[iq, i_config, nu]
            end
        end

        # Apply symmetry
        x_rot = symmetries[j_sym] * x_buf
        y_rot = symmetries[j_sym] * y_buf

        # Step 1: Compute buffer_u and total_wD4
        # buffer_u[iq1, nu1] = sum_nu2 alpha1[p][nu1, nu2] * x_rot[iq2, nu2]
        # where (iq1, iq2) is the pair containing iq1
        # total_wD4 = -sum_pairs (multiplicity) * x_q1^H * alpha1 * x_q2
        total_wD4 = zero(ComplexF64)
        fill!(buffer_u, zero(ComplexF64))

        for p in 1:n_pairs
            iq1 = unique_pairs[p, 1]
            iq2 = unique_pairs[p, 2]

            x_q1 = view(x_rot, (iq1-1)*n_bands+1:iq1*n_bands)
            x_q2 = view(x_rot, (iq2-1)*n_bands+1:iq2*n_bands)

            # buffer_u at iq1: sum_nu2 alpha1[p][nu1, nu2] * x_q2[nu2]
            for nu1 in 1:n_bands
                for nu2 in 1:n_bands
                    buffer_u[iq1, nu1] += alpha1_blocks[p][nu1, nu2] * x_q2[nu2]
                end
            end

            # If iq1 != iq2, also accumulate buffer_u at iq2
            # For the reverse pair (iq2, iq1), the Hermitian Upsilon satisfies:
            #   alpha1(q2,q1)[nu2,nu1] = conj(alpha1(q1,q2)[nu1,nu2])
            # So buffer_u at iq2 uses the Hermitian conjugate of alpha1.
            if iq1 != iq2
                for nu2 in 1:n_bands
                    for nu1 in 1:n_bands
                        buffer_u[iq2, nu2] += conj(alpha1_blocks[p][nu1, nu2]) * x_q1[nu1]
                    end
                end
            end

            # Total weight: conj(x_q1)^T * alpha1 * x_q2
            local_w = zero(ComplexF64)
            for nu1 in 1:n_bands
                for nu2 in 1:n_bands
                    local_w += conj(x_q1[nu1]) * alpha1_blocks[p][nu1, nu2] * x_q2[nu2]
                end
            end
            # For off-diagonal pairs, the reverse pair (iq2,iq1) contributes conj(local_w),
            # so total = local_w + conj(local_w). In real-space (real x), this equals 2*local_w,
            # but in q-space (complex x), we must use the correct Hermitian form.
            if iq1 < iq2
                total_wD4 += local_w + conj(local_w)
            else
                total_wD4 += local_w
            end
        end
        total_wD4 *= -rho[i_config] / 8.0

        # Step 2: Compute total_wb = -sum_{q,nu} conj(buffer_u[q,nu]) * f_psi[nu,q] * y_rot[q,nu]
        total_wb = zero(ComplexF64)
        for iq in 1:n_q
            for nu in 1:n_bands
                y_val = y_rot[(iq-1)*n_bands + nu]
                total_wb -= conj(buffer_u[iq, nu]) * f_psi[nu, iq] * y_val
            end
        end
        total_wb *= rho[i_config] / 4.0

        # Step 3: Apply TOTAL weights to ALL d2v blocks
        for p in 1:n_pairs
            iq1 = unique_pairs[p, 1]
            iq2 = unique_pairs[p, 2]

            x_q1 = view(x_rot, (iq1-1)*n_bands+1:iq1*n_bands)
            y_q1 = view(y_rot, (iq1-1)*n_bands+1:iq1*n_bands)
            x_q2 = view(x_rot, (iq2-1)*n_bands+1:iq2*n_bands)
            y_q2 = view(y_rot, (iq2-1)*n_bands+1:iq2*n_bands)

            for nu1 in 1:n_bands
                r1_1 = f_Y[nu1, iq1] * x_q1[nu1]
                r2_1 = y_q1[nu1]
                for nu2 in 1:n_bands
                    r1_2 = f_Y[nu2, iq2] * x_q2[nu2]
                    r2_2 = y_q2[nu2]

                    # -total_wD4 * (r1*conj(r2)^T + r2*conj(r1)^T)
                    contrib = -total_wD4 * (r1_1 * conj(r2_2) + r2_1 * conj(r1_2))
                    # -total_wb * r1*conj(r1)^T
                    contrib -= total_wb * r1_1 * conj(r1_2)

                    d2v_blocks[p][nu1, nu2] += contrib
                end
            end
        end
    end

    # Normalize
    norm_factor = n_syms * N_eff
    for p in 1:n_pairs
        d2v_blocks[p] ./= norm_factor
    end

    return d2v_blocks
end


"""
    get_f_from_Y_pert_qspace(...)

D3 contribution to f_pert from alpha1/Y1 perturbation.
Mirrors get_f_average_from_Y_pert in tdscha_core.jl.

For each (config, sym):
  total_sum = sum_pairs (mult) * conj(x_q1)^T * alpha1 * x_q2
  buffer_u(q,nu) = sum_nu2 alpha1[...] * x(q2,nu2)
  buf_f_weight = sum_{q,nu} conj(buffer_u) * f_psi * y

  f_pert += (-total_sum/2) * rho/3 * y_rot[q_pert]
  f_pert += (-buf_f_weight) * rho/3 * f_Y[q_pert] * x_rot[q_pert]
"""
function get_f_from_Y_pert_qspace(
    X_q::Array{ComplexF64,3},
    Y_q::Array{ComplexF64,3},
    f_Y::Matrix{Float64},
    f_psi::Matrix{Float64},
    rho::Vector{Float64},
    alpha1_blocks::Vector{Matrix{ComplexF64}},
    symmetries::Vector{SparseMatrixCSC{ComplexF64,Int32}},
    iq_pert::Int64,
    unique_pairs::Matrix{Int32},
    n_bands::Int64,
    n_q::Int64,
    start_index::Int64,
    end_index::Int64
)
    n_pairs = size(unique_pairs, 1)
    n_syms = length(symmetries)
    n_total = n_q * n_bands
    N_eff = sum(rho)

    # Output
    f_pert = zeros(ComplexF64, n_bands)

    # Buffers
    x_buf = zeros(ComplexF64, n_total)
    y_buf = zeros(ComplexF64, n_total)
    buffer_u = zeros(ComplexF64, n_q, n_bands)

    for bigindex in start_index:end_index
        i_config = div(bigindex - 1, n_syms) + 1
        j_sym = mod(bigindex - 1, n_syms) + 1

        # Build combined vector
        for iq in 1:n_q
            for nu in 1:n_bands
                idx = (iq - 1) * n_bands + nu
                x_buf[idx] = X_q[iq, i_config, nu]
                y_buf[idx] = Y_q[iq, i_config, nu]
            end
        end

        # Apply symmetry
        x_rot = symmetries[j_sym] * x_buf
        y_rot = symmetries[j_sym] * y_buf

        # Compute buffer_u and total_sum (same as d2v function)
        total_sum = zero(ComplexF64)
        fill!(buffer_u, zero(ComplexF64))

        for p in 1:n_pairs
            iq1 = unique_pairs[p, 1]
            iq2 = unique_pairs[p, 2]

            x_q1 = view(x_rot, (iq1-1)*n_bands+1:iq1*n_bands)
            x_q2 = view(x_rot, (iq2-1)*n_bands+1:iq2*n_bands)

            # buffer_u at iq1
            for nu1 in 1:n_bands
                for nu2 in 1:n_bands
                    buffer_u[iq1, nu1] += alpha1_blocks[p][nu1, nu2] * x_q2[nu2]
                end
            end

            # buffer_u at iq2 (if not diagonal pair)
            # Reverse pair uses Hermitian conjugate: alpha1(q2,q1) = alpha1(q1,q2)^H
            if iq1 != iq2
                for nu2 in 1:n_bands
                    for nu1 in 1:n_bands
                        buffer_u[iq2, nu2] += conj(alpha1_blocks[p][nu1, nu2]) * x_q1[nu1]
                    end
                end
            end

            # total_sum
            local_w = zero(ComplexF64)
            for nu1 in 1:n_bands
                for nu2 in 1:n_bands
                    local_w += conj(x_q1[nu1]) * alpha1_blocks[p][nu1, nu2] * x_q2[nu2]
                end
            end
            # Reverse pair contributes conj(local_w), not local_w
            if iq1 < iq2
                total_sum += local_w + conj(local_w)
            else
                total_sum += local_w
            end
        end

        # buf_f_weight = sum_{q,nu} conj(buffer_u[q,nu]) * f_psi[nu,q] * y_rot[q,nu]
        buf_f_weight = zero(ComplexF64)
        for iq in 1:n_q
            for nu in 1:n_bands
                y_val = y_rot[(iq-1)*n_bands + nu]
                buf_f_weight += conj(buffer_u[iq, nu]) * f_psi[nu, iq] * y_val
            end
        end

        # Two f_pert contributions:
        # Term 1: (-total_sum/2) * rho/3 * y_rot[q_pert]
        y_pert = view(y_rot, (iq_pert-1)*n_bands+1:iq_pert*n_bands)
        x_pert = view(x_rot, (iq_pert-1)*n_bands+1:iq_pert*n_bands)

        w1 = -total_sum / 2.0 * rho[i_config] / 3.0
        for nu in 1:n_bands
            f_pert[nu] += w1 * y_pert[nu]
        end

        # Term 2: (-buf_f_weight) * rho/3 * f_Y[nu,q_pert] * x_rot[q_pert,nu]
        w2 = -buf_f_weight * rho[i_config] / 3.0
        for nu in 1:n_bands
            f_pert[nu] += w2 * f_Y[nu, iq_pert] * x_pert[nu]
        end
    end

    # Normalize
    f_pert ./= (n_syms * N_eff)

    return f_pert
end


"""
    get_perturb_averages_qspace_fused(...)

Fused single-pass computation of f_pert and d2v_blocks.
Replaces three separate functions (get_d2v_from_R_pert_qspace,
get_f_from_Y_pert_qspace, get_d2v_from_Y_pert_qspace) with a single
loop over (config, sym) that applies the 2 sparse matmuls only once.

For each (config, sym):
  1. Build x_buf, y_buf and apply symmetry rotation (2 sparse matmuls)
  2. Compute D3 weights: weight_R, weight_Rf from R1 perturbation
  3. Compute buffer_u, total_sum (D4 intermediates, shared by f_pert and d2v_v4)
  4. Compute buf_f_weight from buffer_u (shared by f_pert and d2v_v4)
  5. Accumulate f_pert (2 terms)
  6. Accumulate d2v with fused D3 + D4 weights in single inner loop
"""
function get_perturb_averages_qspace_fused(
    X_q::Array{ComplexF64,3},
    Y_q::Array{ComplexF64,3},
    f_Y::Matrix{Float64},
    f_psi::Matrix{Float64},
    rho::Vector{Float64},
    R1::Vector{ComplexF64},
    alpha1_blocks::Vector{Matrix{ComplexF64}},
    symmetries::Vector{SparseMatrixCSC{ComplexF64,Int32}},
    apply_v4::Bool,
    iq_pert::Int64,
    unique_pairs::Matrix{Int32},
    n_bands::Int64,
    n_q::Int64,
    start_index::Int64,
    end_index::Int64
)
    n_pairs = size(unique_pairs, 1)
    n_syms = length(symmetries)
    n_total = n_q * n_bands
    N_eff = sum(rho)

    # Outputs
    d2v_blocks = [zeros(ComplexF64, n_bands, n_bands) for _ in 1:n_pairs]
    f_pert = zeros(ComplexF64, n_bands)

    # Buffers (reused each iteration)
    x_buf = zeros(ComplexF64, n_total)
    y_buf = zeros(ComplexF64, n_total)
    buffer_u = zeros(ComplexF64, n_q, n_bands)

    for bigindex in start_index:end_index
        i_config = div(bigindex - 1, n_syms) + 1
        j_sym = mod(bigindex - 1, n_syms) + 1

        # === Step 1: Build combined vector and apply symmetry (ONCE) ===
        for iq in 1:n_q
            for nu in 1:n_bands
                idx = (iq - 1) * n_bands + nu
                x_buf[idx] = X_q[iq, i_config, nu]
                y_buf[idx] = Y_q[iq, i_config, nu]
            end
        end

        x_rot = symmetries[j_sym] * x_buf
        y_rot = symmetries[j_sym] * y_buf

        # === Step 2: D3 weights from R1 perturbation ===
        x_pert = view(x_rot, (iq_pert-1)*n_bands+1:iq_pert*n_bands)
        y_pert = view(y_rot, (iq_pert-1)*n_bands+1:iq_pert*n_bands)

        weight_R = zero(ComplexF64)
        for nu in 1:n_bands
            weight_R += f_Y[nu, iq_pert] * x_pert[nu] * conj(R1[nu])
        end
        weight_R *= rho[i_config] / 3.0

        weight_Rf = zero(ComplexF64)
        for nu in 1:n_bands
            weight_Rf += conj(R1[nu]) * y_pert[nu]
        end
        weight_Rf *= rho[i_config] / 3.0

        # === Step 3: D4 intermediates (buffer_u, total_sum) ===
        total_sum = zero(ComplexF64)
        fill!(buffer_u, zero(ComplexF64))

        for p in 1:n_pairs
            iq1 = unique_pairs[p, 1]
            iq2 = unique_pairs[p, 2]

            x_q1 = view(x_rot, (iq1-1)*n_bands+1:iq1*n_bands)
            x_q2 = view(x_rot, (iq2-1)*n_bands+1:iq2*n_bands)

            # buffer_u at iq1: sum_nu2 alpha1[p][nu1, nu2] * x_q2[nu2]
            for nu1 in 1:n_bands
                for nu2 in 1:n_bands
                    buffer_u[iq1, nu1] += alpha1_blocks[p][nu1, nu2] * x_q2[nu2]
                end
            end

            # buffer_u at iq2 (reverse pair uses Hermitian conjugate)
            if iq1 != iq2
                for nu2 in 1:n_bands
                    for nu1 in 1:n_bands
                        buffer_u[iq2, nu2] += conj(alpha1_blocks[p][nu1, nu2]) * x_q1[nu1]
                    end
                end
            end

            # total_sum = sum of conj(x_q1)^T * alpha1 * x_q2
            local_w = zero(ComplexF64)
            for nu1 in 1:n_bands
                for nu2 in 1:n_bands
                    local_w += conj(x_q1[nu1]) * alpha1_blocks[p][nu1, nu2] * x_q2[nu2]
                end
            end
            if iq1 < iq2
                total_sum += local_w + conj(local_w)
            else
                total_sum += local_w
            end
        end

        # === Step 4: buf_f_weight from buffer_u ===
        buf_f_weight = zero(ComplexF64)
        for iq in 1:n_q
            for nu in 1:n_bands
                y_val = y_rot[(iq-1)*n_bands + nu]
                buf_f_weight += conj(buffer_u[iq, nu]) * f_psi[nu, iq] * y_val
            end
        end

        # === Step 5: Accumulate f_pert ===
        # Term 1: (-total_sum/2) * rho/3 * y_rot[q_pert]
        w1 = -total_sum / 2.0 * rho[i_config] / 3.0
        for nu in 1:n_bands
            f_pert[nu] += w1 * y_pert[nu]
        end

        # Term 2: (-buf_f_weight) * rho/3 * f_Y[nu,q_pert] * x_rot[q_pert,nu]
        w2 = -buf_f_weight * rho[i_config] / 3.0
        for nu in 1:n_bands
            f_pert[nu] += w2 * f_Y[nu, iq_pert] * x_pert[nu]
        end

        # === Step 6: Fused d2v accumulation (D3 + D4 in single loop) ===
        # D4 total weights (from get_d2v_from_Y_pert_qspace)
        # total_wb = -buf_f_weight * rho/4 (same sum, opposite sign, different rho scaling)
        total_wD4 = zero(ComplexF64)
        total_wb = zero(ComplexF64)
        if apply_v4
            total_wD4 = -total_sum * rho[i_config] / 8.0
            total_wb = -buf_f_weight * rho[i_config] / 4.0
        end

        # Combined weights for fused inner loop
        w_cross = weight_R + total_wD4
        w_diag = weight_Rf + total_wb

        for p in 1:n_pairs
            iq1 = unique_pairs[p, 1]
            iq2 = unique_pairs[p, 2]

            x_q1 = view(x_rot, (iq1-1)*n_bands+1:iq1*n_bands)
            y_q1 = view(y_rot, (iq1-1)*n_bands+1:iq1*n_bands)
            x_q2 = view(x_rot, (iq2-1)*n_bands+1:iq2*n_bands)
            y_q2 = view(y_rot, (iq2-1)*n_bands+1:iq2*n_bands)

            for nu1 in 1:n_bands
                r1_1 = f_Y[nu1, iq1] * x_q1[nu1]
                r2_1 = y_q1[nu1]
                for nu2 in 1:n_bands
                    r1_2 = f_Y[nu2, iq2] * x_q2[nu2]
                    r2_2 = y_q2[nu2]

                    contrib = -w_cross * (r1_1 * conj(r2_2) + r2_1 * conj(r1_2))
                    contrib -= w_diag * r1_1 * conj(r1_2)

                    d2v_blocks[p][nu1, nu2] += contrib
                end
            end
        end
    end

    # Normalize
    norm_factor = n_syms * N_eff
    f_pert ./= norm_factor
    for p in 1:n_pairs
        d2v_blocks[p] ./= norm_factor
    end

    return f_pert, d2v_blocks
end


"""
    get_perturb_averages_qspace(...)

Combined entry point called from Python. Computes f_pert and d2v_blocks,
packs them into a single flat array for MPI reduction.

Returns: [f_pert(n_bands) ; d2v_blocks_flat(n_pairs * n_bands^2)]
"""
function get_perturb_averages_qspace(
    X_q::Array{ComplexF64,3},
    Y_q::Array{ComplexF64,3},
    w_q::Matrix{Float64},
    rho::Vector{Float64},
    R1::Vector{ComplexF64},
    alpha1_flat::Vector{ComplexF64},
    temperature::Float64,
    apply_v4::Bool,
    iq_pert::Int64,
    q_pair_map::Vector{Int32},
    unique_pairs::Matrix{Int32},
    start_index::Int64,
    end_index::Int64
)
    n_q = size(X_q, 1)
    n_bands = size(X_q, 3)
    n_pairs = size(unique_pairs, 1)

    # Get symmetries
    symmetries = _cached_qspace_symmetries[]
    if symmetries === nothing
        error("Q-space symmetries not initialized. Call init_sparse_symmetries_qspace first.")
    end

    # Precompute occupation numbers and scaling factors
    # Acoustic modes (w < threshold) get f_Y=0, f_psi=0 to avoid NaN/Inf
    f_Y = zeros(Float64, n_bands, n_q)
    f_psi = zeros(Float64, n_bands, n_q)
    acoustic_eps = 1e-6

    for iq in 1:n_q
        for nu in 1:n_bands
            w = w_q[nu, iq]
            if w < acoustic_eps
                f_Y[nu, iq] = 0.0
                f_psi[nu, iq] = 0.0
                continue
            end
            if temperature > 0
                nw = 1.0 / (exp(w * RY_TO_K_Q / temperature) - 1.0)
            else
                nw = 0.0
            end
            f_Y[nu, iq] = 2.0 * w / (1.0 + 2.0 * nw)
            f_psi[nu, iq] = (1.0 + 2.0 * nw) / (2.0 * w)
        end
    end

    # Unpack alpha1 blocks
    alpha1_blocks = Vector{Matrix{ComplexF64}}(undef, n_pairs)
    offset = 1
    for p in 1:n_pairs
        alpha1_blocks[p] = reshape(alpha1_flat[offset:offset+n_bands^2-1], n_bands, n_bands)
        offset += n_bands^2
    end

    # Fused single-pass computation
    f_pert, d2v = get_perturb_averages_qspace_fused(
        X_q, Y_q, f_Y, f_psi, rho, R1, alpha1_blocks, symmetries,
        apply_v4, iq_pert, unique_pairs, n_bands, n_q,
        start_index, end_index)

    # Pack result: f_pert followed by flattened d2v blocks
    result = zeros(ComplexF64, n_bands + n_pairs * n_bands^2)
    result[1:n_bands] = f_pert
    offset = n_bands + 1
    for p in 1:n_pairs
        result[offset:offset+n_bands^2-1] = vec(d2v[p])
        offset += n_bands^2
    end

    return result
end
