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
  weight_R  = sum_nu f_Y[nu,iq_pert] * conj(x_rot[iq_pert,nu]) * R1[nu] * rho/3
  weight_Rf = sum_nu R1[nu] * conj(y_rot[iq_pert,nu]) * rho/3

  For each pair p = (q1,q2):
    d2v[p] += -weight_R  * (r1_q1 * r2_q2^T + r2_q1 * r1_q2^T)
    d2v[p] += -weight_Rf * r1_q1 * r1_q2^T

where r1 = f_Y * x, r2 = y.

Convention: pairs satisfy q1 + q2 = q_pert, so all blocks are BILINEAR
(transpose-type) Fourier components, B(q1,q2) = e(q1)^dag M conj(e(q2)).
Contractions of a kernel with the (real-field) ensemble data therefore use
conj(x(q)) where the real-space code uses x, and the dyadic outputs use the
plain (unconjugated) band components; this is what conserves momentum
(<conj r(q_pert) r(q1) r(q2)> has total momentum -q_pert+q1+q2 = 0).
Conjugating the other factors (sesquilinear convention) gives terms of net
momentum 2*q1, whose ensemble average vanishes unless q1 is time-reversal
invariant — which silently destroys the anharmonicity at non-TRI q.
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
            weight_R += f_Y[nu, iq_pert] * conj(x_pert[nu]) * R1[nu]
        end
        weight_R *= rho[i_config] / 3.0

        # weight_Rf = sum_nu conj(R1[nu]) * y_pert[nu] * rho/3
        weight_Rf = zero(ComplexF64)
        for nu in 1:n_bands
            weight_Rf += R1[nu] * conj(y_pert[nu])
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

                    # -weight_R * (r1_q1 * r2_q2^T + r2_q1 * r1_q2^T)
                    contrib = -weight_R * (r1_1 * r2_2 + r2_1 * r1_2)
                    # -weight_Rf * r1_q1 * r1_q2^T
                    contrib -= weight_Rf * r1_1 * r1_2

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

            # buffer_u at iq1: sum_nu2 alpha1[p][nu1, nu2] * conj(x_q2[nu2])
            for nu1 in 1:n_bands
                for nu2 in 1:n_bands
                    buffer_u[iq1, nu1] += alpha1_blocks[p][nu1, nu2] * conj(x_q2[nu2])
                end
            end

            # If iq1 != iq2, also accumulate buffer_u at iq2
            # The bilinear blocks of the (symmetric) Upsilon perturbation satisfy
            #   alpha1(q2,q1)[nu2,nu1] = alpha1(q1,q2)[nu1,nu2]   (transpose)
            # so buffer_u at iq2 uses the transpose of alpha1.
            if iq1 != iq2
                for nu2 in 1:n_bands
                    for nu1 in 1:n_bands
                        buffer_u[iq2, nu2] += alpha1_blocks[p][nu1, nu2] * conj(x_q1[nu1])
                    end
                end
            end

            # Total weight: conj(x_q1)^T * alpha1 * x_q2
            local_w = zero(ComplexF64)
            for nu1 in 1:n_bands
                for nu2 in 1:n_bands
                    local_w += conj(x_q1[nu1]) * alpha1_blocks[p][nu1, nu2] * conj(x_q2[nu2])
                end
            end
            # For off-diagonal pairs, the reverse pair (iq2,iq1) contributes the
            # same value (the bilinear blocks are transpose-symmetric), so total
            # = 2*local_w. The weight is complex: the perturbation is a complex
            # Bloch field at +q_pert.
            if iq1 < iq2
                total_wD4 += 2 * local_w
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
                total_wb -= buffer_u[iq, nu] * f_psi[nu, iq] * conj(y_val)
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

                    # -total_wD4 * (r1*r2^T + r2*r1^T)
                    contrib = -total_wD4 * (r1_1 * r2_2 + r2_1 * r1_2)
                    # -total_wb * r1*r1^T
                    contrib -= total_wb * r1_1 * r1_2

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
                    buffer_u[iq1, nu1] += alpha1_blocks[p][nu1, nu2] * conj(x_q2[nu2])
                end
            end

            # buffer_u at iq2 (if not diagonal pair)
            # Reverse pair uses the transpose: alpha1(q2,q1) = alpha1(q1,q2)^T
            if iq1 != iq2
                for nu2 in 1:n_bands
                    for nu1 in 1:n_bands
                        buffer_u[iq2, nu2] += alpha1_blocks[p][nu1, nu2] * conj(x_q1[nu1])
                    end
                end
            end

            # total_sum
            local_w = zero(ComplexF64)
            for nu1 in 1:n_bands
                for nu2 in 1:n_bands
                    local_w += conj(x_q1[nu1]) * alpha1_blocks[p][nu1, nu2] * conj(x_q2[nu2])
                end
            end
            # Reverse pair contributes the same value (transpose-symmetric blocks)
            if iq1 < iq2
                total_sum += 2 * local_w
            else
                total_sum += local_w
            end
        end

        # buf_f_weight = sum_{q,nu} conj(buffer_u[q,nu]) * f_psi[nu,q] * y_rot[q,nu]
        buf_f_weight = zero(ComplexF64)
        for iq in 1:n_q
            for nu in 1:n_bands
                y_val = y_rot[(iq-1)*n_bands + nu]
                buf_f_weight += buffer_u[iq, nu] * f_psi[nu, iq] * conj(y_val)
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

The vertex renormalization factors scale3 and scale4 multiply the D3-type
(3-field) and D4-type (4-field) averages respectively. They implement the
N_c -> N_f mode-space vertex rescaling of the q-mesh interpolation
(d3 ~ N^-1/2, d4 ~ N^-1, see Interpolation_plan.md §6):
  scale3 = sqrt(N_coarse / N_fine),  scale4 = N_coarse / N_fine.
Both default to 1.0 (commensurate/no-interpolation behavior).
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
    end_index::Int64,
    scale3::Float64=1.0,
    scale4::Float64=1.0
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
            weight_R += f_Y[nu, iq_pert] * conj(x_pert[nu]) * R1[nu]
        end
        weight_R *= rho[i_config] / 3.0 * scale3

        weight_Rf = zero(ComplexF64)
        for nu in 1:n_bands
            weight_Rf += R1[nu] * conj(y_pert[nu])
        end
        weight_Rf *= rho[i_config] / 3.0 * scale3

        # === Step 3: D4 intermediates (buffer_u, total_sum) ===
        total_sum = zero(ComplexF64)
        fill!(buffer_u, zero(ComplexF64))

        for p in 1:n_pairs
            iq1 = unique_pairs[p, 1]
            iq2 = unique_pairs[p, 2]

            x_q1 = view(x_rot, (iq1-1)*n_bands+1:iq1*n_bands)
            x_q2 = view(x_rot, (iq2-1)*n_bands+1:iq2*n_bands)

            # buffer_u at iq1: sum_nu2 alpha1[p][nu1, nu2] * conj(x_q2[nu2])
            for nu1 in 1:n_bands
                for nu2 in 1:n_bands
                    buffer_u[iq1, nu1] += alpha1_blocks[p][nu1, nu2] * conj(x_q2[nu2])
                end
            end

            # buffer_u at iq2 (reverse pair uses the transpose of alpha1)
            if iq1 != iq2
                for nu2 in 1:n_bands
                    for nu1 in 1:n_bands
                        buffer_u[iq2, nu2] += alpha1_blocks[p][nu1, nu2] * conj(x_q1[nu1])
                    end
                end
            end

            # total_sum = sum of conj(x_q1)^T * alpha1 * conj(x_q2)  (= x^T Y1 x)
            local_w = zero(ComplexF64)
            for nu1 in 1:n_bands
                for nu2 in 1:n_bands
                    local_w += conj(x_q1[nu1]) * alpha1_blocks[p][nu1, nu2] * conj(x_q2[nu2])
                end
            end
            if iq1 < iq2
                total_sum += 2 * local_w
            else
                total_sum += local_w
            end
        end

        # === Step 4: buf_f_weight from buffer_u ===
        buf_f_weight = zero(ComplexF64)
        for iq in 1:n_q
            for nu in 1:n_bands
                y_val = y_rot[(iq-1)*n_bands + nu]
                buf_f_weight += buffer_u[iq, nu] * f_psi[nu, iq] * conj(y_val)
            end
        end

        # === Step 5: Accumulate f_pert ===
        # Term 1: (-total_sum/2) * rho/3 * y_rot[q_pert]  (D3-type -> scale3)
        w1 = -total_sum / 2.0 * rho[i_config] / 3.0 * scale3
        for nu in 1:n_bands
            f_pert[nu] += w1 * y_pert[nu]
        end

        # Term 2: (-buf_f_weight) * rho/3 * f_Y[nu,q_pert] * x_rot[q_pert,nu]  (D3-type -> scale3)
        w2 = -buf_f_weight * rho[i_config] / 3.0 * scale3
        for nu in 1:n_bands
            f_pert[nu] += w2 * f_Y[nu, iq_pert] * x_pert[nu]
        end

        # === Step 6: Fused d2v accumulation (D3 + D4 in single loop) ===
        # D4 total weights (from get_d2v_from_Y_pert_qspace)
        # total_wb = -buf_f_weight * rho/4 (same sum, opposite sign, different rho scaling)
        total_wD4 = zero(ComplexF64)
        total_wb = zero(ComplexF64)
        if apply_v4
            total_wD4 = -total_sum * rho[i_config] / 8.0 * scale4
            total_wb = -buf_f_weight * rho[i_config] / 4.0 * scale4
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

                    contrib = -w_cross * (r1_1 * r2_2 + r2_1 * r1_2)
                    contrib -= w_diag * r1_1 * r1_2

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
    end_index::Int64,
    valid_modes_q::Matrix{Bool},  # Mask from Python: false for acoustic/small-w modes
    scale3::Float64=1.0,          # D3 vertex rescaling sqrt(N_c/N_f) for interpolation
    scale4::Float64=1.0,          # D4 vertex rescaling N_c/N_f for interpolation
    prefiltered::Bool=false       # X_q fields already carry f_Y; f_psi folded in alpha1
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
    # Masked modes (valid_modes_q == false) get f_Y=0, f_psi=0 to avoid NaN/Inf
    #
    # In prefiltered mode the X_q fields already carry the f_Y filter
    # (applied on the COARSE grid, where it exactly strips the phonon
    # propagator dressing of every displacement leg by the Gaussian
    # integration-by-parts identity), and the fine-side f_psi factors are
    # folded into the alpha1 blocks by the Python caller. Both tables then
    # reduce to the validity mask. See Interpolation_plan.md section 5.6.
    f_Y = zeros(Float64, n_bands, n_q)
    f_psi = zeros(Float64, n_bands, n_q)

    for iq in 1:n_q
        for nu in 1:n_bands
            if !valid_modes_q[nu, iq]  # Masked mode -> zero out
                f_Y[nu, iq] = 0.0
                f_psi[nu, iq] = 0.0
                continue
            end
            if prefiltered
                f_Y[nu, iq] = 1.0
                f_psi[nu, iq] = 1.0
                continue
            end
            w = w_q[nu, iq]
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
        start_index, end_index, scale3, scale4)

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


# =========================================================================
# Slot-resolved kernel for the windowed ("stochastic centering") estimator
# =========================================================================
#
# The multitaper interpolation (Interpolation_plan.md sections 5.2-5.5)
# evaluates the anharmonic estimator on WINDOWED per-configuration fields,
# with (in general) a different window on each field slot of the trilinear
# D3 contractions:
#   slot z : the q_pert leg   (weight_R, weight_Rf, the f_pert outputs)
#   slot w : the first pair leg  (rows of the d2v blocks, q1)
#   slot v : the second pair leg (columns of the d2v blocks, q2)
# The design is symmetrized in w <-> v by the caller (each pass is run in
# both orientations and averaged), which preserves the transpose symmetry
# of the pair blocks and the Hermiticity of L.
#
# compute_d3 / compute_d4 select the D3 terms (d2v from R1 + f_pert from
# alpha1) and the D4 terms (d2v from alpha1). The windowed passes run with
# compute_d4 = false; a single plain-window pass runs with compute_d3 =
# false to add the D4 terms (plan section 5.4: plain window for D4).
#
# With X_z == X_w == X_v (and Y likewise), compute_d3 = compute_d4 = true,
# this kernel is algebraically identical to
# get_perturb_averages_qspace_fused (regression-tested from Python).

function get_perturb_averages_qspace_slots_kernel(
    Xz::Array{ComplexF64,3}, Yz::Array{ComplexF64,3},
    Xw::Array{ComplexF64,3}, Yw::Array{ComplexF64,3},
    Xv::Array{ComplexF64,3}, Yv::Array{ComplexF64,3},
    f_Y::Matrix{Float64},
    f_psi::Matrix{Float64},
    rho::Vector{Float64},
    R1::Vector{ComplexF64},
    alpha1_blocks::Vector{Matrix{ComplexF64}},
    symmetries::Vector{SparseMatrixCSC{ComplexF64,Int32}},
    compute_d3::Bool,
    compute_d4::Bool,
    iq_pert::Int64,
    unique_pairs::Matrix{Int32},
    n_bands::Int64,
    n_q::Int64,
    start_index::Int64,
    end_index::Int64,
    scale3::Float64,
    scale4::Float64,
    d3_force_z::Bool=true,
    d3_force_w::Bool=true,
    d3_force_v::Bool=true,
    d4_force_ew::Bool=true,
    d4_force_ev::Bool=true,
    d4_force_iw::Bool=true,
    d4_force_iv::Bool=true
)
    n_pairs = size(unique_pairs, 1)
    n_syms = length(symmetries)
    n_total = n_q * n_bands
    N_eff = sum(rho)

    d2v_blocks = [zeros(ComplexF64, n_bands, n_bands) for _ in 1:n_pairs]
    f_pert = zeros(ComplexF64, n_bands)

    # Buffers
    xz_buf = zeros(ComplexF64, n_total); yz_buf = zeros(ComplexF64, n_total)
    xw_buf = zeros(ComplexF64, n_total); yw_buf = zeros(ComplexF64, n_total)
    xv_buf = zeros(ComplexF64, n_total); yv_buf = zeros(ComplexF64, n_total)
    bu_w = zeros(ComplexF64, n_q, n_bands)   # alpha1 . conj(x_v), lives on q1
    bu_v = zeros(ComplexF64, n_q, n_bands)   # alpha1^T . conj(x_w), lives on q2

    for bigindex in start_index:end_index
        i_config = div(bigindex - 1, n_syms) + 1
        j_sym = mod(bigindex - 1, n_syms) + 1

        for iq in 1:n_q
            for nu in 1:n_bands
                idx = (iq - 1) * n_bands + nu
                xz_buf[idx] = Xz[iq, i_config, nu]; yz_buf[idx] = Yz[iq, i_config, nu]
                xw_buf[idx] = Xw[iq, i_config, nu]; yw_buf[idx] = Yw[iq, i_config, nu]
                xv_buf[idx] = Xv[iq, i_config, nu]; yv_buf[idx] = Yv[iq, i_config, nu]
            end
        end

        S = symmetries[j_sym]
        xz_rot = S * xz_buf; yz_rot = S * yz_buf
        xw_rot = S * xw_buf; yw_rot = S * yw_buf
        xv_rot = S * xv_buf; yv_rot = S * yv_buf

        xz_pert = view(xz_rot, (iq_pert-1)*n_bands+1:iq_pert*n_bands)
        yz_pert = view(yz_rot, (iq_pert-1)*n_bands+1:iq_pert*n_bands)

        # === D3 weights from R1 (slot z) ===
        weight_R = zero(ComplexF64)
        weight_Rf = zero(ComplexF64)
        if compute_d3 && (d3_force_w || d3_force_v)
            for nu in 1:n_bands
                weight_R += f_Y[nu, iq_pert] * conj(xz_pert[nu]) * R1[nu]
            end
            weight_R *= rho[i_config] / 3.0 * scale3
        end
        if compute_d3 && d3_force_z
            for nu in 1:n_bands
                weight_Rf += R1[nu] * conj(yz_pert[nu])
            end
            weight_Rf *= rho[i_config] / 3.0 * scale3
        end

        # === alpha1 intermediates (slots w, v) ===
        total_sum = zero(ComplexF64)
        fill!(bu_w, zero(ComplexF64))
        fill!(bu_v, zero(ComplexF64))

        for p in 1:n_pairs
            iq1 = unique_pairs[p, 1]
            iq2 = unique_pairs[p, 2]
            xw1 = view(xw_rot, (iq1-1)*n_bands+1:iq1*n_bands)
            xv2 = view(xv_rot, (iq2-1)*n_bands+1:iq2*n_bands)

            for nu1 in 1:n_bands
                acc = zero(ComplexF64)
                for nu2 in 1:n_bands
                    acc += alpha1_blocks[p][nu1, nu2] * conj(xv2[nu2])
                end
                bu_w[iq1, nu1] += acc
            end
            if iq1 != iq2
                for nu2 in 1:n_bands
                    acc = zero(ComplexF64)
                    for nu1 in 1:n_bands
                        acc += alpha1_blocks[p][nu1, nu2] * conj(xw1[nu1])
                    end
                    bu_v[iq2, nu2] += acc
                end
            end

            local_w = zero(ComplexF64)
            for nu1 in 1:n_bands
                for nu2 in 1:n_bands
                    local_w += conj(xw1[nu1]) * alpha1_blocks[p][nu1, nu2] * conj(xv2[nu2])
                end
            end
            total_sum += (iq1 < iq2 ? 2 * local_w : local_w)
        end

        buf_f_weight_w = zero(ComplexF64)
        buf_f_weight_v = zero(ComplexF64)
        for iq in 1:n_q
            for nu in 1:n_bands
                idx = (iq-1)*n_bands + nu
                buf_f_weight_w += bu_w[iq, nu] * f_psi[nu, iq] * conj(yw_rot[idx])
                buf_f_weight_v += bu_v[iq, nu] * f_psi[nu, iq] * conj(yv_rot[idx])
            end
        end

        # === f_pert (D3 from alpha1, outputs on slot z) ===
        if compute_d3
            w1 = d3_force_z ? -total_sum / 2.0 * rho[i_config] / 3.0 * scale3 : zero(ComplexF64)
            bfw = (d3_force_w ? buf_f_weight_w : zero(ComplexF64)) +
                  (d3_force_v ? buf_f_weight_v : zero(ComplexF64))
            w2 = -bfw * rho[i_config] / 3.0 * scale3
            for nu in 1:n_bands
                if d3_force_z
                    f_pert[nu] += w1 * yz_pert[nu]
                end
                if d3_force_w || d3_force_v
                    f_pert[nu] += w2 * f_Y[nu, iq_pert] * xz_pert[nu]
                end
            end
        end

        # === D4 total weights ===
        # The four D4 force-leg channels (analogue of the D3 force split):
        #   ew: force on the EXTERNAL w leg  -> the y_w(q1) * f_Y x_v(q2) part
        #   ev: force on the EXTERNAL v leg  -> the f_Y x_w(q1) * y_v(q2) part
        #   iw: force on the INTERNAL w leg  -> the buf_f_weight_w part
        #   iv: force on the INTERNAL v leg  -> the buf_f_weight_v part
        # All four true reproduces the original fused D4 term exactly.
        total_wD4_ew = zero(ComplexF64)
        total_wD4_ev = zero(ComplexF64)
        total_wb = zero(ComplexF64)
        if compute_d4
            wd4 = -total_sum * rho[i_config] / 8.0 * scale4
            total_wD4_ew = d4_force_ew ? wd4 : zero(ComplexF64)
            total_wD4_ev = d4_force_ev ? wd4 : zero(ComplexF64)
            total_wb = -((d4_force_iw ? buf_f_weight_w : zero(ComplexF64)) +
                         (d4_force_iv ? buf_f_weight_v : zero(ComplexF64))) *
                       rho[i_config] / 4.0 * scale4
        end

        if weight_R != 0 || weight_Rf != 0 || total_wD4_ew != 0 ||
           total_wD4_ev != 0 || total_wb != 0
            for p in 1:n_pairs
                iq1 = unique_pairs[p, 1]
                iq2 = unique_pairs[p, 2]
                xw1 = view(xw_rot, (iq1-1)*n_bands+1:iq1*n_bands)
                yw1 = view(yw_rot, (iq1-1)*n_bands+1:iq1*n_bands)
                xv2 = view(xv_rot, (iq2-1)*n_bands+1:iq2*n_bands)
                yv2 = view(yv_rot, (iq2-1)*n_bands+1:iq2*n_bands)

                for nu1 in 1:n_bands
                    r1_1 = f_Y[nu1, iq1] * xw1[nu1]
                    r2_1 = yw1[nu1]
                    for nu2 in 1:n_bands
                        r1_2 = f_Y[nu2, iq2] * xv2[nu2]
                        r2_2 = yv2[nu2]
                        contrib = zero(ComplexF64)
                        if compute_d3 && d3_force_v
                            contrib -= weight_R * r1_1 * r2_2
                        end
                        if compute_d3 && d3_force_w
                            contrib -= weight_R * r2_1 * r1_2
                        end
                        if compute_d3 && d3_force_z
                            contrib -= weight_Rf * r1_1 * r1_2
                        end
                        if compute_d4
                            contrib -= total_wD4_ev * r1_1 * r2_2
                            contrib -= total_wD4_ew * r2_1 * r1_2
                            contrib -= total_wb * r1_1 * r1_2
                        end
                        d2v_blocks[p][nu1, nu2] += contrib
                    end
                end
            end
        end
    end

    norm_factor = n_syms * N_eff
    f_pert ./= norm_factor
    for p in 1:n_pairs
        d2v_blocks[p] ./= norm_factor
    end

    return f_pert, d2v_blocks
end


"""
    get_perturb_averages_qspace_slots(...)

Entry point for the slot-resolved (windowed) estimator, called from Python
once per (pass, orientation). Returns the same packed layout as
get_perturb_averages_qspace: [f_pert(n_bands); d2v_blocks_flat].
"""
function get_perturb_averages_qspace_slots(
    Xz::Array{ComplexF64,3}, Yz::Array{ComplexF64,3},
    Xw::Array{ComplexF64,3}, Yw::Array{ComplexF64,3},
    Xv::Array{ComplexF64,3}, Yv::Array{ComplexF64,3},
    w_q::Matrix{Float64},
    rho::Vector{Float64},
    R1::Vector{ComplexF64},
    alpha1_flat::Vector{ComplexF64},
    temperature::Float64,
    compute_d3::Bool,
    compute_d4::Bool,
    iq_pert::Int64,
    unique_pairs::Matrix{Int32},
    start_index::Int64,
    end_index::Int64,
    valid_modes_q::Matrix{Bool},
    scale3::Float64,
    scale4::Float64,
    prefiltered::Bool,
    batched::Bool=true,
    d3_force_z::Bool=true,
    d3_force_w::Bool=true,
    d3_force_v::Bool=true,
    d4_force_ew::Bool=true,
    d4_force_ev::Bool=true,
    d4_force_iw::Bool=true,
    d4_force_iv::Bool=true
)
    n_q = size(Xz, 1)
    n_bands = size(Xz, 3)
    n_pairs = size(unique_pairs, 1)

    symmetries = _cached_qspace_symmetries[]
    if symmetries === nothing
        error("Q-space symmetries not initialized. Call init_sparse_symmetries_qspace first.")
    end

    f_Y = zeros(Float64, n_bands, n_q)
    f_psi = zeros(Float64, n_bands, n_q)
    for iq in 1:n_q
        for nu in 1:n_bands
            if !valid_modes_q[nu, iq]
                continue
            end
            if prefiltered
                f_Y[nu, iq] = 1.0
                f_psi[nu, iq] = 1.0
                continue
            end
            w = w_q[nu, iq]
            nw = temperature > 0 ? 1.0 / (exp(w * RY_TO_K_Q / temperature) - 1.0) : 0.0
            f_Y[nu, iq] = 2.0 * w / (1.0 + 2.0 * nw)
            f_psi[nu, iq] = (1.0 + 2.0 * nw) / (2.0 * w)
        end
    end

    alpha1_blocks = Vector{Matrix{ComplexF64}}(undef, n_pairs)
    offset = 1
    for p in 1:n_pairs
        alpha1_blocks[p] = reshape(alpha1_flat[offset:offset+n_bands^2-1], n_bands, n_bands)
        offset += n_bands^2
    end

    kernel = batched ? _slots_kernel_batched : get_perturb_averages_qspace_slots_kernel
    f_pert, d2v = kernel(
        Xz, Yz, Xw, Yw, Xv, Yv, f_Y, f_psi, rho, R1, alpha1_blocks,
        symmetries, compute_d3, compute_d4, iq_pert, unique_pairs,
        n_bands, n_q, start_index, end_index, scale3, scale4,
        d3_force_z, d3_force_w, d3_force_v,
        d4_force_ew, d4_force_ev, d4_force_iw, d4_force_iv)

    result = zeros(ComplexF64, n_bands + n_pairs * n_bands^2)
    result[1:n_bands] = f_pert
    offset = n_bands + 1
    for p in 1:n_pairs
        result[offset:offset+n_bands^2-1] = vec(d2v[p])
        offset += n_bands^2
    end
    return result
end


# =========================================================================
# Batched (BLAS-3) version of the slot-resolved kernel (M4)
# =========================================================================
#
# Same mathematics as get_perturb_averages_qspace_slots_kernel, restructured
# over chunks of configurations at fixed symmetry:
#   - the symmetry rotation becomes one sparse x dense product per field,
#   - the alpha1 intermediates become (nb x nb)(nb x B) gemms,
#   - the d2v accumulation becomes three (nb x B)(B x nb) gemms per pair.
# Scalar-kernel equivalence is regression-tested from Python.

function _slots_kernel_batched(
    Xz::Array{ComplexF64,3}, Yz::Array{ComplexF64,3},
    Xw::Array{ComplexF64,3}, Yw::Array{ComplexF64,3},
    Xv::Array{ComplexF64,3}, Yv::Array{ComplexF64,3},
    f_Y::Matrix{Float64},
    f_psi::Matrix{Float64},
    rho::Vector{Float64},
    R1::Vector{ComplexF64},
    alpha1_blocks::Vector{Matrix{ComplexF64}},
    symmetries::Vector{SparseMatrixCSC{ComplexF64,Int32}},
    compute_d3::Bool,
    compute_d4::Bool,
    iq_pert::Int64,
    unique_pairs::Matrix{Int32},
    n_bands::Int64,
    n_q::Int64,
    start_index::Int64,
    end_index::Int64,
    scale3::Float64,
    scale4::Float64,
    d3_force_z::Bool=true,
    d3_force_w::Bool=true,
    d3_force_v::Bool=true,
    d4_force_ew::Bool=true,
    d4_force_ev::Bool=true,
    d4_force_iw::Bool=true,
    d4_force_iv::Bool=true
)
    n_pairs = size(unique_pairs, 1)
    n_syms = length(symmetries)
    n_total = n_q * n_bands
    N_eff = sum(rho)
    nb = n_bands

    d2v_blocks = [zeros(ComplexF64, nb, nb) for _ in 1:n_pairs]
    f_pert = zeros(ComplexF64, nb)

    # chunk size: bounded working set (~32 MB of chunk matrices)
    B_max = clamp(div(32 * 1024 * 1024, n_total * 16 * 12), 8, 256)

    # preallocated chunk workspaces (n_total x B_max)
    raw = [zeros(ComplexF64, n_total, B_max) for _ in 1:6]
    rot = [zeros(ComplexF64, n_total, B_max) for _ in 1:6]
    BUw = zeros(ComplexF64, n_total, B_max)
    BUv = zeros(ComplexF64, n_total, B_max)
    T1 = zeros(ComplexF64, nb, B_max)
    A1 = zeros(ComplexF64, nb, B_max)
    A2 = zeros(ComplexF64, nb, B_max)

    fpsi_vec = zeros(Float64, n_total)
    for iq in 1:n_q, nu in 1:nb
        fpsi_vec[(iq-1)*nb + nu] = f_psi[nu, iq]
    end
    aR1 = [f_Y[nu, iq_pert] * R1[nu] for nu in 1:nb]   # f_Y(qp) .* R1
    qp_rows = (iq_pert-1)*nb+1 : iq_pert*nb

    fields = (Xz, Yz, Xw, Yw, Xv, Yv)

    for j_sym in 1:n_syms
        # configs c whose bigindex (c-1)*n_syms + j_sym is in [start, end]
        c_lo = max(1, cld(start_index - j_sym, n_syms) + 1)
        c_hi = min(size(Xz, 2), fld(end_index - j_sym, n_syms) + 1)
        c_lo > c_hi && continue
        S = symmetries[j_sym]

        c = c_lo
        while c <= c_hi
            B = min(B_max, c_hi - c + 1)
            cols = 1:B

            # gather + rotate all six fields
            for (fi, F) in enumerate(fields)
                Rw = raw[fi]
                for b in 1:B
                    cfg = c + b - 1
                    for iq in 1:n_q
                        base = (iq-1)*nb
                        @inbounds for nu in 1:nb
                            Rw[base+nu, b] = F[iq, cfg, nu]
                        end
                    end
                end
                mul!(view(rot[fi], :, cols), S, view(Rw, :, cols))
            end
            Xz_r = view(rot[1], :, cols); Yz_r = view(rot[2], :, cols)
            Xw_r = view(rot[3], :, cols); Yw_r = view(rot[4], :, cols)
            Xv_r = view(rot[5], :, cols); Yv_r = view(rot[6], :, cols)

            rho_c = view(rho, c:c+B-1)

            # === D3 weights from R1 (slot z) ===
            weight_R = zeros(ComplexF64, B)
            weight_Rf = zeros(ComplexF64, B)
            if compute_d3 && (d3_force_w || d3_force_v || d3_force_z)
                Xz_qp = view(Xz_r, qp_rows, :)
                Yz_qp = view(Yz_r, qp_rows, :)
                for b in 1:B
                    wr = zero(ComplexF64); wrf = zero(ComplexF64)
                    if d3_force_w || d3_force_v
                        @inbounds for nu in 1:nb
                            wr += conj(Xz_qp[nu, b]) * aR1[nu]
                        end
                    end
                    if d3_force_z
                        @inbounds for nu in 1:nb
                            wrf += R1[nu] * conj(Yz_qp[nu, b])
                        end
                    end
                    weight_R[b] = wr * rho_c[b] / 3.0 * scale3
                    weight_Rf[b] = wrf * rho_c[b] / 3.0 * scale3
                end
            end

            # === alpha1 intermediates ===
            fill!(view(BUw, :, cols), zero(ComplexF64))
            fill!(view(BUv, :, cols), zero(ComplexF64))
            total_sum = zeros(ComplexF64, B)

            for p in 1:n_pairs
                iq1 = unique_pairs[p, 1]; iq2 = unique_pairs[p, 2]
                r1s = (iq1-1)*nb+1 : iq1*nb
                r2s = (iq2-1)*nb+1 : iq2*nb
                Xw1 = view(Xw_r, r1s, :)
                Xv2 = view(Xv_r, r2s, :)

                # T1 = alpha1[p] * conj(Xv2)   (nb x B)
                T1v = view(T1, :, cols)
                mul!(T1v, alpha1_blocks[p], conj.(Xv2))
                view(BUw, r1s, cols) .+= T1v

                # total_sum += mult * sum(conj(Xw1) .* T1, dims=1)
                mult = iq1 < iq2 ? 2.0 : 1.0
                for b in 1:B
                    acc = zero(ComplexF64)
                    @inbounds for nu in 1:nb
                        acc += conj(Xw1[nu, b]) * T1v[nu, b]
                    end
                    total_sum[b] += mult * acc
                end

                if iq1 != iq2
                    # BUv[q2] += alpha1^T * conj(Xw1)
                    mul!(T1v, transpose(alpha1_blocks[p]), conj.(Xw1))
                    view(BUv, r2s, cols) .+= T1v
                end
            end

            buf_f_w = zeros(ComplexF64, B)
            buf_f_v = zeros(ComplexF64, B)
            for b in 1:B
                acc_w = zero(ComplexF64)
                acc_v = zero(ComplexF64)
                @inbounds for i in 1:n_total
                    acc_w += BUw[i, b] * fpsi_vec[i] * conj(Yw_r[i, b])
                    acc_v += BUv[i, b] * fpsi_vec[i] * conj(Yv_r[i, b])
                end
                buf_f_w[b] = acc_w
                buf_f_v[b] = acc_v
            end

            # === f_pert (D3 from alpha1, slot z outputs) ===
            if compute_d3
                Xz_qp = view(Xz_r, qp_rows, :)
                Yz_qp = view(Yz_r, qp_rows, :)
                for b in 1:B
                    w1 = d3_force_z ? -total_sum[b] / 2.0 * rho_c[b] / 3.0 * scale3 : zero(ComplexF64)
                    bfw = (d3_force_w ? buf_f_w[b] : zero(ComplexF64)) +
                          (d3_force_v ? buf_f_v[b] : zero(ComplexF64))
                    w2 = -bfw * rho_c[b] / 3.0 * scale3
                    @inbounds for nu in 1:nb
                        if d3_force_z
                            f_pert[nu] += w1 * Yz_qp[nu, b]
                        end
                        if d3_force_w || d3_force_v
                            f_pert[nu] += w2 * f_Y[nu, iq_pert] * Xz_qp[nu, b]
                        end
                    end
                end
            end

            # === combined d2v weights per column ===
            w_cross_v = zeros(ComplexF64, B)
            w_cross_w = zeros(ComplexF64, B)
            w_diag = zeros(ComplexF64, B)
            any_w = false
            for b in 1:B
                wcv = (compute_d3 && d3_force_v) ? weight_R[b] : zero(ComplexF64)
                wcw = (compute_d3 && d3_force_w) ? weight_R[b] : zero(ComplexF64)
                wd = (compute_d3 && d3_force_z) ? weight_Rf[b] : zero(ComplexF64)
                if compute_d4
                    # D4 force-leg channels (see the scalar kernel):
                    # ev -> w_cross_v (f_Y x_w * y_v), ew -> w_cross_w
                    # (y_w * f_Y x_v), iw/iv -> the internal-force diagonal.
                    wd4 = -total_sum[b] * rho_c[b] / 8.0 * scale4
                    wcv += d4_force_ev ? wd4 : zero(ComplexF64)
                    wcw += d4_force_ew ? wd4 : zero(ComplexF64)
                    wd += -((d4_force_iw ? buf_f_w[b] : zero(ComplexF64)) +
                            (d4_force_iv ? buf_f_v[b] : zero(ComplexF64))) *
                          rho_c[b] / 4.0 * scale4
                end
                w_cross_v[b] = wcv
                w_cross_w[b] = wcw
                w_diag[b] = wd
                any_w = any_w || (wcv != 0) || (wcw != 0) || (wd != 0)
            end

            if any_w
                for p in 1:n_pairs
                    iq1 = unique_pairs[p, 1]; iq2 = unique_pairs[p, 2]
                    r1s = (iq1-1)*nb+1 : iq1*nb
                    r2s = (iq2-1)*nb+1 : iq2*nb

                    A1v = view(A1, :, cols)
                    A2v = view(A2, :, cols)
                    T1v = view(T1, :, cols)

                    # r1w = f_Y(q1) .* Xw(q1); r2w = Yw(q1)
                    # r1v = f_Y(q2) .* Xv(q2); r2v = Yv(q2)
                    @inbounds for b in 1:B, nu in 1:nb
                        A1v[nu, b] = f_Y[nu, iq1] * Xw_r[r1s[1]+nu-1, b]   # r1w
                        A2v[nu, b] = Yw_r[r1s[1]+nu-1, b]                 # r2w
                    end
                    # T1 = r2v scaled by force-v / D4 cross columns
                    @inbounds for b in 1:B, nu in 1:nb
                        T1v[nu, b] = -w_cross_v[b] * Yv_r[r2s[1]+nu-1, b]
                    end
                    mul!(d2v_blocks[p], A1v, transpose(T1v), 1.0, 1.0)
                    # T1 = r1v scaled by force-w / D4 cross columns
                    @inbounds for b in 1:B, nu in 1:nb
                        T1v[nu, b] = -w_cross_w[b] * f_Y[nu, iq2] * Xv_r[r2s[1]+nu-1, b]
                    end
                    mul!(d2v_blocks[p], A2v, transpose(T1v), 1.0, 1.0)
                    # T1 = r1v scaled by -w_diag
                    @inbounds for b in 1:B, nu in 1:nb
                        T1v[nu, b] = -w_diag[b] * f_Y[nu, iq2] * Xv_r[r2s[1]+nu-1, b]
                    end
                    mul!(d2v_blocks[p], A1v, transpose(T1v), 1.0, 1.0)
                end
            end

            c += B
        end
    end

    norm_factor = n_syms * N_eff
    f_pert ./= norm_factor
    for p in 1:n_pairs
        d2v_blocks[p] ./= norm_factor
    end

    return f_pert, d2v_blocks
end
