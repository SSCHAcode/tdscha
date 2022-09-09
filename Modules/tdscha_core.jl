# Import all the necessary packages

using SparseArrays
using LinearAlgebra
using LinearAlgebra.BLAS
LinearAlgebra.BLAS.set_num_threads(1)
struct Ensemble{T<: AbstractFloat} 
    X:: Matrix{T}
    Y:: Matrix{T}
    ω::Vector{T}
end


struct SymmetriesInfo{T<: AbstractFloat}
    symmetries::Array{T, 4}#Vector{SparseMatrixCSC{T,Int}}
    n_degeneracies::Vector{Int32}
    degenerate_space::Matrix{Int32}
    blocks::Vector{Int32}
end

const RY_TO_K = 157887.32400374097

function create_sparse_matrix_from_symmetries(sym_info::SymmetriesInfo{T}) where {T<: AbstractFloat}
    nblocks = size(sym_info.symmetries, 1)
    
    nsyms = size(sym_info.symmetries, 2)
    mysym = Vector{SparseMatrixCSC{T,Int32}}(undef, nsyms)


    for sindex in 1:nsyms
        mode_x = Vector{Int32}()
        mode_y = Vector{Int32}()
        values = Vector{T}()
        for i in 1:nblocks
            for mu in 1:sym_info.n_degeneracies[i]
                for nu in 1:sym_info.n_degeneracies[i]
                    # + 1 for py to julia indexing
                    append!(mode_x, sym_info.degenerate_space[i,mu] + 1) 
                    append!(mode_y, sym_info.degenerate_space[i,nu] + 1)
                    append!(values, sym_info.symmetries[i, sindex, mu, nu])
                end
            end 
        end 
        A = sparse(mode_x, mode_y, values)
        mysym[sindex] = A
    end

    return mysym
end 

function get_d2v_dR2_from_R_pert_sym_fast(ensemble::Ensemble{T}, symmetries::Vector{SparseMatrixCSC{T,Int32}}, temperature::T, R1::Vector{T}, ω_is::Vector{T}) where {T<: AbstractFloat}
    n_modes = length(ensemble.ω)
    n_configs = size(ensemble.X, 2)
    n_symmetries = length(symmetries)

    forces = similar(ensemble.ω)
    displacements = similar(forces)
    n_ω = temperature > 0 ? 1 ./ (exp.(ensemble.ω .* RY_TO_K / temperature) .- 1) : zeros(T, n_modes)
    f_ψ = (1 .+ 2 .* n_ω) ./ (2 .* ensemble.ω)
    f_Y =  2 .* ensemble.ω ./ (1 .+ 2 .* n_ω)

    # Define private buffers
    d2v_dR2_s = [zeros(T, (n_modes, n_modes)) for i = 1:Threads.nthreads()]
    r1_aux_s = [zeros(T, (n_modes, 1)) for i = 1:Threads.nthreads()]
    r2_aux_s = [zeros(T, (n_modes, 1)) for i = 1:Threads.nthreads()]

    
    Threads.@threads  for i in 1:n_configs
        thread_id = Threads.threadid()
        for j in 1:n_symmetries
            # Compute the symmetrized forces and displacements
            mul!(forces, symmetries[j], view(ensemble.Y, :, i))
            mul!(displacements, symmetries[j], view(ensemble.X, :, i))

            r1_aux_s[thread_id][:, 1] .= f_Y .* displacements
            r2_aux_s[thread_id][:, 1] .= forces
            
            
            weight = view(r1_aux_s[thread_id], :,1)' * R1 
            #println("i = $(i-1), j = $(j-1), weight = $weight")

            weight *= ω_is[i] / 3

            # Outer product between r1 and r2 (with symmetrization)
            BLAS.gemm!('N', 'T', -weight, r1_aux_s[thread_id], r2_aux_s[thread_id], 1.0, d2v_dR2_s[thread_id])
            BLAS.gemm!('N', 'T', -weight, r2_aux_s[thread_id], r1_aux_s[thread_id], 1.0, d2v_dR2_s[thread_id])


            # The second part 
            weight = (R1' * forces) * ω_is[i] / 3 
            # Outer product between r1 and r1
            BLAS.gemm!('N', 'T', -weight, r1_aux_s[thread_id], r1_aux_s[thread_id], 1.0, d2v_dR2_s[thread_id])

            #println("Total sum: $(sum(d2v_dR2))")
        end
    end

    # Now reduce and rescale
    d2v_dR2 = sum(d2v_dR2_s) ./ (n_symmetries * sum(ω_is))
    return d2v_dR2
end 


function get_d2v_dR2_from_Y_pert_sym_fast(ensemble::Ensemble{T}, symmetries::Vector{SparseMatrixCSC{T,Int32}}, temperature::T, Y1::Matrix{T}, ω_is::Vector{T}) where {T<: AbstractFloat}
    n_modes = length(ensemble.ω)
    n_configs = size(ensemble.X, 2)
    n_symmetries = length(symmetries)

    forces = [similar(ensemble.ω) for i = 1:Threads.nthreads()]
    displacements = similar(forces)
    n_ω = temperature > 0 ? 1 ./ (exp.(ensemble.ω .* RY_TO_K / temperature) .- 1) : zeros(T, n_modes)
    f_ψ = (1 .+ 2 .* n_ω) ./ (2 .* ensemble.ω)
    f_Y =  2 .* ensemble.ω ./ (1 .+ 2 .* n_ω)

    d2v_dR2 = zeros(T, (n_modes, n_modes))

    r1_aux = [zeros(T, (n_modes, 1)) for i = 1:Threads.nthreads()]
    r2_aux = [zeros(T, (n_modes, 1)) for i = 1:Threads.nthreads()]
    buffer_f = [zeros(T, n_modes) for i = 1:Threads.nthreads()]
    buffer_u = [zeros(T, n_modes) for i = 1:Threads.nthreads()]


    Threads.@threads for i in 1:n_configs
        id = Threads.threadid()
        for j in 1:n_symmetries
            # Compute the symmetrized forces and displacements
            mul!(forces, symmetries[j], view(ensemble.Y, :, i))
            mul!(displacements, symmetries[j], view(ensemble.X, :, i))
            mul!(buffer_u[id], Y1, displacements)
            buffer_f[id] = f_ψ .* forces

            weight = -displacements' * buffer_u[id]
            weight *= ω_is[i] / 8
            

            r1_aux[:, 1] .= f_Y .* displacements
            r2_aux[:, 1] .= forces

            BLAS.gemm!('N', 'T', -weight, r1_aux, r2_aux, 1.0, d2v_dR2)
            BLAS.gemm!('N', 'T', -weight, r2_aux, r1_aux, 1.0, d2v_dR2)

            weight = - buffer_u' * buffer_f
            weight *= ω_is[i] / 4

            BLAS.gemm!('N', 'T', -weight, r1_aux, r1_aux, 1.0, d2v_dR2)
        end
    end
    return d2v_dR2 / (n_symmetries * sum(ω_is))
end



function get_f_average_from_Y_pert(ensemble::Ensemble{T}, symmetries::Vector{SparseMatrixCSC{T,Int32}}, temperature::T, Y1::Matrix{T}, ω_is::Vector{T}) where {T<: AbstractFloat}
    n_modes = length(ensemble.ω)
    n_configs = size(ensemble.X, 2)
    n_symmetries = length(symmetries)

    forces = similar(ensemble.ω)
    displacements = similar(forces)
    n_ω = temperature > 0 ? 1 ./ (exp.(ensemble.ω .* RY_TO_K / temperature) .- 1) : zeros(T, n_modes)
    f_ψ = (1 .+ 2 .* n_ω) ./ (2 .* ensemble.ω)
    f_Y =  2 .* ensemble.ω ./ (1 .+ 2 .* n_ω)
    # Compute f_psi

    f_average = zeros(T, n_modes) 

    buffer_u = zeros(T, n_modes)
    buffer_f = zeros(T, n_modes)
    buffer_f1 = zeros(T, n_modes)


    Threads.@threads for i in 1:n_configs
        for j in 1:n_symmetries
            # Get forces and threads
            #=
            trial
            =#

            mul!(forces, symmetries[j], view(ensemble.Y, :, i))
            mul!(displacements, symmetries[j], view(ensemble.X, :, i))
            mul!(buffer_u, Y1, displacements)
            buffer_f .= forces
            buffer_f .*= f_ψ 
            buffer_f1 .= f_Y .* displacements

            weight = -displacements' * buffer_u / 2
            weight *= ω_is[i] / 3
            f_average .+= weight .* forces

            weight = - buffer_u' * buffer_f 
            weight *= ω_is[i] / 3.
            f_average .+= weight .* buffer_f1
        end  
    end 

    f_average ./= n_symmetries * sum(ω_is)


    return f_average
end



# This is the function that speaks with the python code
function get_perturb_averages_sym(X::Matrix{T}, Y::Matrix{T}, ω::Vector{T}, rho::Vector{T}, 
        R1::Vector{T}, Y1::Matrix{T}, temperature::T, apply_v4::Bool, symmetries::Array{T, 4}, 
        n_degeneracies::Vector{Int32}, 
        degenerate_space::Matrix{Int32}, blocks::Vector{Int32}) where {T<:AbstractFloat}


    # Prepare the symmetry sym_info
    sym_info = SymmetriesInfo(symmetries, n_degeneracies, degenerate_space, blocks)

    # Convert to a sparse matrix for fast linear LinearAlgebra
    new_symmetries = create_sparse_matrix_from_symmetries(sym_info)

    # Create the ensemble
    ensemble = Ensemble(X, Y, ω)

    # Get the average force
    f_average = get_f_average_from_Y_pert(ensemble, new_symmetries, temperature, Y1, rho)
    d2v_dr2 = get_d2v_dR2_from_R_pert_sym_fast(ensemble, new_symmetries, temperature, R1, rho)

    if apply_v4
        d2v_dr2 += get_d2v_dR2_from_Y_pert_sym_fast(ensemble, new_symmetries, temperature, Y1, rho)
    end 

    return f_average, d2v_dr2
end 