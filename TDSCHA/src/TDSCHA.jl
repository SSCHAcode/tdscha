module TDSCHA 
using SparseArrays
using LinearAlgebra
using BLAS

struct Ensemble{T<: AbstractFloat} 
    X:: Matrix{T}
    Y:: Matrix{T}
    ω::Vector{T}
end


struct SymmetriesInfo{T<: AbstractFloat}
    symmetries::Array{T, 4}#Vector{SparseMatrixCSC{T,Int}}
    n_degeneracies::Vector{Int}
    degenerate_space::Matrix{Int}
    blocks::Vector{Int}
end

const RY_TO_K = 157887.32400374097

function create_sparse_matrix_from_symmetries(sym_info::SymmetriesInfo{T}) where {T<: AbstractFloat}
    nblocks = length(sym_info.bloks)
    
    nsyms = size(sym_info.symmetries, 2)
    symmetries = Vector{SparseMatrixCSC{T,Int}}(nsyms)

    for sindex in 1:nsyms
        mode_x = Vector{Int}()
        mode_y = Vector{Int}()
        values = Vector{T}()
        for i in 1:nblocks
            for mu in 1:sym_info.n_degeneracies[i]
                for nu in 1:sym_info.n_degeneracies[i]
                    append!(mode_x, sym_info.degenerate_space[i][mu])
                    append!(mode_y, sym_info.degenerate_space[i][nu])
                    append!(values, sym_info.symmetries[i, sindex, mu, nu])
                end
            end 

            symmetries[i] = sparse(mode_x, mode_y, values)
        end 
    end

    return symmetries
end 

function get_d2v_dR2_from_R_pert_sym_fast(ensemble::Ensemble{T}, symmetries::Vector{SparseMatrixCSC{T,Int}}, temperature::T, R1::Vector{T}, ω_is::Vector{T}) where {T<: AbstractFloat}
    n_modes = length(ensemble.ω)
    n_configs = size(ensemble.X, 1)
    n_symmetries = length(symmetries)

    forces = similar(ensemble.ω)
    displacements = similar(forces)
    n_ω = temperature > 0 ? 1 ./ (exp.(ensemble.ω .* RY_TO_K / temperature) .- 1) : zeros(T, n_modes)
    f_ψ = (1 .+ 2 .* n_ω) ./ (2 .* ensemble.ω)
    f_Y =  2 .* ensemble.ω ./ (1 .+ 2 .* n_ω)

    d2v_dR2 = zeros(T, (n_modes, n_modes))

    r1_aux = zeros(T, (n_modes, 1))
    r2_aux = zeros(T, (n_modes, 1))

    
    Threads.@threads for i in 1:n_configs
        Threads.@threads for j in 1:n_symmetries
            # Compute the symmetrized forces and displacements
            mul!(forces, symmetries[j], view(ensemble.Y, :, i))
            mul!(displacements, symmetries[j], view(ensemble.X, :, i))

            r1_aux[:, 0] .= f_Y .* displacements
            r2_aux[:, 0] .= forces
            
            
            weight =  view(r1_aux[:,0])' * R1 
            weight *= ω_is[i] / 3

            # Outer product between r1 and r2 (with symmetrization)
            BLAS.gemm!('N', 'T', -weight, r1_aux, r2_aux, 0.0, d2v_dR2)
            BLAS.gemm!('N', 'T', -weight, r2_aux, r1_aux, 1.0, d2v_dR2)


            # The second part 
            weight = (R1' * forces) * ω_is[i] / 3 
            # Outer product between r1 and r1
            BLAS.gemm!('N', 'T', -weight, r1_aux, r1_aux, 1.0, d2v_dR2)
        end
    end

    # Now rescale
    d2v_dR2 ./=  n_symmetries * sum(ω_is)
    return d2v_dR2
end 


function get_d2v_dR2_from_Y_pert_sym_fast(ensemble::Ensemble{T}, symmetries::Vector{SparseMatrixCSC{T,Int}}, temperature::T, Y1::Matrix{T}, ω_is::Vector{T}) where {T<: AbstractFloat}
    n_modes = length(ensemble.ω)
    n_configs = size(ensemble.X, 1)
    n_symmetries = length(symmetries)

    forces = similar(ensemble.ω)
    displacements = similar(forces)
    n_ω = temperature > 0 ? 1 ./ (exp.(ensemble.ω .* RY_TO_K / temperature) .- 1) : zeros(T, n_modes)
    f_ψ = (1 .+ 2 .* n_ω) ./ (2 .* ensemble.ω)
    f_Y =  2 .* ensemble.ω ./ (1 .+ 2 .* n_ω)

    d2v_dR2 = zeros(T, (n_modes, n_modes))

    r1_aux = zeros(T, (n_modes, 1))
    r2_aux = zeros(T, (n_modes, 1))

    Threads.@threads for i in 1:n_configs
        Threads.@threads for j in 1:n_symmetries
            # Compute the symmetrized forces and displacements
            mul!(forces, symmetries[j], view(ensemble.Y, :, i))
            mul!(displacements, symmetries[j], view(ensemble.X, :, i))


            weight = -displacements' * Y1 * displacements 
            weight *= ω_is[i] / 8

            r1_aux[:, 0] .= f_Y .* displacements
            r2_aux[:, 0] .= forces

            BLAS.gemm!('N', 'T', -weight, r1_aux, r2_aux, 0.0, d2v_dR2)
            BLAS.gemm!('N', 'T', -weight, r2_aux, r1_aux, 1.0, d2v_dR2)

            weight = - displacements' * Y1 * (f_ψ .* forces)  / 4
            weight -= (f_ψ .* forces)' * Y1 * displacements / 4
            BLAS.gemm!('N', 'T', -weight, r1_aux, r1_aux, 1.0, d2v_dR2)
        end
    end
    return d2v_dR2 / (n_symmetries * sum(ω_is))
end



function get_f_average_from_Y_pert(ensemble::Ensemble{T}, symmetries::Vector{SparseMatrixCSC{T,Int}}, temperature::T, Y1::Matrix{T}, ω_is::Vector{T}) where {T<: AbstractFloat}
    n_modes = length(ensemble.ω)
    n_configs = size(ensemble.X, 1)
    n_symmetries = length(symmetries)

    forces = similar(ensemble.ω)
    displacements = similar(forces)
    n_ω = temperature > 0 ? 1 ./ (exp.(ensemble.ω .* RY_TO_K / temperature) .- 1) : zeros(T, n_modes)
    f_ψ = (1 .+ 2 .* n_ω) ./ (2 .* ensemble.ω)
    f_Y =  2 .* ensemble.ω ./ (1 .+ 2 .* n_ω)
    # Compute f_psi

    f_average = zeros(T, n_modes) 

    
    Threads.@threads for i in 1:n_configs
        Threads.@threads for j in 1:n_symmetries
            # Get forces and threads
            #=
            trial
            =#

            mul!(forces, symmetries[j], view(ensemble.Y, :, i))
            mul!(displacements, symmetries[j], view(ensemble.X, :, i))


            weight = -displacements' * Y1 * displacements / 2
            f_average .+= ω_is[i] * weight / 3 .* forces

            weight = - displacements' * Y1 * (f_ψ .* forces)  / 4
            weight -= (f_ψ .* forces)' * Y1 * displacements / 4

            f_average .+= ω_is[i] * weight * 2 / 3 .* f_Y .* displacements
        end  
    end 

    f_average ./= n_symmetries * sum(ω_is)

    return f_average
end



# This is the function that speaks with the python code
function get_perturb_averages_sym(X::Matrix{T}, Y::Matrix{T}, ω::Vector{T}, rho::Vector{T}, 
        R1::Vector{T}, Y1::Matrix{T}, temperature::T, apply_v4::bool, 
        n_symmetries::Int, symmetries::Array{T, 4}, n_degeneracies::Vector{Int}, 
        degenerate_space::Matrix{Int}, blocks::Vector{Int}) where {T<:AbstractFloat}


    # Prepare the symmetry sym_info
    sym_info = SymmetriesInfo(symmetries, n_degeneracies, degenerate_space, blocks)

    # Convert to a sparse matrix for fast linear LinearAlgebra
    new_symmetries = create_sparse_matrix_from_symmetries(sym_info)

    # Create the ensemble
    ensemble = Ensemble(X', Y', ω)

    # Get the average force
    f_average = get_f_average_from_Y_pert(ensemble, new_symmetries, temperature, Y1, rho)
    d2v_dr2 = get_d2v_dR2_from_R_pert_sym_fast(ensemble, new_symmetries, temperature, R1, rho)

    if apply_v4
        d2v_dr2 += get_d2v_dR2_from_Y_pert_sym_fast(ensemble, new_symmetries, temperature, Y1, rho)
    end 

    return f_average, d2v_dr2
end