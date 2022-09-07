module TDSCHA 
using SparseArrays
using LinearAlgebra

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

            weight = - displacements' * Y1 * f_ψ  / 4
            weight -= f_ψ' * Y1 * displacements / 4

            f_average .+= ω_is[i] * weight * 2 / 3 .* f_Y .* displacements
        end  
    end 

    f_average ./= n_symmetries * sum(ω_is)

    return f_average
end

end