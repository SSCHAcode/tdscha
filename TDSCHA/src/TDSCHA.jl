module TDSCHA 
using SparseArrays
using LinearAlgebra

struct Ensemble{T<: AbstractFloat} 
    X:: Matrix{T}
    Y:: Matrix{T}
    ω::Vector{T}
end


struct Symmetry{T<:AbstractFloat}
    blocks::Vector{Matrix{T}}
end 

struct SymmetriesInfo{T<: AbstractFloat}
    symmetries::Vector{SparseMatrixCSC{T,Int}}
    n_degeneracies::Vector{Int}
    degeneracies::Matrix{Int}
    blocks::Vector{Int}
end

function SymmetriesInfo()

const RY_TO_K = 157887.32400374097

function get_f_average_from_Y_pert(ensemble::Ensemble{T}, sym_info::SymmetriesInfo{T}, temperature::T, Y1::Matrix{T}, ω_is::Vector{T}) where {T<: AbstractFloat}
    n_modes = length(ensemble.ω)
    n_configs = size(ensemble.X, 1)
    n_symmetries = lenght(sym_info.symmetries)

    forces = similar(ensemble.ω)
    displacements = similar(forces)
    n_ω = temperature > 0 ? 1 ./ (exp.(ensemble.ω .* RY_TO_K / temperature) .- 1) : zeros(T, n_modes)
    f_ψ = (1 .+ 2 .* n_ω) ./ (2 .* ensemble.ω)
    f_υ =  2 .* ensemble.ω ./ (1 .+ 2 .* n_ω)
    # Compute f_psi

    f_average = zeros(T, n_modes) 

    
    Threads.@threads for i in 1:n_configs
        Threads.@threads for j in 1:n_symmetries
            # Get forces and threads
            #=
            trial
            =#

            mul!(forces, sym_info.symmetries[j], view(ensemble.Y, :, i))
            mul!(displacements, sym_info.symmetries[j], view(ensemble.X, :, i))


            weight = -displacements' * Y1 * displacements / 2
            f_average .+= ω_is[i] * weight / 3 .* forces

            weight = - displacements' * Y1 * f_ψ  / 4
            weight -= f_ψ' * Y1 * displacements / 4

            f_average .+= ω_is[i] * weight * 2 / 3 .* f_υ .* displacements
        end  
    end 

    f_average ./= n_symmetries * sum(ω_is)

    return f_average
end

end