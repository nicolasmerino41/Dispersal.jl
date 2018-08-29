"Neighborhoods for dispersal"
abstract type AbstractDispersalNeighborhood <: AbstractNeighborhood end

abstract type AbstractDispersalGrid <: AbstractDispersalNeighborhood end

"""
A neighborhood built from a dispersal kernel function and a cell radius.
Can be built directly by passing in the array, radius and overflow
arguments, but preferably use the keyword constructor to build the array from
a dispersal kernel function.
"""
@limits @flattenable struct DispersalNeighborhood{T,F,P,K,C,I,O} #<: AbstractDispersalNeighborhood
    f::F        | false | _
    param::P    | true  | (0.0, 10.0)
    kernel::K   | false | _
    cellsize::C | true  | (0.0, 10.0)
    radius::I   | true  | (1, 10)
    overflow::O | false | _
    function DispersalNeighborhood{T,F,P,K,C,I,O}(f::F, param::P, init_kernel::K, 
                                                  cellsize::C, radius::I, overflow::O
                                                 ) where {T,F,P,K,C,I,O}
        kernel = build_dispersal_kernel(f, param, init_kernel, cellsize, radius)
        new{T,F,P,typeof(kernel),C,I,O}(f, param, kernel, cellsize, radius, overflow)
    end
end

DispersalNeighborhood(; dir=:inwards, f=exponential, param=1.0, init=[], cellsize=1.0, 
                      radius=3, overflow=Skip()) = begin
    DispersalNeighborhood{dir, typeof.((f, param, init, cellsize, radius, overflow))...
                         }(f, param, init, cellsize, radius, overflow)
end


struct HudginsDispersalGrid{K} <: AbstractDispersalGrid
    kernel::K
end



@mix @limits @flattenable @with_kw struct Dispersal{S,T}
    # "A number or Unitful.jl distance."
    cellsize::S = 1.0                                          | false | _
    # "Minimum habitat suitability index."
    suitability_threshold::T = 0.1                             | true  | (0.0, 1.0)
end

@mix @limits @flattenable @with_kw struct Neighbors{N}
    # "Neighborhood to disperse to or from"
    neighborhood::N = DispersalNeighborhood(cellsize=cellsize) | true  | _
end

@mix @limits @flattenable struct Probabilistic{P}
    # "A real number between one and zero."
    prob_threshold::P = 0.1 | true | (0.0, 1.0)
end

@mix @limits @flattenable struct SpotRange{S}
    # "A number or Unitful.jl distance with the same units as cellsize"
    spotrange::S = 30.0     | true | (0.0, 100.0)
end

"Extend to modify [`InwardsLocalDispersal`](@ref)"
abstract type AbstractInwardsDispersal <: AbstractModel end

"""
Local dispersal within a [`DispersalNeighborhood`](@ref) or other neighborhoods.
Inwards dispersal calculates dispersal *to* the current cell from cells in the neighborhood.
"""
@Probabilistic @Dispersal @Neighbors struct InwardsLocalDispersal{} <: AbstractInwardsDispersal end

"Extend to modify [`OutwardsLocalDispersal`](@ref)"
abstract type AbstractOutwardsDispersal <: AbstractPartialModel end

"""
Local dispersal within a [`DispersalNeighborhood`](@ref)

Outwards dispersal calculates dispersal *from* the current cell to cells
in its neighborhood. This should be more efficient than inwards
dispersal when a small number of cells are occupied, but less efficient when a large
proportion of the grid is occupied.
"""
@Probabilistic @Dispersal @Neighbors struct OutwardsLocalDispersal{} <: AbstractOutwardsDispersal end

@Dispersal struct HudginsDispersal{} <: AbstractOutwardsDispersal
    pop_threshold::Float64 = 0.0006227 | true | (0.0, 0.1) 
    growthrate::Float64 = 2.4321       | true | (0.0, 10.0)
end

"Extend to modify [`JumpDispersal`](@ref)"
abstract type AbstractJumpDispersal <: AbstractPartialModel end

"Jump dispersal within a [`DispersalNeighborhood`](@ref)] or other neighborhoods."
@Probabilistic @SpotRange @Dispersal struct JumpDispersal{} <: AbstractJumpDispersal end

"Extend to modify [`HumanDispersal`](@ref)"
abstract type AbstractHumanDispersal <: AbstractPartialModel end
"Human dispersal model."
@Probabilistic @SpotRange @Dispersal struct HumanDispersal{} <: AbstractHumanDispersal end


