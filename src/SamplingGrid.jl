########## Sampling grid for stochastic process ##########

"""
Abstract sampling grid.
"""
const SamplingGrid{T} = AbstractArray{T,1}
# abstract type SamplingGrid{T} <: AbstractArray{T<:TimeStyle,1} end
# const SamplingGrid{T<:TimeStyle} = AbstractArray{T,1}
# const SamplingGrid{T} = AbstractArray{T<:TimeStyle,1}

const DiscreteTimeSamplingGrid = SamplingGrid{DiscreteTime}
const ContinuousTimeSamplingGrid = SamplingGrid{ContinuousTime}

# StepRangeLen{T,Base.TwicePrecision{T},Base.TwicePrecision{T}} where T

# const DiscreteTimeSamplingGrid = StepRangeLen{}
# const ContinuousTimeSamplingGrid = SamplingGrid{ContinuousTime}

"""
Test if a sampling grid is in a strictly increasing order.
"""
isvalidgrid(g::SamplingGrid) = any(diff(g) .> 0)

"""
Return the sampling step of a grid if possible.
"""
step(g::SamplingGrid) = isdefined(g, :step) ? g.step : NaN

"""
Regular grid with fixed sampling step.
"""
struct RegularGrid{T} <: SamplingGrid{T}
    start::T
    step::T  # step
    stop::T
    len::Int64

    """
    Inner constructor.
    """
    function RegularGrid{T}(start::T, step::T, stop::T) where T
        @assert step > 0
        @assert start <= stop
        len = Integer(div(stop + step - start, step))
        new(start, step, stop, len)
    end

    """
    Inner constructor from `AbstractRange` type.
    """
    function RegularGrid{T}(g::AbstractRange) where T
        RegularGrid{T}(T(g[1]), T(g[2]-g[1]), T(g[end]))
    end
end

"""
Outer constructor.
"""
function RegularGrid(start, step, stop)
    T0 = promote_type(typeof(start), typeof(step), typeof(stop))
    T = T0 <: DiscreteTime ? DiscreteTime : ContinuousTime
    RegularGrid{T}(start, step, stop)
end

function RegularGrid(g::AbstractRange)
    # T0 = typeof(g[1])
    # T = T0 <: DiscreteTime ? DiscreteTime : ContinuousTime
    # step = T(g[2]-g[1])
    # step = typeof(g)<:Union{StepRangeLen, StepRange} ? T(g.step) : T(1)
    # RegularGrid(T(g[1]), T(step), T(g[end]))
    RegularGrid(g[1], g[2]-g[1], g[end])
end

const DiscreteTimeRegularGrid = RegularGrid{DiscreteTime}
const ContinuousTimeRegularGrid = RegularGrid{ContinuousTime}

length(g::RegularGrid) = g.len
size(g::RegularGrid) = (g.len,)
# IndexStyle(::Type{<:RegularGrid}) = IndexLinear()

function getindex(g::RegularGrid{T}, i::Integer) where T
    @boundscheck ((i > 0) && (i<=g.len)) || throw(BoundsError(g, i))
    return convert(T, g.start + g.step * (i-1))
#     @boundscheck ((i > 0) & (ret <= g.stop) & (ret >= g.start)) || throw(BoundsError(g, i))
end

convert(::Type{RegularGrid}, g::AbstractRange) = RegularGrid(g[1], g[2]-g[1], g[end])

# function convert(::Type{RegularGrid}, g::AbstractRange)
#     T = DiscreteTime ? typeof(g[1])<:Integer : ContinuousTime
#     return convert(Type{RegularGrid{T}}, g)
# end


##### Multifractional related #####
"""
    val2grid(X::Vector{Float64}, δ0::Float64=2.5e-2)

Construct a regular grid from a vector of continuous values with a given step.
"""
function val2grid(X::Vector{Float64}, δ0::Float64=2.5e-2)
    xmin, xmax = minimum(X), maximum(X)
    δ0 > 0 || error("Step of grid must be > 0.")
    # (hmin>0 && hmax<1) || error("Husrt exponent must be bounded in (0,1).")
    δ = min(δ0, xmax-xmin)  # hstep smaller than 2.5e-2 is unstable
    grid::Vector{Float64} = (δ > 0) ? collect(xmin:δ:xmax) : [xmin]
    M, N = length(grid), length(X)
    idx::Vector{Integer} = (δ > 0) ? min.(M, max.(1, ceil.(Int, M*(X-xmin)/(xmax-xmin)))) : ones(Integer, N)
    return grid, idx
end
