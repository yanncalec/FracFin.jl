const DiscreteTime = Integer
const ContinuousTime = Real

promote_rule(ContinuousTime, DiscreteTime) = ContinuousTime

"""
Type of time index of a stochastic process: discrete time (Integer) or continuous time (Real).
"""
const TimeStyle = Union{DiscreteTime, ContinuousTime}

"""
Abstract sampling grid.
"""
# abstract type SamplingGrid{T} <: AbstractArray{T<:TimeStyle,1} end
# const SamplingGrid{T<:TimeStyle} = AbstractArray{T,1}
# const SamplingGrid{T} = AbstractArray{T<:TimeStyle,1}
const SamplingGrid{T} = AbstractArray{T,1}

const DiscreteTimeSamplingGrid = SamplingGrid{DiscreteTime}
const ContinuousTimeSamplingGrid = SamplingGrid{ContinuousTime}

"""
Test if a sampling grid is valid.
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

    function RegularGrid{T}(start::T, step::T, stop::T) where T
        @assert step > 0
        @assert start <= stop
        len = Integer(div(stop + step - start, step))
        new(start, step, stop, len)
    end
end

function RegularGrid(start, step, stop)
    T = promote_type(TimeStyle, typeof(start), typeof(step), typeof(stop))
    RegularGrid{T}(start, step, stop)
end

function RegularGrid(g::Range)
    T = typeof(g[1])
    step = T(g[2]-g[1])
    # step = typeof(g)<:Union{StepRangeLen, StepRange} ? T(g.step) : T(1)
    # RegularGrid(T(g[1]), T(step), T(g[end]))
    RegularGrid(g[1], step, g[end])
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

convert(::Type{RegularGrid{T}}, g::Range) where T = RegularGrid{T}(g[1], T(g[2]-g[1]), g[end])

function convert(::Type{RegularGrid}, g::Range)
    T = Integer ? typeof(g[1])<:Integer : Real
    return convert(Type{RegularGrid{T}}, g)
end
