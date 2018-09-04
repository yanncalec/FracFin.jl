########## Common definitions ##########

# const DiscreteTime = Union{Integer, Int64}
# const ContinuousTime = Union{Real, Float64}
const DiscreteTime = Integer
const ContinuousTime = Real
promote_rule(ContinuousTime, DiscreteTime) = ContinuousTime

"""
Type of time index of a stochastic process: discrete time (Integer) or continuous time (Real).
"""
const TimeStyle = Union{DiscreteTime, ContinuousTime}
