__precompile__()

module FracFin

using LinearAlgebra
using Statistics
using StatsBase
using Formatting

import Base: rand, length, size, show, promote_rule, binomial
import StatsBase: autocov!, autocov
import Statistics: mean, cov

import SpecialFunctions: gamma, lgamma

import DSP
import DSP: conv, fft, ifft

import Wavelets

import DataFrames
import GLM
import QuadGK
import Optim

import Dates
import Dates:AbstractTime, AbstractDateTime, TimePeriod

import TimeSeries
import TimeSeries: TimeArray

import RCall

import PyCall
# @PyCall.pyimport pywt
# @PyCall.pyimport pywt.swt as pywt_swt
# @PyCall.pyimport pywt.iswt as pywt_iswt
const pywt = PyCall.PyNULL()

# function __init__()
#     copy!(pywt, PyCall.pyimport("pywt"))
# end

# Metaprogramming for class of exceptions
for ErrType in [:NotImplementedError, :ValueError]
    @eval begin
        struct $ErrType <: Exception
            errmsg::AbstractString
                
            $ErrType(msg::AbstractString="") = new(msg)
        end
        show(io::IO, exc::$ErrType) = print(io, string("$ErrType: ",exc.errmsg))        
    end
end

# # which is equivalent to the following:
#
# """
#     Exception for not implemented methods.
# """
# struct NotImplementedError <: Exception
#     errmsg::AbstractString
#     # errpos::Int64
#     NotImplementedError() = new("")
#     NotImplementedError(msg::AbstractString) = new(msg)
# end
#
# show(io::IO, exc::NotImplementedError) = print(io, string("NotImplementedError: ",exc.errmsg))


# export
#     StochasticProcess,
#     covmat,


include("Common.jl")
include("StochasticProcess.jl")
include("Sampler.jl")
include("Tool.jl")
include("CHA.jl")
include("Stat.jl")
include("Estimator.jl")
include("Trading.jl")

end # module
