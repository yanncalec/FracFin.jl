# __precompile__()

module FracFin

# using Distributions
# using PDMats
# using StatsBase

import Base: convert, rand!, rand, length, size, show
import Distributions: VariateForm, Univariate, Multivariate, ValueSupport, Discrete, Continuous, Sampleable
import StatsBase: autocov!, autocov

"""
    Exception for not implemented methods.
"""
struct NotImplementedError <: Exception
    errmsg::AbstractString
    # errpos::Int64
end
show(io::IO, exc::NotImplementedError) = print(io, string("NotImplementedError:\n",exc.errmsg))

export
    StochasticProcess,
    StationaryStochasticProcess,
    SamplingGrid,
    autocov,
    autocov!,
    partcorr,
    partcorr!,
    FractionalGaussianNoise,
    FractionalBrownianMotion,
    ARFIMA,
    CholeskySampler,
    LevinsonDurbin,
    LevinsonDurbinSampler,
    CircSampler,
    rand,
    rand!,
    rand_otf,
    rand_otf!

include("StochasticProcess.jl")
include("Sampler.jl")

end # module
