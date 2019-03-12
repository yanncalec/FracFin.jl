######### Estimators for fBm and related processes #########

# abstract type AbstractEstimator end
# abstract type AbstractRollingEstimator <: AbstractEstimator end
# abstract type AbstractfBmEstimator <: AbstractEstimator end

include("Estimator_Variogram.jl")
include("Estimator_Scalogram.jl")
include("Estimator_MLE.jl")