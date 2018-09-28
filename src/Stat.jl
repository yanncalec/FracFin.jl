########## General statistics ##########

"""
    cov(X::AbstractVecOrMat, Y::AbstractVecOrMat, w::StatsBase.AbstractWeights)

Reweighted covariance between `X` and `Y`, where each row is an observation.
"""
function cov(X::AbstractVecOrMat, Y::AbstractVecOrMat, w::StatsBase.AbstractWeights)
    # w is always a column vector, by definition of AbstractWeights
    @assert size(X, 1) == size(Y, 1) == length(w)
    # weighted mean
    mX = mean(X, w, 1)
    mY = mean(Y, w, 1)
    return (X .- mX)' * (w .* (Y .- mY)) / sum(w)
end


######## Linear models ########
"""
Multi-linear regression in the column direction. 
"""
function multi_linear_regression_colwise(Y::AbstractVecOrMat{T}, X::AbstractVecOrMat{T}, w::StatsBase.AbstractWeights) where {T<:Real}
    @assert size(Y,1)==size(X,1)==length(w)

    μy = mean(Y, w, 1)[:]  # Julia function (version <= 0.7) `mean` does not take keyword argument `dims=1` if weight is passed.
    μx = mean(X, w, 1)[:]
    Σyx = cov(Y, X, w)  # this calls user defined cov function
    Σxx = cov(X, X, w)
    A = Σyx / Σxx  # scalar or matrix, i.e., Σyx * inv(Σxx), 
    β = μy - A * μx  # scalar or vector
    E = Y - (A * X' .+ β)'
    Σ = cov(E, E, w)
    return (A, β), E, Σ
end

"""
    linear_regression(Y::AbstractMatrix, X::AbstractMatrix, w::StatsBase.AbstractWeights; dims::Int=1)

Reweighted (multi-)linear regression of data matrix `Y` versus `X` in the given dimension (dimension of observation).

# Returns
- (A, β): explanatory slope (scalar or matrix) and intercept (scalar or vector)
- E: residual
- Σ: covariance of the residual
"""
function linear_regression(Y::AbstractMatrix, X::AbstractMatrix, w::StatsBase.AbstractWeights; dims::Int=1)    
    if dims==1
        return multi_linear_regression_colwise(Y, X, w)
    else
        return multi_linear_regression_colwise(Y', X', w)
    end
end

linear_regression(Y::AbstractMatrix, X::AbstractMatrix; dims::Int=1) =
    linear_regression(Y, X, StatsBase.weights(ones(size(Y,1))); dims=dims)

function linear_regression(Y::AbstractVector, X::AbstractVector, w::StatsBase.AbstractWeights)
    return multi_linear_regression_colwise(Y, X, w)
end

linear_regression(Y::AbstractVector, X::AbstractVector) = linear_regression(Y, X, StatsBase.weights(ones(size(Y,1))))


######## Nonlinear models ########
"""
IRLS
"""
function IRLS(Y::AbstractVecOrMat{T}, X::AbstractVecOrMat{T}, pnorm::Real=2.; maxiter::Int=10^3, tol::Float64=10^-3, vreg::Float64=1e-8) where {T<:Real}
    @assert pnorm > 0

    wfunc = E -> (sqrt.(sum(E.*E, dims=2) .+ vreg).^(pnorm-2))[:]  # function for computing weight vector
    
    (A, β), E, Σ = linear_regression(Y, X)  # initialization
    w0::Vector{Float64} =  wfunc(E) # weight vector
    n::Int = 1
    err::Float64 = 0.

    for n=1:maxiter
        (A, β), E, Σ = linear_regression(Y, X, StatsBase.weights(w0))
        w = wfunc(E)
        err = norm(w - w0) / norm(w0)
        w0 = w
        if  err < tol
            break
        end
    end
    # println(n)
    # println(err)

    return (A, β), w0, E, sum(sqrt.(sum(E.*E, dims=2)).^pnorm)
end
