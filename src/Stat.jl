########## General statistics ##########

"""
Compute the symmetric polynomial weight vector.

# Args
- N: length of time window
- p: order of polynomial, p=0 gives uniform weight
"""
function poly_weight(N::Integer, p::Integer)
    # println("poly_weight $(N)")
    w = ones(N)
    for n=0:p-1
        w = cumsum(w)
    end
    if N>=2
        w[(N÷2):end] = reverse(w[1:N-(N÷2)+1])
    end
    return w/sum(w)
end

"""
Causal polynomial weight for a time window, i.e. the weight is polynomial increasing from the left end to the right end of a time window.

# Note
`p=0` gives uniform weight, `p=1` gives linear weight etc.
"""
causal_weight(N, p) = p == 0 ? poly_weight(N,p) : cumsum(poly_weight(N,p-1))


"""
Auto-Correlation function by RCall.
"""
function acf(X::AbstractVector{T}, lagmax::Int) where {T<:Real}
    res = RCall.rcopy(RCall.rcall(:acf, X, lagmax, plot=false, na_action=:na_pass))
    return res[:acf][2:end]
end

"""
Partial Auto-Correlation function by RCall.
"""
function pacf(X::AbstractVector{T}, lagmax::Int) where {T<:Real}
    res = RCall.rcopy(RCall.rcall(:pacf, X, lagmax, plot=false, na_action=:na_pass))
    return res[:acf][:]
end


"""
Auto-Correlation function of increment process.
"""
function acf_incr(X::AbstractVector{T}, dlags::Union{Int, AbstractVector{Int}}, lagmax::Int; method::Symbol=:acf) where {T<:Real}
    # for single value of dlag: convert to a list
    if typeof(dlags) <: Integer
        dlags = [dlags]
    end

    A = if method==:acf
        [acf((X[l+1:end]-X[1:end-l])[100:end-100], lagmax) for l in dlags]
    elseif method==:pacf
        [pacf((X[l+1:end]-X[1:end-l])[100:end-100], lagmax) for l in dlags]
    else
        error("Invalid method: $(method)")
    end

    # A = []  # output of acf
    # for l in dlags
    #     dX = X[l+1:end]-X[1:end-l]
    #     if method==:acf
    #         push!(A, acf(dX, lagmax))
    #         # push!(A, [cor(dX[t+1:end], dX[1:end-t]) for t in tidx])
    #     elseif method==:pacf
    #         push!(A, pacf(dX, lagmax))
    #     else
    #         error("Invalid method: $(method)")
    #     end
    # end
    return hcat(A...)
end


"""
    cov(X::AbstractVecOrMat, Y::AbstractVecOrMat, w::AbstractWeights)

Reweighted covariance between `X` and `Y`, where each row is an observation.
"""
function cov(X::AbstractVecOrMat, Y::AbstractVecOrMat, w::AbstractWeights)
    # w is always a column vector, by definition of AbstractWeights
    @assert size(X, 1) == size(Y, 1) == length(w)
    # weighted mean
    mX = mean(X, w, 1)
    mY = mean(Y, w, 1)
    return (X .- mX)' * (w .* (Y .- mY)) / sum(w)
end


"""
Robust estimator of variance
"""
function robustvar(X::AbstractArray{T}; dims::Int=1) where {T<:Number}
    lqt = -0.393798799600891  # log of 3/4 quantil of standard normal distribution
    Q = log.((X .- mean(X, dims=dims)).^2)
    return exp.(median(Q, dims=dims) .- 2lqt)
end


"""Principal Component Analysis.

# Args
-X (2d array): each row represents a variable and each column represents an observation
-nc (int): number of components to hold

# Returns
- C, U : coefficients and corresponded principal directions
"""
function pca(X::AbstractMatrix, nc::Int, w::AbstractWeights, center=false)
    U, S, V = svd(cov(X, X, w))
    mX = mean(X, w, 1)
    C = center ? (X .- mX) * U' : X * U'

    return C[:,1:nc], U[:,1:nc], S[1:nc]
end

pca(X::AbstractMatrix, nc::Int, center=false) = pca(X, nc, StatsBase.weights(ones(size(X,1))), center)


######## Linear models ########
"""
Multi-linear regression in the column direction.
"""
function multi_linear_regression_colwise(Y::AbstractVecOrMat{T}, X::AbstractVecOrMat{T}, w::AbstractWeights) where {T<:Real}
    @assert size(Y,1)==size(X,1)==length(w)
    # println("size(Y)=",size(Y))
    # println("size(X)=",size(X))
    μy = mean(Y, w, 1)[:]  # Julia function (version <= 0.7) `mean` does not take keyword argument `dims=1` if weight is passed.
    μx = mean(X, w, 1)[:]
    Σyx = cov(Y, X, w)   # this calls user defined cov function
    Σxx = cov(X, X, w)

    # Σxx += 1e-5 * Matrix{Float64}(I,size(Σxx))  # perturbation
    # A = Σyx / Σxx  # scalar or matrix, i.e., Σyx * inv(Σxx)

    A = Σyx * pinv(Σxx)  # scalar or matrix, i.e., Σyx * inv(Σxx)

    # A = try
    #     Σyx * pinv(Σxx)  # scalar or matrix, i.e., Σyx * inv(Σxx)
    # catch
    #     zero(Σyx)
    # end

    β = μy - A * μx  # scalar or vector
    E = Y - (A * X' .+ β)'
    Σ = cov(E, E, w)
    return (A, β), E, Σ
end

"""
    linear_regression(Y::AbstractMatrix, X::AbstractMatrix, w::AbstractWeights; dims::Int=1)

Reweighted (multi-)linear regression of data matrix `Y` versus `X` in the given dimension (dimension of observation).

# Returns
- (A, β): explanatory slope (scalar or matrix) and intercept (scalar or vector)
- E: residual
- Σ: covariance of the residual
"""
function linear_regression(Y::AbstractMatrix, X::AbstractMatrix, w::AbstractWeights; dims::Int=1)
    if dims==1
        return multi_linear_regression_colwise(Y, X, w)
    else
        return multi_linear_regression_colwise(Y', X', w)
    end
end

linear_regression(Y::AbstractMatrix, X::AbstractMatrix; dims::Int=1) =
    linear_regression(Y, X, StatsBase.weights(ones(size(Y,1))); dims=dims)

function linear_regression(Y::AbstractVector, X::AbstractVector, w::AbstractWeights)
    return multi_linear_regression_colwise(Y, X, w)
end

linear_regression(Y::AbstractVector, X::AbstractVector) = linear_regression(Y, X, StatsBase.weights(ones(size(Y,1))))


function linear_prediction(R::Tuple, X0::AbstractVecOrMat, Y0::AbstractVecOrMat=[])
    A,β = R[1]
    Yp = A * X0 .+ β
    Ep = isempty(Y0) ? [] : Y0-Yp  # error of prediction
    return Yp, Y0, Ep
end


######## Nonlinear models ########
"""
IRLS
"""
function IRLS(Y::AbstractVecOrMat{T}, X::AbstractVecOrMat{T}, pnorm::Real=2.; maxiter::Int=10^3, tol::Float64=10^-3, vreg::Float64=1e-8) where {T<:Real}
    @assert pnorm > 0

    wfunc = E -> (sqrt.(sum(E.*E, dims=2) .+ vreg).^(pnorm-2))[:]  # function for computing weight vector

    (A, β), E, Σ = linear_regression(Y, X)  # initialization
    w0::Vector{Float64} =  wfunc(E) # weight vector
    w0 /= sum(w0)
    n::Int = 1
    err::Float64 = 0.

    for n=1:maxiter
        (A, β), E, Σ = linear_regression(Y, X, StatsBase.weights(w0))
        w = wfunc(E)
        err = norm(w - w0) / norm(w0)
        w0 = w / sum(w)
        if  err < tol
            break
        end
    end
    # println(n)
    # println(err)

    return (A, β), w0, E, sum(sqrt.(sum(E.*E, dims=2)).^pnorm)
end

