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


"""
    moment_incr(X::AbstractVector{<:Number}, d::Integer, p::Real, w=StatsBase.AbstractWeights[])

Comput the `p`-th moment for the increment vector of `X` with lag `d`.
"""
function moment_incr(X::AbstractVector{<:Number}, d::Integer, p::Real, w=StatsBase.AbstractWeights[])
    return if length(w) > 0
        mean((abs.(X[d+1:end] - X[1:end-d])).^p, w)
    else
        mean((abs.(X[d+1:end] - X[1:end-d])).^p)
    end
end

######## Linear models ########
"""
Multi-linear regression in the column direction.
"""
function multi_linear_regression_colwise(Y::AbstractVecOrMat{<:Real}, X::AbstractVecOrMat{<:Real}, w::AbstractWeights)
    @assert size(Y,1)==size(X,1)==length(w)
    # println("size(Y)=",size(Y))
    # println("size(X)=",size(X))
    μy = mean(Y, w, 1)  # Julia function (version <= 0.7) `mean` does not take keyword argument `dims=1` if weight is passed. This yields a scalar of Y is vector.
    μx = mean(X, w, 1)
    Σyx = cov(Y, X, w)   # This calls user defined cov function. This yields scalar if Y and X are vectors.
    Σxx = cov(X, X, w)

    # Σxx += 1e-5 * Matrix{Float64}(I,size(Σxx))  # perturbation
    # A = Σyx / Σxx  # scalar or matrix, i.e., Σyx * inv(Σxx)

    A = Σyx * pinv(Σxx)  # scalar or matrix, i.e., Σyx * inv(Σxx)

    # A = try
    #     Σyx * pinv(Σxx)  # scalar or matrix, i.e., Σyx * inv(Σxx)
    # catch
    #     zero(Σyx)
    # end

    β = μy - A * μx'  # scalar or vector, transpose has no effect on scalar
    E = Y - (A * X' .+ β)'
    Σ = cov(E, E, w)
    return A, β, E, Σ
end

"""
    linear_regression(Y::AbstractMatrix, X::AbstractMatrix, w::AbstractWeights; dims::Integer=1)

Reweighted (multi-)linear regression of data matrix `Y` versus `X` in the given dimension (dimension of observation).

# Returns
- (A, β): explanatory slope (scalar or matrix) and intercept (scalar or vector)
- E: residual
- Σ: covariance of the residual
"""
function linear_regression(Y::AbstractMatrix, X::AbstractMatrix, w::AbstractWeights; dims::Integer=1)
    if dims==1
        return multi_linear_regression_colwise(Y, X, w)
    else
        return multi_linear_regression_colwise(Y', X', w)
    end
end

linear_regression(Y::AbstractMatrix, X::AbstractMatrix; dims::Integer=1) =
    linear_regression(Y, X, StatsBase.weights(ones(size(Y,dims))); dims=dims)

function linear_regression(Y::AbstractVector, X::AbstractVecOrMat, w::AbstractWeights)
    return multi_linear_regression_colwise(Y, X, w)
end

linear_regression(Y::AbstractVector, X::AbstractVecOrMat) = linear_regression(Y, X, StatsBase.weights(ones(size(Y,1))))


# function linear_prediction(R::Tuple, X0::AbstractVecOrMat, Y0::AbstractVecOrMat=[])
#     A,β = R[1]
#     Yp = A * X0 .+ β
#     Ep = isempty(Y0) ? [] : Y0-Yp  # error of prediction
#     return Yp, Y0, Ep
# end


"""
    conv_regression(X::AbstractVector{<:Real}, Y::AbstractVector{<:Real}, w::Integer; mode::Symbol=:causal, nan::Symbol=:ignore)

Estimate by linear regression the convolution kernel and bias such that `kernel * X + bias = Y`.

# Args
- X: input vector
- Y: output vector
- w: length of the convolution kernel
- mode: mode of convolution, :causal or :anticausal
- nan: how NaN values are handled: :ignore for ignoring or :zero for replacing by zero

# Returns
A, b, E, σ^2: kernel, bias, error and variance of error
"""
function conv_regression(X::AbstractVector{<:Real}, Y::AbstractVector{<:Real}, w::Integer, p::Integer=1; mode::Symbol=:causal, nan::Symbol=:ignore)
    @assert length(X) == length(Y)

    tidx, Xm0 = rolling_vectorize(X, w, 1, p; mode=mode)
    Xm = transpose(Xm0)
    Yr = Y[tidx]

    if nan==:ignore
        idx = findall((any(isnan.(Xm), dims=2)[:] .+ isnan.(Yr)).==0)
        # println((any(isnan.(Xm), dims=2)[:] .+ isnan.(Yr)).==0)
        # println(any(isnan.(Xm), dims=2)[:])
        # println(isnan.(Yr))
        # print(idx)
        Yr = Yr[idx]
        Xm = Xm[idx,:]
    elseif nan==:zero
        Yr[isnan.(Yr)] .= 0
        Xm[isnan.(Xm)] .= 0
    else
        error("Unknown method for NaN values")
    end

    res = linear_regression(Yr, Xm)
    # res = IRLS(Yr, Xm, .5)

    A, b, E, σ2 = reverse(res[1][:]), res[2][end], res[3], sqrt(res[4][end])
    return A, b, E, σ2
end


######## Nonlinear models ########
"""
IRLS
"""
function IRLS(Y::AbstractVecOrMat{<:Real}, X::AbstractVecOrMat{<:Real}, pnorm::Real=2.; maxiter::Integer=10^3, tol::Real=10^-3, vreg::Real=1e-8)
    @assert pnorm > 0

    wfunc = E -> (sqrt.(sum(E.*E, dims=2) .+ vreg).^(pnorm-2))[:]  # function for computing weight vector

    A, β, E, Σ = linear_regression(Y, X)  # initialization
    w0 =  wfunc(E) # weight vector
    # w0 /= sum(w0)
    err = 0.

    for n=1:maxiter
        A, β, E, Σ = linear_regression(Y, X, StatsBase.weights(w0))
        w = wfunc(E)
        err = LinearAlgebra.norm(w - w0)/LinearAlgebra.norm(w0)
        w0 = w # / sum(w)
        if  err < tol
            break
        end
    end
    # println(n)
    # println(err)
    Σ = sum(sqrt.(sum(E.*E, dims=2)).^pnorm)
    return A, β, E, Σ, w0
end


###### Online Statistics ######

# function exponential_moving_average!(X::AbstractVector{<:Number}, α::Real)
#     # @assert 0<α<=1
#     for t=2:size(X,1)
#         X[t] = α * X[t] + (1-α) * (isnan(X[t-1]) ? 0. : X[t-1])  # NaN safe
#     end
#     return X
# end

function exponential_moving_average!(X::AbstractVecOrMat{<:Number}, α::Real)
    @assert 0<α<=1
    X[findall(isnan.(X))] .= 0.  # NaN safe

    for t=2:size(X,1)  # first axis is the time
        W = view(X,t,:)
        W .= α * W + (1-α) * view(X,t-1,:)
    end
    return X
end

"""
EMA of a vector or a matrix (in row direction).
"""
function exponential_moving_average(X::AbstractVecOrMat{<:Number}, α::Real, n::Integer=1)
    @assert 0<α<=1
    X1 = copy(X)
    Xm = copy(X1)
    exponential_moving_average!(Xm, α)

    for t=2:n
        X1 += X-Xm  # N-fold EMA
        # or the zero-lag EMA filter: X1 += X-circshift(X, 1)
        Xm = copy(X1)
        exponential_moving_average!(Xm, α)
    end
    return Xm
end


# function simple_moving_average!(X::AbstractVector{<:Number}, n::Integer)
#     @assert n>0
#     for t=2:size(X,1)
#         X[t] = (X[t] + X[t-1] * min(n, t-1) - (t>n ? X[t-n] : 0)) / min(n, t)
#     end
#     return X
# end

function simple_moving_average(X0::AbstractVecOrMat{<:Number}, n::Integer)
    @assert n>0

    X1 = copy(X0)  # X0 must be saved in the causal implementation
    X1[findall(isnan.(X1))] .= 0.  # NaN safe
    X = copy(X1)

    for t=2:n
        X[t,:] .= (X[t,:] + X[t-1,:] * (t-1)) / t
    end
    for t=n+1:size(X,1)
        X[t,:] .= (X[t,:] + X[t-1,:] * n - X1[t-n,:]) / n
    end
    for t=2:size(X,1)
        W = view(X,t,:)
        W .= (W + view(X,t-1,:) * min(n, t-1) .- (t>n ? view(X1,t-n,:) : 0)) / min(n, t)
        # # equivalent to
        # X[t,:] .= (X[t,:] + X[t-1,:] * min(n, t-1) .- (t>n ? X[t-n,:] : 0)) / min(n, t)
    end
    return X
end
