###### MLE ######

"""
    xiAx(A, X; ρ=0)

Safe evaluation of the inverse quadratic form
    trace(X' * inv(A) * X)
where the matrix `A` is symmetric and positive definite. Small eigen values of `A` are truncated.
"""
function xiAx(A::AbstractMatrix{<:Real}, X::AbstractVecOrMat{<:Real}; ρ::Real=0)
    # # Sanity check
    # @assert issymmetric(A)
    # @assert size(X, 1) == size(A, 1)

    # # Direct inversion
    # return tr(X' * pinv(A + 1e-3*I) * X)

    # Eigen decomposition, S are the eigen values in increasing order
    S, U = eigen(A)  # so that U * Diagonal(S) * inv(U) == A, in particular, U' == inv(U)
    # On covariance matrix this is equivalent to
    # U, S, V = svd(A)

    # shrinkage of small eigen values for stability
    # idx = findall(abs.(S./maximum(S)) .>= ρ)   # naive way
    Rs = cumsum(reverse(abs.(S)))/sum(abs.(S))
    p = findfirst(Rs .>= 1-ρ)
    idx = isnothing(p) ? (1:length(S)) : max(1,length(S)-p):length(S)

    return sum((U[:,idx]'*X).^2 ./ S[idx])  # this may raise error if idx is empty
end


"""
    log_likelihood(A, X)

Log-likelihood of a general Gaussian vector.

# Args
- A: covariance matrix
- X: sample vector or matrix. For matrix each column is a sample.
"""
function log_likelihood(A::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}; kwargs...)
    # # Sanity check
    # @assert issymmetric(A)
    # @assert size(X, 1) == size(A, 1)

    N = size(X,2) # number of i.i.d. samples in data
    return -1/2 * (xiAx(A,X, kwargs...) + N*logdet(A) + length(X)*log(2π))
end

log_likelihood(A::AbstractMatrix, X::AbstractVector) = log_likelihood(A, reshape(X,:,1))


"""
    log_likelihood_H(A, X)

Safe evaluation of the log-likelihood of a fBm model with the implicite σ (optimal in the MLE sense).

# Args
- A: covariance matrix
- X: sample vector or matrix. For matrix each column is a sample.

# Notes
- This function is common to all MLEs with the covariance matrix of form `σ²A(h)`, where `{σ, h}` are unknown parameters. This kind of MLE can be carried out in `h` uniquely and `σ` is obtained from `h`.
- `log_likelihood_H(Σ, X)` and `log_likelihood(σ^2*Σ, X)` are equivalent with the optimal `σ = sqrt(xiAx(Σ, X) / length(X))`
"""
function log_likelihood_H(A::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}; ρ::Real=0)
    # Sanity check
    # @assert issymmetric(A)
    # @assert size(X, 1) == size(A, 1)

    N = size(X,2) # number of i.i.d. samples in data

    # Compute the log-likelihood: log_likelihood(A,X)
    S, U = eigen(A)  # so that U * Diagonal(S) * inv(U) == A, in particular, U' == inv(U)

    # shrinkage of small eigen values for stability
    # idx = findall(abs.(S)/maximum(abs.(S)) .>= ρ)   # naive way
    Rs = cumsum(reverse(abs.(S)))/sum(abs.(S))
    p = findfirst(Rs .>= 1-ρ)
    # idx = isnothing(p) ? (1:length(S)) : max(length(S)÷2,length(S)-p):length(S)
    idx = isnothing(p) ? (1:length(S)) : max(1,length(S)-p):length(S)

    # println(p)
    # println(size(A))

    val = -1/2 * (length(X)*log(sum((U[:,idx]'*X).^2 ./ S[idx])) + N*sum(log.(S[idx])))  # non-constant part of log-likelihood

    return val - length(X) * log(2π*exp(1)/length(X))/2  # with the constant part
end

log_likelihood_H(A::AbstractMatrix, X::AbstractVector, args...) = log_likelihood_H(A, reshape(X,:,1), args...)


#### fWn (bank)-MLE ####

"""
    fWn_log_likelihood_H(X, H, F, G)

Log-likelihood of a fWn bank model with the optimal volatility.
"""
function fWn_log_likelihood_H(X::AbstractVecOrMat{<:Real}, H::Real, F::AbstractVector{<:AbstractVector{<:Real}}, G::AbstractVector{<:Integer}; mode::Symbol=:causal, kwargs...)
    # # Sanity check
    # @assert size(X,1) == length(G) * length(F)  "Mismatched dimensions."

    # the process and the number of filters in the filter bank
    proc = FractionalWaveletNoiseBank(H, F, mode)  # mode of wavelet transform
    return log_likelihood_H(covmat(proc, G), X; kwargs...)
end


"""
    fWn_MLE_estim(X, F, G; method)

Maximum likelihood estimation of Hurst exponent and volatility for fWn bank.

# Args
- X: sample matrix.
- F: array of filters used for computing `X`
- G: integer time grid of `X`.
- method: :optim for optimization based or :table for look-up table based solution.

# Returns
- hurst, σ: estimation
- loglikelihood: log-likelihood of estimation
- optimizer: object of optimizer, for method==:optim only

# Notes
- The fWnb process is multivariate. An observation at time `t` is a vector that the dimension equals to the number of filters used in fWnb. A sample vector is the concatenation of observations made on some time grid `G` and the row dimension of `X` must be `length(F) * length(G)`. Columns of `X` are treated as i.i.d. sample vectors hence the horizontal axis (i.e. last dimension) should NOT be interpreted as time axis (even it is physically the case, e.g. wavelet coefficients of a time series). In fact the time axis is "in" the vertical axis (i.e. the first dimension) that the grid `G` must be conformal with.
"""
function fWn_MLE_estim(X::AbstractMatrix{<:Real}, F::AbstractVector{<:AbstractVector{<:Real}}, G::AbstractVector{<:Integer}; method::Symbol=:optim, mode::Symbol=:causal, kwargs...)
    if length(X) == 0 || any(isnan.(X))  # for empty input or input containing nans
        return (hurst=NaN, σ=NaN, loglikelihood=NaN, optimizer=nothing)
    else
        # Sanity check
        @assert size(X,1) == length(F) * length(G)  "Mismatched dimensions."
        @assert length(G) == length(unique(G))  "All elements of the grid must be distinct."

        func = h -> -fWn_log_likelihood_H(X, h, F, G; kwargs...)

        opm = nothing
        hurst = nothing
        ε::Real=1e-2  # search Hurst in the range [ɛ, 1-ɛ]

        if method == :optim
            # Gradient-free constrained optimization
            opm = Optim.optimize(func, ɛ, 1-ɛ, Optim.Brent())
            # # Gradient-based optimization
            # optimizer = Optim.GradientDescent()  # e.g. Optim.BFGS(), Optim.GradientDescent()
            # opm = Optim.optimize(func, ε, 1-ε, [0.5], Optim.Fminbox(optimizer))
            hurst = Optim.minimizer(opm)[1]
        elseif method == :table
            Hs = collect(ε:ε:1-ε)
            hurst = Hs[argmin([func(h) for h in Hs])]
        else
            throw("Unknown method: ", method)
        end

        proc = FractionalWaveletNoiseBank(hurst, F, mode)

        Σ = covmat(proc, G)

        # Estimation of volatility
        σ = sqrt(xiAx(Σ, X) / length(X))

        # Log-likelihood
        L = log_likelihood_H(Σ, X)
        # # or equivalently
        # L = log_likelihood(σ^2*Σ, X)
        L /= length(X)  # normalization

        return (hurst=hurst, σ=σ, loglikelihood=L, optimizer=opm)
    end
end


"""
Accelerated MLE by dividing a large vector of samples into smaller ones.

# Args
- X, F: wavelet coefficient matrix and filters
- s: sub window size
- u: downsampling factor
- l: length of decorrelation
"""
function fWn_MLE_estim(X::AbstractMatrix{<:Real}, F::AbstractVector{<:AbstractVector{<:Real}}, s::Integer, u::Integer, l::Integer; kwargs...)
    t, V = rolling_vectorize(X, s, u, l; mode=:causal)  # TODO: case s = size(X,2)
    fWn_MLE_estim(V, F, u*(1:s); kwargs...)  # regular grid is implicitely used here.
end


#### B-Spline MLE ####

"""
Filter of a fWn.

# Note
- The extra 1/sqrt(s) factor is due to the definition of B-Spline wavelet filter and DCWT.
"""
bspline_fWn_filter = (s,v) -> intscale_bspline_filter(s,v) / sqrt(s)


"""
    bspline_MLE_estim(X::AbstractMatrix{<:Real}, sclrng::AbstractVector{<:Integer}, vm::Integer, G::AbstractVector{<:Integer}; kwargs...)

Maximum likelihood estimation of Hurst exponent and volatility for fractional Wavelet noise bank.

# Args
- X: matrix of B-Spline wavelet coefficients. Each row corresponds to a scale.
- sclrng: scales of the rows in `X`.
- vm: vanishing moment of the B-Spline wavelet.
- G: integer grid correpsonding to the columns of `X`.

# Notes
- In rolling window estimation, to avoid recomputing the B-Spline filters one can use directly `fWn_MLE_estim`.
"""
function bspline_MLE_estim(X::AbstractMatrix{<:Real}, sclrng::AbstractVector{<:Integer}, vm::Integer, G::AbstractVector{<:Integer}; kwargs...)
    @assert size(X,1) == length(sclrng) * length(G)  "Mismatched dimensions."

    F = [bspline_fWn_filter(s, vm) for s in sclrng]
    fWn_MLE_estim(X, F, G; kwargs...)
end


"""
    bspline_MLE_estim(X::AbstractMatrix{<:Real}, sclrng::AbstractVector{<:Integer}, vm::Integer, s::Integer, u::Integer, l::Integer; kwargs...)

Cross-scale and cross-time B-Spline MLE, by concatenating adjacent columns into a larger vector and using both the cross-scale and the cross-time correlations.
"""
function bspline_MLE_estim(X::AbstractMatrix{<:Real}, sclrng::AbstractVector{<:Integer}, vm::Integer, s::Integer, u::Integer, l::Integer; kwargs...)
    t, V = rolling_vectorize(X, s, u, l; mode=:causal)
    bspline_MLE_estim(V, sclrng, vm, u*(1:s); kwargs...)
end


"""
    bspline_MLE_estim(X::AbstractMatrix{<:Real}, sclrng::AbstractVector{<:Integer}, vm::Integer; kwargs...)

Cross-scale only B-Spline MLE, by treating the columns of `X` as independent observations and using only the cross-scale correleation.
"""
function bspline_MLE_estim(X::AbstractMatrix{<:Real}, sclrng::AbstractVector{<:Integer}, vm::Integer; kwargs...)
    bspline_MLE_estim(X, sclrng, vm, 1:1; kwargs...)
end


"""
    bspline_MLE_estim(X::AbstractVector{<:Real}, scl::Integer, vm::Integer, args...;  kwargs...)

Cross-time only B-Spline MLE (with acceleration).
"""
function bspline_MLE_estim(X::AbstractVector{<:Real}, scl::Integer, vm::Integer, args...; kwargs...)
    # Call two different functions depending on whether the arguments `s, u, l` are passed or not in `args...`.
    if isempty(args)
        # reshape to a column vector with a regular grid
        bspline_MLE_estim(reshape(X,:,1), [scl], vm, 1:length(X); kwargs...)
    else
        bspline_MLE_estim(reshape(X,1,:), [scl], vm, args...; kwargs...)
    end
end


#### fGn-MLE ####

"""
Filter of a fGn.
"""
fGn_filter = d -> vcat(1, zeros(d-1), -1)


"""
Maximum likelihood estimation of Hurst exponent and volatility for fractional Gaussian noise bank.

# Args
- X: sample matrix.
- lags: time lags of the finite difference operator used for computing `X`.
- G: integer time grid of `X`.
"""
function fGn_MLE_estim(X::AbstractMatrix{<:Real}, lags::AbstractVector{<:Integer}, G::AbstractVector{<:Integer}; kwargs...)
    @assert size(X,1) == length(lags) * length(G)  "Mismatched dimensions."

    F = [fGn_filter(l) for l in lags]
    fWn_MLE_estim(X, F, G; kwargs...)
end


"""
Cross-scale and cross-time fGn MLE.
"""
function fGn_MLE_estim(X::AbstractMatrix{<:Real}, lags::AbstractVector{<:Integer}, s::Integer, u::Integer, l::Integer; kwargs...)
    t, V = rolling_vectorize(X, s, u, l; mode=:causal)
    fGn_MLE_estim(V, lags, u*(1:s); kwargs...)
end


"""
Cross-scale only fGn MLE.
"""

function fGn_MLE_estim(X::AbstractMatrix{<:Real}, lags::AbstractVector{<:Integer}; kwargs...)
    fGn_MLE_estim(X, lags, 1:1; kwargs...)
end


"""
Cross-time only fGn MLE (with acceleration).
"""
function fGn_MLE_estim(X::AbstractVector{<:Real}, lag::Integer, args...; kwargs...)
    # Call two different functions depending on whether the arguments `s, u, l` are passed or not in `args...`.
    if isempty(args)
        # reshape to a column vector with a regular grid
        fGn_MLE_estim(reshape(X,:,1), [lag], 1:length(X); kwargs...)
    else
        fGn_MLE_estim(reshape(X,1,:), [lag], args...; kwargs...)
    end
end
