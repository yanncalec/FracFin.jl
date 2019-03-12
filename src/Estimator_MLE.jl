###### MLE ######

"""
Safe evaluation of the inverse quadratic form
    trace(X' * inv(A) * X)
where the matrix `A` is symmetric and positive definite.
"""
function xiAx(A::AbstractMatrix{<:Real}, X::AbstractVecOrMat{<:Real}, ε::Real=0)
    # Sanity check
    @assert issymmetric(A)
    @assert size(X, 1) == size(A, 1)

    # # a simple version would be:
    # return tr(X' * pinv(A) * X)

    # SVD is equivalent to eigen decomposition on covariance matrix
    # U, S, V = svd(A)
    S, U = eigen(A)  # so that U * Diagonal(S) * inv(U) == A, in particular, U' == inv(U)
    idx = (S .> ε)  # shrinkage of small eigen values for stability

    if length(idx) > 0
        return sum((U[:,idx]'*X).^2 ./ S[idx])
    else
        error("Invalide covariance matrix.")
    end
end


"""
    log_likelihood(A, X)

Log-likelihood of a general Gaussian vector.

The value of log-likelihood (up to some additive constant) is
    -1/2 * (N*log(X'*inv(A)*X) + logdet(A))

# Args
- A: covariance matrix
- X: sample vector or matrix. For matrix each column is a sample.
"""
function log_likelihood(A::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real})
    @assert issymmetric(A)
    @assert size(X, 1) == size(A, 1)

    N = size(X,2) # number of i.i.d. samples in data
    return -1/2 * (N*logdet(A) + xiAx(A,X) + length(X)*log(2π))
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
function log_likelihood_H(A::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}, ε::Real=0)
    # Sanity check
    @assert issymmetric(A)
    @assert size(X, 1) == size(A, 1)

    N = size(X,2) # number of i.i.d. samples in data

    # U, S, V = svd(A)
    S, U = eigen(A)  # so that U * Diagonal(S) * inv(U) == A, in particular, U' == inv(U)
    idx = (S .> ε)

    val = -1/2 * (length(X)*log(sum((U[:,idx]'*X).^2 ./ S[idx])) + N*sum(log.(S[idx])))  # non-constant part of log-likelihood

    return val - length(X)*log(2π*exp(1)/length(X))/2  # with the constant part
end

log_likelihood_H(A::AbstractMatrix, X::AbstractVector, args...) = log_likelihood_H(A, reshape(X,:,1), args...)


#### fWn-MLE ####

"""
    fWn_log_likelihood_H(X, H, ψ, G)

Log-likelihood of a fWn model with the optimal volatility.
"""
function fWn_log_likelihood_H(X::AbstractVecOrMat{<:Real}, H::Real, ψ::AbstractVector{<:Real}, G::AbstractVector{<:Integer})
    @assert length(G) == size(X,1)
    proc = FractionalWaveletNoise(H, ψ)
    return log_likelihood_H(covmat(proc, G), X)

    # Σ::AbstractMatrix = if length(G)>0
    #     @assert length(G) == size(X,1)
    #     covmat(proc, G)
    # else
    #     covmat(proc, size(X,1))
    # end
    # return log_likelihood_H(Σ, X)
end


"""
Maximum likelihood estimation of Hurst exponent and volatility for fractional Wavelet noise.

# Args
- X: sample vector or matrix. For matrix each column is a sample.
- ψ: wavelet filter used for computing `X`.
- G: integer time grid of `X`, by default the regular grid `0:size(X,1)-1` is used.
- method: `:optim` for optimization based or `:table` for lookup table based procedure

# Returns
- H, σ, L, obj: estimation of Hurst exponent, of volatility, log-likelihood, object of optimizer

# Notes
- The MLE is known for its sensitivity to mis-specification of model, as well as to missing value (NaN) and outliers.
- The starting point of the grid `G` has no importance since fWn is stationary.
"""
function fWn_MLE_estim(X::AbstractVecOrMat{<:Real}, ψ::AbstractVector{<:Real}, G::AbstractVector{<:Integer}; method::Symbol=:optim)
    if length(X) == 0 || any(isnan.(X))  # for empty input or input containing nans
        return (hurst=NaN, σ=NaN, loglikelihood=NaN, optimizer=nothing)
    else
        # Force the zero-mean condition: this might be dangerous and gives lower estimations
        # X = (ndims(X) == 1) ? X : X .- mean(X, dims=2)

        @assert length(G) == size(X,1)  "Mismatched dimensions."
        @assert minimum(abs.(diff(sort(G)))) > 0  "All elements of the grid must be distinct."
        # if length(G)>0
        #     @assert length(G) == size(X,1)
        #     @assert minimum(abs.(diff(sort(G)))) > 0  # all elements are distinct
        # end

        func = h -> -fWn_log_likelihood_H(X, h, ψ, G)

        opm = nothing
        hurst = nothing
        ε::Real=1e-2   # search hurst in the range [ε, 1-ε]

        if method == :optim
            # Gradient-free constrained optimization
            opm = Optim.optimize(func, ε, 1-ε, Optim.Brent())
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

        proc = FractionalWaveletNoise(hurst, ψ)

        # Σ = covmat(proc, length(G)>0 ? G : size(X,1))
        Σ = covmat(proc, G)

        # Estimation of volatility
        σ = sqrt(xiAx(Σ, X) / length(X))
        # σ *= mean(diff(G))^(hurst)  # <- why this?

        # Log-likelihood
        L = log_likelihood_H(Σ, X)
        # # or equivalently
        # L = log_likelihood(σ^2*Σ, X)

        L /= length(X)  # normalization
        return (hurst=hurst, σ=σ, loglikelihood=L, optimizer=opm)
    end
end


"""
Accelerated fWn-MLE by dividing a large vector of samples into smaller ones.

The MLE method can be computationally expensive on data of large dimensions due to the inversion of covariance matrix. This function accelerates the MLE method by applying rolling vectorization on the large vector `X` with the parameter `(s,u,l)`. In this way the original vector is divided into smaller vectors which are then treated by MLE as i.i.d. samples.

# Args
- X: sample path of a fWn.
- ψ: wavelet filter used for computing `X`.
- s: sub window size
- u: downsampling factor
- l: length of decorrelation

# Notes
The rolling vectorization returns
- empty, if `s>size(X)[end]`
- `X` in matrix form (i.e. same as `reshape(X, :, 1)`), if `s==size(X)[end]`
"""
function fWn_MLE_estim(X::AbstractVector{<:Real}, ψ::AbstractVector{<:Real}, s::Integer, u::Integer, l::Integer; kwargs...)
    t, V = rolling_vectorize(X, s, u, l; mode=:causal)

    fWn_MLE_estim(V, ψ, u*(0:size(V,1)-1); kwargs...)
end


function bspline_MLE_estim(X::AbstractVector{<:Real}, scl::Integer, vm::Integer, s::Integer, u::Integer, l::Integer; kwargs...)
    fWn_MLE_estim(X, intscale_bspline_filter(scl, vm)/sqrt(scl), s, u, l; kwargs...)
end


#### fGn-MLE ####

"""
Filter of a fGn.
"""
fGn_filter = d -> vcat(1, zeros(d-1), -1)


"""
    fGn_MLE_estim(X, d, G; kwargs...)

Maximum likelihood estimation of Hurst exponent and volatility for fractional Gaussian noise.

# Args
- X: sample vector or matrix of fGn. For matrix each column is a sample.
- d: time lag of the finite difference operator used for computing `X`.
- G: integer time grid of `X`, by default the regular grid `1:size(X,1)` is used.
- kwargs: see `fWn_MLE_estim()`.

# Notes
- The implementation here is based on fWn (a fGn is a fWn with the filter of type `[1,-1]`) and it is just a wrapper of `fWn_MLE_estim()`. See the file `Misc.jl` for an implementation based on fGn.
"""
function fGn_MLE_estim(X::AbstractVecOrMat{<:Real}, d::Integer, G::AbstractVector{<:Integer}; kwargs...)
    fWn_MLE_estim(X, fGn_filter(d), G; kwargs...)
end


"""
Accelerated fGn-MLE by dividing a large vector of samples into smaller ones.

# Args
- X: sample path of fGn.
- d: time lag of the finite difference operator used for computing `X`.
- u: downsampling factor
- s: sub window size
- l: length of decorrelation
"""
function fGn_MLE_estim(X::AbstractVector{<:Real}, d::Integer, s::Integer, u::Integer, l::Integer; kwargs...)
    # @assert d%u == 0  # downsampling factor must divide time scale
    fWn_MLE_estim(X, fGn_filter(d), s, u, l; kwargs...)
end


# """
# Function for compatibility purpose.
# """
# function fGn_MLE_estim(X::AbstractVector{<:Real}, d::Integer, s::Integer, l::Integer; kwargs...)
#     fWn_MLE_estim(X, fGn_filter(d), s, 1, l; kwargs...)
# end


#### fWn (bank)-MLE ####

"""
    fWn_log_likelihood_H(X, H, F, G)

Log-likelihood of a fWn bank model with the optimal volatility.
"""
function fWn_log_likelihood_H(X::AbstractVecOrMat{<:Real}, H::Real, F::AbstractVector{<:AbstractVector{<:Real}}, G::AbstractVector{<:Integer})
    @assert size(X,1) == length(G) * length(F) "Mismatched dimensions."
    # the process and the number of filters in the filter bank
    proc = FractionalWaveletNoiseBank(H, F)
    return log_likelihood_H(covmat(proc, G), X)

    # # covariance matrix of fWnb
    # Σ::AbstractMatrix = if length(G)>0
    #     # @assert size(X,1) == length(F) * length(G)
    #     covmat(proc, G)
    # else
    #     covmat(proc, size(X,1)÷length(F))  # max time lag
    # end
    # return log_likelihood_H(Σ, X)
end


"""
    fWn_MLE_estim(X, F, G; method, ε)

Maximum likelihood estimation of Hurst exponent and volatility for fWn bank.

# Args
- X: sample vector or matrix. For
- F: array of filters used for computing `X`
- G: integer time grid of `X`, by default the regular grid is used.
- method: :optim for optimization based or :table for look-up table based solution.
- ε: this defines the bounded constraint [ε, 1-ε], and for method==:table this is also the step of search for Hurst exponent.

# Returns
- hurst, σ: estimation
- L: log-likelihood of estimation
- opm: object of optimizer, for method==:optim only

# Notes
- The fWnb process is multivariate. An observation at time `t` is a `d`-dimensional vector, where `d` equals to the number of filters used in fWnb. A vector `X` is the concatenation of observations made on some time grid `G`, while a matrix `X` is a collection of i.i.d. sample vectors. Hence the row dimension of `X` must be `length(F) * length(G)`, if `G` is ever provided.
"""
function fWn_MLE_estim(X::AbstractVecOrMat{<:Real}, F::AbstractVector{<:AbstractVector{<:Real}}, G::AbstractVector{<:Integer}; method::Symbol=:optim)
    if length(X) == 0 || any(isnan.(X))  # for empty input or input containing nans
        return (hurst=NaN, σ=NaN, loglikelihood=NaN, optimizer=nothing)
    else
        @assert size(X,1) == length(F) * length(G)  "Mismatched dimensions."
        @assert minimum(abs.(diff(sort(G)))) > 0  "All elements of the grid must be distinct."

        func = h -> -fWn_log_likelihood_H(X, h, F, G)

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

        proc = FractionalWaveletNoiseBank(hurst, F)

        Σ = covmat(proc, G)

        # Estimation of volatility
        σ = sqrt(xiAx(Σ, X) / length(X))
        # σ *= mean(diff(G))^(hurst)  # <- why this?

        # Log-likelihood
        L = log_likelihood_H(Σ, X)
        # # or equivalently
        # L = log_likelihood(σ^2*Σ, X)
        L /= length(X)  # normalization

        return (hurst=hurst, σ=σ, loglikelihood=L, optimizer=opm)
    end
end


"""
Accelerated MLE.
"""
function fWn_MLE_estim(X::AbstractMatrix{<:Real}, F::AbstractVector{<:AbstractVector{<:Real}}, s::Integer, u::Integer, l::Integer; kwargs...)
    t, V = rolling_vectorize(X, s, u, l; mode=:causal)  # TODO: case s = size(X,2)

    return fWn_MLE_estim(V, F, u*(0:size(V,1)-1); kwargs...)  # regular grid is implicitely used here.
end


"""
    fWn_bspline_MLE_estim(X, sclrng, v, args...; kwargs...)

fWn-MLE based on B-Spline wavelet transform.

# Args
- X: DCWT coefficients, each column corresponding to a vector of coefficients. See `cwt_bspline()`.
- sclrng: integer scales of DCWT
- v: vanishing moments of B-Spline wavelet
- s, l: if given call the accelerated version of MLE
"""
function bspline_MLE_estim(X::AbstractMatrix{<:Real}, sclrng::AbstractVector{<:Integer}, vm::Integer,  s::Integer, u::Integer, l::Integer; kwargs...)
    F = [intscale_bspline_filter(s, vm)/sqrt(s) for s in sclrng]  # extra 1/sqrt(s) factor due to the implementation of DCWT
    return fWn_MLE_estim(X, F, s, u, l; kwargs...)
end
