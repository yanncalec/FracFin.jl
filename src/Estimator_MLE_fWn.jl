#### fWn-MLE ####

# This files contains a version of MLE for a single scale fWn.
# For legacy only, all functions can be realized via fWn bank MLE.

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

    fWn_MLE_estim(V, ψ, u*(0:s-1); kwargs...)
end

