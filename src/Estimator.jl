######### Estimators for fBm and related processes #########

# abstract type AbstractEstimator end
# abstract type AbstractRollingEstimator <: AbstractEstimator end
# abstract type AbstractfBmEstimator <: AbstractEstimator end

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
function fWn_MLE_estim(X0::AbstractVecOrMat{<:Real}, ψ::AbstractVector{<:Real}, G::AbstractVector{<:Integer}; method::Symbol=:optim)
    # @assert 0. < ε < 1.
    if length(X0) == 0 || any(isnan.(X0))  # for empty input or input containing nans
        return (hurst=NaN, σ=NaN, loglikelihood=NaN, optimizer=nothing)
    else
        # forcing zero-mean condition
        # X = (ndims(X0) == 1) ? X0 : X0 .- mean(X0, dims=2)        
        X = X0

        @assert length(G) == size(X,1)  "Mismatched dimension."
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
function fWn_log_likelihood_H(X::AbstractVecOrMat{<:Real}, H::Real, F::AbstractVector{<:AbstractVector{<:Real}}, G::AbstractVector{<:Integer}=Int[])
    # the process and the number of filters in the filter bank
    proc = FractionalWaveletNoiseBank(H, F)

    # covariance matrix of fWnb
    Σ::AbstractMatrix = if length(G)>0
        # @assert size(X,1) == length(F) * length(G)
        covmat(proc, G)
    else
        covmat(proc, size(X,1)÷length(F))  # max time lag
    end

    return log_likelihood_H(Σ, X)
end


"""
    fWn_MLE_estim(X, F, G; method, ε)

Maximum likelihood estimation of Hurst exponent and volatility for fWn bank.

# Args
- X: sample vector or matrix
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
function fWn_MLE_estim(X::AbstractVecOrMat{<:Real}, F::AbstractVector{<:AbstractVector{<:Real}}, G::AbstractVector{<:Integer}=Int[]; method::Symbol=:optim, ε::Real=1e-2)
    # @assert 0. < ε < 1.

    if length(X) == 0  # for empty input
        return (hurst=NaN, σ=NaN, loglikelihood=NaN, optimizer=nothing)
    else
        if length(G)>0
            @assert size(X,1) == length(F) * length(G)
            @assert minimum(abs.(diff(sort(G)))) > 0  # all elements are distinct
        else
            @assert size(X,1) % length(F) == 0
        end

        func = h -> -fWn_log_likelihood_H(X, h, F, G)
        opm = nothing
        hurst = nothing

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

        proc = FractionalWaveletNoiseBank(hurst, F)
        Σ = covmat(proc, length(G)>0 ? G : size(X,1)÷length(F))
        σ = sqrt(xiAx(Σ, X) / length(X))
        L = log_likelihood_H(Σ, X)

        return (hurst=hurst, σ=σ, loglikelihood=L, optimizer=opm)
    end
end


"""
Accelerated MLE.
"""
function fWn_MLE_estim(X::AbstractMatrix{<:Real}, F::AbstractVector{<:AbstractVector{<:Real}}, s::Integer, l::Integer; kwargs...)
    t, V = rolling_vectorize(X, s, 1, l; mode=:causal)  # TODO: case s = size(X,2)

    return fWn_MLE_estim(V, F; kwargs...)  # regular grid is implicitely used here.
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
function fWn_bspline_MLE_estim(X::AbstractMatrix{<:Real}, sclrng::AbstractVector{<:Integer}, v::Integer, args...; kwargs...)
    F = [intscale_bspline_filter(s, v)/sqrt(s) for s in sclrng]  # extra 1/sqrt(s) factor due to the implementation of DCWT
    return fWn_MLE_estim(X, F, args...; kwargs...)
end

# const fBm_bspline_MLE_estim = fWn_bspline_MLE_estim


###### variogram estimator ######

"""
Variance function of the empirical variogram
"""
function variogram_variance(H::Real, lags::AbstractVector{<:Integer}, N::Integer)
    V = zeros(Float64, length(lags))

    for (j, d) in enumerate(lags)
        proc = FractionalGaussianNoise(H, d)
        V[j] = autocov(proc, 0)^2 + sum((1-i/N) * autocov(proc, i)^2 for i=0:N-1)
    end
    return 4/N * V
end


"""
    powlaw_estim(X, lags; p=2., method=:optim)

Power-law estimator for Hurst exponent and volatility.

# Args
- X: matrix of fGn, each row is a fGn of some time lag and each column is an observation.
- lags: time lags (increment step) used to compute each component of `X`
- p: power of the moment
- methods: method of estimation: {:optim, :lm}

# Returns
- hurst, σ: estimation of Hurst and volatility, as well as an object of optimizer
- (xp, yp): vectors of regression
- opm: optimizer

# Notes
- `X` is computed from fBm by taking finite differences. The second dimension corresponds to time. Example, let `W` be a fBm sample path then the following command computes `X`:
```julia
julia> lags = 2:10
julia> X = transpose(lagdiff(W, lags, mode=:causal))
```
- `p=1` is robust against quantization error.
"""
function powlaw_estim(X::AbstractMatrix{<:Real}, lags::AbstractVector{<:Integer}; pow::Real=2., method::Symbol=:lm)
    # remove columns containing NaN
    idx = findall(vec(.!any(isnan.(X), dims=1)))
    X = X[:,idx] # view(X,:,idx)

    if length(X) == 0
        return (hurst=NaN, σ=NaN, vars=nothing, optimizer=nothing)
    else
        @assert length(lags) == size(X,1) > 1  "Dimension mismatch."
        @assert all(lags .>= 1)  "Lags must be larger than or equal to 1."
        @assert pow>0  "Moment must be positive."

        # explanatory and observation vectors
        xp = pow * log.(lags)
        μX = mean(X, dims=2)
        yp = vec(log.(mean((abs.(X.-μX)).^pow, dims=2)))
        # yp = vec(log.(mean((abs.(X)).^pow, dims=2)))
        # hurst, η, res, err = NaN, NaN, NaN, (NaN, NaN)

        # old version with weights
        # # polynomial order of the weight for samples, if 0 the uniform weight is used
        # kt::Integer = 0  # non-zero value puts more weight on most recent samples (i.e. those at large column numbers).
        # wt = StatsBase.weights(causal_weight(size(X,2), kt))
        # μX = mean(X, wt, 2)
        # yp = vec(log.(mean((abs.(X.-μX)).^p, wt, 2)))
        # # yp = vec(log.(mean((abs.(X)).^p, wt, 2)))  # <- this gives lower SNR
        # xp = p * log.(lags)
        # # weight for scales
        # ks::Integer = 0  # hard-coded: polynomial order of weight for scales, if 0 the uniform weight is used
        # ws = StatsBase.weights(poly_weight(length(yp), ks))
        # yc = yp .- mean(yp, ws)
        # xc = xp .- mean(xp, ws)
        # func = h -> 1/2 * sum(ws .* (yc - h*xc).^2)
        # # func = h -> 1/2 * sum(ws .* abs.(yc - h*xc))

        # compute the weighting vector:
        # Run first an estimation of Hurst by linear regression, then use this estimate to compute the weighting vector.
        dg = DataFrames.DataFrame(xvar=xp, yvar=yp)
        opm = GLM.lm(@GLM.formula(yvar~xvar), dg)
        η, hurst = GLM.coef(opm)  # intercept and slope
        res = GLM.deviance(opm)  # residual
        # err = try  # std error of estimates
        #     GLM.stderror(opm)
        # catch
        #     (NaN, NaN)
        # end

        ws =  variogram_variance(0 < hurst < 1 ? hurst : 0.5, lags, size(X,2)) .^ -1
        # ws =  variogram_variance(0.5, lags, size(X,2)) .^ -1  # in practice this is good enough?
        # ws = ones(length(lags))  # uniform weight
        # ws ./= sum(ws)

        # estimation of H and η

        if method == :optim
            yc = yp .- mean(yp)
            xc = xp .- mean(xp)
            func = h -> sum(ws .* (yc - h*xc).^2)
            # func = h -> sum(ws .* abs.(yc - h*xc).^qnorm)
            # Gradient-free constrained optimization
            ɛ = 1e-2  # search hurst in the interval [ɛ, 1-ɛ]
            opm = Optim.optimize(func, ε, 1-ε, Optim.Brent())
            # # Gradient-based optimization
            # optimizer = Optim.GradientDescent()  # e.g. Optim.BFGS(), Optim.GradientDescent()
            # opm = Optim.optimize(func, ε, 1-ε, [0.5], Optim.Fminbox(optimizer))
            hurst = Optim.minimizer(opm)[1]
            η = mean(yp - hurst*xp)
            # relative residual
            res = sqrt(opm.minimum / length(xp) / var(yp))
        elseif method == :lm  # using GLM package
            dg = DataFrames.DataFrame(xvar=xp, yvar=yp)
            opm = GLM.glm(@GLM.formula(yvar~xvar), dg, GLM.Normal(), GLM.IdentityLink(), wts=sqrt.(ws))
            # opm = GLM.lm(@GLM.formula(yvar~xvar), dg)
            η, hurst = GLM.coef(opm)
            # GLM.deviance is by definition the RSS
            res = sqrt(GLM.deviance(opm) / length(xp) / var(yp))

            # # or equivalently, by manual inversion
            # Ap = hcat(xp, ones(length(xp))) # design matrix
            # hurst, η = Ap \ yp
        else
            error("Unknown method $(method).")
        end

        cp = normal_moment_factor(pow)  # constant factor depending on p
        σ = (0.05 < hurst < 0.95) ? exp((η-log(cp))/pow) : NaN

        # return hurst, σ, (xp, yp)
        return (hurst=hurst, σ=σ, η=η, residual=res, vars=(xp,yp), optimizer=opm)
    end
end


"""
    powlaw_estim(X::AbstractVector{<:Real}, lags::AbstractVector{<:Integer}; kwargs...)

# Args
- X: sample path of fBm, e.g. the log-price
- lags: time lags used to compute finite differences
"""
function powlaw_estim(X::AbstractVector{<:Real}, lags::AbstractVector{<:Integer}; mode::Symbol=:causal, kwargs...)
    dX = transpose(lagdiff(X, lags; mode=mode))  # take transpose s.t. each column is an observation
    return powlaw_estim(dX, lags; kwargs...)
end


"""
# Args
- estimator: function of estimator having the interface `estimator(X, p, args...; kwargs...)`. Here `X` is the input array, `p` is the power of moment.
"""
function multifractal_estim(estimator::Function, X::AbstractArray, pows::AbstractVector{<:Real}, args...; kwargs...)
    @assert all(pows .> 0) "Powers of moment must be positive"

    res = [estimator(X, p, args...; kwargs...) for p in pows]
    ws = StatsBase.weights([x.residual for x in res].^ -1)
    Hs = [x.hurst for x in res]
    σs = [x.σ for x in res]
    Rs = [x.residual for x in res]

    # return (hurst=mean(Hs, ws), σ=mean(σs, ws), residual=mean(Rs, ws))
    # return (hurst=median(Hs, ws), σ=median(σs, ws), residual=median(Rs, ws))
    return (hurst=median(Hs), σ=median(σs), residual=median(Rs))
end

const variogram_estim = powlaw_estim


####### Scalogram estimator #######

"""
Variance function of the empirical scalogram of B-Spline wavelet
"""
function bspline_scalogram_variance(H::Real, vm::Integer, sclrng::AbstractVector{<:Integer}, N::Integer)
    V = zeros(Float64, length(sclrng))

    for (j, s) in enumerate(sclrng)
        # B-Spline filter: extra 1/sqrt(s) factor is due to the implementation of DCWT
        filter = intscale_bspline_filter(s, vm)/sqrt(s)
        proc = FractionalWaveletNoise(H, filter)
        V[j] = autocov(proc, 0)^2 + sum((1-i/N) * autocov(proc, i)^2 for i=0:N-1)
    end
    return 4/N * V
end


"""

B-Spline scalogram estimator for Hurst exponent and volatility.

# Args
- X: matrix of wavelet coefficients. Each row corresponds to a scale.
- sclrng: scale of wavelet transform. Each number in `sclrng` corresponds to one row in the matrix X.
- v: vanishing moments of the wavelet
- p: power of the scalogram
"""
function bspline_scalogram_estim(X::AbstractMatrix{<:Real}, sclrng::AbstractVector{<:Integer}, vm::Integer; pow::Real=2., method::Symbol=:optim)
    # remove columns containing NaN
    idx = findall(vec(.!any(isnan.(X), dims=1)))
    X = X[:,idx] # view(X,:,idx)

    if length(X) == 0
        return (hurst=NaN, σ=NaN, vars=nothing, optimizer=nothing)
    else
        @assert length(sclrng) == size(X,1) > 1  "Dimension mismatch."
        @assert any(sclrng .% 2 .== 0) && any(sclrng .> 0)  "All scales must be positive even number."
        @assert pow>0  "Moment must be positive."

        xp = pow * log.(sclrng)
        μX = mean(X, dims=2)
        yp = vec(log.(mean(abs.(X.-μX).^pow, dims=2)))
        # hurst, η, res = NaN, NaN, NaN

        # compute the weighting vector:
        # Run first an estimation of Hurst by linear regression, then use this estimate to compute the weighting vector.
        dg = DataFrames.DataFrame(xvar=xp, yvar=yp)
        opm = GLM.lm(@GLM.formula(yvar~xvar), dg)
        coef = GLM.coef(opm)
        η, hurst = coef[1], coef[2] - 1/2  # intercept and slope
        res = GLM.deviance(opm)  # residual
        ws =  bspline_scalogram_variance(0 < hurst < 1 ? hurst : 0.5, vm, sclrng, size(X,2)) .^ -1
        # ws ./= sum(ws)

        # estimation of H and η
        if method == :optim
            yc = yp .- mean(yp)
            xc = xp .- mean(xp)

            # original implentation
            # func = h -> sum(ws .* abs.(yc - h*xc).^1)
            func = h -> sum(ws .* (yc - h*xc).^2)
            # Gradient-free constrained optimization
            ɛ = 1e-2  # search hurst in the interval 0.5+[ɛ, 1-ɛ]
            opm = Optim.optimize(func, 0.5+ε, 1.5-ε, Optim.Brent())
            hurst = Optim.minimizer(opm)[1] - 1/2
            η = mean(yp - (hurst+1/2)*xp)

            # # implementation of Knut with weight
            # ɛ = 1e-2  # search hurst in the interval 0.5+[ɛ, 1-ɛ]
            # wv1 = 1 ./ (sclrng)
            # func = h -> sum(wv1 .* abs.(yc - h*xc).^2)
            # # func = h -> sum(wv1 .* (yc - h*xc).^2)
            # opm = Optim.optimize(func, 0.5+ε, 1.5-ε, Optim.Brent())
            # hurst1 = Optim.minimizer(opm)[1] - 1/2
            # wv2 = 1 ./ (sclrng .^ 3)
            # func = h -> sum(wv2 .* abs.(yc - h*xc).^2)
            # # func = h -> sum(wv2 .* (yc - h*xc).^2)
            # opm = Optim.optimize(func, 0.5+ε, 1.5-ε, Optim.Brent())
            # hurst2 = Optim.minimizer(opm)[1] - 1/2
            # hurst = max(hurst1, hurst2)
            # # println("Hurst1=$(hurst1), Hurst2=$(hurst2)")
            # η = mean(yp - (hurst+1/2)*xp)

            res = sqrt(opm.minimum / length(xp) / var(yp))
        elseif method == :lm  # using GLM package
            dg = DataFrames.DataFrame(xvar=xp, yvar=yp)
            opm = GLM.glm(@GLM.formula(yvar~xvar), dg, GLM.Normal(), GLM.IdentityLink(), wts=sqrt.(ws))
            # opm = GLM.lm(@GLM.formula(yvar~xvar), dg)
            coef = GLM.coef(opm)
            η, hurst = coef[1], coef[2] - 1/2
            res = sqrt(GLM.deviance(opm) / length(xp) / var(yp))

        # elseif method == :irls
        #     coef = IRLS(yp, xp, p; maxiter=10^4, tol=10^-4)
        #     hurst = coef[1][1]-1/2
        #     η = coef[2][1]  # returned value is a scalar in a vector form
        #     opm = nothing
        else
            error("Unknown method $(method).")
        end

        cp = normal_moment_factor(pow)  # constant factor depending on p
        σ = if 0.05 < hurst < 0.95
            try
                A = Aψρ_bspline(0, 1, hurst, vm, :center)  # kwargs: mode=:center
                exp((η - log(cp) - log(A)*pow/2)/pow)
            catch
                NaN
            end
        else
            NaN
        end

        return (hurst=hurst, σ=σ, η=η, residual=res, vars=(xp,yp), optimizer=opm)
    end
end


"""
# Args
- X: sample path of fBm, e.g. the log-price
- v: vanishing moments
"""
function bspline_scalogram_estim(X::AbstractVector{<:Real}, sclrng::AbstractVector{<:Integer}, vm::Integer; kwargs...)

    # B-Spline wavelet transform
    W, M = cwt_bspline(X, sclrng, vm, :causal)
    # Wt = [view(W, findall(M[:,n]),n) for n=1:size(W,2)]
    # truncation of boundary points
    t1 = findall(prod(M, dims=2))[1][1]
    t2 = findall(prod(M, dims=2))[end][1]
    Wt = W[t1:t2, :]

    # # Covariance matrix
    # lag = 0
    # Σ = cov(Wt[1:end-lag, :], Wt[lag+1:end,:], dims=1);

    return bspline_scalogram_estim(Wt', sclrng, v; kwargs...)
end


# function bspline_scalogram_estim(X::AbstractMatrix{<:Real}, sclrng::AbstractVector{<:Integer}, v::Integer, pows::AbstractVector{<:Real}; kwargs...)

#     bspline_scalogram_estim()
# end


# """
# B-Spline scalogram estimator with a matrix of DCWT coefficients as input. Each column in `W` is a vector of DCWT coefficients.
# """
# function fBm_bspline_scalogram_estim(W::AbstractMatrix{T}, sclrng::AbstractVector{Int}, v::Int; dims::Int=1, mode::Symbol=:center) where {T<:Real}
#     return fBm_bspline_scalogram_estim(var(W,dims), sclrng, v; mode=mode)
# end

# """
# B-Spline scalogram estimator with an array of DCWT coefficients as input. Each row in `W` corresponds to a scale.
# """
# function fBm_bspline_scalogram_estim(W::AbstractVector{T}, sclrng::AbstractVector{Int}, v::Int; mode::Symbol=:center) where {T<:AbstractVector{<:Real}}
#     return fBm_bspline_scalogram_estim([var(w) for w in W], sclrng, v; mode=mode)
# end


"""
Generalized B-Spline scalogram estimator for Hurst exponent and volatility.

# Args
- Σ: covariance matrix of wavelet coefficients.
- sclrng: scale of wavelet transform. Each number in `sclrng` corresponds to one row in the matrix X
- v: vanishing moments
- r: rational ratio defining a line in the covariance matrix, e.g. r=1 corresponds to the main diagonal.
"""
function gen_bspline_scalogram_estim(Σ::AbstractMatrix{T}, sclrng::AbstractVector{Int}, v::Int, r::Rational=1//1; mode::Symbol=:center) where {T<:Real}
    @assert issymmetric(Σ)
    @assert size(Σ,1) == length(sclrng)
    @assert r >= 1
    if r > 1
        all(diff(sclrng/sclrng[1]) .== 1) || error("Imcompatible scales: the ratio between the k-th and the 1st scale must be k")
    end

    p,q,N = r.num, r.den, length(sclrng)
    @assert N>=2p

    # Σ = cov(X, X, dims=2, corrected=true)  # covariance matrix

    yr = [log(abs(Σ[q*j, p*j])) for j in 1:N if p*j<=N]
    xr = [log(sclrng[q*j] * sclrng[p*j]) for j in 1:N if p*j<=N]

    df = DataFrames.DataFrame(xvar=xr, yvar=yr)
    ols = GLM.lm(@GLM.formula(yvar~xvar), df)
    coef = GLM.coef(ols)

    hurst = coef[2]-1/2
    Aρ = Aρ_bspline(0, r, hurst, v, mode)
    σ = exp((coef[1] - log(abs(Aρ)))/2)
    return (hurst, σ), ols

    # Ar = hcat(xr, ones(length(xr)))  # design matrix
    # H0, η = Ar \ yr  # estimation of H and β
    # hurst = H0-1/2
    # Aρ = Aρ_bspline(0, r, hurst, v, mode)
    # σ = ℯ^((η - log(abs(Aρ)))/2)
    # return hurst, σ
end
const fBm_gen_bspline_scalogram_estim = gen_bspline_scalogram_estim
