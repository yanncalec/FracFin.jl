######### Estimators for fractional processes #########


#### Rolling window ####

"""
Data preparation for MLE.

The purpose of this function is to break long observations into short ones by assuming they are i.i.d. This helps to reduce the size of the covariance matrix and the complexity of the MLE method.

This function constructs i.i.d. samples from the observation of a (multivariate) process. 

# Args
- X0: either a vector or a matrix, e.g. a sample trajectory of a fGn process (scalar) or a fWn process (multivariate)
- w,d,n: 

# Examples
```julia

julia> wsize, dlen, nobs = 100, 1, 0
julia> dX = FracFin.MLE_prepare_data(dX0, (wsize, dlen, nobs))
julia> (hurst_estim, σ_estim0), obj = FracFin.fGn_MLE_estim(dX)
julia> σ_estim = σ_estim0 / lag^hurst  # obtain true estimation of σ by rescaling
```
"""
function rolling_apply(func::Function, X::AbstractMatrix{T}, s::Int, d::Int=1) where {T<:Real}
    # @assert s>0
    return if s<=size(X,2)
            hcat([func(X[:,t:t+s-1]) for t=1:d:size(X,2)-s+1]...)
        else 
            []
        end    
    # # X = ndims(X0)>1 ? X0 : reshape(X0, 1, :)  # vec to matrix, create a reference not a copy
    # @assert size(X,2) >= s > 0  # valid window size
    # return hcat([trans(X[:,t:t+s-1]) for t=1:d:size(X,2) if t+s-1<=size(X,2)]...)  # put into a matrix form
    # # N = (n==0) ? size(X1,2) : min(n, size(X1,2))
    # # return view(X1, :, 1:N)  # down-sampling and truncation
end


function rolling_apply(func::Function, X::AbstractVector{T}, s::Int, d::Int=1) where {T<:Real}
    return rolling_apply(func, reshape(X,1,:), s, d)
end


"""
    rolling_estim(estim::Function, X0::AbstractVecOrMat{T}, p::Int, (w,d,n)::Tuple{Int,Int,Int}, trans::Function=(x->vec(x)); mode::Symbol=:causal) where {T<:Real}

Rolling window estimator for 1d or multivariate time series.

# Args
- estim: estimator which accepts either 1d or 2d array
- X0: input, 1d or 2d array with each column being one observation
- p: step of the rolling window
- (w,d,n): size of sub-window; length of decorrelation (no effect if `n==1`); number of sub-windows per rolling window
- trans: function of transformation which returns a vector of fixed size or a scalar,  optional
- mode: :causal or :anticausal

# Returns
- array of estimations on the rolling window

# Notes
The estimator is applied on a rolling window every `p` steps. The rolling window is divided into `n` (possibly overlapping) sub-windows of size `w` at the pace `d`, such that the size of the rolling window equals to `(n-1)*d+w`. For q-variates time series, data on a sub-window is a matrix of dimension `q`-by-`w` which is further transformed by the function `trans` into another vector. The transformed vectors of `n` sub-windows are concatenated into a matrix which is finally passed to the estimator `estim`. 

As example, for `trans = x -> vec(x)` the data on a rolling window is put into a new matrix of dimension `w*q`-by-`n`, and its columns are the column-concatenantions of data on the sub-window. Moreover, different columns of this new matrix are assumed as i.i.d. observations.
"""
function rolling_estim(estim::Function, X0::AbstractVecOrMat{T}, (w,s,d)::Tuple{Int,Int,Int}, p::Int, trans::Function=(x->vec(x)); mode::Symbol=:causal) where {T<:Real}
    # L = (n-1)*d + w  # size of rolling window
    
    res = []
    X = ndims(X0)>1 ? X0 : reshape(X0, 1, :)  # vec to matrix, create a reference not a copy
    @assert size(X,2) >= w >= s

    if mode == :causal  # causal
        for t = size(X,2):-p:1
            xs = rolling_apply(trans, X[:,t:-1:max(1, t-w+1)], s, d)
            # xs = hcat([trans(X[:,(t-i-w+1):(t-i)]) for i in d*(n-1:-1:0) if t-i>=w]...) # concatenation of column vectors
            if length(xs) > 0
                pushfirst!(res, (t,estim(squeezedims(xs))))  # <- Bug: this may give 0-dim array when apply on 1d row vector
            end
        end
    else  # anticausal
        for t = 1:p:size(X,2)
            xs = rolling_apply(trans, X[:,t:min(size(X,2), t+w-1)], s, d)
            # xs = hcat([trans(X[:, (t+i):(t+i+w-1)]) for i in d*(0:n-1) if t+i+w-1<=length(X)]...)
            if length(xs) > 0
                push!(res, (t,estim(squeezedims(xs))))
            end
        end
    end
    return res
end


function rolling_estim(func::Function, X0::AbstractVecOrMat{T}, w::Int, p::Int, trans::Function=(x->x); mode::Symbol=:causal) where {T<:Real}
    return rolling_estim(func, X0, (w,w,1), p, trans; mode=mode)
end


"""
Rolling mean in row direction.
"""
function rolling_mean(X0::AbstractVecOrMat{T}, w::Int, d::Int=1) where {T<:Real}
    return rolling_apply(x->mean(x, dims=2), X0, w, d)
end

function rolling_vectorize(X0::AbstractVecOrMat{T}, w::Int, d::Int=1) where {T<:Real}
    return rolling_apply(x->vec(x), X0, w, d)
end

######## Estimators for fBm ########

"""
Compute the p-th moment of the increment of time-lag `d` of a 1d array.
"""
moment_incr(X,d,p) = mean((abs.(X[d+1:end] - X[1:end-d])).^p)

"""
Power-law estimator for Hurst exponent and volatility.

# Args
- X: sample path
- lags: array of the increment step
- p: power

# Returns
- (hurst, σ), ols: estimation of Hurst and volatility, as well as the GLM ols object.
"""
function fBm_powlaw_estim(X::AbstractVector{T}, lags::AbstractVector{Int}, p::T=2.) where {T<:Real}
    @assert length(lags) > 1 && all(lags .> 1)
    @assert p > 0.

    C = 2^(p/2) * gamma((p+1)/2)/sqrt(pi)

    yp = map(d -> log(moment_incr(X, d, p)), lags)
    xp = p * log.(lags)
    
    # estimation of H and β
    # by manual inversion
    # Ap = hcat(xp, ones(length(xp))) # design matrix
    # hurst, β = Ap \ yp
    # or by GLM
    dg = DataFrames.DataFrame(xvar=xp, yvar=yp)
    ols = GLM.lm(@GLM.formula(yvar~xvar), dg)
    β, hurst = GLM.coef(ols)

    σ = exp((β-log(C))/p)

    return (hurst, σ), ols
end


##### Generalized scalogram #####

"""
B-Spline scalogram estimator for Hurst exponent and volatility.

# Args
- S: vector of scalogram, ie, variance of the wavelet coefficients per scale.
- sclrng: scale of wavelet transform. Each number in `sclrng` corresponds to one row in the matrix X
- v: vanishing moments
"""
function fBm_bspline_scalogram_estim(S::AbstractVector{T}, sclrng::AbstractVector{Int}, v::Int; mode::Symbol=:center) where {T<:Real}
    @assert length(S) == length(sclrng)

    df = DataFrames.DataFrame(xvar=log.(sclrng.^2), yvar=log.(S))
    ols = GLM.lm(@GLM.formula(yvar~xvar), df)
    coef = GLM.coef(ols)

    hurst = coef[2]-1/2
    Aρ = Aρ_bspline(0, 1, hurst, v, mode)
    σ = exp((coef[1] - log(abs(Aρ)))/2)
    return (hurst, σ), ols

    # Ar = hcat(xr, ones(length(xr)))  # design matrix
    # H0, η = Ar \ yr  # estimation of H and β
    # hurst = H0-1/2
    # A = Aρ_bspline(0, r, hurst, v, mode)
    # σ = exp((η - log(abs(A)))/2)
    # return hurst, σ
end

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
function fBm_gen_bspline_scalogram_estim(Σ::AbstractMatrix{T}, sclrng::AbstractVector{Int}, v::Int, r::Rational=1//1; mode::Symbol=:center) where {T<:Real}
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


###### MLE ######


"""
Safe evaluation of the inverse quadratic form
    trace(X' * inv(A) * X)
where the matrix A is symmetric and positive definite.
"""
function xiAx(A::AbstractMatrix{T}, X::AbstractVecOrMat{T}, ε::Real=0) where {T<:Real}
    @assert issymmetric(A)
    @assert size(X, 1) == size(A, 1)

    S, U = eigen(A)  # so that U * Diagonal(S) * inv(U) == A, in particular, U' == inv(U)
    idx = (S .> ε)

    # U, S, V = svd(A)
    # idx = S .> ε
    return sum((U[:,idx]'*X).^2 ./ S[idx])

    #     iA = pinv(A)
    #     return tr(X' * iA * X)
end


"""
Safe evaluation of the log-likelihood of a fBm model with the implicit optimal volatility (in the MLE sense).

The value of log-likelihood (up to some additif constant) is
    -1/2 * (N*log(X'*inv(A)*X) + logdet(A))

# Args
- A: covariance matrix
- X: vector of matrix of observation. For matrix input the columns are i.i.d. observations.

# Notes
- This function is common to all MLEs with the covariance matrix of form σ²A(h), where {σ, h} are unknown parameters. This kind of MLE can be carried out in h uniquely and σ is obtained from h.
"""
function log_likelihood_H(A::AbstractMatrix{T}, X::AbstractVecOrMat{T}, ε::Real=0) where {T<:Real}
    @assert issymmetric(A)
    @assert size(X, 1) == size(A, 1)

    N = ndims(X)>1 ? size(X,2) : 1  # number of i.i.d. samples in data
    # d = size(X,1)  # such that N*d == length(X)

    S, U = eigen(A)  # so that U * Diagonal(S) * inv(U) == A, in particular, U' == inv(U)
    idx = (S .> ε)
    # U, S, V = svd(A)
    # idx = S .> ε

    val = -1/2 * (length(X)*log(sum((U[:,idx]'*X).^2 ./ S[idx])) + N*sum(log.(S[idx])))  # non-constant part of log-likelihood
    return val - length(X)*log(2π*ℯ/length(X))/2  # with the constant part
end


function log_likelihood(A::AbstractMatrix{T}, X::AbstractVecOrMat{T}) where {T<:Real}
    @assert issymmetric(A)
    @assert size(X, 1) == size(A, 1)

    N = ndims(X)>1 ? size(X,2) : 1
    # d = size(X,1), # such that N*d == length(X)

    return -1/2 * (N*logdet(A) + xiAx(A,X) + length(X)*log(2π))
end


#### fWn-MLE ####
# A fWn is the filtration of a fBm time series by a bank of high pass filters, eg, multiscale wavelet filters.

"""
Compute the covariance matrix of a fWn at some time lag.

# Args
- F: array of filters
- l: time lag
- H: Hurst exponent
"""
function fWn_covmat_lag(F::AbstractVector{<:AbstractVector{T}}, l::Int, H::Real) where {T<:Real}
    L = maximum([length(f) for f in F])  # maximum length of filters
    M = [abs(l+(n-m))^(2H) for n=0:L-1, m=0:L-1]    
    Σ = -1/2 * [ψi' * view(M, 1:length(ψi), 1:length(ψj)) * ψj for ψi in F, ψj in F]
end


"""
Compute the covariance matrix of a time-concatenated fWn.
"""
function fWn_covmat(F::AbstractVector{<:AbstractVector{T}}, lmax::Int, H::Real) where {T<:Real}
    J = length(F)
    Σ = zeros(((lmax+1)*J, (lmax+1)*J))
    Σs = [fWn_covmat_lag(F, d, H) for d = 0:lmax]

    for r = 0:lmax
        for c = 0:lmax
            Σ[(r*J+1):(r*J+J), (c*J+1):(c*J+J)] = (c>=r) ? Σs[c-r+1] : transpose(Σs[r-c+1])
        end
    end

    return Matrix(Symmetric(Σ))  #  forcing symmetry
end


"""
H-dependent log-likelihood of fraction wavelet noise (fWn) with optimal σ.
"""
function fWn_log_likelihood_H(X::AbstractVecOrMat{T}, F::AbstractVector{<:AbstractVector{T}}, H::Real) where {T<:Real}
    @assert 0 < H < 1
    @assert size(X,1) % length(F) == 0
    
    Σ = Matrix(Symmetric(fWn_covmat(F, size(X,1)÷length(F)-1, H)))
    return log_likelihood_H(Σ, X)
end


"""
General fWn-MLE of Hurst exponent and volatility.

# Args
- X: transformed coefficients, each column is a vector of coefficient; or concatenation of vectors.
- F: array of filters, each corresponding to a row in X
- method: :optim for optimization based or :table for look-up table based solution.
- ε: this defines the bounded constraint [ε, 1-ε], and for method==:table this is also the step of search for Hurst exponent.

# Returns
- (hurst, σ): estimation
- L: log-likelihood of estimation
- opm: object of optimizer, for method==:optim only

# Notes
- X can also be the concatenation of vectors at at consecutive instants.
"""
function fWn_MLE_estim(X::AbstractVecOrMat{T}, F::AbstractVector{<:AbstractVector{T}}; method::Symbol=:optim, ε::Real=1e-2) where {T<:Real}
    @assert 0. < ε < 1.
    @assert size(X,1) % length(F) == 0

    func = h -> -fWn_log_likelihood_H(X, F, h)

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
    
    Σ = Matrix(Symmetric(fWn_covmat(F, size(X,1)÷length(F)-1, hurst)))
    σ = sqrt(xiAx(Σ, X) / length(X))
    L = log_likelihood_H(Σ, X)
    
    return (hurst, σ), L, opm
end


"""
fWn-MLE based on B-Spline wavelet transform.

# Args
- X: DCWT coefficients, each column corresponding to a vector of coefficients. See `cwt_bspline()`.
- sclrng: integer scales of DCWT
- v: vanishing moments of B-Spline wavelet
"""
function fBm_bspline_MLE_estim(X::AbstractVecOrMat{T}, sclrng::AbstractVector{Int}, v::Int; method::Symbol=:optim, ε::Real=1e-2) where {T<:Real}
    F = [_intscale_bspline_filter(s, v)/sqrt(s) for s in sclrng]  # extra 1/sqrt(s) factor due to the implementation of DCWT
    return fWn_MLE_estim(X, F; method=method, ε=ε)
end


#### fGn-MLE ####
# A special case of fWn-MLE which deserves it own implementation.

function covmat(s::AbstractVector{T}) where {T<:Real}
    [s[abs(n-m)+1] for n=1:length(s), m=1:length(s)]
end

function fGn_covfunc(H::Real, n::Int, d::Int)
    return 1/2 * (abs(n+d)^(2H) + abs(n-d)^(2H) - 2*abs(n)^(2H))
end

fGn_covseq(H::Real, lags::AbstractVector{Int}, d::Int) where {T<:Real} = [fGn_covfunc(H,n,d) for n in lags]
                               
fGn_covmat(H::Real, N::Int, d::Int) = covmat(fGn_covseq(H, 0:N-1, d))

function fGn_log_likelihood_H(X::AbstractVecOrMat{T}, H::Real, d::Int) where {T<:Real}
    @assert 0 < H < 1

    # Σ = Matrix(Symmetric(covmat(FractionalGaussianNoise(H, 1.), size(X,1))))
    Σ = Matrix(Symmetric(fGn_covmat(H, size(X,1), d)))
    return log_likelihood_H(Σ, X)
end


"""
Maximum likelihood estimation of Hurst exponent and volatility using fractional Gaussian noise model.

# Args
- X: observation vector or matrix of a fGn process. For matrix input each column is an i.i.d. observation.
- d: time-lag of the finite difference operator used for computing `X`.
- method, ε: see `fWn_MLE_estim()`.

# Notes
- This method is very expensive for data of large dimensions, see docs in `MLE_prepare_data()`.
"""
function fGn_MLE_estim(X::AbstractVecOrMat{T}, d::Int; method::Symbol=:optim, ε::Real=1e-2) where {T<:Real}
    @assert 0. < ε < 1.
    func = h -> -fGn_log_likelihood_H(X, h, d)

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
    
    # Σ = Matrix(Symmetric(covmat(FractionalGaussianNoise(hurst, 1.), size(X,1))))
    Σ = Matrix(Symmetric(fGn_covmat(hurst, size(X,1), d)))
    σ = sqrt(xiAx(Σ, X) / length(X))
    L = log_likelihood_H(Σ, X)

    return (hurst, σ), L, opm
end


"""
TODO
"""
function msfGn_MLE_estim(X::AbstractVector{T}, lags::AbstractVector{Int}, (w,d)::Tuple{Int,Int}) where {T<:Real}
    Hs = zeros(length(lags))
    Σs = zeros(length(lags))

    for (n,lag) in enumerate(lags)  # time lag for finite difference
        dXm = mean(vec2mat(X[lag+1:end]-X[1:end-lag], lag), dims=1)
        dX = squeezedims(rolling_vectorize(dXm, w, d))
        # println(size(dX))
        (hurst_estim, σ_estim), obj = fGn_MLE_estim(dX, 1)
        # # σ_estim = σ_estim0 / lag^hurst_estim  # obtain true estimation of σ by rescaling

        Hs[n] = hurst_estim
        Σs[n] = σ_estim
    end
    return Hs, Σs
end


##### B-Spline DCWT MLE (Not maintained) #####
# Implementation based on DCWT formulation, not working well in practice.

function fBm_bspline_covmat_lag(H::Real, v::Int, l::Int, sclrng::AbstractVector{Int}, mode::Symbol)
    return Amat_bspline(H, v, l, sclrng) .* [sqrt(i*j) for i in sclrng, j in sclrng].^(2H+1)
end


"""
Compute the covariance matrix of B-Spline DCWT coefficients of a pure fBm.

The full covariance matrix of `J`-scale transform and of time-lag `N` is a N*J-by-N*J symmetric matrix.

# Args
- l: maximum time-lag
- sclrng: scale range
- v: vanishing moments of B-Spline wavelet
- H: Hurst exponent
- mode: mode of convolution
"""
function fBm_bspline_covmat(l::Int, sclrng::AbstractVector{Int}, v::Int, H::Real, mode::Symbol)    
    J = length(sclrng)
    Σ = zeros(((l+1)*J, (l+1)*J))
    Σs = [fBm_bspline_covmat_lag(H, v, d, sclrng, mode) for d = 0:l]

    for r = 0:l
        for c = 0:l
            Σ[(r*J+1):(r*J+J), (c*J+1):(c*J+J)] = (c>=r) ? Σs[c-r+1] : transpose(Σs[r-c+1])
        end
    end

    return Matrix(Symmetric(Σ))  #  forcing symmetry
    # return [(c>=r) ? Σs[c-r+1] : Σs[r-c+1]' for r=0:N-1, c=0:N-1]
end


"""
Evaluate the log-likelihood of B-Spline DCWT coefficients.
"""
function fBm_bspline_log_likelihood_H(X::AbstractVecOrMat{T}, sclrng::AbstractVector{Int}, v::Int, H::Real, mode::Symbol) where {T<:Real}
    @assert 0 < H < 1
    @assert size(X,1) % length(sclrng) == 0

    L = size(X,1) ÷ length(sclrng)  # integer division: \div
    # N = ndims(X)>1 ? size(X,2) : 1

    Σ = fBm_bspline_covmat(L-1, sclrng, v, H, mode)  # full covariance matrix

    # # strangely, the following does not work (logarithm of a negative value)
    # iΣ = pinv(Σ)  # regularization by pseudo-inverse
    # return -1/2 * (J*N*log(trace(X'*iΣ*X)) + logdet(Σ))

    return log_likelihood_H(Σ, X)
end


"""
B-Spline wavelet-MLE estimator.
"""
function fBm_bspline_DCWT_MLE_estim(X::AbstractVecOrMat{T}, sclrng::AbstractVector{Int}, v::Int, mode::Symbol; method::Symbol=:optim, ε::Real=1e-2) where {T<:Real}
    @assert size(X,1) % length(sclrng) == 0
    # number of wavelet coefficient vectors concatenated into one column of X
    L = size(X,1) ÷ length(sclrng)  # integer division: \div
    # N = ndims(X)>1 ? size(X,2) : 1
    
    func = x -> -fBm_bspline_log_likelihood_H(X, sclrng, v, x, mode)

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
    
    Σ = fBm_bspline_covmat(L-1, sclrng, v, hurst, mode)
    σ = sqrt(xiAx(Σ, X) / length(X))

    return (hurst, σ), opm
end

