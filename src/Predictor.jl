function fGn_predict_cond(X::AbstractVector{<:Real}, α::AbstractMatrix, u::Integer=1; bias::Bool=true, kwargs...)
    # estimation of bias  β = μy - α * μx
    n, k = size(α)  # n: step of prediction, k: number of samples used for prediction

    gy = (t,n) -> t .+ collect(1:n)  # grid of prediction, starting from t
    # gx = (t,k,u) -> collect((t-k*u+1):u:t)  # grid of samples, ending by t
    gx = (t,k,u) -> reverse(collect(t:-u:(t-k*u+1)))  # grid of samples, ending by t

    β::AbstractVector = if bias
        βm = hcat([X[gy(t,n)] - α*X[gx(t,k,u)] for t=(k*u):(length(X)-n)]...)  # t=(k*u):(length(X)-n)
        mean(βm, dims=2)[:]
    else
        zeros(n)
    end

    # conditional mean and covariance
    μ::AbstractVector = α * X[gx(length(X),k,u)] + β
#     Σ::AbstractMatrix = Σyy - Σyx * iΣxx * Σyx'

    return (μ=μ, β=β)
end


function fGn_predict_replict(X::AbstractVector{<:Real}, hurst::Real, u::Integer=1, k::Integer=1; kwargs...)
    N = length(X)
    μ = sign(hurst-0.5) * mean(X[end:-u:(end-k*u+1)])
#     μ = sign(hurst-0.5) * mean(X[end-k+1:end])

    return (μ=μ,)
end


"""
MLE of fGn model and prediction by replication.

# Args
- s: sub window size
- l: length of decorrelation
- k: number of samples used for prediction
"""
function fGn_MLE_estim_predict_replict(X::AbstractVector{<:Real}, d::Integer, s::Integer, l::Integer, k::Integer=1; kwargs...)
    est = fGn_MLE_estim(X, d, s, l; kwargs...)
    μ = sign(est.hurst-0.5) * mean(X[end-k+1:end])

    return (hurst=est.hurst, σ=est.σ, μ=μ)
end


"""
MLE of fGn model and prediction by conditional statistics.

# Args
- s: sub window size
- l: length of decorrelation
- k: number of samples used for prediction
- u: downsampling factor for samples
- n: step of prediction
"""
function fGn_MLE_estim_predict_cond(X::AbstractVector{<:Real}, d::Integer, s::Integer, l::Integer, k::Integer, u::Integer=1, n::Integer=d; bias::Bool=false, kwargs...)
    @assert 0 < k && 0 < n
    # @assert n+k*u <= length(X)

    # estimation of hurst and volatility
    est = fGn_MLE_estim(X, d, s, l; kwargs...)

    P = FractionalGaussianNoise(est.hurst, d)  # fGn with estimated hurst at step d
    gy = t -> t .+ collect(1:n)  # grid of prediction, starting from t
    gx = t -> collect(reverse(t:-u:(t-k*u+1)))  # grid of samples, ending by t

    # covariance matrices
    Σyy = covmat(P, gy(0))
    Σyx = covmat(P, gy(0), gx(0))
    Σxx = covmat(P, gx(0))
    iΣxx = pinv(Matrix(Σxx))
    Σ = Σyy - Σyx * iΣxx * Σyx'

    α::AbstractMatrix = Σyx * iΣxx  # kernels of conditional mean
    # estimation of bias  β = μy - α * μx
    β::AbstractVector = if bias
        # βm = hcat([X[gy(t)] - α*X[gx(t)] for t=1:length(X) if gx(t)[1]>0 && gy(t)[end]<=length(X)]...)  # t=(k*u):(length(X)-n)
        βm = hcat([X[gy(t)] - α*X[gx(t)] for t=k*u:length(X)-n]...)  # t=(k*u):(length(X)-n)
        mean(βm, dims=2)[:]
    else
        zeros(n)
    end

    # conditional mean and covariance
    μ = α * X[gx(length(X))] + β

    return (hurst=est.hurst, σ=est.σ, μ=μ, Σ=Σ, α=α, β=β)
end

"""
# Args
- X: each row corresponds to a symbol
"""
function fGn_MLE_estim_predict_cond(X::AbstractMatrix{<:Real}, d::Integer, s::Integer, l::Integer, k::Integer, u::Integer=1, n::Integer=d; bias::Bool=false, kwargs...)
    @assert 0 < k && 0 < n
    # @assert n+k*u <= length(X)

    # estimation of hurst and volatility
    res = []  # results of estimation
    for r=1:size(X,1)
        push!(res, fGn_MLE_estim(X[r,:], d, s, l; kwargs...))
    end

    Hs = [x.hurst for x in res]  # vector of Hurst
    # Hμ = mean(Hs)
    # Hσ = std(Hs)
    Vs = [x.σ for x in res]  # vector of volatility
    # Vμ = mean(Vs)
    # Vσ = std(Vs)

    # hurst = mean(Hs)
    hurst = median(Hs)

    P = FractionalGaussianNoise(hurst, d)  # fGn with estimated hurst at step d
    gy = t -> t .+ collect(1:n)  # grid of prediction, starting from t
    gx = t -> collect(reverse(t:-u:(t-k*u+1)))

    # covariance matrices
    Σyy = covmat(P, gy(0))
    Σyx = covmat(P, gy(0), gx(0))
    Σxx = covmat(P, gx(0))
    iΣxx = pinv(Matrix(Σxx))
    Σ = Σyy - Σyx * iΣxx * Σyx'

    α::AbstractMatrix = Σyx * iΣxx  # kernels of conditional mean
    # estimation of bias  β = μy - α * μx
    β::AbstractMatrix = if bias
        βm = [X[:,gy(t)]' - α*X[:,gx(t)]' for t=k*u:size(X,2)-n]  # t=(k*u):(length(X)-n)
        mean(βm)
    else
        zeros(size(X,1),n)
    end

    # conditional mean and covariance
    μs = X[:,gx(size(X,2))] * α' + β
    # for r=1:size(X,1)
    #     push!(μs, α*X[r,gx(size(X,2))] + β)
    # end

    return (hurst=Hs, σ=Vs, μ=μs, Σ=Σ, α=α, β=β)
end


function fGn_MLE_estim_predict_old(X::AbstractVector{<:Real}, d::Integer, s::Integer, l::Integer, k::Integer, u::Integer, n::Integer; dmode::Symbol=:regular, kwargs...)
    N = length(X)  # number of samples

    # when sub window equals to rolling window then `rolling_vectorize()` has no effect
    est = fGn_MLE_estim(X, d, s, l; kwargs...)

    # convention of time arrow: from left to right
    sgrid::AbstractVector{<:Integer} = Int[]  # grid of historical samples
    S::AbstractVector{<:Real} = Real[]  # value of historical samples

    if dmode == :logscale  # logscale downsampling
        tidx = findall(reverse(logtrain(N, N÷u)))
        # select the last k samples
        sgrid = (k==0 || k>=length(tidx)) ? tidx : tidx[end-k+1:end]
        S = X[sgrid]
    else # regular downsampling
        Sm = vec2mat(X, u, keep=:tail)
        S0 = mean(Sm, dims=1)[:]
        # S0 = Sm[1,:]
        S = (k==0 || k>=length(S0)) ? S0 : S0[end-k+1:end]
        sgrid = u*(1:length(S))
    end

    # conditional mean and covariance
    pgrid = sgrid[end] .+ (1:n)  # grid of prediction
    prd = cond_mean_cov(FractionalGaussianNoise(est.hurst, d), pgrid, sgrid, S)
    C = prd.C[end,:]

    return (hurst=est.hurst, σ=est.σ, μ=prd.μ, C=C, Σ=prd.Σ)
    # return hurst, σ, μ, σ^2*Σ
end


"""
Power-law estimator and predictor.

# Args
- X: input matrix. The first row is the observation of fBm and the others are fGn
- lags: integer time lags (increment step) used to compute each component of `X` starting from the second row. Values in `lags` must be all distinct.
- s: time lag for prediction
- d: (average) downsampling factor

# Returns
- H, σ, (μr, Vr), (μy, Vy): estimation of hurst, volatility, conditional mean and covariance of fGn and fBm.

# Notes
- This function is intended to be used with `rolling_apply` where `X` is some observation on a time window. - The time arrow is on the second dimension (i.e. horizontal) from left to right.
- μr and

"""
function powlaw_estim_predict(X::AbstractMatrix{<:Real}, lags::AbstractVector{<:Integer}, s::Integer, d::Integer, k::Integer=0; kwargs...)
    # @assert s in lags
    # @assert "all values in lags must be distinct"

    N = size(X, 2)  # number of rows
    Y = view(X, 1, :)  # first row is an observation of fBm
    R = view(X, 2:size(X,1), :)  # the other rows are observations of fGn, corresponding to `lags`
    sidx = findall(lags.==s)[1]  # index of `s` in `lags`

    H, σ, opm = powlaw_estim(R, lags; kwargs...)

    # convention of time arrow: from left to right
    # method 1: logscale downsampling
    tidx0 = findall(reverse(logtrain(N, N÷d)))
    # @assert k <= length(tidx0)
    tidx = k>0 ? tidx0[end-k+1:end] : tidx0  # select the last k samples
    S = R[sidx,tidx]

    # # method 2: regular downsampling with averaging
    # S0 = mean(vec2mat(R[sidx,:], d, keep=:tail), dims=1)[:]
    # # S0 = transpose(vec2mat(R[sidx,:], d, keep=:tail))[1,:]
    # S = k>0 ? S0[end-k+1:end] : S0
    # tidx = d*(1:size(S,1))

    # conditional mean and covariance
    G = tidx[end]+1:tidx[end]+s  # grid of prediction
    # fGn: `S` is the observation of fGn corresponding to `s`
    μr, Σr = cond_mean_cov(FractionalGaussianNoise(H, s), G, tidx, S)
    # fBm: `Y` is the observation of fBm
    μy, Σy = cond_mean_cov(FractionalBrownianMotion(H), G, tidx, Y[tidx])

    return H, σ, (μr, σ^2*Σr), (μy, σ^2*Σy)
end


# """
# EXPERIMENTAL: prediction by recursive conditional mean
# No visible improvements over classical conditional mean for long range prediction.
# """
# function powlaw_estim_predict(X::AbstractMatrix{<:Real}, lags::AbstractVector{<:Integer}, s::Integer, k::Integer; kwargs...)
#     # @assert k>0 && d>0
#     H, σ, opm = FracFin.powlaw_estim(X, lags; kwargs...)
#     l = size(X,2)÷4
#     Cv = cond_mean_coeff(FractionalGaussianNoise(H, lags[s]), k, l)
#     μc = Cv * X[s, end-l+1:end]
#     return H, σ, μc
# end
