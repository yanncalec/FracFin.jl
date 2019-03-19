###### variogram estimator ######

"""
Multifractal estimator.

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
function powlaw_estim(X::AbstractMatrix{<:Real}, lags::AbstractVector{<:Integer}; pow::Real=2., method::Symbol=:lm, reweight::Bool=true)
    # remove columns containing NaN
    idx = findall(vec(.!any(isnan.(X), dims=1)))
    X = X[:,idx] # view(X,:,idx)

    if length(X) == 0
        return (hurst=NaN, σ=NaN, η=NaN, residual=NaN, vars=nothing, optimizer=nothing)
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
        # Note that the reweighting scheme is exact only for pow=2.
        dg = DataFrames.DataFrame(xvar=xp, yvar=yp)
        opm = GLM.lm(@GLM.formula(yvar~xvar), dg)
        η, hurst = GLM.coef(opm)  # intercept and slope
        res = GLM.deviance(opm)  # residual
        # err = try  # std error of estimates
        #     GLM.stderror(opm)
        # catch
        #     (NaN, NaN)
        # end

        ws =  if reweight
            variogram_variance(0 < hurst < 1 ? hurst : 0.5, lags, size(X,2)) .^ (-1)
            # variogram_variance(0.5, lags, size(X,2)) .^ -1  # in practice this is good enough?
        else
            ones(length(lags))  # uniform weight
        end
        ws ./= sum(ws)

        # estimation of H and η
        if method == :optim
            yc = yp .- mean(yp)
            xc = xp .- mean(xp)
            func = h -> sum(ws .* (yc - h*xc).^2)
            # func = h -> sum(ws .* abs.(yc - h*xc).^qnorm)
            # Gradient-free constrained optimization
            # ɛ = 1e-2  # search hurst in the interval [ɛ, 1-ɛ]
            opm = Optim.optimize(func, 0., 1., Optim.Brent())
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
            res = sqrt(GLM.deviance(opm) / var(yp)) # / length(xp)

            # # or equivalently, by manual inversion
            # Ap = hcat(xp, ones(length(xp))) # design matrix
            # hurst, η = Ap \ yp
        else
            error("Unknown method $(method).")
        end

        cp = normal_moment_factor(pow)  # constant factor depending on p
        σ = exp((η-log(cp))/pow)

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

const variogram_estim = powlaw_estim

