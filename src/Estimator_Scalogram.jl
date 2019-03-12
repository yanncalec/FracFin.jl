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
function bspline_scalogram_estim(X::AbstractMatrix{<:Real}, sclrng::AbstractVector{<:Integer}, vm::Integer; pow::Real=2., method::Symbol=:optim, cmode::Symbol=:causal)
    # remove columns containing NaN
    idx = findall(vec(.!any(isnan.(X), dims=1)))
    X = X[:,idx] # view(X,:,idx)

    if length(X) == 0
        return (hurst=NaN, σ=NaN, η=NaN, residual=NaN, vars=nothing, optimizer=nothing)
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
            opm = Optim.optimize(func, 0.5+ɛ, 1.5-ɛ, Optim.Brent())
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
        σ = try
            A = Aψρ_bspline(0, 1, hurst, vm, cmode)  # kwargs: mode=:center
            exp((η - log(cp) - log(A)*pow/2)/pow)
        catch
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

