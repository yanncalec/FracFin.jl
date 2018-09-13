
# function scalogram_estim(Cxx::Matrix{Float64}, sclrng::AbstractArray{Int}, ρmax::Int=1)
#     nr, nc = size(Cxx)
#     @assert nr == nc == length(sclrng)
#     @assert ρmax >= 1
#     xvar = Float64[]
#     yvar = Float64[]
#     for ρ=1:ρmax
#         toto = [Cxx[j, ρ*j] for j in 1:nr if ρ*j<=nr]
#         xvar = vcat(xvar, sclrng[1:length(toto)])
#         yvar = vcat(yvar, abs.(toto))
#     end
#     df = DataFrames.DataFrame(
#         xvar=log2.(xvar),
#         yvar=log2.(yvar)
#     )
#     ols_hurst = GLM.lm(@GLM.formula(yvar~xvar), df)
#     hurst_estim = (GLM.coef(ols_hurst)[2]-1)/2    
#     return hurst_estim, ols_hurst
# end


function powlaw_estim_old(X::Vector{Float64}, lags::AbstractArray{Int}, pows::AbstractArray{T}) where {T<:Real}
    # Define the function for computing the p-th moment of the increment
    moment_incr(X,d,p) = mean((abs.(X[d+1:end] - X[1:end-d])).^p)

    # Estimation of Hurst exponent
    Y = zeros(Float64, (length(lags), length(pows)))
    for (n,p) in enumerate(pows)
        Y[:,n] = map(d -> log(moment_incr(X, d, p)), lags)
    end
    dY = diff(Y, 1)
    df = DataFrames.DataFrame(xvar=(diff(log.(lags)) * pows')[:],
                              yvar=dY[:])
    ols_hurst = GLM.lm(@GLM.formula(yvar ~ xvar), df)
    Hurst = GLM.coef(ols_hurst)[2]  # estimation of hurst exponent

    # Estimation of volatility
    # Constant in the p-th moment of normal distribution, see
    # https://en.wikipedia.org/wiki/Normal_distribution#Moments
    cps = [2^(p/2) * gamma((p+1)/2)/sqrt(pi) for p in pows]
    Z = Y -  Hurst * (log.(lags) * pows') - ones(lags) * log.(cps)'
    dg = DataFrames.DataFrame(xvar=(ones(lags) * pows')[:],
                              yvar=Z[:])
    ols_sigma = GLM.lm(@GLM.formula(yvar ~ xvar), dg)

    return Y, Dict('H'=>ols_hurst, 'σ'=>ols_sigma)
end

"""
Extract the estimates of Hurst and voaltility from the result of `powlaw_estim`.
"""
function powlaw_coeff(ols::Dict, h::Float64)
    H = GLM.coef(ols['H'])[2]
    σ = exp(GLM.coef(ols['σ'])[2] - H * log(h))

    return H, σ
end


##### Special functions #####

"""
Compute the continued fraction involved in the upper incomplete gamma function using the modified Lentz's method.
"""
function _uigamma_cf(s::Complex, z::Complex; N=100, epsilon=1e-20)
#     a::Complex = 0
#     b::Complex = 0
#     d::Complex = 0
    u::Complex = s
    v::Complex = 0
    p::Complex = 0

    for n=1:N
#         a, b = (n%2==1) ? ((-div(n-1,2)-s)*z, s+n) : (div(n,2)*z, s+n)
        a, b = (n%2==1) ? ((-div(n-1,2)-s), (s+n)/z) : (div(n,2), (s+n)/z)
        u = b + a / u
        v = 1/(b + a * v)
        d = log(u * v)
        (abs(d) < epsilon) ? break : (p += d)
#         println("$(a), $(b), $(u), $(v), $(d), $(p), $(exp(p))")
    end
    return s * exp(p)
end

doc"""
    uigamma0(z::Complex; N=100, epsilon=1e-20)

Upper incomplete gamma function with vanishing first argument:
$$ \Gamma(0,z) = \lim_{a\rightarrow 0} \Gamma(a,z) $$

Computed using the series expansion of the [exponential integral](https://en.wikipedia.org/wiki/Exponential_integral) $E_1(z)$.
"""
function uigamma0(z::Number; N=100, epsilon=1e-20)
    #     A::Vector{Complex} = [(-z)^k / k / exp(lgamma(k+1)) for k=1:N]
    #     s = sum(A[abs.(A)<epsilon])
    s::Complex = 0
    for k=1:N
        d = (-z)^k / k / exp(lgamma(k+1))
        (abs(d) < epsilon) ? break : (s += d)
    end
    r = -(eulergamma + log(z) + s)
    return (typeof(z) <: Real ? real(r) : r)
end

# """
# Upper incomplete gamma function.
# """
# function uigamma(a::Real, z::T; N=100, epsilon=1e-8) where {T<:Number}
#     z == 0 && return gamma(a)
#     u::T = z
#     v::T = 0
#     f::T = z
# #     f::Complex = log(z)
#     for n=1:N
#         an, bn = (n%2==1) ? (div(n+1,2)-a, z) : (div(n,2), 1)
#         u = bn + an / u
#         v = bn + an * v
#         f *= (u/v)
# #         f += (log(α) - log(β))
#         println("$(an), $(bn), $(u), $(v), $(f)")
#         if abs(u/v-1) < epsilon
#             break
#         end
#     end
#     return z^a * exp(-z) / f
# #     return z^a * exp(-z-f)
# end


doc"""
    uigamma(s::Complex, z::Complex; N=100, epsilon=1e-20)

Upper incomplete gamma function $\Gamma(s,z)$ with complex arguments.

Computed using the [continued fraction representation](http://functions.wolfram.com/06.06.10.0005.01).
The special case $\Gamma(0,z)$ is computed via the series expansion of the exponential integral $E_1(z)$.

# Reference
- [Upper incomplete gamma function](https://en.wikipedia.org/wiki/Incomplete_gamma_function)
- [Continued fraction representation](http://functions.wolfram.com/06.06.10.0005.01)
- [Exponential integral](https://en.wikipedia.org/wiki/Exponential_integral)
"""

function uigamma(s::Number, z::Number; N=100, epsilon=1e-20)
    if abs(s) == 0
        return uigamma0(z; N=N, epsilon=epsilon)
    end

    r = gamma(s) - z^s * exp(-z) / _uigamma_cf(Complex(s), Complex(z); N=N, epsilon=epsilon)
    return (typeof(s)<:Real && typeof(z)<:Real) ? real(r) : r
end

doc"""
    ligamma(s::Complex, z::Complex; N=100, epsilon=1e-20)

Lower incomplete gamma function $\gamma(s,z)$ with complex arguments.
"""
function ligamma(s::Number, z::Number; N=100, epsilon=1e-20)
    return gamma(s) - uigamma(s, z; N=N, epsilon=epsilon)
end



function wavelet_MLE_obj(X::Matrix{Float64}, sclrng::AbstractArray{Int}, v::Int, H::Real, σ::Real; mode::Symbol=:center)
    N, d = size(X)  # length and dim of X
    A = [sqrt(i*j) for i in sclrng, j in sclrng].^(2H+1)
    C1 = [C1rho(0, j/i, H, v, mode) for i in sclrng, j in sclrng]
    
    Σ = σ^2 * C1 .* A
    # Σ += Matrix(1.0I, size(Σ)) * max(1e-8, mean(abs.(Σ))*1e-5)
    # println("H=$H, σ=$σ, det(Σ)=$(det(Σ))")

    # method 1:
    # iX = Σ \ X'  # <- unstable!
    
    # method 2:
    # iΣ = pinv(Σ)  # regularization by pseudo-inverse
    # iX = iΣ * X'

    # method 3:
    iX = lsqr(Σ, X')
    
    return -1/2 * (tr(X*iX) + N*logdet(Σ) + N*d*log(2π))
end


function grad_wavelet_MLE_obj(X::Matrix{Float64}, sclrng::AbstractArray{Int}, v::Int, H::Real, σ::Real; mode::Symbol=:center)
    N, d = size(X)  # length and dim of X
    A = [sqrt(i*j) for i in sclrng, j in sclrng].^(2H+1)
    C1 = [C1rho(0, j/i, H, v, mode) for i in sclrng, j in sclrng]
    dAda = [log(i*j) for i in sclrng, j in sclrng] .* A
    dC1da = [diff_C1rho(0, j/i, H, v, mode) for i in sclrng, j in sclrng]

    Σ = σ^2 * C1 .* A
    # Σ += Matrix(1.0I, size(Σ)) * max(1e-8, mean(abs.(Σ))*1e-5)

    dΣda = σ^2 * (dC1da .* A + C1 .* dAda)
    dΣdb = 2σ * C1 .* A

    # method 1:
    # iX = Σ \ X'
    # da = N * tr(Σ \ dΣda) - tr(iX' * dΣda * iX)
    # db = N * tr(Σ \ dΣdb) - tr(iX' * dΣdb * iX)

    # method 2:
    # iΣ = pinv(Σ)  # regularization by pseudo-inverse
    # iX = iΣ * X'
    # da = N * tr(iΣ * dΣda) - tr(iX' * dΣda * iX)
    # db = N * tr(iΣ * dΣdb) - tr(iX' * dΣdb * iX)

    # method 3:
    iX = lsqr(Σ, X')
    da = N * tr(lsqr(Σ, dΣda)) - tr(iX' * dΣda * iX)
    db = N * tr(lsqr(Σ, dΣdb)) - tr(iX' * dΣdb * iX)

    return  -1/2 * [da, db]
end