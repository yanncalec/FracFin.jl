##### Algebra #####

function norm(X::AbstractMatrix, p::Real=2; dims::Int=1)
    if dims==1
        return [LinearAlgebra.norm(X[:,n],p) for n=1:size(X,2)]
    else
        return [LinearAlgebra.norm(X[n,:],p) for n=1:size(X,1)]
    end
end


function pinv_iter(A::AbstractMatrix, method::Symbol=:lsqr)
    iA = zeros(Float64, size(A'))
    try
        iA = pinv(A)
    catch
        for c = 1:size(iA, 2)
            b = zeros(Float64, size(A,1))
            b[c] = 1.
            iA[:,c] = IterativeSolvers.lsqr(A, b)
        end
    end
    return iA
end


"""
Compute A^-1 * B using the lsqr iterative method.
"""
function lsqr(A::AbstractMatrix{T}, B::AbstractMatrix{T}; kwargs...) where {T<:Real}
    # println("My lsqr")
    X = zeros(Float64, (size(A,2), size(B,2)))
    for n=1:size(B,2)
        X[:,n] = lsqr(A, B[:,n])
    end
    return X
end

"""
    LevinsonDurbin(cseq::Vector{Float64})

Diagonalization of a symmetric positive definite Toeplitz matrix using Levinson-Durbin (LD) method by providing `cseq`,
the covariance sequence of a stationary process.

# Returns
- `pseq::Vector{Vector{Float64}}`: linear prediction coefficients
- `sseq`: variances of residual
- `rseq`: partial correlation coefficients

# Explanation
`pseq` forms the lower triangular matrix diagonalizing the covariance matrix Γ, and `sseq` forms the resulting diagonal matrix. `rseq[n]` is just `pseq[n][n]`.
"""
function LevinsonDurbin(cseq::Vector{Float64})
    N = length(cseq)

    if N > 1
        # check that cseq is a validate covariance sequence
        @assert cseq[1] > 0
        @assert all(abs.(cseq[2:end]) .<= cseq[1])
        # @assert all(diff(abs.(cseq)) .<= 0)

        # initialization
        # pseq: linear prediction coefficients
        pseq = Vector{Vector{Float64}}(N-1); pseq[1] = [cseq[2]/cseq[1]]
        # sseq: variances of residual
        sseq = zeros(N); sseq[1] = cseq[1]; sseq[2] = (1-pseq[1][1]^2) * sseq[1]
        # rseq: partial correlation coefficients
        rseq = zeros(N-1); rseq[1] = pseq[1][1]

        # recursive construction of the prediction coefficients and variances
        for n=2:N-1
            pseq[n] = zeros(n)
            pseq[n][n] = (cseq[n+1] - cseq[2:n]' * pseq[n-1][end:-1:1]) / sseq[n]
            pseq[n][1:n-1] = pseq[n-1] - pseq[n][n] * pseq[n-1][end:-1:1]
            sseq[n+1] = (1 - pseq[n][n]^2) * sseq[n]
            rseq[n] = pseq[n][n]
        end
    else
        pseq = Vector{Float64}[]
        sseq = copy(cseq)
        rseq = Float64[]
    end
    return pseq, sseq, rseq
end

LevinsonDurbin(p::StationaryProcess{T}, g::RegularGrid{<:T}) where T = LevinsonDurbin(covseq(p, g))


"""
Cholesky decomposition based on SVD.
"""
function chol_svd(W::Matrix{Float64})
    Um, Sm, Vm = svd((W+W')/2)  # svd of forced symmetric matrix
    Ss = sqrt.(Sm[Sm.>0])  # truncation of negative singular values
    return Um*diagm(Ss)
end


"""
Vandermonde matrix.
"""
function vandermonde(dim::Tuple{Int,Int})
    nrow, ncol = dim
    V = zeros(Float64, dim)
    for c = 1:dim[2]
        V[:,c] = collect((1:dim[1]).^(c-1))
    end
    return V
end

vandermonde(nrow::Int, ncol::Int) = vandermonde((nrow, ncol))

function col_normalize(A::Matrix, p::Real=2)
    return A / diagm([norm(A[:,n], p) for n=1:size(A,2)])
end

function col_normalize!(A::Matrix, p::Real=2)
    for n=1:size(A,2)
        A[:,n] ./= norm(A[:,n], p)
    end
    return A
end

row_normalize(A) = col_normalize(transpose(A))
row_normalize!(A) = col_normalize!(transpose(A))

##### Useful functions #####

"""
Sigmoid function.
"""
sigmoid(α::Real) = exp(α)/(1+exp(α))

"""
Derivative of sigmoid function.
"""
diff_sigmoid(α::Real) = exp(α)/(1+2*exp(α)+exp(2α))


##### Wavelet transform #####

# Normalizations of Wavelets.jl for different families are not coherent:
# Assuming the sqrt(2) factor in the cascade algorithm (see wavefun), then the filters of the following
# family have to be rescaled by the corresponding factor
# - Daubechies: 1
# - Coiflet: 1
# - Symlet: 1/sqrt(2)
# - Battle-Lemarie: sqrt(2)
# - Beylkin: 1
# - Vaidyanathan: 1

"""
Construct the matrix of a convolution kernel h.
"""
function convolution_matrix(h::Vector{T}, N::Int) where T
    @assert N>0
    M = length(h)
    K = zeros(T, (M+N-1, M+N-1))
    for r=1:M
        for d=1:M+N-r
            K[d+r-1,d] = h[r]
        end
    end
    return K[:,1:N]
end


"""
Compute the scaling and the wavelet function using the cascade algorithm.

The implementation follows the reference 2, but with a modified initialization.

# Returns
- ϕ, ψ, g: scaling, wavelet function and the associated sampling grid (computed for the Daubechies wavelet)

# References
- https://en.wikipedia.org/wiki/Daubechies_wavelet
- https://en.wikipedia.org/wiki/Cascade_algorithm
- http://cnx.org/content/m10486/latest/
"""
function wavefunc(lo::Vector{Float64}, hi::Vector{Float64}=Float64[]; level::Int=10, nflag::Bool=true)
    if isempty(hi)
        hi = (lo .* (-1).^(1:length(lo)))[end:-1:1]
    else
        length(lo)==length(hi) || error("Invalid high-pass filter.")
    end

    # Initialization of the cascade algorithm
    # Method 1: using constant 1, this gives the best results (criteria of orthogonality etc.)
    ϕ = [1]
#
#     # Method 2: using the original filter
#     ϕ = copy(lo)
#
#     # Method 3: take one specific eigen vector of the decimated convolution matrix, see Reference 2.
#     K = convolution_matrix(lo, length(lo))[1:2:end, :]
#     μ, V = eig(K)  # such that (K * V) - V * diagm(μ) = 0
#     idx = find(abs.(μ - 1/sqrt(2)) .< 1e-3)[1]
#     ϕ = real(V[:, idx])

    # Normalization: this is necessary to get the correct numerical range
    ϕ ./= sum(ϕ)
    ψ = Float64[]

    fct = sqrt(2)  # scaling factor

    # Iteration of the cascade algorithm
    for n = 1:level
        # up-sampling of low-pass filter
        s = 2^(n-1)
        l = (length(lo)-1) * s + 1
        lo_up = zeros(Float64, l)
        lo_up[1:s:end] = lo

        if n==level
            # Last iteration only
            # up-sampling of high-pass filter
            hi_up = zeros(Float64, l)
            hi_up[1:s:end] = hi
            ψ = conv(hi_up, ϕ) * fct
        end
        ϕ = conv(lo_up, ϕ) * fct
    end

    # sampling grid:
    # the Daubechies wavelet of N vanishing moments has support [0, 2N-1] and its qmf filter has length 2N
    g = (length(lo)-1) * collect(0:(length(ϕ)-1))/length(ϕ)

    if nflag # force unit norm
        δ = g[2]-g[1]  # step of the sampling grid
        ϕ ./= sqrt(ϕ' * ϕ * δ)
        ψ ./= sqrt(ψ' * ψ * δ)
    end

    return ϕ, ψ, g
end


"""
Compute the masks for convolution for a mode of truncation.

For a input signal `x` and a kernel `h`, the full convolution x*h has length `length(x)+length(h)-1`. This function computes two masks:
- `kmask`: corresponds to the left/center/right part in x*h having the length of `x`
- `vmask`: corresponds to the valide coefficients (boundary free) in x*h

Note that `vmask` does not depend on `mode` and `vmask[kmask]` gives the mask of length of `x` corresponding to the valid coefficients in `kmask`.

# Args
- nx: length of input signal
- nh: length of kernel
- mode: {:left, :right, :center}

# Returns
- kmask, vmask

# Examples
```julia-repl
julia> y = conv(x, h)
julia> kmask, vmask = convmask(length(x), length(h), :center)
julia> tmask = vmask[kmask]
julia> y[kmask] # center part, same size as x
julia> y[vmask] # valide part, same as y[kmask][dmask]
```
"""
function convmask(nx::Int, nh::Int, mode::Symbol)
    kmask = zeros(Bool, nx+nh-1)

    if mode == :left
        kmask[1:nx] .= true
    elseif mode == :right
        kmask[nh:end] .= true  # or mask0[end-nx+1:end] = true
    elseif mode == :center
        m = max(1, div(nh, 2))
        kmask[m:m+nx-1] .= true
    else
        error("Unknown mode: $mode")
    end

    vmask = zeros(Bool, nx+nh-1); vmask[nh:nx] .= true
    return kmask, vmask
end


"""
Down-sampling operator.
"""
function downsampling(x::AbstractArray{<:Any, 1}, s::Int=2)
    return x[1:s:end]
end

"""
Up-sampling operator.
"""
function upsampling(x::AbstractArray{<:Any, 1}, s::Int=2; tight::Bool=true)
    y = zeros(length(x)*s)
    y[1:s:end] = x
    return tight ? y[1:(end-(s-1))] : y
end

↑ = upsampling  # \uparrow
↓ = downsampling  # \downarrow

# ∗(x,y) = conv(x,y[end:-1:1])  # correlation, \ast
∗(x,y) = conv(x,y)  # convolution, \ast
⊛(x,y) = ↓(x ∗ ↑(y, 2, tight=true), 2)  # up-down convolution, \circledast


"""
Compute filters of Wavelet Packet transform.

# Args
- lo: low-pass filter
- hi: high-pass filter
- n: level of decomposition. If n=0 the original filters are returned.

# Returns
- a matrix of size ?-by-2^n that each column is a filter.
"""
function wpt_filter(lo::Vector{T}, hi::Vector{T}, n::Int) where T<:Number
    F0::Vector{Vector{T}}=[[1]]
    for l=1:n+1
        F1::Vector{Vector{T}}=[]
        for f in F0
            push!(F1, f ⊛ lo)
            push!(F1, f ⊛ hi)
        end
        F0 = F1
    end
    return hcat(F0...)
end


"""
N-fold convolution of two filters.

Compute
    x_0 ∗ x_1 ∗ ... x_{n-1}
where x_i ∈ {lo, hi}, i.e. either the low or the high filter.

# Returns
- a matrix of size ?-by-(level+1) that each column is a filter.
"""
function biconv_filter(lo::Vector{T}, hi::Vector{T}, n::Int) where T<:Number
    @assert n>=0
    F0::Vector{Vector{T}}=[]
    for l=0:n+1
        s = reduce(∗, [lo for i=l+1:n+1]; init=reduce(∗, [hi for i=1:l]))
        push!(F0, s)
    end
    return hcat(F0...)
end


"""
Dyadic scale stationary wavelet transform using à trous algorithm.

# Returns
- ac: matrix of approximation coefficients with increasing scale index in column direction
- dc: matrix of detail coefficients
- mc: mask for valide coefficients
"""
function swt(x::Vector{Float64}, level::Int, lo::Vector{Float64}, hi::Vector{Float64}=Float64[];
        mode::Symbol=:center)
    @assert level > 0

    # if high pass filter is not given, use the qmf.
    if isempty(hi)
        hi = (lo .* (-1).^(1:length(lo)))[end:-1:1]
    else
        @assert length(lo) == length(hi)
    end

    nx = length(x)
    fct = 1  # unlike in the à trous algorithm of `wavefunc`, here the scaling factor must be 1.
    ac = zeros(Float64, (length(x), level))
    dc = zeros(Float64, (length(x), level))
    mc = zeros(Bool, (length(x), level))

    # Finest level transform
    nk = length(lo)
    km, vm = convmask(nx, nk, mode)
    xd = conv(hi, x) * fct
    xa = conv(lo, x) * fct
    ac[:,1], dc[:,1], mc[:,1] = xa[km], xd[km], vm[km]

    # Iteration of the cascade algorithm
    for n = 2:level
        # up-sampling of qmf filters
        s = 2^(n-1)
        l = (length(lo)-1) * s + 1
        lo_up = zeros(Float64, l)
        lo_up[1:s:end] = lo
        hi_up = zeros(Float64, l)
        hi_up[1:s:end] = hi
        nk += l-1  # actual kernel length

        km, vm = convmask(nx, nk, mode)
        xd = conv(hi_up, xa) * fct
        xa = conv(lo_up, xa) * fct
        ac[:,n], dc[:,n], mc[:,n] = xa[km], xd[km], vm[km]
    end

    return ac, dc, mc
end


"""
Continuous wavelet transform based on quadrature.

# Args
- x: input signal
- wfunc: function for evaluation of wavelet at integer scales
"""
function cwt_quad(x::Vector{Float64}, wfunc::Function, sclrng::AbstractArray{Int}, mode::Symbol=:center)
    Ns = length(sclrng)
    Nx = length(x)

    dc = zeros((Nx, Ns))
    mc = zeros(Bool, (Nx, Ns))

    for (n,k) in enumerate(sclrng)
        f = wfunc(k)
        km, vm = convmask(Nx, length(f), mode)

        Y = conv(x, f[end:-1:1])
        dc[:,n] = Y[km] / sqrt(k)
        mc[:,n] = vm[km]
    end
    return dc, mc
end


"""
Evaluate the wavelet function at integer scales by looking-up table.

# Args
- k: scale
- ψ: values of compact wavelet function on a grid
- Sψ: support of ψ
- v: desired number of vanishing moments of ψ

# Returns
- f: the vector (ψ(n/k))_n such that n/k lies in Sψ.

# Note
For accuracy, increase the density of grid for pre-evaluation of ψ.
"""
function _intscale_wavelet_filter(k::Int, ψ::Vector{Float64}, Sψ::Tuple{Real,Real}, v::Int=0)
    # @assert k > 0
    # @assert Sψ[2] > Sψ[1]

    Nψ = length(ψ)
    dh = (Sψ[2]-Sψ[1])/Nψ  # sampling step
    # @assert k < 1/dh  # upper bound of scale range

    Imin, Imax = ceil(Int, k*Sψ[1]), floor(Int, k*Sψ[2])
    idx = [max(1, min(Nψ, floor(Int, n/k/dh))) for n in Imin:Imax]
    f::Vector{Float64} = ψ[idx]

    # Forcing vanishing moments: necessary to avoid inhomogenity due to sampling of ψ
    # Projection onto the kernel of a under-determined Vandermonde matrix:
    if v>0
        V = vandermonde((length(f), v))'
        f -= V\(V*f)
    end

    return f
end


"""
Evaluate the wavelet function at integer scales.
"""
function _intscale_wavelet_filter(k::Int, ψ::Function, Sψ::Tuple{Real,Real}, v::Int=0)
    # @assert k > 0
    # @assert Sψ[2] > Sψ[1]

    Imin, Imax = ceil(Int, k*Sψ[1]), floor(Int, k*Sψ[2])
    f::Vector{Float64} = ψ.((Imin:Imax)/k)

    # Forcing vanishing moments
    if v>0
        V = vandermonde((length(f), v))'
        f -= V\(V*f)
    end

    return f
end


"""
Integer (even) scale Haar filter.

The original Haar wavelet takes value 1 on [0,1/2) and -1 on [1/2, 1) and 0 elsewhere.
"""
function _intscale_haar_filter(scl::Int)
    @assert scl > 0 && iseven(scl)

    k::Int = div(scl,2)
    return vcat(ones(Float64, k), -ones(Float64, k))
end


# """
# Compute B-Spline filters of `v` vanishing moments at scale `2k`.
# """
# function bspline_filters(k::Int, v::Int)
#     @assert v>0
#     lo = vcat(ones(k), ones(k))
#     hi = vcat(ones(k),-ones(k))
#     # return col_normalize(wpt_filter(lo, hi, v-1)) * sqrt(k)
#     return col_normalize(biconv_filter(lo, hi, v-1)) * sqrt(k)
# end


"""
Integer (even) scale B-Spline filter.

This B-Spline filter is defined as the auto-convolution of Haar filter.

# Notes
- The true scale is `2k`, like in `_intscale_haar_filter`.
- A trick can be used to improve the scaling law at fine scales, but this corrupts the intercept with a unknown factor in the linear regression.
"""
function _intscale_bspline_filter(scl::Int, v::Int)
    @assert scl > 0 && iseven(scl)
    @assert v>0

    k::Int = div(scl,2)
    hi = vcat(ones(Float64, k), -ones(Float64, k))
    b0 = reduce(∗, [hi for n=1:v]) / (2k)^(v-1)

    return b0  # <-- without forced scaling.
    # return normalize(b0) * sqrt(2k)  # <-- trick: forced scaling.
end


"""
Continous Haar transform.
"""
function cwt_haar(x::Vector{Float64}, sclrng::AbstractArray{Int}, mode::Symbol=:center)
    all(iseven.(sclrng)) || error("Only even integer scale is admitted.")

    return cwt_quad(x, _intscale_haar_filter, sclrng, mode)
end


"""
    cwt_bspline(x::Vector{Float64}, sclrng::AbstractArray{Int}, v::Int, mode::Symbol=:center)

Continous B-Spline transform at integer (even) scales.

# Args
- x: Input vector
- sclrng: vector of integer scales, all numbers must be even
- v: vanishing moment
- mode: mode of convolution

# TODO: parallelization
"""
function cwt_bspline(x::Vector{Float64}, sclrng::AbstractArray{Int}, v::Int, mode::Symbol=:center)
    all(iseven.(sclrng)) || error("Only even integer scale is admitted.")

    # bsfilter = k->normalize(_intscale_bspline_filter(k, v))
    bsfilter = k->_intscale_bspline_filter(k, v)
    return cwt_quad(x, bsfilter, sclrng, mode)
end


mexhat(t::Real) = -exp(-t^2) * (4t^2-2t) / (2*sqrt(2π))

function _intscale_mexhat_filter(k::Int)
    return _intscale_wavelet_filter(k, mexhat, (-5.,5.), 2)  # Mexhat has two vanishing moments
end

"""
Continous Mexican hat transform
"""
function cwt_mexhat(x::Vector{Float64}, sclrng::AbstractArray{Int}, mode::Symbol=:center)
    return cwt_quad(x, _intscale_mexhat_filter, sclrng, mode)
end


"""
Evaluate the Fourier transform of a centered B-Spline wavelet.
"""
function _bspline_ft(ω::Real, v::Int)
#     @assert v>0  # check vanishing moment

    # # Version 1: non centered ψ. This works with convolution mode `:left`` and produces artefacts of radiancy straight lines.
    # return (ω==0) ? 0 : (2π)^((v-1)/2) * (-(1-exp(1im*ω/2))^2/(sqrt(2π)*1im*ω))^(v)  # non-centered bspline: supported on [0, v]

    # Version 2: centered ψ. This works with convolution mode `:center` which is non-causal add produces artefacts of radial circles.
    return (ω==0) ? 0 : (2π)^((v-1)/2) * (-(1-exp(1im*ω/2))^2/(sqrt(2π)*1im*ω) * exp(-1im*ω/2))^(v)  # centered bspline: supported on [-v/2, v/2]
end


#### mBm wavelet analysis ####

"""
The integrand function of G^ψ_ρ(τ, H) with a centered B-spline wavelet.
"""
function Gfunc_bspline_integrand_center(τ::Real, ω::Real, ρ::Real, H::Real, v::Int)
    return 1/16^v * 1/2π * (ω^2)^(v-(H+1/2)) * (sinc(ω*√ρ/4π)*sinc(ω/√ρ/4π))^(2v) * cos(ω*τ)
end

"""
Evaluate the integrand function of G^ψ_ρ(τ, H)

# Args
- τ, ω, ρ, H: see definition of G^ψ_ρ(τ, H).
- v: vanishing moments of the Bspline wavelet ψ, e.g. v=1 for Haar wavelet.

# Note
- We use the fact that G^ψ_ρ(τ, H) is a real function to simplify the implementation.
- In Julia the sinc function is defined as `sinc(x)=sin(πx)/(πx)`.
"""
function Gfunc_bspline_integrand(τ::Real, ω::Real, ρ::Real, H::Real, v::Int, mode::Symbol)
    # @assert ρ>0 && 1>H>0 && v>0

    # The integrand is, by definition
    # _bspline_ft(ω*√ρ, v) * conj(_bspline_ft(ω/√ρ, v)) / abs(ω)^(2H+1) * exp(-1im*ω*τ)
    # this should be modulated by
    #    exp(-1im*ω*v*(√ρ-1/√ρ)/2), if the convolution mode is :left, i.e. causal
    # or exp(+1im*ω*v*(√ρ-1/√ρ)/2), if the convolution mode is :right, i.e. anti-causal
    # However, such implementation is numerically unstable due to singularities. We rewrite the function in an equivalent form using the sinc function which is numerically stable.
    if mode == :center
        return Gfunc_bspline_integrand_center(τ, ω, ρ, H, v)
    elseif mode == :left
        return Gfunc_bspline_integrand_center(τ+v*(√ρ-1/√ρ)/2, ω, ρ, H, v)
    elseif mode == :right
        return Gfunc_bspline_integrand_center(τ-v*(√ρ-1/√ρ)/2, ω, ρ, H, v)
    else
        throw(UndefRefError("Unknown mode: $(mode)"))
    end
end

"""
Evaluate the function G^ψ_ρ(τ,H) by numerical integration.
"""
function Gfunc_bspline(τ::Real, ρ::Real, H::Real, v::Int, mode::Symbol; rng::Tuple{Real, Real}=(-50, 50))
    f = ω -> Gfunc_bspline_integrand(τ, ω, ρ, H, v, mode)
    # println("τ=$τ, ρ=$ρ, H=$H, v=$v") #

    res = QuadGK.quadgk(f, rng...)[1]
    # res = 1e-4 * sum(f.(rng[1]:1e-4:rng[2]))

    return res
end

"""
Derivative w.r.t. H
"""
function diff_Gfunc_bspline(τ::Real, ρ::Real, H::Real, v::Int, mode::Symbol; rng::Tuple{Real, Real}=(-50, 50))
    f = ω -> ((ω==0) ? 0 : (-log(ω^2) * Gfunc_bspline_integrand(τ, ω, ρ, H, v, mode)))
    res = QuadGK.quadgk(f, rng...)[1]
    # res = try
    #     # QuadGK.quadgk(f, -100, 100, order=10)[1]
    #     QuadGK.quadgk(f, rng...)[1]
    # catch
    #     1e-3 * sum(f.(rng[1]:1e-3:rng[2]))
    # end
    return res
end

"""
Evaluate G-matrix in DCWT

# Notes
- The true scale is two times the scale index due to the special implementation of B-Spline wavelet, see also `_intscale_bspline_filter()`.

# TODO
- parallelization
"""
function Gmat_bspline(H::Real, v::Int, lag::Real, sclrng::AbstractArray{Int}, mode::Symbol)
    all(iseven.(sclrng)) || error("Only even integer scale is admitted.")
    return [Gfunc_bspline(lag/sqrt(i*j), j/i, H, v, mode) for i in sclrng, j in sclrng]
end

"""
Function C^1_ρ(τ, H)

# Args
- τ, ρ, H: see definition
- v: vanishing moments of the wavelet ψ
- mode: {:left, :center, :right} for causal, centered, anti-causal ψ
"""
C1rho(τ::Real, ρ::Real, H::Real, v::Int, mode::Symbol) = gamma(2H+1) * sin(π*H) * Gfunc_bspline(τ, ρ, H, v, mode)

diff_gamma = x -> ForwardDiff.derivative(gamma, x)

"""
Derivative w.r.t. H
"""
function diff_C1rho(τ::Real, ρ::Real, H::Real, v::Int, mode::Symbol)
    d1 = 2 * diff_gamma(2H+1) * sin(π*H) * Gfunc_bspline(τ, ρ, H, v, mode)
    d2 = gamma(2H+1) * cos(π*H) * π * Gfunc_bspline(τ, ρ, H, v, mode)
    # d3 = gamma(2H+1) * sin(π*H) * ForwardDiff.derivative(H->Gfunc_bspline(τ, ρ, H, v, mode), H)
    d3 = gamma(2H+1) * sin(π*H) * diff_Gfunc_bspline(τ, ρ, H, v, mode)
    return d1 + d2 + d3
end


#### Statistics ####

function cov(X::AbstractVecOrMat, Y::AbstractVecOrMat, w::StatsBase.AbstractWeights)
    # w is always a column vector, by definition of AbstractWeights
    @assert size(X, 1) == size(Y, 1) == length(w)
    # weighted mean
    mX = mean(X, w, 1)
    mY = mean(Y, w, 1)
    return (X .- mX)' * (w .* (Y .- mY)) / sum(w)
end


"""
    multi_linear_regression_colwise(Y::Matrix{Float64}, X::Matrix{Float64}, w::StatsBase.AbstractWeights)

Multi-linear regression of data matrix Y versus X in the column direction, i.e. each row in Y is an observation.
"""
function multi_linear_regression_colwise(Y::Matrix{Float64}, X::Matrix{Float64}, w::StatsBase.AbstractWeights)
    @assert size(Y,1)==size(X,1)==length(w)

    μy = mean(Y, w, 1)[:]  # do not take keyword argument `dims=1` if weight is passed.
    μx = mean(X, w, 1)[:]
    Σyx = cov(Y, X, w)  # this calls user defined cov function
    Σxx = cov(X, X, w)
    A::Matrix{Float64} = Σyx / Σxx  # i.e., Σyx * inv(Σxx)
    β::Vector{Float64} = μy - A * μx
    E = Y - (A * X' .+ β)'
    Σ = cov(E, E, w)
    return (A, β), E, Σ
end

function multi_linear_regression(Y::Matrix{Float64}, X::Matrix{Float64}, w::StatsBase.AbstractWeights; dims::Int=1)
    if dims==1
        return multi_linear_regression_colwise(Y, X, w)
    else
        return multi_linear_regression_colwise(Y', X', w)
    end
end

multi_linear_regression(Y::Matrix{Float64}, X::Matrix{Float64}; dims::Int=1) =
    multi_linear_regression(Y, X, StatsBase.weights(ones(size(Y,1))); dims=dims)

"""
value of regularization constant
"""
function IRLS(Y::Matrix{Float64}, X::Matrix{Float64}, pnorm::Real=2.; maxiter::Int=10^3, tol::Float64=10^-3, vreg::Float64=1e-8)
    @assert pnorm > 0

    wfunc = E -> (sqrt.(sum(E.*E, dims=2) .+ vreg).^(pnorm-2))[:]  # function for computing weight vector
    (A, β), E, Σ = multi_linear_regression(Y, X)  # initialization
    w0::Vector{Float64} =  wfunc(E) # weight vector
    n::Int = 1
    err::Float64 = 0.

    for n=1:maxiter
        (A, β), E, Σ = multi_linear_regression(Y, X, StatsBase.weights(w0))
        w = wfunc(E)
        err = norm(w - w0) / norm(w0)
        w0 = w
        if  err < tol
            break
        end
    end
    println(n)
    println(err)

    return (A, β), w0, E, sum(sqrt.(sum(E.*E, dims=2)).^pnorm)
end
