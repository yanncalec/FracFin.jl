##### Algebra #####

function col_normalize(A::Matrix, p::Real=2)
    return A / diagm([norm(A[:,n], p) for n=1:size(A,2)])
end

function col_normalize!(A::Matrix, p::Real=2)
    for n=1:size(A,2)
        A[:,n] /= norm(A[:,n], p)
    end
    return A
end

row_normalize(A) = col_normalize(A.')
row_normalize!(A) = col_normalize!(A.')


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
* ϕ, ψ, g: scaling, wavelet function and the associated sampling grid (computed for the Daubechies wavelet)

# References
* https://en.wikipedia.org/wiki/Daubechies_wavelet
* https://en.wikipedia.org/wiki/Cascade_algorithm
* http://cnx.org/content/m10486/latest/
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
    ϕ /= sum(ϕ)
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
        ϕ /= sqrt(ϕ' * ϕ * δ)
        ψ /= sqrt(ψ' * ψ * δ)
    end

    return ϕ, ψ, g
end


"""
Compute the masks for convolution for a mode of truncation.

For a input signal `x` and a kernel `h`, the full convolution x*h has length `length(x)+length(h)-1`.
This function computes two masks:
- `kmask`: corresponds to the left/center/right part in x*h having the length of `x`
- `vmask`: corresponds to the valide coefficients (boundary free) in x*h

Note that `vmask` does not depend on `mode` and `vmask[kmask]` gives the mask of length of `x` corresponding
to the valid coefficients in `kmask`.

# Args
* nx: length of input signal
* nh: length of kernel
* mode: {:left, :right, :center}

# Returns
* kmask, vmask

# Examples
julia> y = conv(x, h)
julia> kmask, vmask = convmask(length(x), length(h), :center)
julia> tmask = vmask[kmask]
julia> y[kmask] # center part, same size as x
julia> y[vmask] # valide part, same as y[kmask][dmask]
"""
function convmask(nx::Int, nh::Int, mode::Symbol)
    kmask = zeros(Bool, nx+nh-1)

    if mode == :left
        kmask[1:nx] = true
    elseif mode == :right
        kmask[nh:end] = true  # or mask0[end-nx+1:end] = true
    elseif mode == :center
        m = max(1, div(nh, 2))
        kmask[m:m+nx-1] = true
    else
        error("Unknown mode: $mode")
    end

    vmask = zeros(Bool, nx+nh-1); vmask[nh:nx] = true
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
* lo: low-pass filter
* hi: high-pass filter
* n: level of decomposition. If n=0 the original filters are returned.

# Return
* a matrix of size ?-by-2^n that each column is a filter.
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

# Return
* a matrix of size ?-by-(level+1) that each column is a filter.
"""
function biconv_filter(lo::Vector{T}, hi::Vector{T}, n::Int) where T<:Number
    @assert n>=0
    F0::Vector{Vector{T}}=[]
    for l=0:n+1
        s = reduce(∗, reduce(∗, [1], [hi for i=1:l]), [lo for i=l+1:n+1])
        push!(F0, s)
    end
    return hcat(F0...)
end


"""
Dyadic scale stationary wavelet transform using à trous algorithm.

# Returns
* ac: matrix of approximation coefficients with increasing scale index in column direction
* dc: matrix of detail coefficients
* mc: mask for valide coefficients
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
Vandermonde matrix.
"""
function vandermonde(dim::Tuple{Int,Int})
    nrow, ncol = dim
    V = zeros(Float64, dim)
    for c=1:dim[2]
        V[:,c] = collect((1:dim[1]).^(c-1))
    end
    return V
end

vandermonde(nrow::Int, ncol::Int) = vandermonde((nrow, ncol))


"""
Continuous wavelet transform based on quadrature.

# Args
* x: input signal
* wfunc: function for evaluation of wavelet at integer scales
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
* k: scale
* ψ: compact wavelet function evaluated on a grid
* Sψ: support of ψ
* v: desired number of vanishing moments of ψ

# Return
f: the vector (ψ(n/k))_n such that n/k lies in Sψ.

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
Integer scale Haar filter.

# Notes
- The true scale is `2k`.
- The filter is not normalized. Normalization comes from the 1/sqrt(k) factor in cwt_quad.
"""
function _intscale_haar_filter(k::Int)
    return vcat(ones(Float64, k), -ones(Float64, k)) / √2  # L2 norm = √k
end


"""
Compute B-Spline filters.

# Args
* v: number of vanishing moments
"""
function bspline_filters(k::Int, v::Int)
    @assert v>0
    lo = vcat(ones(k), ones(k))
    hi = vcat(ones(k),-ones(k))

    # return col_normalize(wpt_filter(lo, hi, v-1)) * sqrt(k)
    return col_normalize(biconv_filter(lo, hi, v-1)) * sqrt(k)
end


"""
B-Spline filter as the auto-convolution of Haar filter.

# Notes
- The true scale is `2k`, like in `_intscale_haar_filter`.
- This filter is not normalized. A trick is used here to find the correct scaling.
"""
function _intscale_bspline_filter(k::Int, v::Int)
    @assert v>0
    hi = vcat(ones(Float64, k), -ones(Float64, k))
    # Analogy of the continuous case:
    # the l^2 norm of the rescaled filter ψ[⋅/k] is √k
    b0 = reduce(∗, [1], [hi for n=1:v])

    # # force even-length kernel
    # if mod(length(b0),2) == 1
    #     b0 = vcat(b0, 0)
    # end

    return normalize(b0) * sqrt(k)  # <- Trick!
end

# function _intscale_bspline_filter_tight(k::Int, v::Int)
#     @assert v>0
#     ko = max(1, ones(k÷2))
#     hi = if k%2 == 1 vcat(ko, -ko) else vcat(ko, 0, -ko) end
#     b0 = reduce(∗, [1], [hi for n=1:v])
#     return normalize(b0) * sqrt(k)  # <- Trick!
# end


"""
Continous Haar transform.
"""
function cwt_haar(x::Vector{Float64}, sclrng::AbstractArray{Int}, mode::Symbol=:center)
    return cwt_quad(x, _intscale_haar_filter, sclrng, mode)
end


"""
Continous B-Spline transform.

# TODO: parallelization
"""
function cwt_bspline(x::Vector{Float64}, sclrng::AbstractArray{Int}, v::Int, mode::Symbol=:center)
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
Evaluate the Fourier transform of B-Spline wavelet.
"""
function _bspline_ft(ω::Real, v::Int)
#     @assert v>0  # check vanishing moment
    # return (ω==0) ? 0 : (2π)^((v-1)/2) * (-(1-exp(1im*ω/2))^2/(√2*1im*ω))^(v)  # non-centered bspline: supported on [0, v]
    return (ω==0) ? 0 : (2π)^((v-1)/2) * (-(1-exp(1im*ω/2))^2/(√2*1im*ω) * exp(-1im*ω/2))^(v)  # centered bspline: supported on [-v/2, v/2]
end


"""
Evaluate the integrand function of G^ψ_{ρ}

# Args
* ω: frequency
* v: vanishing moments
"""
function Gfunc_bspline_integrand(τ::Real, ω::Real, ρ::Real, H::Real, v::Int)
    #     @assert ρ>0
    #     @assert 1>H>0
    s = √ρ
    return (ω==0) ? 0 : real(_bspline_ft(ω*s, v) * conj(_bspline_ft(ω/s, v)) / abs(ω)^(2H+1) * exp(-1im*ω*τ))
end

"""
Expanded and centered version.
"""
function Gfunc_bspline_integrand_expand(τ::Real, ω::Real, ρ::Real, H::Real, v::Int)
    #     @assert ρ>0
    #     @assert 1>H>0
    s = √ρ
    # # Version 1: non centered ψ. This works with convolution mode `:left`` and produces artefacts of radiancy straight lines
    # return (ω==0) ? 0 : (2π)^(v-1) * 2^v * (1-cos(ω*s/2))^v * (1-cos(ω/s/2))^v * cos(ω*v*(s-1/s)/2 - ω*τ) / abs(ω)^(2v+2H+1)
    # Version 2: centered ψ. This works with convolution mode `:center`
    return (ω==0) ? 0 : (2π)^(v-1) * 2^v * (1-cos(ω*s/2))^v * (1-cos(ω/s/2))^v * cos(ω*τ) / abs(ω)^(2v+2H+1)
end

"""
Evaluate the G^ψ_{ρ} function by numerical integration.

# Args
"""
function Gfunc_bspline(τ::Real, ρ::Real, H::Real, v::Int)
    f = ω -> Gfunc_bspline_integrand_expand(τ, ω, ρ, H, v)
    # f = ω -> Gfunc_bspline_integrand(τ, ω, ρ, H, v)

    # res = QuadGK.quadgk(f, -100, 100, order=10)
    res = QuadGK.quadgk(f, -50, 50)
    return res[1]
end


"""
Evaluate G-matrix in DCWT

# Notes
- The true scale is two times scale index due to the special implementation of B-Spline wavelet, see also `_intscale_bspline_filter()`.
- TODO: parallelization!
"""
function Gmat_bspline(H::Real, v::Int, lag::Real, sclrng::AbstractArray)
    # true scale is 2i hence the extra 1/2 factor
    return [Gfunc_bspline(lag/sqrt(i*j)/2, j/i, H, v) for i in sclrng, j in sclrng]
end


#     A = zeros((length(sclrng),length(sclrng)))

#     # Parallelization!
#     for (c,i) in enumerate(sclrng)
#         for (r,j) in enumerate(sclrng)
#             A[r,c] = Cbspline_func(lag/sqrt(i*j), sqrt(i*j), H, v)
#             # f(ω) = Cbspline_intfunc(lag/sqrt(i*j), ω, sqrt(i*j), H, v)
#             # res = QuadGK.quadgk(f, -20, 20)
#             # A[r,c] = res[1]
#         end
#     end
#     return A
# end

