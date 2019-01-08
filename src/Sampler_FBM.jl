######## Sampler for pure fBm and related processes ########

#### Cholesky ####
"""
Cholesky sampler for general Gaussian process.

# Notes
- This method generates a sample trajectory on any type of sampling grid (regular or arbitrary) adapted to the underlying process.
"""
struct CholeskySampler{T, P, G} <: Sampler{T, P, G}
    proc::P  # instance of the stochastic process
    grid::G  # grid in use for sampling
    cmat::AbstractMatrix  # covariance matrix
    lmat::AbstractMatrix  # lower-triangular matrix: lmat * lmat' = cmat

    function CholeskySampler{T, P, G}(p::P, g::G) where {T, P, G}
        # check the grid for fBM
        if (P<:FractionalBrownianMotion) && (0 in g)
            error("The sampling grid must not contain the origin.")
        end
        # construct the auto-covariance matrix
        cmat = covmat(p, g)
        # cholesky decomposition yields lower and upper triangular matrices L, U such that L*U = cmat, and L=U'
        # lmat = cholesky(cmat).U'
        lmat = cholesky(cmat).L
        return new(p, g, cmat, lmat)
    end
end

# outer constructor as a shortcut
# CholeskySampler(p::P, g::G) where {P, G} = CholeskySampler{typeof(p), typeof(g)}(p, g)
CholeskySampler(p::StochasticProcess{T}, g::AbstractVector{<:T}) where T<:TimeStyle = CholeskySampler{T, typeof(p), typeof(g)}(p, g)

function rand!(x::Vector{<:AbstractFloat}, s::CholeskySampler)
    return copyto!(x, s.lmat * randn(length(s)))
end


#### Circulant Embedding ####
"""
Circulant embedding sampler for stationary Gaussian process.

# Notes
- This method generates a sample trajectory on any sampling grid adapted to the underlying process.
"""
struct CircSampler{T, P<:StationaryProcess{T}, G<:AbstractVector{<:T}} <: Sampler{T, P, G}
    proc::P  # instance of the stochastic process
    grid::G  # grid in use for sampling
    cseq::AbstractVector  # covariance sequence
    fseq::AbstractVector  # square-root of the Fourier coefficients of cseq

    function CircSampler{T, P, G}(p::P, g::G) where {T, P, G}
        @assert isregulargrid(g)  # only works on regular (continuous or discrete) grid

        # Nf = 1 << ceil(Integer, log2(length(p)))  # length of FFT, equivalent to 2^()
        c = covseq(p, g)
        # minimal circulant embedding
        cm = vcat(c, c[end-1:-1:2])  # mirror vector
        M = length(cm)
        # fft() in Julia is not normalized
        # cf = real(fft(cm)/sqrt(M))  # the imaginary part is close to zero
        cf = real(fft(cm))  # the imaginary part is close to zero
        # check the non-negative constraint
        idx = cf .< 0.
        any(idx) && @warn("Negative eigenvalues encountered, using Wood-Chan approximation.")
        cf[idx] .= 0.
        new(p, g, c, sqrt.(cf))
    end
end

CircSampler(p::StationaryProcess{T}, g::AbstractVector{<:T}) where T = CircSampler{T, typeof(p), typeof(g)}(p, g)

# function rand!(x::Vector{Complex128}, s::CircSampler{P, G}) where {P, G}
function rand!(x::Vector{<:AbstractFloat}, s::CircSampler)
    # @assert length(x) <= length(s)
    M = length(s.fseq)
    z = randn(M) + im * randn(M)
    # y = real(ifft(s.fseq .* fft(z)))
    y = real(ifft(s.fseq .* z) * sqrt(M))
    return copyto!(x, y[1:length(x)])
end

"""
Functional implementation of the circulant embedding method. It takes the sequence of covariance and return a sample path of the same length.
"""
function circulant_embedding(c::Vector{<:AbstractFloat})
    cm = vcat(c, c[end-1:-1:2])  # mirror vector
    M = length(cm)
    # fft() in Julia is not normalized
    # cf = real(fft(cm)/sqrt(M))  # the imaginary part is close to zero
    cf = real(fft(cm))  # the imaginary part is close to zero
    # check the non-negative constraint
    idx = cf .< 0.
#     any(idx) && warn("Negative eigenvalues encountered, using Wood-Chan approximation.")
#         println("length(idx)=$(length(idx)), cf[idx]=$(cf[idx])")
    cf[idx] = 0.
    fseq = sqrt.(cf)
    z = randn(M) + im * randn(M)
    # y = real(ifft(s.fseq .* fft(z)))
    return real(ifft(fseq .* z) * sqrt(M))[1:length(c)]
end


#### Hosking ####
"""
Hosking (Levinson-Durbin) sampler for stationary Gaussian process.

# Notes
* This method draws the sample trajectory on any sampling grid adapted to the underlying process.
"""
struct HoskingSampler{T, P<:StationaryProcess{T}, G<:AbstractVector{<:T}} <: Sampler{T, P, G}
    proc::P  # instance of the stochastic process
    grid::G  # grid in use for sampling
    cseq::AbstractVector  # covariance sequence
    rseq::AbstractVector  # sequence of partial correlation
    pseq::AbstractVector{<:AbstractVector}  # the coefficients of consecutive projections
    sseq::AbstractVector  # sequence of variance of residual of projections
    # pmat::SparseMatrixCSC{AbstractFloat}  # upper triangular matrix diagonalizing the covariance matrix
    # cmat::Matrix{<:AbstractFloat}  # matrix of covariance

    function HoskingSampler{T, P, G}(p::P, g::G) where {T, P, G}
        @assert isregulargrid(g)  # only works on regular (continuous or discrete) grid

        cseq = covseq(p, g)
        pseq, sseq, rseq = LevinsonDurbin(cseq)

        # N = length(g)
        # pmat = 1 * speye(N,N)
        # for c = 2:N
        #     pmat[1:c-1, c] = -pseq[c-1][end:-1:1]
        # end
        # cmat = covmat(p, g)
        return new(p, g, cseq, rseq, pseq, sseq) #, pmat, cmat)
    end
end

HoskingSampler(p::StationaryProcess{T}, g::AbstractVector{<:T}) where T = HoskingSampler{T, typeof(p), typeof(g)}(p, g)


# function rand!(x::Vector{<:AbstractFloat}, s::HoskingSampler{P, G}) where {P,G}
function rand!(x::Vector{<:AbstractFloat}, s::HoskingSampler)
    # length(x) == length(s) || error()
    # generate the first sample
    x[1] = sqrt(s.sseq[1]) * randn()
    # recursive conditional sampling
    for n = 1:length(x)-1
        μ = s.pseq[n]' * x[n:-1:1]
        σ = sqrt(s.sseq[n+1])
        x[n+1] = μ + σ * randn()
    end
    return x
end

"""
    rand_otf!(x::Vector{<:AbstractFloat}, s::HoskingSampler{P, G}) where {P, G}

On-the-fly sampling using Levinson-Durbin algorithm. The partial auto-correlation function must be known for the process, for example, the FARIMA{0,d,0} process.
"""
function rand_otf!(x::Vector{<:AbstractFloat}, p::StationaryProcess{T}, g::AbstractVector{<:T}) where T
    # println("Stationary otf")
    # check dimension
    # @assert length(x) <= length(g)

    # generate the first sample
    γ0 = autocov(p, 0)
    x[1] = sqrt(γ0) * randn()

    σ = sqrt(γ0)
    ϕ = zeros(length(x)); ϕ[1] = partcorr(p, 1)

    # recursive conditional sampling
    for n = 1:length(x)-1
        μ = ϕ[1:n]' * x[n:-1:1]
        σ *= sqrt(1-ϕ[n]^2)
        x[n+1] = μ + σ * randn()
        ϕ[n+1] = partcorr(p, n+1)
        ϕ[1:n] -= ϕ[n+1] * ϕ[n:-1:1]
    end
    return x
end

# function rand_otf!(x::Vector{<:AbstractFloat}, s::HoskingSampler{P, G}) where {P,G}
#     # println("Stationary otf")
#     # length(x) == length(s) || error()

#     # generate the first sample
#     γ0 = autocov(s.proc, 0)
#     x[1] = sqrt(γ0) * randn()
#     # recursive conditional sampling
#     σ = sqrt(γ0)
#     ϕ = [partcorr(s.proc, 1)]
#     for n = 1:length(x)-1
#         μ = ϕ' * x[n:-1:1]
#         σ *= sqrt(1-ϕ[end]^2)
#         x[n+1] = μ + σ * randn()
#         r = partcorr(s.proc, n+1)
#         ϕ = [ϕ - r*ϕ[end:-1:1]; r]  # <- memory reallocation!
#     end
#     return x
# end


"""
Sampling a general FARIMA process via a FARIMA{0,d,0} process.
"""
function rand(p::FARIMA, s::HoskingSampler{T, <:FractionalIntegrated, G}) where {T, G}
    p.d == s.proc.d || error("The order of fractional differential in FARIMA{P,Q} and FARIMA{0,0} must be identical.")
    ar_len, ma_len = length(p.ar), length(p.ma)
    x = zeros(length(s))

    # step 1: sample an FARIMA{0,d,0} process
    rand!(x, s)
    # or equivalently, invoke(rand!, Tuple{Vector{<:AbstractFloat}, HoskingSampler}, x, s)

    # step 2: generate an ARMA process
    # REVIEW 15/11/2018: NOT SURE about this implementation!!
    x = conv(x, [1; p.ma])[1:end-ma_len]
    if ar_len > 0
        for t = (ar_len+1):length(s)
            # singular case: Float64[]' * Float64[] = 0.0
            x[t] += p.ar' * x[(t-1):-1:(t-ar_len)]
        end
    end
    return x
end


#### CRMD ####
"""
Conditionalized random midpoint displacement (CRMD) sampler for the increment process of SSSI process, e.g. a fGn.

# Notes
- The step of increment process (e.g. a fGn) must be equal to 1, this is crucial since the Yule-Walker equations in our implementation is de-scaled.
- The original CRMD method draws the sample trajectory on the interval [0,1]. However in our implementation a scaling factor is applied at the end so that the final trajectory are samples of a discrete fGn process (i.e. of unit step).
- The sampling grid used in this method corresponds to the index of the sampling points but not their physical position in [0,1].
"""
struct CRMDSampler{P<:IncrementProcess} <: DiscreteTimeSampler{P}
    proc::P  # instance of the stochastic process
    grid::DiscreteTimeGrid  # grid in use for sampling
    jmin::Integer  # coarse scale index
    coarse_sampler::Sampler  # exact sampler for the coarse scale

    # sclrng::RegularGrid  # range of dyadic scales jmin..jmax
    wsize::Integer  # window size for the conditionalized sampling
    init_coef::AbstractMatrix  # coefficients of the initial sampler
    init_lmat::AbstractMatrix  # lower triangular matrix of the initial sampler
    rfn_coef::AbstractVector  # coefficients of the refinement sampler
    rfn_std::AbstractFloat  # square-root of variance of the refinement (scale independant)
    # cmat::AbstractMatrix  # system matrix of the refinement sampler
    # cvec::AbstractVector  # RHS vector of the refinement sampler
    # init_pos::Symbol  # window position of initial samples, :left, :right, :center

    """
    Constructor of CRMDSampler.

    # Args
    - p: object of `IncrementSSSIProcess`.
    - g: discrete-time regular sampling grid, the step can be >= 1.
    - w: moving window size.
    - jmin: index of the coarsest scale.

    # Notes
    - The step of increment process `p` (e.g. a fGn) must be equal to 1, this is crucial since the Yule-Walker equations in our implementation is de-scaled.
    - the grid `g` is used only at the end: the generated trajectory with unit step is restricted on `g` to obtain the (down-sampled) trajectory of desired length.
    """
    function CRMDSampler{P}(p::P, g::DiscreteTimeGrid, w::Integer, jmin::Integer) where {P}
        @assert isregulargrid(g)  # only works on regular (continuous or discrete) grid

        step(p) == 1 || error("Step of increment of the underlying process must be 1.")
        @assert 1 <= w <= 2^jmin
        # 2^(Int(round(log2(length(g))))) == length(g) || error("Length of the sampling grid must be power of 2.")
        # jmin = 10  # coarse scale length = 2^jmin
        # jmax = max(jmin, ceil(Int, log2(length(G))))

        H = ss_exponent(p)

        # full covariance matrix
        # Note: without explicite conversion RegularGrid(...) the autocov function
        # `autocov(X::IncrementProcess{T, P}, t::T, s::T) where {T, P}` will be called.
        Cf = covmat(p, 1:2*w)
        Ch = Cf[1:w, 1:w]  # half matrix
        Cd = Cf[1:2:end, 1:2:end]  # down-sampled
        Af = [autocov(p, 2*(l-m)) + autocov(p, 2*(l-m)+1) for l=0:2w-1, m=0:2w-1]

        # initial sampler
        Ah = Af[1:w, 1:w] # [autocov(p, 2*(l-m)) + autocov(p, 2*(l-m)+1) for l=0:w-1, m=0:w-1]
        ξ = (2^(2*H) * Ch) \ Ah
        M = Symmetric(Cd - ξ' * Ah)  # reinforce symmetry
        L = cholesky(M).U'

        # system matrix
        b11 = 2^(2*H) * Cf
        b12 = Af[:, 1:w]
        # b12 = [autocov(p, 2*(l-m)) + autocov(p, 2*(l-m)+1) for l=0:2*w-1, m=0:w-1]
        # b21 = [autocov(p, 2*(m-l)) + autocov(p, 2*(m-l)-1) for m=0:w-1, l=0:2*w-1]
        b22 = Cd
        Γ = [b11 b12; b12' b22]

        # RHS vector
        b1 = [autocov(p, 2*(l-w)) + autocov(p, 2*(l-w)+1) for l=0:2w-1]
        b2 = [autocov(p, 2*(m-w)) for m=0:w-1]
        γ = [b1; b2]

        φ = Γ \ γ  # coefficients of prediction
        σ = sqrt(autocov(p, 0) - φ' * γ)  # sqaure-root of variance

        coarse_sampler = CholeskySampler(p, 1:2^jmin)
        return new(p, g, jmin, coarse_sampler, w, ξ, L, φ, σ)
    end
end

CRMDSampler(p::P, g::DiscreteTimeGrid, w::Integer=10, jmin::Integer=10) where {P} = CRMDSampler{P}(p, g, w, jmin)

function rand!(x::Vector{<:AbstractFloat}, s::CRMDSampler)
    # @assert length(x) <= length(s)

    H = ss_exponent(s.proc)
    # sampling at the coarse scale using an exact sampler
    x0 = 2^(-s.jmin * H) * rand(s.coarse_sampler)

    # dyadic refinement, until the trajectory grows to the end of the sampling grid
    while length(x0) < s.grid[end]
        x0 = rand_rfn(x0, s)
    end
    # x0 is a sample trajectory of ${X^\delta_H(n)}_{n=0...N-1} $ with $\delta=1/N$ on the interval $[0,1]$. To obtain a sample trajectory of $X^1_H$, apply the scaling factor $N^H$.
    x0 .*= length(x0)^H

    return copyto!(x, x0[s.grid][1:length(x)])
end

function rand_rfn(x0::Vector{<:AbstractFloat}, s::CRMDSampler)
    j = ceil(Int, log2(length(x0)))  # scale index
    2^j == length(x0) || error("Length of the input vector must be a power of 2.")

    H = ss_exponent(s.proc)
    α = 2^(-j * H)  # scaling factor
    w = s.wsize

    x1 = zeros(2*length(x0))  # refined samples
    xo = view(x1, 1:2:length(x1))  # refined samples of odd positions
    xe = view(x1, 2:2:length(x1))  # refined samples of even positions
    xz = zeros(length(x0) + w); xz[1:length(x0)] = x0  # zero-padding to avoid boundary check
    wn = randn(length(x0))  # white noise vector

    # initialization: generate the w samples at the leftmost
    xo[1:w] = s.init_coef' * xz[1:w] + α * s.init_lmat * wn[1:w]

    # left to right (forward) propagation
    for t = (w+1):length(x0)
        xo[t] = s.rfn_coef' * [xz[t-w:t+(w-1)]; xo[t-w:t-1]] + α * s.rfn_std * wn[t]
    end
    xe[:] = x0 - xo  # [:] creates a reference, without [:] there will be re-allocation of memory.

    return x1
end


#### Wavelet ####
"""
Wavelet sampler for fractional integrated (fIt) process with d ∈ (-1/2, 1/2).

# Notes
- This method can generated FARIMA(0,H-1/2,0) as well as FARIMA(0, H+1/2,0) process for H in (0,1). The later corresponds to the partial sum of the former. The sampling grid corresponds to the index of the discrete samples. In the case of FARIMA(0, H+1/2, 0) the trajectory can approximate a fBm on the interval [0,1] by proper rescaling.
"""
struct WaveletSampler <: DiscreteTimeSampler{FractionalIntegrated}
    proc::FractionalIntegrated  # instance of the stochastic process
    grid::DiscreteTimeGrid  # grid in use for sampling
    coarse_sampler::Sampler  # exact sampler for the coarse scale
    psflag::Bool  # flag of partial sum
    jmin::Integer  # coarse scale index
    qmf_ori::Tuple{AbstractVector, AbstractVector}  # original qmf filter (lo, hi)
    # qmf_mod::Tuple{AbstractVector, AbstractVector}  # modified qmf filter (lo, hi)
    qmf_fra::Tuple{AbstractVector, AbstractVector}  # fractionnal qmf filter  (lo, hi)

    """
    Constructor of WaveletSampler.

    # Args
    - p: fractional integrated process, with the parameters d=H-1/2 and H the Hurst exponent
    - r: regularity of the wavelet function, must be strictly larger than s=H+1/2
    - psflag: if true generate a trajectory of FARIMA(0, H+1/2, 0), otherwise generate FARIMA(0, H-1/2, 0).
    """
    function WaveletSampler(p::FractionalIntegrated, g::DiscreteTimeGrid; r=5, jmin=10, psflag=true)
        @assert isregulargrid(g)  # only works on regular (continuous or discrete) grid

        # jmin = 10  # coarse scale index
        trunc_eps = 1e-8  # precision of truncation
        H = p.d + 1/2  # Hurst exponent

        r > H+1/2 || error("Regularity of the wavelet must be larger than H+1/2!")
        v = psflag ? H+1/2 : H-1/2  # fractional exponent for H+1/2 or H-1/2

        # s = H + 1/2; d = H - 1/2   #  or simply: s = p.d + 1; d = p.d
        # r > s || error("Regularity of the wavelet must be larger than H+1/2!")
        # v = psflag ? s : d  # fractional exponent for H+1/2 or H-1/2

        # original qmf filters
        lo_ori = Wavelets.WT.daubechies(2*r)
        hi_ori = qmf(lo_ori)  # or: hi_ori = reverse(mirror(lo_ori))

        # modified qmf filters
        lo_mod = copy(lo_ori)
        hi_mod = copy(hi_ori)
        knl = (-1).^(0:length(lo_mod)-1)
        for n=1:r
            lo_mod = conv(knl, lo_mod)[1:length(lo_mod)]
            hi_mod = cumsum(hi_mod)
        end

        lmax = 1000  # maximum length of truncation
        f0 = [binomial(v+r, k) for k in 0:lmax]
        g0 = [(-1)^k * binomial(r-v, k) for k in 0:lmax]
        lt = max(sum(abs.(f0).>trunc_eps), sum(abs.(g0).>trunc_eps))
        lo_fra = conv(f0[1:lt], lo_mod)
        hi_fra = conv(g0[1:lt], hi_mod)

        coarse_sampler = CholeskySampler(p, 1:2^jmin)
        return new(p, g, coarse_sampler, psflag, jmin, (lo_ori, hi_ori), (lo_fra, hi_fra))
    end
end

# binomial(n::Real, k::Real) = gamma(n+1)/gamma(n-k+1)/gamma(k+1)
binomial(n::Complex, k::Complex) = exp(lgamma(n+1)-lgamma(n-k+1)-lgamma(k+1))
# binomial(n::Complex, k::Complex) = exp(lgamma(k-n)-lgamma(-n)-lgamma(k+1))
binomial(n::Real, k::Real) = real(binomial(Complex(n), Complex(k)))
# binomial(n::Number, k::Number) = binomial(Complex(n), Complex(k))

"""
Dyadic refinement sampling.
"""
function rand_rfn(x0::Vector{<:AbstractFloat}, s::WaveletSampler)
    x = zeros(2*length(x0))  # up-sampled samples
    w = zeros(2*length(x0))  # white noise
    x[1:2:end] = x0
    w[1:2:end] = randn(length(x0))

    return conv(s.qmf_fra[1], x) + conv(s.qmf_fra[2], w)
end

# function rand_rfn_ori(x0::Vector{<:AbstractFloat}, s::WaveletSampler)
#     x = zeros(2*length(x0))  # up-sampled samples
#     w = zeros(2*length(x0))  # white noise
#     x[1:2:end] = x0
#     w[1:2:end] = randn(length(x0))
#     return conv(s.qmf_ori[1], x) + conv(s.qmf_ori[2], w)
# end


function rand!(x::Vector{<:AbstractFloat}, s::WaveletSampler)
    # @assert length(x) <= length(s)

    # sampling at the coarse scale using an exact sampler
    x0 = rand(s.coarse_sampler)
    if s.psflag
        x0 = cumsum(x0)
    end

    # dyadic refinement
    while length(x0) < 2 * s.grid[end]
        x0 = rand_rfn(x0, s)
    end

    Ng = s.grid[end]-s.grid[1]  # full range of grid
    # keep only the central part and force to start from 0
    n = max((length(x0)-Ng)÷2, 1)
    x1 = x0[n:(n+Ng)] .- x0[n]
    return copyto!(x, x1[s.grid][1:length(x)])
end

