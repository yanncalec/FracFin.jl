########## Simulation methods (sampler) for stochastic process ##########

"""
Abstract sampler for a stochastic process.

# Members
- proc: process to be sampled
- grid: sampling grid adapted to proc
"""
abstract type Sampler{T<:TimeStyle, P<:StochasticProcess{T}, G<:SamplingGrid{<:T}} end

"""
Sampler for continuous time stochastic process.
"""
const ContinuousTimeSampler{P, G} = Sampler{ContinuousTime, P, G}

"""
Sampler for discrete time stochastic process (time-series).
"""
const DiscreteTimeSampler{P, G} = Sampler{DiscreteTime, P, G}


"""
Random sampling function for an initialized sampler.
"""
rand!(x::Vector{Float64}, s::Sampler) = throw(NotImplementedError("rand!(::Vector{Float64}, ::$(typeof(s)))"))
rand(s::Sampler) = rand!(zeros(Float64, length(s)), s)


"""
Random sampling function for a process of increment.
"""
function rand(::Type{R}, s::Sampler{DiscreteTime, <:IncrementProcess{DiscreteTime,R}, DiscreteTimeRegularGrid}) where R
    if s.grid.step == 1
        return cumsum(rand(s))
    else
        throw(InexactError("Step of the sampling grid must be 1"))
    end
end


"""
Random sampling function with on-the-fly implementation.
"""
rand_otf!(x::Vector{Float64}, p::StochasticProcess{T}, g::SamplingGrid{<:T}) where T = throw(NotImplementedError("rand_otf!(::Vector{Float64}, ::$(typeof(p)), ::$(typeof(g)))"))
rand_otf(p::StochasticProcess{T}, g::SamplingGrid{<:T}) where T = rand_otf!(Vector{Float64}(length(g)), p, g)

length(s::Sampler) = length(s.grid)
size(s::Sampler) = size(s.grid,)

# The convert function is implicitly called by `new()` in the constructor of a sampler
# convert(::Type{S}, s::Sampler) where {S<:Sampler} = S(s.proc, s.grid)


"""
    init_sampler_fGn(H::Float64, N::Integer, name::String)

Initialize a sampler for fGn.

# Args
- H: hurst exponent
- N: number of points of the regular sampling grid
- name: name of the sampling method: {"CHOLESKY", "CIRCULANT", "HOSKING", "MIDPOINT", "WAVELET"}

# Returns
- an object of sampler
"""
function init_sampler_fGn(H::Float64, N::Integer, name::String)
    @assert 0. < H < 1.

    name = uppercase(name)
    fGn = FractionalGaussianNoise(H)  # use unit step of increment
    fIt = FractionalIntegrated(H-1/2)

    Rgrid = ContinuousTimeRegularGrid((1:N)/N)  # sampling grid on [0,1]
    Zgrid = DiscreteTimeRegularGrid(1:N)  # sampling grid 1,2...

    # the scaling factor \delta for arbitrary N comes from the following reasoning: to transform a fBm defined on [0,s] to [0,t], just apply the scaling factor (t/s)^H on the original fBm.
    # δ = Tmax * Rgrid.step  # scaling factor

    if name == "CHOLESKY"
        sampler = CholeskySampler(fGn, Rgrid)
    elseif name == "CIRCULANT"
        sampler = CircSampler(fGn, Zgrid)
    elseif name == "HOSKING"
        sampler = HoskingSampler(fGn, Zgrid)
    elseif name == "MIDPOINT"
        sampler = CRMDSampler(fGn, Zgrid)
    elseif name == "WAVELET"
        sampler = WaveletSampler(fIt, Zgrid, psflag=false)
    else
        error("Unknown method $(name).")
    end

    return sampler
end


"""
    rand_fGn(sampler::Sampler, Tmax::Real=1.)

Generate a sample trajectory of fGn on the interval [0, `Tmax`] by applying an initialized sampler.
"""
function rand_fGn(sampler::Sampler, Tmax::Real=1.)
    T = typeof(sampler)
    N = length(sampler)
    H = ss_exponent(sampler.proc)
    δ = Tmax / N  # sampling step

    if T <: CholeskySampler
        X = rand(sampler)
    elseif T <: Union{CircSampler, HoskingSampler, CRMDSampler, WaveletSampler}
        X = δ^H * rand(sampler)
    end

    return X
end


"""
    init_sampler_fBm(H::Float64, N::Integer, name::String)

Initialize a sampler for fBm.

# Args
- H: hurst exponent
- N: number of points of the regular sampling grid
- name: name of the sampling method: {"CHOLESKY", "CIRCULANT", "HOSKING", "MIDPOINT", "WAVELET"}

# Returns
- an object of sampler
"""
function init_sampler_fBm(H::Float64, N::Integer, name::String)
    @assert 0. < H < 1.

    name = uppercase(name)
    fBm = FractionalBrownianMotion(H)
    fGn = FractionalGaussianNoise(H)  # use unit step of increment
    fIt = FractionalIntegrated(H-1/2)

    Rgrid = ContinuousTimeRegularGrid((1:N)/N)  # sampling grid on [0,1]
    Zgrid = DiscreteTimeRegularGrid(1:N)  # sampling grid 1,2...

    # the scaling factor \delta for arbitrary N comes from the following reasoning: to transform a fBm defined on [0,s] to [0,t], just apply the scaling factor (t/s)^H on the original fBm.
    # δ = Tmax * Rgrid.step  # scaling factor

    if name == "CHOLESKY"
        sampler = CholeskySampler(fBm, Rgrid)
    elseif name == "CIRCULANT"
        sampler = CircSampler(fGn, Zgrid)
    elseif name == "HOSKING"
        sampler = HoskingSampler(fGn, Zgrid)
    elseif name == "MIDPOINT"
        sampler = CRMDSampler(fGn, Zgrid)
    elseif name == "WAVELET"
        sampler = WaveletSampler(fIt, Zgrid, psflag=true)
    else
        error("Unknown method $(name).")
    end

    return sampler
end


"""
    rand_fBm(sampler::Sampler, Tmax::Real=1.)

Generate a sample trajectory of fBm on the interval [0, `Tmax`] by applying an initialized sampler.
"""
function rand_fBm(sampler::Sampler, Tmax::Real=1.)
    T = typeof(sampler)
    N = length(sampler)
    H = ss_exponent(sampler.proc)
    δ = Tmax / N  # sampling step

    if T <: CholeskySampler
        X = rand(sampler)
    elseif T <: Union{CircSampler, HoskingSampler, CRMDSampler}
        X = δ^H * cumsum(rand(sampler))
    elseif T <: WaveletSampler
        # B_H = 2^(-J*H) * rand(sampler) is a fBm trajectory on [0,1] of length 2^J. Taking the first N samples and rescaling by (2^J/N)^H * B_H[1:N] gives a fBm trajectory on [0,1] of length N. Finally multiplying by Tmax^H it gives a trajectory on [0, Tmax], which gives the (same) scaling factor (Tmax/N)^H
        X = δ^H * rand(sampler)
    end

    X .-= X[1]  # force starting from 0
    return X
end


######## Sampler for pure fBm and related processes ########

#### Cholesky ####
"""
Cholesky sampler for general Gaussian process.

# Notes
- This method generates a sample trajectory on any type of sampling grid (regular or arbitrary) adapted to the underlying process.
"""
struct CholeskySampler{T, P, G}<:Sampler{T, P, G}
    proc::P  # instance of the stochastic process
    grid::G  # grid in use for sampling
    cmat::Matrix{Float64}  # covariance matrix
    lmat::Matrix{Float64}  # lower-triangular matrix: lmat * lmat' = cmat

    function CholeskySampler{T, P, G}(p::P, g::G) where {T, P, G}
        # check the grid for fBM
        if (P<:FractionalBrownianMotion) && (0 in g)
            error("The sampling grid must not contain the origin.")
        end
        # construct the auto-covariance matrix
        cmat = autocov(p, g)
        lmat = (cholesky(cmat)).U'  # cholesky decomposition yields an upper-triangular matrix
        return new(p, g, cmat, lmat)
    end
end

# outer constructor as a shortcut
# CholeskySampler(p::P, g::G) where {P, G} = CholeskySampler{typeof(p), typeof(g)}(p, g)
CholeskySampler(p::StochasticProcess{T}, g::SamplingGrid{<:T}) where T = CholeskySampler{T, typeof(p), typeof(g)}(p, g)

function rand!(x::Vector{Float64}, s::CholeskySampler)
    return copyto!(x, s.lmat * randn(length(s)))
end


#### Circulant Embedding ####
"""
Circulant embedding sampler for stationary Gaussian process.

# Notes
- This method generates a sample trajectory on any sampling grid adapted to the underlying process.
"""
struct CircSampler{T, P<:StationaryProcess{T}, G<:RegularGrid{<:T}}<:Sampler{T, P, G}
    proc::P  # instance of the stochastic process
    grid::G  # grid in use for sampling
    cseq::Vector{Float64}  # covariance sequence
    fseq::Vector{Float64}  # square-root of the Fourier coefficients of cseq

    function CircSampler{T, P, G}(p::P, g::G) where {T, P, G}
        # Nf = 1 << ceil(Int64, log2(length(p)))  # length of FFT, equivalent to 2^()
        c = covseq(p, g)
        # minimal circulant embedding
        cm = vcat(c, c[end-1:-1:2])  # mirror vector
        M = length(cm)
        # fft() in Julia is not normalized
        # cf = real(fft(cm)/sqrt(M))  # the imaginary part is close to zero
        cf = real(fft(cm))  # the imaginary part is close to zero
        # check the non-negative constraint
        idx = cf .< 0.
        any(idx) && warn("Negative eigenvalues encountered, using Wood-Chan approximation.")
        cf[idx] .= 0.
        new(p, g, c, sqrt.(cf))
    end
end

CircSampler(p::StationaryProcess{T}, g::RegularGrid{<:T}) where T = CircSampler{T, typeof(p), typeof(g)}(p, g)

# function rand!(x::Vector{Complex128}, s::CircSampler{P, G}) where {P, G}
function rand!(x::Vector{Float64}, s::CircSampler)
    # @assert length(x) <= length(s)
    M = length(s.fseq)
    z = randn(M) + im * randn(M)
    # y = real(ifft(s.fseq .* fft(z)))
    y = real(ifft(s.fseq .* z) * sqrt(M))
    return copyto!(x, y[1:length(x)])
end

"""
    circulant_embedding(c::Vector{Float64})

Functional implementation of the circulant embedding method. It takes the sequence of covariance and return a sample path of the same length.
"""
function circulant_embedding(c::Vector{Float64})
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
Hosking (Levinson-Durbin) sampler for staionary Gaussian process.

# Notes
* This method draws the sample trajectory on any sampling grid adapted to the underlying process.
"""
struct HoskingSampler{T, P<:StationaryProcess{T}, G<:RegularGrid{<:T}}<:Sampler{T, P, G}
    proc::P  # instance of the stochastic process
    grid::G  # grid in use for sampling
    cseq::Vector{Float64}  # covariance sequence
    rseq::Vector{Float64}  # sequence of partial correlation
    pseq::Vector{Vector{Float64}}  # the coefficients of consecutive projections
    sseq::Vector{Float64}  # sequence of variance of residual of projections
    # pmat::SparseMatrixCSC{Float64}  # upper triangular matrix diagonalizing the covariance matrix
    # cmat::Matrix{Float64}  # matrix of covariance

    function HoskingSampler{T, P, G}(p::P, g::G) where {T, P, G}
        # cseq =
        #     try
        #         cseq = covseq(p, g)
        #     catch msg  # method autocov!() not defined for the given process
        #         warn(msg)
        #         Float64[]
        #     end
        cseq = covseq(p, g)
        pseq, sseq, rseq = LevinsonDurbin(cseq)

        # N = length(g)
        # pmat = 1 * speye(N,N)
        # for c = 2:N
        #     pmat[1:c-1, c] = -pseq[c-1][end:-1:1]
        # end
        # cmat = autocov(p, g)
        return new(p, g, cseq, rseq, pseq, sseq) #, pmat, cmat)
    end
end

HoskingSampler(p::StationaryProcess{T}, g::RegularGrid{<:T}) where T = HoskingSampler{T, typeof(p), typeof(g)}(p, g)


# function rand!(x::Vector{Float64}, s::HoskingSampler{P, G}) where {P,G}
function rand!(x::Vector{Float64}, s::HoskingSampler)
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
    rand_otf!(x::Vector{Float64}, s::HoskingSampler{P, G}) where {P, G}

On-the-fly sampling using Levinson-Durbin algorithm. The partial auto-correlation function must be known for the process, for example, the FARIMA{0,d,0} process.
"""
function rand_otf!(x::Vector{Float64}, p::StationaryProcess{T}, g::RegularGrid{<:T}) where T
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

# function rand_otf!(x::Vector{Float64}, s::HoskingSampler{P, G}) where {P,G}
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
    # or equivalently, invoke(rand!, Tuple{Vector{Float64}, HoskingSampler}, x, s)

    # step 2: generate an ARMA process
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
struct CRMDSampler{P<:IncrementSSSIProcess}<:DiscreteTimeSampler{P, DiscreteTimeRegularGrid}
    proc::P  # instance of the stochastic process
    grid::DiscreteTimeRegularGrid  # grid in use for sampling
    jmin::Int64  # coarse scale index
    coarse_sampler::DiscreteTimeSampler  # exact sampler for the coarse scale

    # sclrng::RegularGrid  # range of dyadic scales jmin..jmax
    wsize::Int64  # window size for the conditionalized sampling
    init_coef::Matrix{Float64}  # coefficients of the initial sampler
    init_lmat::Matrix{Float64}  # lower triangular matrix of the initial sampler
    rfn_coef::Vector{Float64}  # coefficients of the refinement sampler
    rfn_std::Float64  # square-root of variance of the refinement (scale independant)
    # cmat::Matrix{Float64}  # system matrix of the refinement sampler
    # cvec::Vector{Float64}  # RHS vector of the refinement sampler
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
    function CRMDSampler{P}(p::P, g::DiscreteTimeRegularGrid, w::Int64, jmin::Int64) where {P}
        step(p) == 1 || error("Step of increment of the underlying process must be 1.")
        @assert 1 <= w <= 2^jmin
        # 2^(Int(round(log2(length(g))))) == length(g) || error("Length of the sampling grid must be power of 2.")
        # jmin = 10  # coarse scale length = 2^jmin
        # jmax = max(jmin, ceil(Int, log2(length(G))))

        H = ss_exponent(p)

        # full covariance matrix
        # Note: without explicite conversion RegularGrid(...) the autocov function
        # `autocov(X::IncrementProcess{T, P}, t::T, s::T) where {T, P}` will be called.
        Cf = covmat(p, RegularGrid(1:2*w))
        Ch = Cf[1:w, 1:w]  # half matrix
        Cd = Cf[1:2:end, 1:2:end]  # down-sampled
        Af = [autocov(p, 2*(l-m)) + autocov(p, 2*(l-m)+1) for l=0:2w-1, m=0:2w-1]

        # initial sampler
        Ah = Af[1:w, 1:w] # [autocov(p, 2*(l-m)) + autocov(p, 2*(l-m)+1) for l=0:w-1, m=0:w-1]
        ξ = (2^(2*H) * Ch) \ Ah
        M = full(Symmetric(Cd - ξ' * Ah))  # reinforce symmetry
        L = full(chol(M)')

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

        coarse_sampler = CholeskySampler(p, DiscreteTimeRegularGrid(1:2^jmin))
        return new(p, g, jmin, coarse_sampler, w, ξ, L, φ, σ)
    end
end

CRMDSampler(p::P, g::DiscreteTimeRegularGrid, w::Int64=10, jmin::Int64=10) where {P} = CRMDSampler{P}(p, g, w, jmin)

function rand!(x::Vector{Float64}, s::CRMDSampler)
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

function rand_rfn(x0::Vector{Float64}, s::CRMDSampler)
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
struct WaveletSampler<:DiscreteTimeSampler{FractionalIntegrated, DiscreteTimeRegularGrid}
    proc::FractionalIntegrated  # instance of the stochastic process
    grid::DiscreteTimeRegularGrid  # grid in use for sampling
    coarse_sampler::DiscreteTimeSampler  # exact sampler for the coarse scale
    psflag::Bool  # flag of partial sum
    jmin::Int64  # coarse scale index
    qmf_ori::Tuple{Vector{Float64}, Vector{Float64}}  # original qmf filter (lo, hi)
    # qmf_mod::Tuple{Vector{Float64}, Vector{Float64}}  # modified qmf filter (lo, hi)
    qmf_fra::Tuple{Vector{Float64}, Vector{Float64}}  # fractionnal qmf filter  (lo, hi)

    """
    Constructor of WaveletSampler.

    # Args
    - p: fractional integrated process, with the parameters d=H-1/2 and H the Hurst exponent
    - r: regularity of the wavelet function, must be strictly larger than s=H+1/2
    - psflag: if true generate a trajectory of FARIMA(0, H+1/2, 0), otherwise generate FARIMA(0, H-1/2, 0).
    """
    function WaveletSampler(p::FractionalIntegrated, g::DiscreteTimeRegularGrid; r=5, jmin=10, max_len=40, psflag=true)
        # jmin = 10  # coarse scale index
        trunc_eps = 1e-8  # precision of truncation
        H = p.d + 1/2  # Hurst exponent
        s = H + 1/2; d = H - 1/2
        r > s || error("Regularity of the wavelet must be larger than H+1/2!")
        v = psflag ? s : d  # fractional exponent for H+1/2 or H-1/2

        # original qmf filters
        lo_ori = daubechies(2*r)
        hi_ori = reverse(mirror(lo_ori))

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
        lo_fra = conv(f0[1:lt], lo_mod)[1:max_len]
        hi_fra = conv(g0[1:lt], hi_mod)[1:max_len]

        coarse_sampler = CholeskySampler(p, DiscreteTimeRegularGrid(1:2^jmin))
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
function rand_rfn(x0::Vector{Float64}, s::WaveletSampler)
    x = zeros(2*length(x0))  # up-sampled samples
    w = zeros(2*length(x0))  # white noise
    x[1:2:end] = x0
    w[1:2:end] = randn(length(x0))

    return conv(s.qmf_fra[1], x) + conv(s.qmf_fra[2], w)
end

# function rand_rfn_ori(x0::Vector{Float64}, s::WaveletSampler)
#     x = zeros(2*length(x0))  # up-sampled samples
#     w = zeros(2*length(x0))  # white noise
#     x[1:2:end] = x0
#     w[1:2:end] = randn(length(x0))
#     return conv(s.qmf_ori[1], x) + conv(s.qmf_ori[2], w)
# end


function rand!(x::Vector{Float64}, s::WaveletSampler)
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
    n = max(Int(round((length(x0)-Ng)/2)), 1)
    x1 = x0[n:(n+Ng)] - x0[n]
    return copyto!(x, x1[s.grid][1:length(x)])
end


######## Sampler for multifractional Brownian motion (mBm) and related processes ########

# Auxiliary functions for the covariance of the  multifractional field
_D(h1,h2) = (h1==h2) ? 1. : sqrt(gamma(2*h1+1)*sin(pi*h1)*gamma(2*h2+1)*sin(pi*h2)) / (gamma(h1+h2+1)*sin(pi*(h1+h2)/2))
# or equivalently
# _F(h) = lgamma(2h+1) + log(sin(π*h))
# _D(h1,h2) = exp((_F(h1)+_F(h2))/2 - _F((h1+h2)/2))

_gn(t,s,h) = (abs(t)^(h) + abs(s)^(h) - abs(t-s)^(h))/2  # non-stationary
_gs(t,h) = (abs(t+1)^(h) + abs(t-1)^(h) - 2*abs(t)^(h))/2  # stationary

"""
    mBm_cov(t1::Real,t2::Real,h1::Real,h2::Real)

Covariance of multifractional Brownian motion.
"""
function mBm_cov(t1::Real,t2::Real,h1::Real,h2::Real)
    _D(h1,h2) * _gn(t1,t2,h1+h2)
end

"""
    mGn_cov(t::Real,h1::Real,h2::Real)

Covariance of multifractional Gaussian noise.
"""
function mGn_cov(t::Real,h1::Real,h2::Real)
    _D(h1,h2) * _gs(t,h1+h2)
end


"""
    linprd_fGns(Hgrid::Vector{Float64}, N::Integer, epsilon=1e-8)

Compute the coefficients of linear prediction for a sequence of fGn.

# Args
- Hgrid: grid points of Hurst exponents
- N: length of stationary process
- epsilon: threshold for identical Hgrid exponents

# Returns
- J: bool index of spanning subsequence
- Ψ: coefficients of basis
- Φ: coefficients of linear prediction
- R: variance of residual
- C: covariance sequence of residual
"""
function linprd_SfGn(Hgrid::Vector{Float64}, N::Integer, epsilon=1e-8)
    # Initialization
    M = length(Hgrid)
    ρ2 = 1. * N   # residual
    ρ = sqrt(abs(ρ2))
#     J::Vector{Integer} = [1] # index of spanning processes
    J::Vector{Bool} = zeros(M)  # bool index of spanning processes
    J[1] = true
#     A::Matrix{Float64} = eye(N)  # triangular system matrix for ϕ
#     F::Matrix{Float64} = zeros(N,N)  # triangular system matrix for ψ
#     F[1,1] = ρ
    iF::Matrix{Float64} = zeros(N,N)  # triangular system matrix for ψ, inverse
    iF[1,1] = 1/ρ

    Ψ = Vector{Vector{Float64}}(M)  # coeff of linear prediction
    Φ = Vector{Vector{Float64}}(M)  # coeff of orthonormal basis
    R = zeros(M); R[1] = ρ2
    C = zeros(N, M)
    cvec = zeros(N)

    # Iteration
    for j=2:M
        Jidx = find(J)
        γ = N * [mGn_cov(0, Hgrid[j], Hgrid[i]) for i in Jidx]  #RHS vector
#         G = N * [mGn_cov(0, Hgrid[i1], Hgrid[i2]) for i1 in Jidx, i2 in Jidx]  # system matrix of process, such that ϕ = G \ γ
        k = length(Jidx)
#         Ak = view(A, 1:k, 1:k)
#         Fk = view(F, 1:k, 1:k)
        iFk = view(iF, 1:k, 1:k)

        # Update by innovation
#         ϕ = Ak' * diagm(1 ./ R[Jidx]) * Ak * γ
#         ψ = Fk' * ϕ
        # another way, which seems more stable than the previous one
        ψ = iFk * γ
        ϕ = iFk' * ψ

        # Construct covariance matrix of the residual
        for n=0:N-1
            u = ϕ' * [mGn_cov(n, Hgrid[j], Hgrid[i]) for i in Jidx]
            v = ϕ' * [mGn_cov(n, Hgrid[i1], Hgrid[i2]) for i1 in Jidx, i2 in Jidx] * ϕ
            cvec[n+1] = mGn_cov(n, Hgrid[j], Hgrid[j]) - 2*u + v
        end

        # Residual
        ρ2 = N*cvec[1]
#         ρ2 = N - 2 * ϕ' * γ + ϕ' * G * ϕ
#         ρ2 = N - ϕ' * γ  # This gives negative values!
        ρ = sqrt(abs(ρ2))

        Ψ[j] = ψ
        Φ[j] = ϕ
        C[:,j] = cvec
        R[j] = ρ2

        # current process is accepted in the list of spanning processes
        if ρ/sqrt(N) > epsilon
            J[j] = true
            k = sum(J)
#             A[k,1:(k-1)] = -ϕ'
#             F[k,1:(k-1)] = ψ'
#             F[k,k] = ρ
            iF[k,1:(k-1)] = -1/ρ * ϕ'
            iF[k,k] = 1/ρ
        end
    end
    return J, Ψ, Φ, R, C
end

"""
Conditional sampling of a sequence of fGn
"""
function condsampl_SfGn(Hgrid::Vector{Float64},
                        J::Vector{Bool},
                        Ψ::Vector{Vector{Float64}},
                        Φ::Vector{Vector{Float64}},
                        R::Vector{Float64},
                        C::Matrix{Float64},
                        epsilon=1e-8)
    N = size(C,1)
    M = length(Hgrid)

    # Initialization
    Z = circulant_embedding(covseq(FractionalGaussianNoise(Hgrid[1]), DiscreteTimeRegularGrid(1:N)))
    # Z = rand(CircSampler(FractionalGaussianNoise(Hgrid[1]), DiscreteTimeRegularGrid(1:N)))

    X = zeros(N, M)  # historical sample paths
    X[:,1] = Z
    ρ = sqrt(abs(R[1]))
#     E = zeros(N, M)  # historical sample paths of residual
#     E[:,1] = Z/ρ

    for j=2:M
        ρ = sqrt(abs(R[j]))
        ϕ, ψ = Φ[j], Ψ[j]
        Jidx = find(J[1:j-1])

        # Conditional mean
        μ = X[:,Jidx] * ϕ
        # or use psi
#         μ = E[:,Jidx] * ψ

        # Conditional error
        Z = (ρ/sqrt(N) > epsilon) ? circulant_embedding(C[:,j]) : zeros(N)
        X[:,j] = μ + Z
#         E[:,j] = Z/ρ
    end
    return X
end


"""
Conditionalized sampling of a sequence of fGn (SfGn).
"""
struct SfGnSampler<:DiscreteTimeSampler{FractionalGaussianNoise, DiscreteTimeRegularGrid}
    Hurst::Vector{Float64}
    grid::DiscreteTimeRegularGrid  # grid in use for sampling
    hgrid::Vector{Float64}  # grid for Hurst exponents
    hidx::Vector{Integer}  # index of Hurst exponents in `hrid`

    J::Vector{Bool}  # bool index of spanning processes
    Ψ::Vector{Vector{Float64}}  # coeff of linear prediction
    Φ::Vector{Vector{Float64}}  # coeff of orthonormal basis
    R::Vector{Float64}  # square of residual
    C::Matrix{Float64}  # sequence of covariance

    function SfGnSampler(Hurst::Vector{Float64}, δ::Float64=2.5e-2)
        N = length(Hurst)
        grid = DiscreteTimeRegularGrid(1:N)
        hgrid, hidx = val2grid(Hurst, δ)
        J, Ψ, Φ, R, C = linprd_SfGn(hgrid, N)
        new(Hurst, grid, hgrid, hidx, J, Ψ, Φ, R, C)
    end
end

function rand!(x::Vector{Float64}, s::SfGnSampler)
    # @assert length(x) <= length(s)

    dW0 = condsampl_SfGn(s.hgrid, s.J, s.Ψ, s.Φ, s.R, s.C)
    N = length(s.grid)
    dY = ((1/N).^ s.Hurst) .* [dW0[n, s.hidx[n]] for n=1:N]
    return copyto!(x, cumsum(dY)[1:length(x)])
end

# function rand_mBm(Hurst0, )
#     J, Ψ, Φ, R, C = FracFin.linprd_fGns(Hgrid, N);
