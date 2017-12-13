
"""
Sampler for a stochastic process
"""
abstract type Sampler{P<:StochasticProcess, G<:SamplingGrid} end

# Random sampling function with initialized sampler
rand!(x::Vector{Float64}, s::Sampler) = throw(NotImplementedError("rand!(::Vector{Float64}, ::$(typeof(s)))"))
rand(s::Sampler) = rand!(Vector{Float64}(length(s)), s)

# Random sampling function with on-the-fly implementation
rand_otf!(x::Vector{Float64}, p::StochasticProcess, g::SamplingGrid) = throw(NotImplementedError("rand_otf!(::Vector{Float64}, ::$(typeof(p)), ::$(typeof(g)))"))
rand_otf(p::StochasticProcess, g::SamplingGrid) = rand_otf!(Vector{Float64}(length(g)), p, g)

length(s::Sampler) = length(s.grid)
size(s::Sampler) = size(s.grid)


"""
Cholesky sampler.
"""
struct CholeskySampler{P, G}<:Sampler{P, G}
    proc::P  # instance of the stochastic process
    grid::G  # grid in use for sampling
    cmat::Matrix{Float64}  # covariance matrix
    wmat::Matrix{Float64}  # cholesky decomposition such that wmat * wmat' = cmat

    function CholeskySampler{P,G }(p::P, g::G) where {P, G}
        # function CholeskySampler{P,G}(p::P, g::G) where {P<:StochasticProcess, G<:SamplingGrid}
        # function CholeskySampler{P<:StochasticProcess, G<:SamplingGrid}(p::P, g::G)
        (0 in g) && error("The sampling grid must not contain the origin.")
        # construct the auto-covariance matrix
        cmat = autocov(p, g)
        wmat = chol(cmat)
        new(p, g, cmat, wmat)
    end
end

# outer constructor as a shortcut
CholeskySampler(p::P, g::G) where {P, G} = CholeskySampler{P, G}(p, g)

function rand!(x::Vector{Float64}, s::CholeskySampler{P, G}) where {P, G}
    return copy!(x, s.wmat' * randn(length(s)))
end


"""
    LevinsonDurbin(::Vector{Float64})

Decomposition of a Toeplitz matrix using Levinson-Durbin (LD) method.

# Explanation
Provided the covariance sequence gamma(cdot) of a real stationary stochastic process, the LD method computes the lower triangular matrix s.t.
A Gamma A' = diag
"""
function LevinsonDurbin(cseq::Vector{Float64})
    N = length(cseq)

    if N > 1
        # check that cseq is a validate covariance sequence
        @assert cseq[1] > 0
        @assert all(diff(abs.(cseq)) .<= 0)

        # initialization
        pseq = Vector{Vector{Float64}}(N-1); pseq[1] = [cseq[2]/cseq[1]]
        sseq = zeros(N); sseq[1] = cseq[1]; sseq[2] = (1-pseq[1][1]^2) * sseq[1]
        rseq = zeros(N-1); rseq[1] = pseq[1][1]
        # recursive construction of the projection coefficients and variances
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


"""
# Note
For ARFIMA{0,0} process the partial correlation is explicitly given by:
    TODO
    [pseq[n][n] == partcorr(p, n) || error("Error in partcorr() of ARFIMA process.") for n in 1:N-1]
end

construct the triangular Cholesky decomposition pmat satisfying the identity:
pmat' * cmat * pmat = diag(sseq)
"""
struct LevinsonDurbinSampler{P<:StationaryStochasticProcess, G<:UnitRange{Int64}}<:Sampler{P, G}
    proc::P  # instance of the stochastic process
    grid::G  # grid in use for sampling
    cseq::Vector{Float64}  # covariance sequence
    rseq::Vector{Float64}  # sequence of partial correlation
    pseq::Vector{Vector{Float64}}  # the coefficients of consecutive projections
    sseq::Vector{Float64}  # sequence of variance of residual of projections
    # pmat::SparseMatrixCSC{Float64}  # upper triangular matrix diagonalizing the covariance matrix
    # cmat::Matrix{Float64}  # matrix of covariance

    function LevinsonDurbinSampler{P,G}(p::P, g::G) where {P,G}
        cseq =
            try
                autocov!(zeros(length(g)), p, g)
            catch msg  # method covseq() not defined for the given process
                warn(msg)
                Float64[]
            end
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

LevinsonDurbinSampler(p::P, g::G) where {P, G} = LevinsonDurbinSampler{P, G}(p, g)


# function rand!(x::Vector{Float64}, s::LevinsonDurbinSampler{P, G}) where {P,G}
function rand!(x::Vector{Float64}, s::LevinsonDurbinSampler)
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
    rand_otf!(x::Vector{Float64}, s::LevinsonDurbinSampler{P, G}) where {P, G}

On-the-fly sampling using Levinson-Durbin algorithm. The partial auto-correlation function must be known for the process, for example, the ARFIMA{0,d,0} process.
"""
function rand_otf!(x::Vector{Float64}, p::StationaryStochasticProcess, g::UnitRange{Int64})
    # println("Stationary otf")
    # check dimension
    @assert length(x) == length(g)

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

# function rand_otf!(x::Vector{Float64}, s::LevinsonDurbinSampler{P, G}) where {P,G}
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
    rand(p::ARFIMA, s::LevinsonDurbinSampler{P, G}) where {P<:ARFIMA{0,0}, G}

Sampling a general ARFIMA process via a ARFIMA{0,0} process.
"""
function rand(p::ARFIMA, s::LevinsonDurbinSampler{P, G}) where {P<:ARFIMA{0,0}, G}
    p.d == s.proc.d || error("The order of fractional differential in ARFIMA{P,Q} and ARFIMA{0,0} must be identical.")
    ar_len, ma_len = length(p.ar), length(p.ma)
    x = zeros(length(s))

    # step 1: sample an ARFIMA{0,0} process
    rand!(x, s)
    # or equivalently, invoke(rand!, Tuple{Vector{Float64}, LevinsonDurbinSampler}, x, s)
    # step 2: generate an ARMA process
    x = conv(x, [1; p.ma])[1:end-ma_len]
    if ar_len > 0
        for t = (ar_len+1):length(x)
            # singular case: Float64[]' * Float64[] = 0.0
            x[t] += p.ar' * x[(t-1):-1:(t-ar_len)]
        end
    end
    return x
end


"""
Circulant embeding method
"""
struct CircSampler{P<:StationaryStochasticProcess, G<:UnitRange{Int64}}<:Sampler{P, G}
    proc::P  # instance of the stochastic process
    grid::G  # grid in use for sampling
    cseq::Vector{Float64}  # covariance sequence
    fseq::Vector{Float64}  # square-root of the Fourier coefficients of cseq

    function CircSampler{P, G}(p::P, g::G) where {P, G}
        # Nf = 1 << ceil(Int64, log2(length(p)))  # length of FFT, equivalent to 2^()
        c = autocov!(zeros(length(g)), p, g)
        # minimal circulant embedding
        cm = vcat(c, c[end-1:-1:2])  # mirror vector
        cf = real(fft(cm))  # the imaginary part is close to zero
        # check the non-negative constraint
        idx = cf .< 0.
        any(idx) && warn("Negative eigenvalues encountered, using Wood-Chan approximation.")
        cf[idx] = 0.
        new(p, g, c, sqrt.(cf))
    end
end

# CircSampler{P, G}(p::P, g::G) where {P<:FractionalGaussianNoise, G} = CircSampler{P,G}(convert(FractionalGaussianNoise, p), g)
# CircSampler(p::FractionalBrownianMotion, g::G) where G = CircSampler{FractionalGaussianNoise, G}(convert(FractionalGaussianNoise, p), g)

CircSampler(p::P, g::G) where {P, G} = CircSampler{P, G}(p, g)


# function rand!(x::Vector{Complex128}, s::CircSampler{P, G}) where {P, G}
function rand!(x::Vector{Float64}, s::CircSampler)
    # println("Calling rand!()")
    @assert length(x) == length(s)
    z = randn(length(s.fseq)) + im * randn(length(s.fseq))
    y = real(ifft(s.fseq .* fft(z)))[1:length(x)]
    return copy!(x, y)
end

function rand(s::CircSampler)
    z = randn(length(s.fseq)) + im * randn(length(s.fseq))
    return ifft(s.fseq .* fft(z))[1:length(s)]
end
