# Wavelet filters

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

# References
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
            ψ = conv(hi_up, ϕ) * sqrt(2)
        end
        ϕ = conv(lo_up, ϕ) * sqrt(2)
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
Stationary wavelet transform using à trous algorithm.

# Returns
* Ma: matrix of approximation coefficients with increasing scale index in row direction
* Md: matrix of detail coefficients
* nbem: number of left side boundary elements
"""
function swt(x::Vector{Float64}, level::Int, lo::Vector{Float64}, hi::Vector{Float64}=Float64[])
    # @assert level > 0
    # @assert length(lo) == length(hi)

    # if high pass filter is not given, use the qmf.
    if isempty(hi)
        hi = (lo .* (-1).^(1:length(lo)))[end:-1:1]
    end

    ac::Array{Vector{Float64},1} = []
    dc::Array{Vector{Float64},1} = []
    push!(ac, x)
    klen = zeros(Int, level)

    # Iteration of the cascade algorithm
    for n = 1:level
        # up-sampling of qmf filters
        s = 2^(n-1)
        l = (length(lo)-1) * s + 1
        lo_up = zeros(Float64, l)
        lo_up[1:s:end] = lo
        hi_up = zeros(Float64, l)
        hi_up[1:s:end] = hi
        klen[n] = l
        # (n > 1) ? (l + klen[n-1] - 1) : l-1
        push!(ac, conv(lo_up, ac[end]) * sqrt(2))
        push!(dc, conv(hi_up, ac[end-1]) * sqrt(2))
    end

    Ma = zeros(Float64, (level, length(x)))
    Md = zeros(Float64, (level, length(x)))
    nbem = cumsum(klen-1)  # number of left side boundary elements

    for n = 1:level
#         println(length(ac[n+1]))
#         println(length(dc[n]))
#         println(klen[n])
        Ma[n, :] = ac[n+1][nbem[n]+1:end]
        Md[n, :] = dc[n][nbem[n]+1:end]
    end

    return Ma, Md, nbem
end


"""
Continuous wavelet transform using parametric wavelet.
"""
function cwt(x::Vector{Float64}, lo::Vector{Float64}, level::Int)
    @assert level>0

#     xc = zeros(Float64, (level, length(x)))
    ac::Array{Vector{Float64},1} = []
    dc::Array{Vector{Float64},1} = []
    klen = zeros(Int, level)

    for n = 1:level
        ϕ, ψ, g = wavefunc(lo, level=n, nflag=true)
        push!(ac, conv(x, ϕ))
        push!(dc, conv(x, ψ))
        klen[n] = length(ϕ)
    end
    return ac, dc
end

function morlet()

end

"""
Mexican hat function.

# Reference
* https://en.wikipedia.org/wiki/Mexican_hat_wavelet
"""
function mexhat(N::Int, a::Float64)
    cst = 2 / (sqrt(3 * a) * (pi^0.25))
    X = collect(0, N-1) - N/2
    X = linspace(-3a, 3a, N)
    return cst * (1 - (X/a).^2) .* exp(- (X/a).^2/2)
end