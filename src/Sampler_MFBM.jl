######## Sampler for multifractional Brownian motion (mBm) and related processes ########

"""
    val2grid(X::Vector{Float64}, δ0::Float64=2.5e-2)

Construct a regular grid from a vector of continuous values with a given step.
"""
function val2grid(X::Vector{Float64}, δ0::Float64=2.5e-2)
    xmin, xmax = minimum(X), maximum(X)
    δ0 > 0 || error("Step of grid must be > 0.")
    # (hmin>0 && hmax<1) || error("Husrt exponent must be bounded in (0,1).")
    δ = min(δ0, xmax-xmin)  # hstep smaller than 2.5e-2 is unstable
    grid::Vector{Float64} = (δ > 0) ? collect(xmin:δ:xmax) : [xmin]
    M, N = length(grid), length(X)
    idx::Vector{Integer} = (δ > 0) ? min.(M, max.(1, ceil.(Int, M*(X-xmin)/(xmax-xmin)))) : ones(Integer, N)
    return grid, idx
end


"""
    linprd_fGns(Hgrid::Vector{<:AbstractFloat}, N::Integer, epsilon=1e-8)

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
function linprd_SfGn(Hgrid::Vector{<:AbstractFloat}, N::Integer, epsilon=1e-8)
    # Initialization
    M = length(Hgrid)
    ρ2 = 1. * N   # residual
    ρ = sqrt(abs(ρ2))
#     J::Vector{Integer} = [1] # index of spanning processes
    J::Vector{Bool} = zeros(M)  # bool index of spanning processes
    J[1] = true
#     A::Matrix{<:AbstractFloat} = eye(N)  # triangular system matrix for ϕ
#     F::Matrix{<:AbstractFloat} = zeros(N,N)  # triangular system matrix for ψ
#     F[1,1] = ρ
    iF::Matrix{<:AbstractFloat} = zeros(N,N)  # triangular system matrix for ψ, inverse
    iF[1,1] = 1/ρ

    Ψ = Vector{Vector{<:AbstractFloat}}(M)  # coeff of linear prediction
    Φ = Vector{Vector{<:AbstractFloat}}(M)  # coeff of orthonormal basis
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
function condsampl_SfGn(Hgrid::Vector{<:AbstractFloat},
                        J::Vector{Bool},
                        Ψ::Vector{Vector{<:AbstractFloat}},
                        Φ::Vector{Vector{<:AbstractFloat}},
                        R::Vector{<:AbstractFloat},
                        C::Matrix{<:AbstractFloat},
                        epsilon=1e-8)
    N = size(C,1)
    M = length(Hgrid)

    # Initialization
    Z = circulant_embedding(covseq(FractionalGaussianNoise(Hgrid[1]), 1:N))
    # Z = rand(CircSampler(FractionalGaussianNoise(Hgrid[1]), DiscreteTimeGrid(1:N)))

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
struct SfGnSampler <: DiscreteTimeSampler{FractionalGaussianNoise}
    Hurst::Vector{<:AbstractFloat}
    grid::DiscreteTimeGrid  # grid in use for sampling
    hgrid::Vector{<:AbstractFloat}  # grid for Hurst exponents
    hidx::Vector{Integer}  # index of Hurst exponents in `hrid`

    J::Vector{Bool}  # bool index of spanning processes
    Ψ::Vector{Vector{<:AbstractFloat}}  # coeff of linear prediction
    Φ::Vector{Vector{<:AbstractFloat}}  # coeff of orthonormal basis
    R::Vector{<:AbstractFloat}  # square of residual
    C::Matrix{<:AbstractFloat}  # sequence of covariance

    function SfGnSampler(Hurst::Vector{<:AbstractFloat}, δ::Float64=2.5e-2)
        N = length(Hurst)
        grid = 1:N
        hgrid, hidx = val2grid(Hurst, δ)
        J, Ψ, Φ, R, C = linprd_SfGn(hgrid, N)
        new(Hurst, grid, hgrid, hidx, J, Ψ, Φ, R, C)
    end
end

function rand!(x::Vector{<:AbstractFloat}, s::SfGnSampler)
    # @assert length(x) <= length(s)

    dW0 = condsampl_SfGn(s.hgrid, s.J, s.Ψ, s.Φ, s.R, s.C)
    N = length(s.grid)
    dY = ((1/N).^ s.Hurst) .* [dW0[n, s.hidx[n]] for n=1:N]
    return copyto!(x, cumsum(dY)[1:length(x)])
end

# function rand_mBm(Hurst0, )
#     J, Ψ, Φ, R, C = FracFin.linprd_fGns(Hgrid, N);
