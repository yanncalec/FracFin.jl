import ForwardDiff

diff_gamma = x -> ForwardDiff.derivative(gamma, x)

"""
Derivative w.r.t. H
"""
function diff_Aρ_bspline(τ::Real, ρ::Real, H::Real, v::Int, mode::Symbol)
    d1 = 2 * diff_gamma(2H+1) * sin(π*H) * Cψρ_bspline(τ, ρ, H, v, mode)
    d2 = gamma(2H+1) * cos(π*H) * π * Cψρ_bspline(τ, ρ, H, v, mode)
    # d3 = gamma(2H+1) * sin(π*H) * ForwardDiff.derivative(H->Cψρ_bspline(τ, ρ, H, v, mode), H)
    d3 = gamma(2H+1) * sin(π*H) * diff_Cψρ_bspline(τ, ρ, H, v, mode)
    return d1 + d2 + d3
end
