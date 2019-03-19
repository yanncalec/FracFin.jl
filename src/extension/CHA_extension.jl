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



"""
Compute filters of Wavelet Packet transform.

# Args
- lo: low-pass filter
- hi: high-pass filter
- n: level of decomposition. If n=0 the original filters are returned.

# Returns
- a matrix of size ?-by-2^n that each column is a filter.
"""
function wpt_filter(lo::AbstractVector{<:Number}, hi::AbstractVector{<:Number}, n::Integer)
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
function biconv_filter(lo::Vector{T}, hi::Vector{T}, n::Int) where {T<:Number}
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
- mc: mask for valid coefficients

# Notes
Current implementation is buggy: reconstruction by `iswt` can be very wrong.
"""
function swt(x::AbstractVector{<:Number}, lo::AbstractVector{<:Number}, hi::AbstractVector{<:Number}, level::Integer, mode::Symbol)
    @assert level > 0
    @assert length(lo) == length(hi)

    # # if high pass filter is not given, use the qmf.
    # if isempty(hi)
    #     hi = (lo .* (-1).^(1:length(lo)))[end:-1:1]
    # else
    #     @assert length(lo) == length(hi)
    # end

    nx = length(x)
    # fct = sqrt(2)  # scaling factor
    fct = 1
    ac = zeros(T, (nx, level+1))  # approximation coefficients
    ac[:,1] = x
    dc = zeros(T, (nx, level))  # detail coefficients
    mc = zeros(Bool, (nx, level))  # masks

    # Iteration of the cascade algorithm
    for n = 1:level
        # up-sampling of qmf filters
        lo_up = upsampling(lo, 2^(n-1), tight=true)
        hi_up = upsampling(hi, 2^(n-1), tight=true)
        km, vm = convmask(nx, length(lo_up), mode)

        mc[:,n] = vm[km]
        dc[:,n] = fct * conv(hi_up, ac[:,n])[km]
        ac[:,n+1] = fct * conv(lo_up, ac[:,n])[km]
    end

    return  ac[:,2:end], dc, mc
end


"""
Inverse stationary transform.
"""
function iswt(ac::AbstractVector{<:Number}, dc::AbstractMatrix{<:Number}, lo::AbstractVector{<:Number}, hi::AbstractVector{<:Number}, mode::Symbol)
    @assert length(ac) == size(dc,1)

    level = size(dc, 2)  # number of levels of transform
    nx = length(ac)  # length of resynthesized signal
    # fct = sqrt(2)  # scaling factor
    fct = 1
    mask = zeros(nx)  # mask emulating the up-sampling operator in the decimated transform

    xr = ac  # initalization of resynthesized signal
    # xr = zeros(T, (nx,level))  # reconstructed approximation coefficients at different levels

    for n=level:-1:1
        lo_up = upsampling(lo, 2^(n-1), tight=true)
        hi_up = upsampling(hi, 2^(n-1), tight=true)
        km, vm = convmask(nx, length(lo_up), mode)

        fill!(mask, 0); mask[1:2^(n):end] = 1
        xr = fct * (conv(hi_up, mask .* view(dc,:,n)) + conv(lo_up, mask .* xr))[km]
    end
    return xr
end
