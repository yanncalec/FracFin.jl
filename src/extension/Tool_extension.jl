import IterativeSolvers
import IterativeSolvers: lsqr

function pinv_iter(A::AbstractMatrix{T}, method::Symbol=:lsqr) where {T<:Number}
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
