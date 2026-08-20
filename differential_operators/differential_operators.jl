# Eighth-order periodic finite-difference operators in MPO form.
# This follows QIFS/2D_QIS/numpy/differential_mpo.py. A physical index has
# dimension four. With the Julia/ITensor fused-index convention used here,
# states 1,2,3,4 represent (x_bit,y_bit) as 00, 10, 01, 11.

using ITensorMPS
using ITensors: ITensor, Index, dag, dim, inds
using LinearAlgebra: I

"""Allocate an MPO with specified site indices and bond dimensions."""
function custom_mpo(::Type{T}, sites::AbstractVector{<:Index}, linkdims::AbstractVector{<:Integer}) where {T<:Number}
    n = length(sites)
    n >= 3 || throw(ArgumentError("eighth-order MPOs require at least 3 scale sites"))
    length(linkdims) == n - 1 || throw(ArgumentError("expected $(n - 1) link dimensions"))
    links = [Index(linkdims[k], "Link,l=$k") for k in 1:(n - 1)]
    tensors = Vector{ITensor}(undef, n)
    tensors[1] = ITensor(T, dag(sites[1]), sites[1]', links[1])
    for k in 2:(n - 1)
        tensors[k] = ITensor(T, dag(links[k - 1]), dag(sites[k]), sites[k]', links[k])
    end
    tensors[n] = ITensor(T, dag(links[n - 1]), dag(sites[n]), sites[n]')
    return MPO(tensors)
end

function _shift_matrix(axis::Symbol)
    J = zeros(Float64, 4, 4)
    if axis === :x
        J[2, 1] = 1
        J[4, 3] = 1
    elseif axis === :y
        J[3, 1] = 1
        J[4, 2] = 1
    else
        throw(ArgumentError("axis must be :x or :y"))
    end
    return J
end

function _fill_eighth_order!(mpo::MPO, J::AbstractMatrix, a, derivative_order::Integer)
    identity = Matrix{eltype(J)}(I, 4, 4)
    Jt = transpose(J)
    carry = J + Jt
    n = length(mpo)
    for (site, tensor) in enumerate(mpo)
        data = zeros(eltype(J), dim.(inds(tensor)))
        if site == 1
            data[:, :, 1] = identity
            data[:, :, 2] = carry
            data[:, :, 3] = carry
        elseif site == n - 1
            data[1, :, :, 1] = identity
            data[1, :, :, 2] = Jt
            data[1, :, :, 3] = J
            data[2, :, :, 2] = J
            data[3, :, :, 3] = Jt
            data[2, :, :, 4] = identity
            data[3, :, :, 5] = identity
        elseif site == n
            data[1, :, :] = a[1] * identity + a[2] * Jt + a[3] * J
            data[2, :, :] = a[2] * J + a[4] * identity + a[5] * Jt
            if derivative_order == 1
                data[3, :, :] = a[3] * Jt - a[4] * identity - a[5] * J
                data[4, :, :] = a[5] * J + a[6] * identity
                data[5, :, :] = -a[5] * Jt - a[6] * identity
            else
                data[3, :, :] = a[3] * Jt + a[4] * identity + a[5] * J
                data[4, :, :] = a[5] * J + a[6] * identity
                data[5, :, :] = a[5] * Jt + a[6] * identity
            end
        else
            data[1, :, :, 1] = identity
            data[1, :, :, 2] = Jt
            data[1, :, :, 3] = J
            data[2, :, :, 2] = J
            data[3, :, :, 3] = Jt
        end
        mpo[site] = ITensor(data, inds(tensor))
    end
    return mpo
end

function _diff_8(h::Real, sites, axis::Symbol, derivative_order::Integer)
    length(sites) >= 3 || throw(ArgumentError("eighth-order MPOs require at least 3 scale sites"))
    all(dim(site) == 4 for site in sites) || throw(ArgumentError("QIFS scale sites must have dimension 4"))
    h > 0 || throw(ArgumentError("grid spacing h must be positive"))
    coefficients = if derivative_order == 1
        [0.0, -4 / 5, 4 / 5, 1 / 5, -4 / 105, 1 / 280] / h
    elseif derivative_order == 2
        [-205 / 72, 8 / 5, 8 / 5, -1 / 5, 8 / 315, -1 / 560] / h^2
    else
        throw(ArgumentError("derivative order must be 1 or 2"))
    end
    n = length(sites)
    linkdims = [k == n - 1 ? 5 : 3 for k in 1:(n - 1)]
    mpo = custom_mpo(Float64, sites, linkdims)
    return _fill_eighth_order!(mpo, _shift_matrix(axis), coefficients, derivative_order)
end

"""Eighth-order first derivative in x with periodic boundary conditions."""
Diff_1_8_x(h::Real, sites) = _diff_8(h, sites, :x, 1)

"""Eighth-order first derivative in y with periodic boundary conditions."""
Diff_1_8_y(h::Real, sites) = _diff_8(h, sites, :y, 1)

"""Eighth-order second derivative in x with periodic boundary conditions."""
Diff_2_8_x(h::Real, sites) = _diff_8(h, sites, :x, 2)

"""Eighth-order second derivative in y with periodic boundary conditions."""
Diff_2_8_y(h::Real, sites) = _diff_8(h, sites, :y, 2)

