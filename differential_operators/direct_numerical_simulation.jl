# Dense-grid reference routines following
# QIFS/2D_QIS/numpy/differential_operators_numpy.py.

using LinearAlgebra: norm

"""
    initial_velocity_field(nbits; L=1.0)

Return a periodic Taylor-Green velocity field `(u, v)` on a `2^nbits x
2^nbits` grid. Array dimension 1 is x and dimension 2 is y, matching the
QIFS differential-operator convention.
"""
function initial_velocity_field(nbits::Integer; L::Real=1.0)
    nbits > 0 || throw(ArgumentError("nbits must be positive"))
    N = 2^nbits
    h = L / N
    coordinates = (0:(N - 1)) .* h
    # Different frequencies in x and y ensure that an accidental axis swap is
    # detected by both the first- and second-derivative comparisons. The 1/2
    # factor keeps the field divergence-free.
    u = [sin(2pi * x / L) * cos(2pi * y / L) for x in coordinates, y in coordinates]
    v = [-0.5 * cos(2pi * x / L) * sin(2pi * y / L) for x in coordinates, y in coordinates]
    return u, v, h
end

function _periodic_shift(field::AbstractMatrix, offset::Integer, axis::Integer)
    axis in (1, 2) || throw(ArgumentError("axis must be 1 (x) or 2 (y)"))
    shifts = axis == 1 ? (offset, 0) : (0, offset)
    return circshift(field, shifts)
end

"""Direct eighth-order periodic first derivative along dense-grid `axis`."""
function dns_first_derivative(field::AbstractMatrix, h::Real, axis::Integer)
    coefficients = (4 / 5, -1 / 5, 4 / 105, -1 / 280)
    derivative = zeros(promote_type(eltype(field), typeof(float(h))), size(field))
    for (distance, coefficient) in enumerate(coefficients)
        # np.roll(f, -distance) - np.roll(f, distance)
        derivative .+= (coefficient / h) .* (
            _periodic_shift(field, -distance, axis) .-
            _periodic_shift(field, distance, axis)
        )
    end
    return derivative
end

"""Direct eighth-order periodic second derivative along dense-grid `axis`."""
function dns_second_derivative(field::AbstractMatrix, h::Real, axis::Integer)
    coefficients = (8 / 5, -1 / 5, 8 / 315, -1 / 560)
    derivative = (-205 / (72h^2)) .* field
    for (distance, coefficient) in enumerate(coefficients)
        derivative .+= (coefficient / h^2) .* (
            _periodic_shift(field, -distance, axis) .+
            _periodic_shift(field, distance, axis)
        )
    end
    return derivative
end

Dx_dns(field, h) = dns_first_derivative(field, h, 1)
Dy_dns(field, h) = dns_first_derivative(field, h, 2)
Dxx_dns(field, h) = dns_second_derivative(field, h, 1)
Dyy_dns(field, h) = dns_second_derivative(field, h, 2)

"""Return maximum absolute and relative L2 errors between two dense fields."""
function field_errors(actual::AbstractMatrix, reference::AbstractMatrix)
    size(actual) == size(reference) || throw(DimensionMismatch("field sizes differ"))
    difference = actual .- reference
    absolute = maximum(abs, difference)
    relative = norm(difference) / max(norm(reference), eps(Float64))
    return (; absolute, relative)
end
