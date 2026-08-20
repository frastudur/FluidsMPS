using ITensors
using ITensorMPS
using Printf

include("../differential_operators/differential_operators.jl")
include("../differential_operators/direct_numerical_simulation.jl")

"""Encode a dense QIFS field exactly as a scale-resolved ITensor MPS."""
function dense_field_to_mps(field::AbstractMatrix, sites::Vector{<:Index}; cutoff::Real=0.0)
    N, Ny = size(field)
    N == Ny || throw(ArgumentError("field must be square"))
    nbits = round(Int, log2(N))
    N == 2^nbits || throw(ArgumentError("field side length must be a power of two"))
    length(sites) == nbits || throw(ArgumentError("number of sites does not match field"))

    full_tensor = ITensor(eltype(field), sites...)
    local_states = Vector{Int}(undef, nbits)
    for x in 0:(N - 1), y in 0:(N - 1)
        for site in 1:nbits
            shift = nbits - site
            xbit = (x >> shift) & 1
            ybit = (y >> shift) & 1
            # ITensor/Julia column-major fusion makes x the faster digit:
            # state 1,2,3,4 correspond to 00,10,01,11.
            local_states[site] = 1 + xbit + 2ybit
        end
        full_tensor[(sites .=> local_states)...] = field[x + 1, y + 1]
    end
    return MPS(full_tensor, sites; cutoff)
end

"""Decode a scale-resolved ITensor MPS into a dense `(x,y)` field."""
function mps_to_dense_field(state::MPS, sites::Vector{<:Index})
    nbits = length(sites)
    N = 2^nbits
    full_tensor = contract(state)
    field = zeros(eltype(full_tensor), N, N)
    local_states = Vector{Int}(undef, nbits)
    for x in 0:(N - 1), y in 0:(N - 1)
        for site in 1:nbits
            shift = nbits - site
            xbit = (x >> shift) & 1
            ybit = (y >> shift) & 1
            local_states[site] = 1 + xbit + 2ybit
        end
        field[x + 1, y + 1] = full_tensor[(sites .=> local_states)...]
    end
    return field
end

function compare_dns_and_mpo(; nbits::Integer=4, tolerance::Real=1e-10)
    u, v, h = initial_velocity_field(nbits)
    sites = siteinds("Qudit", nbits; dim=4)
    u_mps = dense_field_to_mps(u, sites)
    v_mps = dense_field_to_mps(v, sites)

    cases = (
        ("D1x(u)", Diff_1_8_x(h, sites), u_mps, Dx_dns(u, h)),
        ("D1y(u)", Diff_1_8_y(h, sites), u_mps, Dy_dns(u, h)),
        ("D2x(u)", Diff_2_8_x(h, sites), u_mps, Dxx_dns(u, h)),
        ("D2y(u)", Diff_2_8_y(h, sites), u_mps, Dyy_dns(u, h)),
        ("D1x(v)", Diff_1_8_x(h, sites), v_mps, Dx_dns(v, h)),
        ("D1y(v)", Diff_1_8_y(h, sites), v_mps, Dy_dns(v, h)),
        ("D2x(v)", Diff_2_8_x(h, sites), v_mps, Dxx_dns(v, h)),
        ("D2y(v)", Diff_2_8_y(h, sites), v_mps, Dyy_dns(v, h)),
    )

    println("QIFS dense-vs-MPO derivative test ($(2^nbits) x $(2^nbits) grid)")
    println(rpad("operator", 12), rpad("max abs error", 20), "relative L2 error")
    all_passed = true
    results = Dict{String,NamedTuple}()
    for (name, operator, state, reference) in cases
        # cutoff=0 prevents truncation from contaminating the operator test.
        mpo_result = apply(operator, state; alg="naive", cutoff=0.0)
        actual = mps_to_dense_field(mpo_result, sites)
        errors = field_errors(actual, reference)
        results[name] = errors
        passed = errors.relative <= tolerance
        all_passed &= passed
        @printf("%-12s  %.12e    %.12e  %s\n", name, errors.absolute,
                errors.relative, passed ? "PASS" : "FAIL")
    end
    all_passed || error("at least one MPO differs from the DNS stencil; see errors above")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    compare_dns_and_mpo()
   #sites = siteinds("Qudit", 2; dim=4)
   #full_tensor = ITensor(eltype(Float64), sites...)
   #local_states = Vector{Int}(undef, 2)
   #local_states[1] = 1
   #local_states[2] = 1
   # full_tensor[(sites .=> local_states)...] = 1.0
end
