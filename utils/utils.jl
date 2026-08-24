function mps_to_dense_field2D(state::MPS, sites::Vector{<:Index})
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