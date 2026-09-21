# Mapping an MPS to a two-dimensional field

This note follows the convention established where the field MPS is created in
`time_evolve/script.jl`. It is not an arbitrary choice: it comes from the
default `unfoldingscheme=:fused` of `QuanticsGrids.DiscretizedGrid` and is
preserved by `QuanticsTCI`, `TensorTrain`, and `ITensorMPS.MPS`.

The convention is:

- the real-space array is indexed as `field[x + 1, y + 1]`;
- `x` and `y` start from zero, while Julia array indices start from one;
- bit `0` is the **most significant bit** (MSB).

For a two-dimensional grid, QuanticsGrids builds one fused site per resolution
level. Its internal index table lists `(yᵢ,xᵢ)`, so that `xᵢ` is the
faster-varying digit. Consequently, the one-based local index is

```text
sᵢ = 1 + xᵢ + 2yᵢ.
```

QuanticsGrids assigns `bitnumber = 1` the weight `2^(n-1)`. Therefore the first
MPS site contains `(x₀,y₀)`, the coordinate MSBs.

The complete construction chain is therefore

```text
f(x,y)
  -> QuanticsGrids fused indices [1+x₀+2y₀, 1+x₁+2y₁, ...]
  -> QuanticsTCI / TensorTrain (same site order and local indices)
  -> ITensor MPS (same site order and local indices).
```

For `n` bits per coordinate, the field has size

```text
L × L, where L = 2^n.
```

The coordinates are reconstructed as

```text
x = (x₀ x₁ ... xₙ₋₁)₂
y = (y₀ y₁ ... yₙ₋₁)₂
```

or, equivalently,

```text
x = x₀ 2^(n-1) + x₁ 2^(n-2) + ... + xₙ₋₁
y = y₀ 2^(n-1) + y₁ 2^(n-2) + ... + yₙ₋₁.
```

## Small example: `n = 2`

The field is `4 × 4`. Consider the point

```text
x = 1 = 01₂
y = 2 = 10₂.
```

Therefore,

```text
x₀ = 0, x₁ = 1
y₀ = 1, y₁ = 0.
```

In Julia this point is stored at

```julia
field[x + 1, y + 1] == field[2, 3]
```

The same point can be encoded using either of the following MPS layouts.

## 1. One dimension-4 index per scale

The contracted tensor has the layout

```text
T[(x₀,y₀), (x₁,y₁), ..., (xₙ₋₁,yₙ₋₁)].
```

Each pair of bits is fused into one Julia index:

```text
local index = 1 + xᵢ + 2yᵢ.
```

| `(xᵢ,yᵢ)` | Local index |
|---|---:|
| `(0,0)` | 1 |
| `(1,0)` | 2 |
| `(0,1)` | 3 |
| `(1,1)` | 4 |

In the small example,

```text
(x₀,y₀) = (0,1) -> 3
(x₁,y₁) = (1,0) -> 2,
```

so the mapping is

```julia
field[2, 3] = T[3, 2]
```

The general conversion is:

```julia
function qudit_mps_to_field(state::MPS)
    sites = siteinds(state)
    @assert all(dim(s) == 4 for s in sites)

    n = length(sites)
    L = 2^n
    T = Array(contract(state), sites...)
    field = Matrix{eltype(T)}(undef, L, L)
    local_states = Vector{Int}(undef, n)

    for x in 0:L-1, y in 0:L-1
        for i in 1:n
            shift = n - i  # i=1 extracts the MSB
            xbit = (x >> shift) & 1
            ybit = (y >> shift) & 1
            local_states[i] = 1 + xbit + 2ybit
        end
        field[x + 1, y + 1] = T[local_states...]
    end

    return field
end
```

## 2. Factorized dimension-2 indices

Here the `x` and `y` bits are separate MPS sites. The contracted tensor has the
layout

```text
T[x₀, y₀, x₁, y₁, ..., xₙ₋₁, yₙ₋₁].
```

Because Julia indices start from one, a bit `0` selects index `1` and a bit `1`
selects index `2`.

In the small example,

```text
(x₀,y₀,x₁,y₁) = (0,1,1,0),
```

and hence

```julia
field[2, 3] = T[1, 2, 2, 1]
```

The following conversion produces the same `field[x + 1, y + 1]` convention as
the dimension-4 function:

```julia
function factorized_mps_to_field(state::MPS)
    sites = siteinds(state)
    N = length(sites)

    @assert iseven(N)
    @assert all(dim(s) == 2 for s in sites)

    n = N ÷ 2
    L = 2^n
    T = Array(contract(state), sites...)

    # Original axes: x₀,y₀,x₁,y₁,...,xₙ₋₁,yₙ₋₁
    # Reverse each group because Julia's first reshape axis varies fastest.
    x_order = collect(N - 1:-2:1)
    y_order = collect(N:-2:2)
    order = vcat(x_order, y_order)

    reordered = permutedims(T, order)
    return reshape(reordered, L, L)
end
```

For `n = 2`, this permutation is

```text
(x₀,y₀,x₁,y₁) -> (x₁,x₀,y₁,y₀).
```

The reversal does not exchange `x` and `y`. It only makes `x₀` and `y₀` the
most significant bits after Julia's column-major `reshape`.

## Summary

For the same physical point, the two representations give

```text
dimension-4 MPS:
field[x+1,y+1] = T[1+x₀+2y₀, ..., 1+xₙ₋₁+2yₙ₋₁]

factorized MPS:
field[x+1,y+1] = T[x₀+1, y₀+1, ..., xₙ₋₁+1, yₙ₋₁+1].
```

Neither conversion should call `normalize!`: mapping to real space should not
change the amplitude of the stored field. If `Plots.heatmap(x, y, ...)` is used,
pass `field'`, because the plotting function expects rows to correspond to `y`
and columns to `x`.

## 3. Amplitudes indexed by a Yao basis-state integer

Suppose a function `amplitude(b)` returns the amplitude of basis state `b`, and
the Yao qubits are ordered as

```text
(x₀, y₀, x₁, y₁, ..., xₙ₋₁, yₙ₋₁),
```

where `x₀` and `y₀` are the coordinate MSBs. In Yao's basis-state integer,
qubit 1 occupies bit position 0 (the integer's LSB). Thus, site `i` selects a
coordinate bit from MSB to LSB, but writes it into integer positions `2i` and
`2i+1`:

```julia
"""Encode `(x,y)` for Yao qubits `(x₀,y₀,x₁,y₁,...)`."""
function interleave_xy(x::Integer, y::Integer, n::Integer)
    n >= 0 || throw(ArgumentError("n must be nonnegative"))
    0 <= x < 2^n || throw(ArgumentError("x does not fit in $n bits"))
    0 <= y < 2^n || throw(ArgumentError("y does not fit in $n bits"))

    b = zero(promote_type(typeof(x), typeof(y)))
    for i in 0:n-1
        coordinate_shift = n - 1 - i
        xbit = (x >> coordinate_shift) & 1
        ybit = (y >> coordinate_shift) & 1
        b |= xbit << (2i)
        b |= ybit << (2i + 1)
    end
    return b
end

"""Convert Yao amplitudes to `field[x+1,y+1]`."""
function amplitudes_to_field(amplitude, Nqubits::Integer)
    Nqubits >= 0 || throw(ArgumentError("Nqubits must be nonnegative"))
    iseven(Nqubits) || throw(ArgumentError("Nqubits must be even"))

    n = Nqubits ÷ 2
    L = 2^n
    first_value = amplitude(interleave_xy(0, 0, n)).value
    field = Matrix{typeof(first_value)}(undef, L, L)

    for x in 0:L-1, y in 0:L-1
        b = interleave_xy(x, y, n)
        field[x + 1, y + 1] = amplitude(b).value
    end
    return field
end
```

For the `n = 2` example, `x = 01₂` and `y = 10₂`, so the qubits are

```text
(x₀,y₀,x₁,y₁) = (0,1,1,0).
```

Since Yao stores qubit 1 in the integer LSB, the basis-state integer is

```text
b = (0110)₂ = 6
     ↑↑↑↑
     y₁x₁y₀x₀    (usual MSB-to-LSB display of the integer)
```

and the mapping is

```julia
field[2, 3] = amplitude(6).value
```

The older loop that extracted `(x >> k) & 1` wrote the coordinate LSB first.
It therefore corresponded to qubits `(xₙ₋₁,yₙ₋₁,...,x₀,y₀)`, which is the
reverse scale ordering and is not compatible with the convention used above.

## Project Configuration

```toml
name = "FluidsMPS"
uuid = "a145dc3f-64e7-4e99-82c9-01e25e5f26dc"
authors = ["Francesca"]
version = "0.1.0"

[deps]
ArgParse = "c7e460c6-2fb9-53a9-8c5b-16f535851c63"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
ITensorMPS = "0d1a4710-d33b-49a5-8f18-73bdf49b47e2"
ITensors = "9136182c-28ba-11e9-034c-db9fb085ebd5"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
QuanticsGrids = "634c7f73-3e90-4749-a1bd-001b8efc642d"
QuanticsTCI = "b11687fd-3a1c-4c41-97d0-998ab401d50e"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
TCIITensorConversion = "9f0aa9f4-9415-4e6a-8795-331ebf40aa04"
TOML = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
TensorCrossInterpolation = "b261b2ec-6378-4871-b32e-9173bb050604"

[compat]
julia = "1.8"
```