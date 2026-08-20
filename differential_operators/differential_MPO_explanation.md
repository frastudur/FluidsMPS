# Construction of the QIFS differential MPOs

This note explains how QIFS constructs the periodic eighth-order finite-difference MPOs

- `Diff_1_8_x`: first derivative in x;
- `Diff_1_8_y`: first derivative in y;
- `Diff_2_8_x`: second derivative in x;
- `Diff_2_8_y`: second derivative in y.

The primary reference is `QIFS/2D_QIS/numpy/differential_mpo.py`. The Julia implementation is `FluidsMPS/differential_operators.jl`.

## 1. Main idea

A centered finite-difference derivative is a weighted sum of translated copies of a field. If `T_r` denotes periodic translation by `r` grid cells, then

```text
D = sum_{r=-4}^{4} c_r T_r.
```

The eighth-order stencil contains translations by up to four cells. The purpose of the MPO is therefore to represent the periodic translation operators `T_-4, ..., T_4` and their coefficients compactly.

The MPO tensors act like a finite-state machine:

1. Their physical indices update the binary digits of a coordinate.
2. Their bond indices propagate carries or borrows between binary digits.
3. Their final tensor closes the possible paths and applies the finite-difference coefficients.

## 2. Scale-resolved representation of a two-dimensional field

For a `2^n x 2^n` grid, write each coordinate using `n` bits:

```text
x = (x_1 x_2 ... x_n)_2,
y = (y_1 y_2 ... y_n)_2.
```

QIFS groups bits at the same scale. MPS site `k` therefore represents the pair

```text
(x_k, y_k).
```

Each physical site has dimension four:

| ITensor state | Binary pair | Meaning |
|---:|:---:|---|
| 1 | `00` | `x_k = 0`, `y_k = 0` |
| 2 | `10` | `x_k = 1`, `y_k = 0` |
| 3 | `01` | `x_k = 0`, `y_k = 1` |
| 4 | `11` | `x_k = 1`, `y_k = 1` |

For example, with three bits,

```text
x = 5 = (101)_2,
y = 3 = (011)_2
```

becomes the scale-resolved word

```text
(x_1,y_1) (x_2,y_2) (x_3,y_3) = (10)(01)(11),
```

which corresponds to local ITensor states `(2, 3, 4)`.

This state numbering is the Julia/ITensor fused-index convention used by the
MPO implementation: x is the faster local digit. It differs from reading the
two-character labels as an ordinary base-two integer.

The first MPS site contains the most significant pair of bits. The final site contains the least significant pair.

## 3. A periodic translation is binary addition

Translating a field by one grid cell means incrementing or decrementing the corresponding binary coordinate.

For addition by one:

```text
0 + 1 -> 1, and the carry stops;
1 + 1 -> 0, and a carry continues to the next bit.
```

For subtraction by one:

```text
1 - 1 -> 0, and the borrow stops;
0 - 1 -> 1, and a borrow continues to the next bit.
```

This is a local process except for the carry or borrow. An MPO can represent it efficiently because its bond index communicates this information between adjacent scale sites.

Periodic boundary conditions are built into binary overflow:

```text
11...11 + 1 -> 00...00,
00...00 - 1 -> 11...11.
```

## 4. Meaning of the basic MPO bond dimension

For translations by one cell, the MPO needs three bond states:

| Bond state | Meaning |
|---:|---|
| 1 | No active carry: copy the physical state |
| 2 | Propagate the carry associated with addition |
| 3 | Propagate the borrow associated with subtraction |

This explains why most MPO bonds have dimension three.

This bond dimension is not the MPS truncation parameter `chi`. It is the exact number of internal automaton states used to describe the shift operation.

## 5. Local x and y shift matrices

At each site, the physical basis is ordered as

```text
00, 10, 01, 11.
```

An x translation changes the x member of the pair while leaving y unchanged. The Julia code uses

```julia
Jx[2, 1] = 1
Jx[4, 3] = 1
```

An y translation changes the y member while leaving x unchanged:

```julia
Jy[3, 1] = 1
Jy[4, 2] = 1
```

The apparent orientation of these matrices depends on the ordering of the input and output physical indices inside each ITensor. The important point is that `Jx` acts on the x bit and `Jy` acts on the y bit.

The transpose `J'` implements the complementary local transition. Combinations of `J`, `J'`, and the identity matrix form the local rules for positive and negative carry paths.

## 6. Eighth-order finite-difference stencils

The centered eighth-order approximation to the first derivative is

```text
f'(i) ~= [ f(i-4)/280 - 4 f(i-3)/105 + f(i-2)/5 - 4 f(i-1)/5
           + 4 f(i+1)/5 - f(i+2)/5 + 4 f(i+3)/105 - f(i+4)/280 ] / h.
```

Equivalently, its coefficients are

```text
r    -4       -3       -2      -1    0     1      2       3        4
c_r  1/280   -4/105    1/5    -4/5   0    4/5   -1/5     4/105  -1/280
```

The first-derivative stencil is antisymmetric:

```text
c_{-r} = -c_r.
```

The centered eighth-order approximation to the second derivative is

```text
f''(i) ~= [ -f(i-4)/560 + 8 f(i-3)/315 - f(i-2)/5 + 8 f(i-1)/5
            - 205 f(i)/72
            + 8 f(i+1)/5 - f(i+2)/5 + 8 f(i+3)/315 - f(i+4)/560 ] / h^2.
```

The second-derivative stencil is symmetric:

```text
c_{-r} = c_r.
```

The coefficient vectors in the code contain only six values because positive and negative translations share MPO carry paths. The final tensor expands those shared paths with the appropriate symmetric or antisymmetric signs.

## 7. Why one MPO bond grows from dimension 3 to 5

The three basic states are sufficient for translations by `+1` and `-1`. The eighth-order stencil also contains translations by two, three, and four cells.

Near the least significant binary sites, the MPO must distinguish whether a carry path has progressed farther than a one-cell translation. Two additional states are introduced:

| Bond state | Meaning |
|---:|---|
| 1 | Identity/no carry |
| 2 | Ordinary positive carry |
| 3 | Ordinary negative carry |
| 4 | Extended positive carry |
| 5 | Extended negative carry |

Only the bond immediately before the final MPO tensor needs these extra states. Thus the bond dimensions are

```text
3 -- 3 -- ... -- 3 -- 5.
```

Conceptually, the network is

```text
 first tensor       ordinary scale tensors      extended tensor      final tensor
      |                       |                        |                   |
      O ---- 3 ---- O ---- 3 ---- ... ---- 3 ---- O ---- 5 ---- O
      |                       |                        |                   |
```

The penultimate tensor opens the two extended paths. The final tensor closes all paths and gives them the coefficients required for translations through four grid cells.

## 8. Role of the different MPO tensors

### First tensor

The first tensor initializes the automaton paths:

- identity;
- positive translation;
- negative translation.

It has no left MPO bond because it lies at the boundary of the tensor network.

### Ordinary interior tensors

Every ordinary interior tensor propagates:

- the identity path using `I`;
- the addition path using `J` and `J'`;
- the subtraction path using `J` and `J'`.

These tensors all have bond dimension three.

### Penultimate tensor

The penultimate tensor propagates the three ordinary paths and opens the two extended carry paths. Its right bond therefore has dimension five.

### Final tensor

The final tensor has no right MPO bond. It:

1. terminates every allowed carry path;
2. combines paths that represent translations by one through four cells;
3. multiplies the paths by the finite-difference coefficients;
4. applies different signs for odd and even derivatives.

## 9. Why four constructors share the same implementation

All four operators use the same carry network. Only the axis and coefficient symmetry change:

| Constructor | Local shift matrix | Coefficients |
|---|---|---|
| `Diff_1_8_x(h, sites)` | `Jx` | Antisymmetric first derivative, scaled by `1/h` |
| `Diff_1_8_y(h, sites)` | `Jy` | Antisymmetric first derivative, scaled by `1/h` |
| `Diff_2_8_x(h, sites)` | `Jx` | Symmetric second derivative, scaled by `1/h^2` |
| `Diff_2_8_y(h, sites)` | `Jy` | Symmetric second derivative, scaled by `1/h^2` |

This is why the Julia implementation has one internal `_diff_8` constructor and four small public wrapper functions.

## 10. How the implementation should be verified

Successfully constructing an `MPO` is not enough to prove that it represents the intended operator. A strong numerical test should:

1. Choose a small grid, such as `2^4 x 2^4`.
2. Generate a random dense field.
3. Convert the field to the QIFS scale-resolved MPS ordering.
4. Apply the MPO using ITensorMPS.
5. Convert the resulting MPS back to a dense field.
6. Apply the corresponding periodic finite-difference stencil directly to the original dense field.
7. Compare the two dense results entry by entry.

This should be done separately for all four operators.

A second test should measure convergence on smooth periodic functions. If the grid spacing is halved, the discretization error should decrease approximately as

```text
error(h/2) / error(h) ~= 1 / 2^8
```

until floating-point error or tensor truncation becomes significant.

## 11. One-sentence summary

The QIFS differential MPO is a weighted superposition of periodic binary translation automata: physical indices update coordinate bits, bond indices carry addition/subtraction state between scales, and the final tensor assigns the eighth-order stencil coefficients.
