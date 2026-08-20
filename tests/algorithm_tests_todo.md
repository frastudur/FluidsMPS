# Algorithm tests to revisit later

These tests are intentionally not implemented yet. They should be considered
only if the physical validation tests expose unexpected behavior.

## Local tensor representation

- Verify that flattening and reconstructing the two center tensors is an exact
  round trip, including index order and complex element type.
- Verify that concatenating and splitting `cvec` uses the same `x/y` ordering
  as `beta` and `linoperator`.
- Verify that the tensors returned by `lhs` and `rhs` have exactly the open
  indices expected for the current optimization center.

## Linear operator

- Materialize the local matrix on a very small problem by applying
  `linoperator` to every canonical basis vector.
- Compare that matrix with an independently assembled dense block operator
  `I - mu*tau^2*H`.
- Test Hermiticity through both `norm(M - M')` and random-vector identities
  `dot(x, M*y) == dot(M*x, y)`.
- Check that the local operator is positive definite for every parameter set
  used with conjugate gradient.
- Verify explicitly that the `H_yx` action equals the adjoint of `H_xy`.

## Conjugate gradient

- Compare the CG solution with `M \ beta` on a small dense local problem.
- Measure the true relative residual `norm(M*cvec-beta)/norm(beta)` after every
  solve.
- Check convergence from zero, random, and current-center initial guesses.
- Record iteration count versus `mu`, `tau`, bond dimension, and condition
  number.

## Environments and boundaries

- Compare cached left/right environments with contractions rebuilt from
  scratch at every site.
- Test centers `1`, an interior site, and `N` explicitly.
- Verify that `ITensor(1.0)` acts as the empty environment at both boundaries.
- Check that no unwanted link indices remain open after an environment
  contraction.

## Cache invalidation between RK stages

- Call `optimize` with a new `b` whose MPS addition created new link IDs.
- Compare reuse of a correctly reset optimizer with a newly constructed
  `QuantumFluidsOpt` object.
- Confirm that `A`, `H`, and `D` environments are rebuilt whenever their input
  MPS changes.
- Repeat the same optimization twice and check determinism.

## Sweep behavior

- Compare one left-to-right/right-to-left sweep with the reference
  `optimize_sweep` implementation on a small problem.
- Check that moving the orthogonality center does not change the represented
  dense field.
- Track the objective and residual after each local update and complete sweep.
- Add a maximum number of outer sweeps and test both convergence and controlled
  non-convergence.

## MPS accuracy

- Compare complete small-grid results against a dense implementation.
- Repeat with increasing maximum bond dimension and decreasing truncation
  cutoff.
- Record discarded weight and distinguish tensor-compression error from solver
  error.

