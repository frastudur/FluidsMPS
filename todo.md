# TODO

## Differential MPO validation

- [x] Compare `Diff_1_8_x`, `Diff_1_8_y`, `Diff_2_8_x`, and `Diff_2_8_y`
  against the corresponding dense QIFS finite-difference routines.
- [x] Verify the local ITensor state ordering `00, 10, 01, 11` and the x/y
  axis convention.
- [ ] Test eighth-order convergence against exact analytical derivatives on
  several grid resolutions. The error should scale as `O(h^8)` before
  floating-point error or MPS truncation becomes significant.

# DONE