# FluidsMPS

Fluid simulations using matrix product states (MPS).

### 1. Download

You need Git, Julia (the included environment uses Julia 1.10.4), and Bash.

```bash
git clone https://github.com/frastudur/FluidsMPS.git
cd FluidsMPS
```

### 2. Set parameters

Edit [time_evolve/config.toml](time_evolve/config.toml) and save it before running.
Each parameter has a short explanation. Start with the defaults; for a shorter
run, change `ttotal` to `0.1`.

### 3. Run

Save this as `run.sh` in the `FluidsMPS` folder:

```bash
#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"

julia --project=. -e 'using Pkg; Pkg.instantiate()'
output="time_evolve/run_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$output"
echo "Results: $output"
julia --project=. time_evolve/script.jl "$output"
```

Then run from that folder:

```bash
bash run.sh
```

Results are saved in the printed folder: `velocity_field.h5` contains the velocity fields and `config.toml`
records the parameters used.
