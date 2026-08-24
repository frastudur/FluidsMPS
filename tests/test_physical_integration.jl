using Test
using LinearAlgebra
using Printf
using Dates
using ITensors
using ITensorMPS

include("../Optimizer/Optimizer.jl")
using .Optimizer

# Reuse the exact dense <-> QIFS-MPS conversion helpers already used by the
# differential-operator tests. The file's tests do not run when it is included.
include("test_differential_operators.jl")

const DEFAULT_NBITS = 4
const DEFAULT_DT = 1.0e-3
const DEFAULT_VIS = 0.5
const DEFAULT_MU = 2.5e5
const RESULTS_DIR = joinpath(@__DIR__, "physical_integration_results")

csv_escape(value) = "\"" * replace(string(value), "\"" => "\"\"") * "\""

function save_test_results(filename, test_name, description, metrics;
                           nbits, dt, vis, mu, solver_tolerance, maxiter)
    mkpath(RESULTS_DIR)
    timestamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")
    path = joinpath(RESULTS_DIR, filename)
    headers = (
        "timestamp", "test_name", "test_description", "metric",
        "metric_description", "measured_value", "expected_condition",
        "tolerance", "status", "nbits", "grid_size", "dt", "vis",
        "mu", "solver_tolerance", "maxiter",
    )
    open(path, "w") do io
        println(io, join(headers, ","))
        for metric in metrics
            values = (
                timestamp, test_name, description, metric.name,
                metric.description, metric.value, metric.expected,
                metric.tolerance, metric.passed ? "PASS" : "FAIL",
                nbits, 2^nbits, dt, vis, mu, solver_tolerance, maxiter,
            )
            println(io, join(csv_escape.(values), ","))
        end
    end
    println("CSV results saved to: ", path)
    return path
end

function make_optimizer(sites, h; mu=DEFAULT_MU, vis=DEFAULT_VIS, dt=DEFAULT_DT)
    operators = Dict{String,MPO}(
        "d1x" => Diff_1_8_x(h, sites),
        "d1y" => Diff_1_8_y(h, sites),
        "d2x" => Diff_2_8_x(h, sites),
        "d2y" => Diff_2_8_y(h, sites),
    )
    operators["d1x_d1y"] = apply(
        operators["d1x"], operators["d1y"];
        alg="naive", cutoff=0.0,
    )
    return QuantumFluidsOpt(length(sites), operators, mu, vis, dt, h)
end

kinetic_energy(ux, uy, h) = 0.5 * h^2 * (sum(abs2, ux) + sum(abs2, uy))

function relative_velocity_error(ux, uy, reference_x, reference_y)
    numerator = sqrt(norm(ux - reference_x)^2 + norm(uy - reference_y)^2)
    denominator = sqrt(norm(reference_x)^2 + norm(reference_y)^2)
    return numerator / max(denominator, eps(Float64))
end

function relative_divergence(ux, uy, h)
    divergence = Dx_dns(ux, h) + Dy_dns(uy, h)
    velocity_norm = sqrt(norm(ux)^2 + norm(uy)^2)
    return norm(divergence) / max(velocity_norm, eps(Float64))
end

function uniform_velocity(nbits; ux_value=1.0, uy_value=-0.25, L=1.0)
    grid_size = 2^nbits
    h = L / grid_size
    return (
        fill(Float64(ux_value), grid_size, grid_size),
        fill(Float64(uy_value), grid_size, grid_size),
        h,
    )
end

function divergence_free_mode(nbits, mode; amplitude=1.0, L=1.0)
    grid_size = 2^nbits
    h = L / grid_size
    coordinates = (0:(grid_size - 1)) .* h
    wave_number = 2pi * mode / L
    ux = [amplitude * sin(wave_number * x) * cos(wave_number * y)
          for x in coordinates, y in coordinates]
    uy = [-amplitude * cos(wave_number * x) * sin(wave_number * y)
          for x in coordinates, y in coordinates]
    return ux, uy, h, wave_number
end

function evolve_one_step(ux, uy, h; mu=DEFAULT_MU, vis=DEFAULT_VIS,
                         dt=DEFAULT_DT, eps=1.0e-6, maxiter=100, maxsweeps=100)
    nbits = round(Int, log2(size(ux, 1)))
    sites = siteinds("Qudit", nbits; dim=4)
    ux_mps = dense_field_to_mps(ux, sites; cutoff=0.0)
    uy_mps = dense_field_to_mps(uy, sites; cutoff=0.0)
    optimizer = make_optimizer(sites, h; mu, vis, dt)
    evolved_x, evolved_y = RK4(optimizer, ux_mps, uy_mps; eps=eps, maxiter=maxiter, maxsweeps=maxsweeps)
    return (
        mps_to_dense_field(evolved_x, sites),
        mps_to_dense_field(evolved_y, sites),
    )
end

function test_uniform_flow(; nbits=DEFAULT_NBITS, dt=DEFAULT_DT,
                           vis=DEFAULT_VIS, mu=DEFAULT_MU,
                           solver_tolerance=1.0e-6, maxiter=100, maxsweeps=100)
    uniform_tolerance = 1.0e-5
    @testset "Uniform flow is stationary" begin
        ux0, uy0, h = uniform_velocity(nbits)
        ux1, uy1 = evolve_one_step(
            ux0, uy0, h; mu, vis, dt, eps=solver_tolerance, maxiter=maxiter, maxsweeps=maxsweeps,
        )
        error = relative_velocity_error(ux1, uy1, ux0, uy0)
        @printf("uniform-flow relative error: %.6e\n", error)
        save_test_results(
            "uniform_flow.csv",
            "Uniform flow is stationary",
            "Checks that one RK4 step leaves the constant velocity field ux=1.0, uy=-0.25 unchanged.",
            [(name="relative_velocity_error",
              description="Relative L2 error of the evolved velocity against the initial uniform velocity.",
              value=error, expected="value <= tolerance",
              tolerance=uniform_tolerance, passed=error <= uniform_tolerance)];
            nbits, dt, vis, mu, solver_tolerance, maxiter,
        )
        @test error <= uniform_tolerance
    end
end

function test_fourier_mode_decay(; nbits=DEFAULT_NBITS, dt=DEFAULT_DT,
                                 vis=DEFAULT_VIS, mu=DEFAULT_MU,
                                 solver_tolerance=1.0e-6, maxiter=100, maxsweeps=100)
    mode_tolerance = 5.0e-3
    divergence_tolerance = 1.0e-5
    @testset "Divergence-free Fourier mode has the correct decay" begin
        ux0, uy0, h, k = divergence_free_mode(nbits, 1)
        ux1, uy1 = evolve_one_step(
            ux0, uy0, h; mu, vis, dt, eps=solver_tolerance, maxiter=maxiter, maxsweeps=maxsweeps,
        )
        amplitude = exp(-2 * vis^2 * k^2 * dt)
        expected_x = amplitude .* ux0
        expected_y = amplitude .* uy0
        error = relative_velocity_error(ux1, uy1, expected_x, expected_y)
        divergence = relative_divergence(ux1, uy1, h)
        @printf("Fourier-mode relative error: %.6e\n", error)
        @printf("Fourier-mode relative divergence: %.6e\n", divergence)
        save_test_results(
            "fourier_mode_decay.csv",
            "Divergence-free Fourier mode has the correct decay",
            "Checks one-step decay of Fourier mode 1 against exp(-2*vis^2*k^2*dt), and checks that the evolved field remains divergence-free.",
            [(name="relative_velocity_error",
              description="Relative L2 error against the analytically decayed velocity field.",
              value=error, expected="value <= tolerance",
              tolerance=mode_tolerance, passed=error <= mode_tolerance),
             (name="relative_divergence",
              description="L2 norm of discrete divergence divided by the velocity L2 norm.",
              value=divergence, expected="value <= tolerance",
              tolerance=divergence_tolerance,
              passed=divergence <= divergence_tolerance)];
            nbits, dt, vis, mu, solver_tolerance, maxiter,
        )
        @test error <= mode_tolerance
        @test divergence <= divergence_tolerance
    end
end

function test_kinetic_energy_decay(; nbits=DEFAULT_NBITS, dt=DEFAULT_DT,
                                   vis=DEFAULT_VIS, mu=DEFAULT_MU,
                                   solver_tolerance=1.0e-6, maxiter=100, maxsweeps=100)
    mode_tolerance = 5.0e-3
    @testset "Kinetic energy decays at the physical rate" begin
        ux0, uy0, h, k = divergence_free_mode(nbits, 1)
        ux1, uy1 = evolve_one_step(
            ux0, uy0, h; mu, vis, dt, eps=solver_tolerance, maxiter=maxiter, maxsweeps=maxsweeps,
        )
        energy_0 = kinetic_energy(ux0, uy0, h)
        energy_1 = kinetic_energy(ux1, uy1, h)
        expected_energy = energy_0 * exp(-4 * vis^2 * k^2 * dt)
        relative_energy_error = abs(energy_1 - expected_energy) / energy_0
        @printf("initial energy: %.12e\n", energy_0)
        @printf("one-step energy: %.12e\n", energy_1)
        @printf("expected energy: %.12e\n", expected_energy)
        energy_upper_bound = energy_0 * (1 + 1.0e-10)
        save_test_results(
            "kinetic_energy_decay.csv",
            "Kinetic energy decays at the physical rate",
            "Checks that mode-1 kinetic energy does not increase and follows E(dt)=E(0)*exp(-4*vis^2*k^2*dt).",
            [(name="one_step_kinetic_energy",
              description="Kinetic energy after one RK4 step; it must not exceed the initial energy apart from roundoff.",
              value=energy_1, expected="value <= tolerance (upper bound)",
              tolerance=energy_upper_bound, passed=energy_1 <= energy_upper_bound),
             (name="relative_energy_error",
              description="Absolute difference between measured and analytical one-step energy, divided by initial energy.",
              value=relative_energy_error, expected="value <= tolerance",
              tolerance=mode_tolerance,
              passed=relative_energy_error <= mode_tolerance)];
            nbits, dt, vis, mu, solver_tolerance, maxiter,
        )
        @test energy_1 <= energy_0 * (1 + 1.0e-10)
        @test relative_energy_error <= mode_tolerance
    end
end

function test_wavelength_decay(; nbits=DEFAULT_NBITS, dt=DEFAULT_DT,
                               vis=DEFAULT_VIS, mu=DEFAULT_MU,
                               solver_tolerance=1.0e-6, maxiter=100, maxsweeps=100)
    @testset "Shorter wavelengths decay faster" begin
        ux_low, uy_low, h, _ = divergence_free_mode(nbits, 1)
        ux_high, uy_high, _, _ = divergence_free_mode(nbits, 2)
        low_x, low_y = evolve_one_step(
            ux_low, uy_low, h; mu, vis, dt, eps=solver_tolerance, maxiter=maxiter, maxsweeps=maxsweeps,
        )
        high_x, high_y = evolve_one_step(
            ux_high, uy_high, h; mu, vis, dt, eps=solver_tolerance, maxiter=maxiter, maxsweeps=maxsweeps,
        )
        low_ratio = sqrt(kinetic_energy(low_x, low_y, h) /
                         kinetic_energy(ux_low, uy_low, h))
        high_ratio = sqrt(kinetic_energy(high_x, high_y, h) /
                          kinetic_energy(ux_high, uy_high, h))
        @printf("mode-1 amplitude ratio: %.12e\n", low_ratio)
        @printf("mode-2 amplitude ratio: %.12e\n", high_ratio)
        save_test_results(
            "wavelength_decay.csv",
            "Shorter wavelengths decay faster",
            "Compares one-step amplitude decay of Fourier modes 1 and 2; mode 2 must decay more than mode 1.",
            [(name="mode_1_amplitude_ratio",
              description="Square root of final-to-initial kinetic-energy ratio for Fourier mode 1.",
              value=low_ratio, expected="0 <= value <= tolerance (upper bound)",
              tolerance=1 + 1.0e-10,
              passed=0 <= low_ratio <= 1 + 1.0e-10),
             (name="mode_2_amplitude_ratio",
              description="Square root of final-to-initial kinetic-energy ratio for Fourier mode 2.",
              value=high_ratio, expected="0 <= value < mode_1_amplitude_ratio",
              tolerance=low_ratio, passed=0 <= high_ratio < low_ratio)];
            nbits, dt, vis, mu, solver_tolerance, maxiter,
        )
        @test 0 <= high_ratio < low_ratio <= 1 + 1.0e-10
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_uniform_flow()
  #  test_fourier_mode_decay()
  #  test_kinetic_energy_decay()
  #  test_wavelength_decay()
end
