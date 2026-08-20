using Test
using LinearAlgebra
using Printf
using ITensors
using ITensorMPS

include("../Optimizer.jl")
using .Optimizer

# Reuse the exact dense <-> QIFS-MPS conversion helpers already used by the
# differential-operator tests. The file's tests do not run when it is included.
include("test_differential_operators.jl")

const DEFAULT_NBITS = 4
const DEFAULT_DT = 1.0e-3
const DEFAULT_RE = 2.0
const DEFAULT_MU = 2.5e5

function make_optimizer(sites, h; mu=DEFAULT_MU, Re=DEFAULT_RE, dt=DEFAULT_DT)
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
    return QuantumFluidsOpt(length(sites), operators, mu, Re, dt, h)
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

function evolve_one_step(ux, uy, h; mu=DEFAULT_MU, Re=DEFAULT_RE,
                         dt=DEFAULT_DT, eps=1.0e-6, maxiter=100)
    nbits = round(Int, log2(size(ux, 1)))
    sites = siteinds("Qudit", nbits; dim=4)
    ux_mps = dense_field_to_mps(ux, sites; cutoff=0.0)
    uy_mps = dense_field_to_mps(uy, sites; cutoff=0.0)
    optimizer = make_optimizer(sites, h; mu, Re, dt)
    evolved_x, evolved_y = RK4(optimizer, ux_mps, uy_mps; eps, maxiter)
    return (
        mps_to_dense_field(evolved_x, sites),
        mps_to_dense_field(evolved_y, sites),
    )
end

function run_physical_tests(; nbits=DEFAULT_NBITS, dt=DEFAULT_DT,
                            Re=DEFAULT_RE, mu=DEFAULT_MU,
                            solver_tolerance=1.0e-6, maxiter=100)
    # These tolerances include time discretization, the finite-difference
    # stencil, CG error, and possible MPS roundoff. Tighten them as the
    # implementation matures.
    uniform_tolerance = 1.0e-5
    mode_tolerance = 5.0e-3
    divergence_tolerance = 1.0e-5

    @testset "Physical Navier-Stokes tests (diffusion, no convection)" begin
        @testset "Uniform flow is stationary" begin
            ux0, uy0, h = uniform_velocity(nbits)
            ux1, uy1 = evolve_one_step(
                ux0, uy0, h; mu, Re, dt,
                eps=solver_tolerance, maxiter,
            )
            error = relative_velocity_error(ux1, uy1, ux0, uy0)
            @printf("uniform-flow relative error: %.6e\n", error)
            @test error <= uniform_tolerance
        end

        @testset "Divergence-free Fourier mode has the correct decay" begin
            ux0, uy0, h, k = divergence_free_mode(nbits, 1)
            ux1, uy1 = evolve_one_step(
                ux0, uy0, h; mu, Re, dt,
                eps=solver_tolerance, maxiter,
            )

            viscosity = inv(Re^2)
            amplitude = exp(-2 * viscosity * k^2 * dt)
            expected_x = amplitude .* ux0
            expected_y = amplitude .* uy0
            error = relative_velocity_error(ux1, uy1, expected_x, expected_y)
            divergence = relative_divergence(ux1, uy1, h)

            @printf("Fourier-mode relative error: %.6e\n", error)
            @printf("Fourier-mode relative divergence: %.6e\n", divergence)
            @test error <= mode_tolerance
            @test divergence <= divergence_tolerance
        end

        @testset "Kinetic energy decays at the physical rate" begin
            ux0, uy0, h, k = divergence_free_mode(nbits, 1)
            ux1, uy1 = evolve_one_step(
                ux0, uy0, h; mu, Re, dt,
                eps=solver_tolerance, maxiter,
            )

            energy_0 = kinetic_energy(ux0, uy0, h)
            energy_1 = kinetic_energy(ux1, uy1, h)
            expected_energy = energy_0 * exp(-4 * inv(Re^2) * k^2 * dt)
            relative_energy_error = abs(energy_1 - expected_energy) / energy_0

            @printf("initial energy: %.12e\n", energy_0)
            @printf("one-step energy: %.12e\n", energy_1)
            @printf("expected energy: %.12e\n", expected_energy)
            @test energy_1 <= energy_0 * (1 + 1.0e-10)
            @test relative_energy_error <= mode_tolerance
        end

        @testset "Shorter wavelengths decay faster" begin
            ux_low, uy_low, h, _ = divergence_free_mode(nbits, 1)
            ux_high, uy_high, _, _ = divergence_free_mode(nbits, 2)

            low_x, low_y = evolve_one_step(
                ux_low, uy_low, h; mu, Re, dt,
                eps=solver_tolerance, maxiter,
            )
            high_x, high_y = evolve_one_step(
                ux_high, uy_high, h; mu, Re, dt,
                eps=solver_tolerance, maxiter,
            )

            low_ratio = sqrt(
                kinetic_energy(low_x, low_y, h) /
                kinetic_energy(ux_low, uy_low, h)
            )
            high_ratio = sqrt(
                kinetic_energy(high_x, high_y, h) /
                kinetic_energy(ux_high, uy_high, h)
            )

            @printf("mode-1 amplitude ratio: %.12e\n", low_ratio)
            @printf("mode-2 amplitude ratio: %.12e\n", high_ratio)
            @test 0 <= high_ratio < low_ratio <= 1 + 1.0e-10
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_physical_tests()
end

