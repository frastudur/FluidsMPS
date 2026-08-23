using ITensors, ITensorMPS
include("../Optimizer.jl")
using .Optimizer 
using CSV
include("test_differential_operators.jl")
penalty=2.5e5
viscosity=0.0
nbits=4
dq=1.0/(2^nbits-1)
dt = 0.1 * 2.0^-(nbits-1)
@show dt
maxdim=64

sites = siteinds("Qudit", nbits, dim=4)

ops = Dict{String, MPO}()
function IdMPO(s::Vector{Index{Int64}})
    Im=Matrix(1.0I, 4, 4)
    Iop=[op(Im, si) for si in s]
    return MPO(Iop)
end


ops["d1x"] = Diff_1_8_x(dq, sites)
ops["d1y"] = Diff_1_8_y(dq, sites)
ops["d2x"] = apply(ops["d1x"], ops["d1x"], maxdim=maxdim) #Diff_2_8_x(dq, sites)
ops["d2y"] = apply(ops["d1y"], ops["d1y"], maxdim=maxdim) #Diff_2_8_y(dq, sites)
ops["d1x_d1y"] = apply(ops["d1x"], ops["d1y"], maxdim=maxdim)

Opt=QuantumFluidsOpt(nbits, ops, penalty, viscosity, dt, dq)

getinfo(Opt)

function uniform_velocity(nbits, dq; ux_value=1.0, uy_value=-0.25, L=1.0)
    grid_size = 2^nbits
    return (
        fill(Float64(ux_value), grid_size, grid_size),
        fill(Float64(uy_value), grid_size, grid_size),
    )
end

vx, vy= uniform_velocity(nbits, dq)
vx_mps = dense_field_to_mps(vx, sites; cutoff=0.0)
vy_mps = dense_field_to_mps(vy, sites; cutoff=0.0)


vx_t, vy_t = time_evolution(Opt, vx_mps, vy_mps, 500*dt; tol=1e-6, maxiter=10, maxdim=maxdim, maxsweeps=10)