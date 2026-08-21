using ITensors, ITensorMPS
include("../Optimizer.jl")
using .Optimizer 
using CSV

mu=2.5e5
ni=1e+5
nbits=4
dq=1.0/(2^nbits-1)
dt = 0.1 * 2.0^-(nbits-1)
maxdim=16

sites = siteinds("Qudit", nbits, dim=4)

ops = Dict{String, MPO}()

ops["d1x"] = Diff_1_8_x(dq, sites)
ops["d1y"] = Diff_1_8_y(dq, sites)
ops["d2x"] = Diff_2_8_x(dq, sites)
ops["d2y"] = Diff_2_8_y(dq, sites)
ops["d1x_d1y"] = apply(ops["d1x"], ops["d1y"], maxdim=maxdim)


Opt=QuantumFluidsOpt(nbits, ops, mu, ni, dt, dq)

getinfo(Opt)

vx=random_mps(sites)
vy=random_mps(sites)

vx_t, vy_t = time_evolution(Opt, vx, vy, 1.0; eps=1e-6, maxiter=100, maxdim=maxdim)