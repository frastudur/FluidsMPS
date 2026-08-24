using ITensors, ITensorMPS
include("../Optimizer/Optimizer.jl")
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
#randMPS=random_mps(sites)
#randMPO=outer(randMPS', randMPS)
ops["d1x"] = Diff_1_8_x(dq, sites)
ops["d1y"] = Diff_1_8_y(dq, sites)  
ops["d2x"] = apply(ops["d1x"], ops["d1x"], maxdim=maxdim) #Diff_2_8_x(dq, sites)
ops["d2y"] = apply(ops["d1y"], ops["d1y"], maxdim=maxdim) #Diff_2_8_y(dq, sites)
ops["d1x_d1y"] = apply(ops["d1x"], ops["d1y"], maxdim=maxdim)
ops["d1y_d1x"] = apply(ops["d1y"], ops["d1x"], maxdim=maxdim)
ops["rank-3-delta"]=MPO([delta(sites[i], sites[i]', sites[i]'') for i in 1:length(sites)]) #3 rank kronecker delta MPO for convective terms

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


v=Dict(:x => copy(vx_mps), :y => copy(vy_mps))
a=Dict(:x => copy(vx_mps)/4, :y => copy(vy_mps)/4)
b=Dict(:x => copy(vx_mps), :y => copy(vy_mps))


C1xx, C2xx = convective_operators(b[:x], ops["rank-3-delta"], ops["d1x"])
C1yy, C2yy = convective_operators(b[:y], ops["rank-3-delta"], ops["d1y"])


op_convective1=Dict("x"=>C1xx, "y"=>C1yy)
op_convective2=Dict("x"=>C2xx, "y"=>C2yy)

for i=1:nbits
    movecenter!(Opt, i, v, a, b, nbits, op_convective1, op_convective2)
    println("center projection at site : $i")

    e1=environment(Opt.A[:x], i, a[:x][i], nothing)
    e2=environment(Opt.D["x,x,x"], i, b[:x][i], Opt.ops["d2x"][i])
    e3=environment(Opt.C1["x,x,x"], i, b[:x][i], op_convective1["x"][i])
    
    @show inds(e1)

    @show inds(e2)
    
    @show inds(e3)
    
    #exact_contraction=inner(v[:y]',Opt.ops["d2y"], v[:y])
    #E=environment(Opt.H["y,y"], i, v[:y][i], Opt.ops["d2y"][i])
    #contraction=dag(v[:y][i])*E 
    #@show exact_contraction
    #@show contraction
end
    
