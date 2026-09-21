import QuanticsGrids as QG
using QuanticsTCI: quanticscrossinterpolate
import TensorCrossInterpolation as TCI
using TCIITensorConversion
using ITensors,ITensorMPS
include("../Optimizer/Optimizer.jl")
using .Optimizer 
using HDF5

#load parameters
include("loadconfigs.jl")
dq=1.0/(2^nbits-1)
dt = 0.1 * 2.0^-(nbits-1)


grid = QG.DiscretizedGrid{2}(nbits, (0,0), (1,1); includeendpoint = true)

u0=1.0

function maxnorm(f1::Function, f2::Function)
    maxval = 0.0
    num_points = 2^nbits
    xrange=range(0, 1, num_points)
    yrange=range(0, 1, num_points)
    for x in xrange
        for y in yrange
            val = sqrt( f1(x,y)^2 + f2(x,y)^2 )
            if val > maxval
                maxval = val
            end
        end
    end
    return maxval
end

function j1_fun(y; xmin=0.4, xmax=0.6, h=0.005, u0=u0)
    return u0 / 2 * ( tanh( (y - xmin)/h ) - tanh( (y - xmax)/h ) - 1 )
end

function d1_fun(x,y; ymin=0.4, ymax=0.6, h=0.005, u0=u0)
    out = (2 / h^2) * ( (y - ymax) * exp( -(y - ymax)^2 / h^2 ) + (y - ymin) * exp( -(y - ymin)^2 / h^2 ) )
    return out * ( sin(8π * x) + sin(24π * x) + sin(6π * x) ) 
end

function d2_fun(x,y; ymin=0.4, ymax=0.6, h=0.005, u0=u0)
    out = π * ( exp( -(y - ymax)^2 / h^2 ) + exp( -(y - ymin)^2 / h^2 ) )
    return out * ( 8*cos(8π * x) + 24*cos(24π * x) + 6*cos(6π * x) )  
end

deltavar=u0/(40.0 * maxnorm(d1_fun, d2_fun))

D1_fun = (x,y) -> deltavar * d1_fun(x,y)
D2_fun = (x,y) -> deltavar * d2_fun(x,y)

ux = (x,y) -> j1_fun(y) + D1_fun(x,y)
uy = (x,y) -> D2_fun(x,y)
print("Initial velocity field defined. ")


coordinates=range(0, 1, 2^nbits)
norm_ux=sqrt(sum([ux(x,y)^2 for x in coordinates, y in coordinates]))
norm_uy=sqrt(sum([uy(x,y)^2 for x in coordinates, y in coordinates]))

# build mps with QuanticsTCI
u1Q, rank1, error1 = quanticscrossinterpolate(Float64, ux, grid; nrandominitpivot=50, nsearchglobalpivot=50)
u2Q, rank2, error2 = quanticscrossinterpolate(Float64, uy, grid; nrandominitpivot=50, nsearchglobalpivot=50)
@show error1
@show error2
# convert to ITensorMPS format
ttx=TCI.TensorTrain(u1Q.tci)
tty=TCI.TensorTrain(u2Q.tci)

sites = siteinds("Qudit", nbits, dim=4)
ux=ITensorMPS.MPS(ttx, sites=sites)
uy=ITensorMPS.MPS(tty, sites=sites)

truncate!(ux, maxdim=maxdim)
truncate!(uy, maxdim=maxdim)


@show maxlinkdim(ux)
@show maxlinkdim(uy)
@show norm(ux)
@show norm(uy)

@assert isapprox(norm(ux), norm_ux; atol=1e-6)
@assert isapprox(norm(uy), norm_uy; atol=1e-6)

u = [ux, uy]

#prepare output file for the mps
fname="$output_folder/velocity_field.h5"
h5open(fname, "w") do file
    write(file, "ux_t0", ux)
    write(file, "uy_t0", uy)
end


function callback(vx, vy, t)
    h5open(fname, "cw") do file
        write(file, "ux_t$(t)", vx)
        write(file, "uy_t$(t)", vy)
    end
end

#Differential operators
ops = Dict{String, MPO}()
ops["d1x"] = Diff_1_8_x(dq, sites)
ops["d1y"] = Diff_1_8_y(dq, sites)
ops["d2x"] = Diff_2_8_x(dq, sites)
ops["d2y"] = Diff_2_8_y(dq, sites)
ops["d1x_d1y"] = apply(ops["d1x"], ops["d1y"])

ops["rank-3-delta"]=MPO([delta(sites[i], sites[i]', sites[i]'') for i in 1:length(sites)]) #3 rank kronecker delta MPO for convective terms

Opt=QuantumFluidsOpt(nbits, ops, penalty, viscosity, dt, dq)

t0 = time()
vx_t, vy_t = time_evolution(Opt, ux, uy, ttotal; tol=1e-6, maxiter=maxiter, maxdim=maxdim, maxsweeps=maxsweeps, callback=callback)
t1 = time()
println("Time evolution took ", t1 - t0, " seconds.")