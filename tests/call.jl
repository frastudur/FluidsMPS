import QuanticsGrids as QG
using QuanticsTCI: quanticscrossinterpolate
import TensorCrossInterpolation as TCI
using TCIITensorConversion
using ITensors,ITensorMPS
include("../Optimizer/Optimizer.jl")
using .Optimizer 

penalty=2.5e5
viscosity=1e-5
nbits=10 #resolution of the grid
dq=1.0/(2^nbits-1)
dt = 0.1 * 2.0^-(nbits-1)
maxdim=39
dimension=2 #Space dimension
grid = QG.DiscretizedGrid{2}(nbits, (0,0), (1,1); includeendpoint = true)

u0=1.0

#show fields of grid which i do not know in advance
for field in fieldnames(typeof(grid))
    println("Field: ", field)
    println(getfield(grid, field))
end



function maxnorm(f1::Function, f2::Function, grid::QG.DiscretizedGrid{2})
    maxval = 0.0
    xrange=range(grid.lower_bound[1],grid.upper_bound[1], 2^nbits)
    yrange=range(grid.lower_bound[2],grid.upper_bound[2], 2^nbits)
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

function d1_fun(x,y; xmin=0.4, xmax=0.6, h=0.005, u0=u0)
    out = 2 / h^2 * ( (y - xmax) * exp( -(y - xmax)^2 / h^2 ) + (y - xmin) * exp( -(y - xmin)^2 / h^2 ) )
    return out * ( sin(8π * x) + sin(24π * x) + sin(6π * x) ) 
end

function d2_fun(x,y; xmin=0.4, xmax=0.6, h=0.005, u0=u0)
    out = π * ( exp( -(y - xmax)^2 / h^2 ) + exp( -(y - xmin)^2 / h^2 ) )
    return out * ( 8*cos(8π * x) + 24*cos(24π * x) + 6*cos(6π * x) )  
end

deltavar=u0/(40.0 * maxnorm(d1_fun, d2_fun, grid))

D1_fun = (x,y) -> deltavar * d1_fun(x,y)
D2_fun = (x,y) -> deltavar * d2_fun(x,y)

ux = (x,y) -> j1_fun(y) + D1_fun(x,y)
uy = (x,y) -> D2_fun(x,y)


print("Initial velocity field defined. ")

# build mps with QuanticsTCI
u1Q, rank1, error1 = quanticscrossinterpolate(Float64, ux, grid) 
u2Q, rank2, error2 = quanticscrossinterpolate(Float64, uy, grid)

# convert to ITensorMPS format
ttx=TCI.TensorTrain(u1Q.tci)
tty=TCI.TensorTrain(u2Q.tci)

sites = siteinds("Qudit", nbits, dim=2^dimension)
ux=ITensorMPS.MPS(ttx, sites=sites)
uy=ITensorMPS.MPS(tty, sites=sites)

u = [ux, uy]

#Differential operators
ops = Dict{String, MPO}()
ops["d1x"] = Diff_1_8_x(dq, sites)
ops["d1y"] = Diff_1_8_y(dq, sites)
ops["d2x"] = apply(ops["d1x"], ops["d1x"], maxdim=maxdim) #Diff_2_8_x(dq, sites)
ops["d2y"] = apply(ops["d1y"], ops["d1y"], maxdim=maxdim) #Diff_2_8_y(dq, sites)
ops["d1x_d1y"] = apply(ops["d1x"], ops["d1y"], maxdim=maxdim)

ops["rank-3-delta"]=MPO([delta(sites[i], sites[i]', sites[i]'') for i in 1:length(sites)]) #3 rank kronecker delta MPO for convective terms

Opt=QuantumFluidsOpt(nbits, ops, penalty, viscosity, dt, dq)

t0 = time()
vx_t, vy_t = time_evolution(Opt, ux, uy, 2*dt; tol=1e-6, maxiter=100, maxdim=maxdim, maxsweeps=50)
t1 = time()
println("Time evolution took ", t1 - t0, " seconds.")