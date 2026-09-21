include("DNS.jl")
import .DNS
using HDF5

include("loadconfigs.jl")

dq=1.0/(2^nbits-1)
dt = 0.1 * 2.0^-(nbits-1)
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
n=length(coordinates)

U=zeros(n, n, 2)
cinds = CartesianIndices(U[:, :, 1])

for I in cinds
    i, j = Tuple(I)
    x = coordinates[i]
    y = coordinates[j]
    U[i, j, 1] = ux(x, y)
    U[i, j, 2] = uy(x, y)
end

fname="$output_folder/velocity_field.h5"
h5open(fname, "w") do file
    write(file, "ux_t0", U[:, :, 1])
    write(file, "uy_t0", U[:, :, 2])
end


function callback(U, t)
    h5open(fname, "cw") do file
        write(file, "ux_t$(t)", U[:, :, 1])
        write(file, "uy_t$(t)", U[:, :, 2])
    end
end


solver_info = DNS.Info(nbits, dq, dt, viscosity, "RK4")

U_t = DNS.time_evolution(U, solver_info, ttotal; callback=callback)