#=
- Calculate the intermediate velocity $u^*$ by solving the momentum equation without pressure term - aka Burger's equation: 
  $$
   u^∗ = u^s − ∆t(C^s − D^s)  #euler step
   $$
  
- Compute the Fourier transform $\hat{u}^*$.
- Project $\hat{u}^*$ to divergence-free space.
  $$
  u^{s+1} = u^∗ - κ \frac{u^∗ \cdot κ}{κ^2}
 
  $$
- Compute the inverse Fourier transform to retrieve $\hat{u}^{s+1}$.
=#


include("../differential_operators/direct_numerical_simulation.jl")
using FFTW
#using HDF5



#load parameters
nbits=10
dq=1.0/(2^nbits-1)
dt = 0.1 * 2.0^-(nbits-1)
vis = 0.01



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


## Convection term 
function hadamard_prod(A, B)
    return A .* B
end

function convection_first_term(Ax::AbstractArray, Ay::AbstractArray, dq::Float64)
    fields_grads_ux=[Dx_dns(Ax, dq), Dy_dns(Ax, dq)]
    fields_grads_uy=[Dx_dns(Ay, dq), Dy_dns(Ay, dq)]

    Cx = hadamard_prod(Ax, fields_grads_ux[1]) + hadamard_prod(Ay, fields_grads_ux[2])
    Cy = hadamard_prod(Ax, fields_grads_uy[1]) + hadamard_prod(Ay, fields_grads_uy[2])

    return 0.5*Cx, 0.5*Cy
end

function convection_second_term(Ax::AbstractArray, Ay::AbstractArray, dq::Float64)
    H=Dict{String, AbstractArray}()

    H["x,x"]=hadamard_prod(Ax, Ax)
    H["x,y"]=hadamard_prod(Ax, Ay)
    H["y,x"]=hadamard_prod(Ay, Ax)
    H["y,y"]=hadamard_prod(Ay, Ay)

    Cx = Dx_dns(H["x,x"], dq) + Dy_dns(H["y,x"], dq)
    Cy = Dx_dns(H["x,y"], dq) + Dy_dns(H["y,y"], dq)

    return 0.5*Cx, 0.5*Cy
end

function convection(Ax::AbstractArray, Ay::AbstractArray, dq::Float64)
    C1x, C1y = convection_first_term(Ax, Ay, dq)
    C2x, C2y = convection_second_term(Ax, Ay, dq)

    return C1x + C2x, C1y + C2y
end

function diffusion(Ax::AbstractArray, Ay::AbstractArray, dq::Float64, vis::Float64)
    Dx = vis * (Dxx_dns(Ax, dq) + Dyy_dns(Ax, dq))
    Dy = vis * (Dxx_dns(Ay, dq) + Dyy_dns(Ay, dq))

    return Dx, Dy
end


function euler_step(U::AbstractArray, dq::Float64, vis::Float64, dt::Float64)

    Ax = U[:, :, 1]
    Ay = U[:, :, 2]

    Cx, Cy = convection(Ax, Ay, dq)
    Dx, Dy = diffusion(Ax, Ay, dq, vis)

    U_new = similar(U)
    U_new[:, :, 1] = Ax + dt * (-Cx + Dx)
    U_new[:, :, 2] = Ay + dt * (-Cy + Dy)

    return divergence_free_mode(U_new, dq)
end


function RK4(U::AbstractArray, dq::Float64, vis::Float64, dt::Float64)
    k1 = euler_step(U, dq, vis, dt)
    k2 = euler_step(U .+ 0.5 * dt * k1, dq, vis, dt)
    k3 = euler_step(U .+ 0.5 * dt * k2, dq, vis, dt)
    k4 = euler_step(U .+ dt * k3, dq, vis, dt)

    U_new = U .+ (dt / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
    return U_new
end


function divergence_free_mode(U::AbstractArray, dq::Float64)
    # Compute the Fourier transform of the velocity field
    U_hat = fft(U, [1, 2])

    # Fourier symbols of the eighth-order Dx_dns and Dy_dns stencils.
    function modified_wave_numbers(n)
        theta = (2π / n) .* (0:n-1)
        k = (2 / dq) .* ((4/5) .* sin.(theta) .- (1/5) .* sin.(2 .* theta) .+
                        (4/105) .* sin.(3 .* theta) .- (1/280) .* sin.(4 .* theta))
        k[1] = 0.0
        if iseven(n)
            k[n÷2 + 1] = 0.0
        end
        return k
    end

    KX = reshape(modified_wave_numbers(size(U, 1)), :, 1)
    KY = reshape(modified_wave_numbers(size(U, 2)), 1, :)

    # Project to divergence-free space
    k_squared = KX.^2 .+ KY.^2
    # Preserve modes in the nullspace of both first-derivative operators.
    k_squared[k_squared .== 0] .= 1.0
    div_free_factor = (KX .* U_hat[:, :, 1] .+ KY .* U_hat[:, :, 2]) ./ k_squared
    U_hat[:, :, 1] .-= KX .* div_free_factor
    U_hat[:, :, 2] .-= KY .* div_free_factor

    # Inverse Fourier transform to get the updated velocity field
    U_new = ifft(U_hat, [1, 2])
    return real(U_new)
end




coordinates=range(0, 1, 2^nbits)

#defined type for the velocity field component
Tf=AbstractArray{Float64,2}
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

