module DNS
    include("../differential_operators/direct_numerical_simulation.jl")
    using FFTW

    struct Info 
        nbits::Int
        dq::Float64
        dt::Float64
        vis::Float64
        integrator::String
        
        function Info(nbits::Int, dq::Float64, dt::Float64, vis::Float64, integrator::String)
            if integrator != "Euler" && integrator != "RK4"
                error("Unsupported integrator: $integrator")
            end
            return new(nbits, dq, dt, vis, integrator)
        end
    end



    function hadamard_prod(A, B)
        return A .* B
    end


    function convection(Ax::AbstractArray, Ay::AbstractArray, params::Info)
        dq = params.dq

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
        
        C1x, C1y = convection_first_term(Ax, Ay, dq)
        C2x, C2y = convection_second_term(Ax, Ay, dq)
        return C1x + C2x, C1y + C2y
    end



    function diffusion(Ax::AbstractArray, Ay::AbstractArray, params::Info)
        dq = params.dq
        vis = params.vis
        Dx = vis * (Dxx_dns(Ax, dq) + Dyy_dns(Ax, dq))
        Dy = vis * (Dxx_dns(Ay, dq) + Dyy_dns(Ay, dq))
        return Dx, Dy
    end


    function divergence_free_mode(U::AbstractArray, dq::Float64)
        #Compute the Fourier transform of the velocity field
        U_hat = fft(U, [1, 2])

        #Fourier symbols of the eighth-order Dx_dns and Dy_dns stencils.
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

        #Project to divergence-free space
        k_squared = KX.^2 .+ KY.^2
        #Preserve modes in the nullspace of both first-derivative operators.
        k_squared[k_squared .== 0] .= 1.0
        div_free_factor = (KX .* U_hat[:, :, 1] .+ KY .* U_hat[:, :, 2]) ./ k_squared
        U_hat[:, :, 1] .-= KX .* div_free_factor
        U_hat[:, :, 2] .-= KY .* div_free_factor

        #Inverse Fourier transform to get the updated velocity field
        U_new = ifft(U_hat, [1, 2])
        return real(U_new)
    end



    function euler_step(U::AbstractArray, params::Info)
        Ax = U[:, :, 1]
        Ay = U[:, :, 2]

        Cx, Cy = convection(Ax, Ay, params)
        Dx, Dy = diffusion(Ax, Ay, params)

        U_new = similar(U)
        U_new[:, :, 1] = Ax + params.dt * (-Cx + Dx)
        U_new[:, :, 2] = Ay + params.dt * (-Cy + Dy)

        return divergence_free_mode(U_new, params.dq)
    end


    function RK4(U::AbstractArray, params::Info)
        k1 = euler_step(U, params)
        k2 = euler_step(U .+ 0.5 * params.dt * k1, params)
        k3 = euler_step(U .+ 0.5 * params.dt * k2, params)
        k4 = euler_step(U .+ params.dt * k3, params)

        U_new = U .+ (params.dt / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
        return U_new
    end


    function time_evolution(U::AbstractArray, params::Info, tmax; callback::Function=(args...)->nothing)
        t = 0.0
        nsnapshots=50
        snapshot_interval = tmax / nsnapshots
        snapshot_counter = 1
        while t < tmax
            U = RK4(U, params)
            t += params.dt

            if t + eps(Float64(tmax)) >= snapshot_counter * snapshot_interval
                # Save or process the snapshot here
                callback(U, t)
                snapshot_counter += 1
            end
        end

        return U
    end
end