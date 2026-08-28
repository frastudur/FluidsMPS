struct Params
    mu::Float64 #penalty parameter 
    vis::Float64 #Viscosity number
    dt::Float64 #time step
    nbits::Int #number of MPS sites
    dq::Float64 #grid spacing in both x and y directions
end

mutable struct QuantumFluidsOpt
    params::Params #parameters for the optimization
    ops::Dict{String, MPO} #dictionary of MPOs for differential operators
    A::Dict{Symbol, Vector{ITensor}} #hold contraction of current MPS with previous time MPS
    H::Dict{String, Vector{ITensor}} #holds contraction matrix elements Hij 
    D::Dict{String, Vector{ITensor}} #diffusion terms
    C1::Dict{String, Vector{ITensor}} #convective term 1 Hadamard product times partial derivative
    C2::Dict{String, Vector{ITensor}} #convective term 2 partial derivative times Hadamard product
    center::Int
end

function QuantumFluidsOpt(nbits::Int, ops::Dict{String, MPO}, mu::Float64, vis::Float64, dt::Float64, dq::Float64)
    params = Params(mu, vis, dt, nbits, dq)
 
    Ax, Ay=Vector{ITensor}(undef, nbits + 1), Vector{ITensor}(undef, nbits + 1)
    Ax[1]=ITensor(1.0)
    Ax[end]=ITensor(1.0)
    Ay[1]=ITensor(1.0)
    Ay[end]=ITensor(1.0)
    A=Dict(:x=>Ax, :y=>Ay)
    
    keys=["x,x", "x,y", "y,y"]
    values=[Vector{ITensor}(undef, nbits + 1) for _ in 1:length(keys)]
    H=Dict(zip(keys, values))
    for key in keys
        H[key][1]=ITensor(1.0)
        H[key][end]=ITensor(1.0)
    end

    keys=["x,x,x", "x,y,x", "y,x,y", "y,y,y"]  
    D=Dict(key => Vector{ITensor}(undef, nbits + 1) for key in keys)
    C1=Dict(key => Vector{ITensor}(undef, nbits + 1) for key in keys)
    C2=Dict(key => Vector{ITensor}(undef, nbits + 1) for key in keys)

    for key in keys
        D[key][1]=ITensor(1.0)
        D[key][end]=ITensor(1.0)

        C1[key][1]=ITensor(1.0)
        C1[key][end]=ITensor(1.0)
        C2[key][1]=ITensor(1.0)
        C2[key][end]=ITensor(1.0)
    end
    return QuantumFluidsOpt(params, ops, A, H, D, C1, C2, 0)
end


function getinfo(optim::QuantumFluidsOpt)
    println("Parameters:")
    println("penalty term: ", optim.params.mu)
    println("Kinematic viscosity: ", optim.params.vis)
    println("dt: ", optim.params.dt)
    println("Resolution (number of sites of the MPS): ", optim.params.nbits)
    println("spacing (dq): ", optim.params.dq)
    println("Differential operators: ", keys(optim.ops))
    println("Center site: ", optim.center)
end


function reset!(optim::QuantumFluidsOpt)
    #reset the contraction matrices to identity
    nbits=optim.params.nbits
    for i in 1:nbits + 1
        optim.A[:x][i]=ITensor(1.0)
        optim.A[:y][i]=ITensor(1.0)
    end
    for key in keys(optim.H)
        for i in 1:nbits + 1
            optim.H[key][i]=ITensor(1.0)
        end
    end
    for key in keys(optim.D)
        for i in 1:nbits + 1
            optim.D[key][i]=ITensor(1.0)
            optim.C1[key][i]=ITensor(1.0)
            optim.C2[key][i]=ITensor(1.0)
            
        end
    end
    optim.center=0
end


function movecenter!(optim::QuantumFluidsOpt, site::Int, v::Dict{Symbol,MPS}, a::Dict{Symbol,MPS}, b::Dict{Symbol,MPS}, n::Int, op_c1::Dict{String, MPO}, op_c2::Dict{String, MPO})
    for i in [:x, :y]
        orthogonalize!(v[i], site)
        orthogonalize!(a[i], site)
        orthogonalize!(b[i], site)
    end

    if optim.center == 0
        for i=2:site
            buildleft!(optim, v, a, b, i, op_c1, op_c2)
        end
        for i=n:-1:site+1
            buildright!(optim, v, a, b, i, n, op_c1, op_c2)
        end
    elseif site > optim.center
        for i=optim.center+1:site
            buildleft!(optim, v, a, b, i, op_c1, op_c2)
        end
    else
        for i=optim.center:-1:site+1
            buildright!(optim, v, a, b, i, n, op_c1, op_c2)
        end
    end
    optim.center = site
end


function buildleft!(optim::QuantumFluidsOpt, v::Dict{Symbol,MPS}, a::Dict{Symbol,MPS}, b::Dict{Symbol,MPS}, site::Int, op_c1::Dict{String, MPO}, op_c2::Dict{String, MPO})
    vx=dag(v[:x])
    vy=dag(v[:y])

    vxp=prime(vx)
    vyp=prime(vy)

    optim.A[:x][site] = optim.A[:x][site-1] * vx[site-1] * a[:x][site-1]
    optim.A[:y][site] = optim.A[:y][site-1] * vy[site-1] * a[:y][site-1]

    optim.H["x,x"][site] = optim.H["x,x"][site-1] * vxp[site-1] * optim.ops["d2x"][site-1] * v[:x][site-1]
    optim.H["x,y"][site] = optim.H["x,y"][site-1] * vxp[site-1] * optim.ops["d1x_d1y"][site-1] * v[:y][site-1]
    optim.H["y,y"][site] = optim.H["y,y"][site-1] * vyp[site-1] * optim.ops["d2y"][site-1] * v[:y][site-1]

    optim.D["x,x,x"][site] = optim.D["x,x,x"][site-1] * vxp[site-1] * optim.ops["d2x"][site-1] * b[:x][site-1]
    optim.D["x,y,x"][site] = optim.D["x,y,x"][site-1] * vxp[site-1] * optim.ops["d2y"][site-1] * b[:x][site-1]
    optim.D["y,x,y"][site] = optim.D["y,x,y"][site-1] * vyp[site-1] * optim.ops["d2x"][site-1] * b[:y][site-1]
    optim.D["y,y,y"][site] = optim.D["y,y,y"][site-1] * vyp[site-1] * optim.ops["d2y"][site-1] * b[:y][site-1]


    optim.C1["x,x,x"][site] = optim.C1["x,x,x"][site-1] * vxp[site-1] * op_c1["x"][site-1] * b[:x][site-1]
    optim.C1["x,y,x"][site] = optim.C1["x,y,x"][site-1] * vxp[site-1] * op_c1["y"][site-1] * b[:x][site-1]
    optim.C1["y,x,y"][site] = optim.C1["y,x,y"][site-1] * vyp[site-1] * op_c1["x"][site-1] * b[:y][site-1]
    optim.C1["y,y,y"][site] = optim.C1["y,y,y"][site-1] * vyp[site-1] * op_c1["y"][site-1] * b[:y][site-1]


    optim.C2["x,x,x"][site] = optim.C2["x,x,x"][site-1] * vxp[site-1] * op_c2["x"][site-1] * b[:x][site-1]
    optim.C2["x,y,x"][site] = optim.C2["x,y,x"][site-1] * vxp[site-1] * op_c2["y"][site-1] * b[:x][site-1]
    optim.C2["y,x,y"][site] = optim.C2["y,x,y"][site-1] * vyp[site-1] * op_c2["x"][site-1] * b[:y][site-1]
    optim.C2["y,y,y"][site] = optim.C2["y,y,y"][site-1] * vyp[site-1] * op_c2["y"][site-1] * b[:y][site-1]
end

function buildright!(optim::QuantumFluidsOpt, v::Dict{Symbol,MPS}, a::Dict{Symbol,MPS}, b::Dict{Symbol,MPS}, site::Int, n::Int, op_c1::Dict{String, MPO}, op_c2::Dict{String, MPO})
    vx=dag(v[:x])
    vy=dag(v[:y])

    vxp=prime(vx)
    vyp=prime(vy)

    optim.A[:x][site] = optim.A[:x][site+1] * vx[site] * a[:x][site]
    optim.A[:y][site] = optim.A[:y][site+1] * vy[site] * a[:y][site]

    optim.H["x,x"][site] = optim.H["x,x"][site+1] * vxp[site] * optim.ops["d2x"][site] * v[:x][site]
    optim.H["x,y"][site] = optim.H["x,y"][site+1] * vxp[site] * optim.ops["d1x_d1y"][site] * v[:y][site]
    optim.H["y,y"][site] = optim.H["y,y"][site+1] * vyp[site] * optim.ops["d2y"][site] * v[:y][site]

    optim.D["x,x,x"][site] = optim.D["x,x,x"][site+1] * vxp[site] * optim.ops["d2x"][site] * b[:x][site]
    optim.D["x,y,x"][site] = optim.D["x,y,x"][site+1] * vxp[site] * optim.ops["d2y"][site] * b[:x][site]
    optim.D["y,x,y"][site] = optim.D["y,x,y"][site+1] * vyp[site] * optim.ops["d2x"][site] * b[:y][site]
    optim.D["y,y,y"][site] = optim.D["y,y,y"][site+1] * vyp[site] * optim.ops["d2y"][site] * b[:y][site]

    optim.C1["x,x,x"][site] = optim.C1["x,x,x"][site+1] * vxp[site] * op_c1["x"][site] * b[:x][site]
    optim.C1["x,y,x"][site] = optim.C1["x,y,x"][site+1] * vxp[site] * op_c1["y"][site] * b[:x][site]
    optim.C1["y,x,y"][site] = optim.C1["y,x,y"][site+1] * vyp[site] * op_c1["x"][site] * b[:y][site]
    optim.C1["y,y,y"][site] = optim.C1["y,y,y"][site+1] * vyp[site] * op_c1["y"][site] * b[:y][site]

    optim.C2["x,x,x"][site] = optim.C2["x,x,x"][site+1] * vxp[site] * op_c2["x"][site] * b[:x][site]
    optim.C2["x,y,x"][site] = optim.C2["x,y,x"][site+1] * vxp[site] * op_c2["y"][site] * b[:x][site]
    optim.C2["y,x,y"][site] = optim.C2["y,x,y"][site+1] * vyp[site] * op_c2["x"][site] * b[:y][site]
    optim.C2["y,y,y"][site] = optim.C2["y,y,y"][site+1] * vyp[site] * op_c2["y"][site] * b[:y][site]
end




function environment(L::ITensor, R::ITensor, Node::ITensor, Op::Union{Nothing, ITensor})
    #compute the environment of a node in the MPS network, with or without an operator
    # L: left environment tensor
    # R: right environment tensor
    # Node: the node for which we want to compute the environment
    # Op: optional operator to be applied to the node. If Op is nothing, it is ignored.
    # if Op === nothing.              if Op !==nothing

    # |-- Node--|                       |---Node---|
    # L    |    R = C_lik.              |    |     | 
    # |__     __|                       L---Op-----R
    #                                   |    |     |
    #                                   |__      __|
    if Op === nothing
        return L * Node * R
    else
        return L * Node * Op * R
    end
end


function environment(environments::AbstractVector{ITensor}, center::Int,
                     Node::ITensor, Op::Union{Nothing, ITensor}; dagger::Bool=false)
    L=environments[center]
    R=environments[center+1]
    if dagger 
        return noprime(environment(swapprime(dag(L), 1=>0), swapprime(dag(R), 1=>0), Node, swapprime(dag(Op), 1=>0)))
    end
    return noprime(environment(L, R, Node, Op))
end



function lhs(optim::QuantumFluidsOpt, candidate::Vector{ITensor}, dt2::Float64)
    i=optim.center
    cx, cy = candidate[1], candidate[2]
    mu=optim.params.mu
    
    H_xx = environment(optim.H["x,x"], i, cx, optim.ops["d2x"][i])
    H_xy = environment(optim.H["x,y"], i, cy, optim.ops["d1x_d1y"][i])
    H_yy = environment(optim.H["y,y"], i, cy, optim.ops["d2y"][i])
    H_yx = environment(optim.H["x,y"], i, cx, optim.ops["d1x_d1y"][i]; dagger=true)
    Hcx=H_xx + H_xy 
    Hcy=H_yx + H_yy 
    result=[cx - mu*dt2*Hcx, cy - mu*dt2*Hcy]
    return result
end


function linoperator(optim::QuantumFluidsOpt, v::Vector{Float64}, tau2::Float64, xinds, yinds, nx::Int, ny::Int)
    cx = ITensor(v[1:nx], xinds)
    cy = ITensor(v[nx+1:end], yinds)
    candidate=[cx, cy]
    result=lhs(optim, candidate, tau2)
    return vcat(vec.(array.(result))...)
end


function rhs(optim::QuantumFluidsOpt, a::Dict{Symbol, MPS}, b::Dict{Symbol, MPS}, tau::Float64, vis::Float64, op_c1::Dict{String, MPO}, op_c2::Dict{String, MPO})
    i=optim.center
    
    beta_x = environment(optim.A[:x], i, a[:x][i], nothing)
    #diffusion
    beta_x += (tau * vis) * environment(optim.D["x,x,x"], i, b[:x][i], optim.ops["d2x"][i])
    beta_x += (tau * vis) * environment(optim.D["x,y,x"], i, b[:x][i], optim.ops["d2y"][i])
    #convection
    beta_x -= (tau * 0.5) * environment(optim.C1["x,x,x"], i, b[:x][i], op_c1["x"][i]) 
    beta_x -= (tau * 0.5) * environment(optim.C1["x,y,x"], i, b[:x][i], op_c1["y"][i]) 
    beta_x -= (tau * 0.5) * environment(optim.C2["x,x,x"], i, b[:x][i], op_c2["x"][i]) 
    beta_x -= (tau * 0.5) * environment(optim.C2["x,y,x"], i, b[:x][i], op_c2["y"][i]) 


    beta_y = environment(optim.A[:y], i, a[:y][i], nothing)
    beta_y += (tau * vis) * environment(optim.D["y,x,y"], i, b[:y][i], optim.ops["d2x"][i])
    beta_y += (tau * vis) * environment(optim.D["y,y,y"], i, b[:y][i], optim.ops["d2y"][i])

    beta_y -= (tau * 0.5) * environment(optim.C1["y,x,y"], i, b[:y][i], op_c1["x"][i]) 
    beta_y -= (tau * 0.5) * environment(optim.C1["y,y,y"], i, b[:y][i], op_c1["y"][i]) 
    beta_y -= (tau * 0.5) * environment(optim.C2["y,x,y"], i, b[:y][i], op_c2["x"][i]) 
    beta_y -= (tau * 0.5) * environment(optim.C2["y,y,y"], i, b[:y][i], op_c2["y"][i]) 
    return [beta_x, beta_y]
end



#optimization of [vx,vy] with the RK4 methods: it takes intermediate steps [ax,ay] and [bx,by]
function optimize(optim, vx, vy, ax, ay, bx, by, tau; tol=1e-6, maxiter=100, maxsweeps=100)
    v=Dict(:x=>copy(vx), :y=>copy(vy))
    a=Dict(:x=>copy(ax), :y=>copy(ay))
    b=Dict(:x=>copy(bx), :y=>copy(by))

    tau2=tau*tau
    vis=optim.params.vis
    dq=optim.params.dq
    nbits=optim.params.nbits
    ops=optim.ops

    C1xx, C2xx = convective_operators(b[:x], ops["rank-3-delta"], ops["d1x"])
    C1yy, C2yy = convective_operators(b[:y], ops["rank-3-delta"], ops["d1y"])

    op_convective1=Dict("x"=>C1xx, "y"=>C1yy)
    op_convective2=Dict("x"=>C2xx, "y"=>C2yy)
    
    function update!(i::Int)
        movecenter!(optim, i, v, a, b, nbits, op_convective1, op_convective2)

        cx = v[:x][optim.center]
        cy = v[:y][optim.center]

        xinds=inds(cx)
        yinds=inds(cy)

        nx=prod(dim.(xinds))
        ny=prod(dim.(yinds))

        operator= x -> linoperator(optim, x, tau2, xinds, yinds, nx, ny)
        
        M=FunctionMap{Float64,false}(operator, nx+ny)

        beta = rhs(optim, a, b, tau, vis, op_convective1, op_convective2)
        beta_x = permute(beta[1], xinds)
        beta_y = permute(beta[2], yinds)
        beta_vec = vcat(vec.(array.([beta_x, beta_y]))...)
        @assert length(beta_vec) == nx + ny "Length of beta does not match the expected dimension size. Expected: $(nx + ny), got: $(length(beta_vec))"
        

        cvec = vcat(vec.(array.([cx, cy]))...)
        @assert length(cvec) == nx + ny "Length of cvec does not match the expected dimension size. Expected: $(nx + ny), got: $(length(cvec))"
    
        cg!(cvec, M, beta_vec, maxiter=maxiter, reltol=tol, abstol=tol^2, verbose=false, log=false)
        cg_residual = M * cvec - beta_vec
        cg_residual_norm = sqrt(real(dot(cg_residual, cg_residual)))
        println("site $i CG residual norm: $cg_residual_norm")

        v[:x][optim.center] = ITensor(cvec[1:nx], xinds)
        v[:y][optim.center] = ITensor(cvec[nx+1:end], yinds)
    end

    E_0=1e-10
    E_1=2*tol
    it=0
    while abs((E_1 - E_0)/E_0) > tol && it < maxsweeps
        for j in 1:nbits
            update!(j)
            
        end
        for j in nbits-1:-1:1
            update!(j)
        end
        it += 1
        E_0 = E_1
        cvec = vcat(vec.(array.([v[:x][optim.center], v[:y][optim.center]]))...)
        E_1 = dot(cvec, cvec)
        if it == maxsweeps
            println("Warning: Maximum number of sweeps reached without convergence.")
        end
    end
    return v[:x], v[:y]
end


function RK4(optim, ux, uy; tol=1e-6, maxiter=100, maxdim=16, maxsweeps=100)
    #RK4 method for time evolution of the MPS
    #ux, uy: MPS for the velocity field in x and y directions
    #tol: tolerance for the optimization
    #maxiter: maximum number of iterations for the conjugate gradient solver
    dt=optim.params.dt

    U1x, U1y = optimize(optim, ux, uy, ux/4, uy/4, ux, uy, dt/6; tol=tol, maxiter=maxiter, maxsweeps=maxsweeps)


    b = 3 .* [U1x, U1y] +  (1/4) .* [ux, uy]

    reset!(optim)
    U2x, U2y = optimize(optim, ux, uy, ux/4, uy/4, b[1], b[2], dt/3; tol=tol, maxiter=maxiter, maxsweeps=maxsweeps)

    b = (3/2) .* [U2x, U2y] +  (5/8) .* [ux, uy]
    reset!(optim)
    U3x, U3y = optimize(optim, ux, uy, ux/4, uy/4, b[1], b[2], dt/3; tol=tol, maxiter=maxiter, maxsweeps=maxsweeps)

    b = 3 .* [U3x, U3y] +  (1/4) .* [ux, uy]
    reset!(optim)
    U4x, U4y = optimize(optim, ux, uy, ux/4, uy/4, b[1], b[2], dt/6; tol=tol, maxiter=maxiter, maxsweeps=maxsweeps)

    result = [U1x, U1y] + [U2x, U2y] + [U3x, U3y] + [U4x, U4y] 

    truncate!.(result; maxdim=maxdim)
    return result[1], result[2]
end

function time_evolution(optim, vx, vy, tmax; tol=1e-6, maxiter=100, maxdim=16, maxsweeps=100, callback::Function=(args...)->nothing)
    dt=optim.params.dt
    t=0.0
    nsnapshots=50 
    snapshot_interval = tmax / nsnapshots
    snapshot_counter = 0
    while t < tmax
        reset!(optim)
        vx, vy = RK4(optim, vx, vy; tol=tol, maxiter=maxiter, maxdim=maxdim, maxsweeps=maxsweeps)
        t += dt
        if t >= snapshot_counter * snapshot_interval
            @show t
            callback(vx, vy, t)
        end
        snapshot_counter += 1
    end
    return vx, vy
end
