struct Params
    mu::Float64 #penalty parameter 
    ni::Float64 #Viscosity number
    dt::Float64 #time step
    N::Int #number of MPS sites
    dq::Float64 #grid spacing in both x and y directions
end

mutable struct QuantumFluidsOpt
    params::Params #parameters for the optimization
    ops::Dict{String, MPO} #dictionary of MPOs for differential operators
    A::Dict{Symbol, Vector{ITensor}} #hold contraction of current MPS with previous time MPS
    H::Dict{String, Vector{ITensor}} #holds contraction matrix elements Hij or contraction Hij*Cj ?
    D::Dict{String, Vector{ITensor}} #diffusion terms
    center::Int
end

function QuantumFluidsOpt(N::Int, ops::Dict{String, MPO}, mu::Float64, ni::Float64, dt::Float64, dq::Float64)
    params = Params(mu, ni, dt, N, dq)
 
    Ax, Ay=Vector{ITensor}(undef, N), Vector{ITensor}(undef, N)
    Ax[1]=ITensor(1.0)
    Ax[N]=ITensor(1.0)
    Ay[1]=ITensor(1.0)
    Ay[N]=ITensor(1.0)
    A=Dict(:x=>Ax, :y=>Ay)
    
    keys=["x,x", "x,y", "y,y"]
    values=[Vector{ITensor}(undef, N) for _ in 1:length(keys)]
    H=Dict(zip(keys, values))

    keys=["x,x,x", "x,y,x", "y,x,y", "y,y,y"]  
    values=[Vector{ITensor}(undef, N) for _ in 1:length(keys)]

    D=Dict(zip(keys, values))
    for key in keys
        D[key][1]=ITensor(1.0)
        D[key][N]=ITensor(1.0)
    end
    return QuantumFluidsOpt(params, ops, A, H, D, 0)
end


function getinfo(optim::QuantumFluidsOpt)
    #print list of parameters, operators keys 
    println("Parameters:")
    println("penalty term: ", optim.params.mu)
    println("Kinematic viscosity: ", optim.params.ni)
    println("dt: ", optim.params.dt)
    println("Resolution (number of sites of the MPS): ", optim.params.N)
    println("spacing (dq): ", optim.params.dq)
    println("Differential operators: ", keys(optim.ops))
    println("Center site: ", optim.center)
end


function reset!(optim::QuantumFluidsOpt)
    #reset the contraction matrices to identity
    N=optim.params.N
    for i in 1:N
        optim.A[:x][i]=ITensor(1.0)
        optim.A[:y][i]=ITensor(1.0)
    end
    for key in keys(optim.H)
        for i in 1:N
            optim.H[key][i]=ITensor(1.0)
        end
    end
    for key in keys(optim.D)
        for i in 1:N
            optim.D[key][i]=ITensor(1.0)
        end
    end
    optim.center=0
end


function movecenter!(optim::QuantumFluidsOpt, site::Int, v::Dict{Symbol,MPS}, a::Dict{Symbol,MPS}, b::Dict{Symbol,MPS}, n::Int)
    for i in [:x, :y]
        orthogonalize!(v[i], site)
        orthogonalize!(a[i], site)
        orthogonalize!(b[i], site)
    end

    if optim.center == 0
        for i=1:site
            buildleft!(optim, v, a, b, i)
        end
        for i=n:-1:site
            buildright!(optim, v, a, b, i, n)
        end
    elseif site > optim.center
        for i=optim.center:site
            buildleft!(optim, v, a, b, i)
        end
    else
        for i=optim.center:-1:site
            buildright!(optim, v, a, b, i, n)
        end
    end
    optim.center = site
end


function buildleft!(optim::QuantumFluidsOpt, v::Dict{Symbol,MPS}, a::Dict{Symbol,MPS}, b::Dict{Symbol,MPS}, site::Int)
    vx=prime(linkinds, dag(v[:x]))
    vy=prime(linkinds, dag(v[:y]))
    
    vxp=prime(siteinds, vx)
    vyp=prime(siteinds, vy)
    
    if site==2
        optim.A[:x][site-1] = vx[site-1] * a[:x][site-1]
        optim.A[:y][site-1] = vy[site-1] * a[:y][site-1]
        
        optim.H["x,x"][site-1] = vxp[site-1] * optim.ops["d2x"][site-1] * v[:x][site-1]
        optim.H["x,y"][site-1] = vxp[site-1] * optim.ops["d1x_d1y"][site-1] * v[:y][site-1]
        optim.H["y,y"][site-1] = vyp[site-1] * optim.ops["d2y"][site-1] * v[:y][site-1]
        
        optim.D["x,x,x"][site-1] = vxp[site-1] * optim.ops["d2x"][site-1] * b[:x][site-1]
        optim.D["x,y,x"][site-1] = vxp[site-1] * optim.ops["d2y"][site-1] * b[:x][site-1]
        optim.D["y,x,y"][site-1] = vyp[site-1] * optim.ops["d2x"][site-1] * b[:y][site-1]
        optim.D["y,y,y"][site-1] = vyp[site-1] * optim.ops["d2y"][site-1] * b[:y][site-1]
    

    elseif site>2
        optim.A[:x][site-1] = optim.A[:x][site-2] * vx[site-1] * a[:x][site-1]
        optim.A[:y][site-1] = optim.A[:y][site-2] * vy[site-1] * a[:y][site-1]

        optim.H["x,x"][site-1] = optim.H["x,x"][site-2] * vxp[site-1] * optim.ops["d2x"][site-1] * v[:x][site-1]
        optim.H["x,y"][site-1] = optim.H["x,y"][site-2] * vxp[site-1] * optim.ops["d1x_d1y"][site-1] * v[:y][site-1]
        optim.H["y,y"][site-1] = optim.H["y,y"][site-2] * vyp[site-1] * optim.ops["d2y"][site-1] * v[:y][site-1]

        optim.D["x,x,x"][site-1] = optim.D["x,x,x"][site-2] * vxp[site-1] * optim.ops["d2x"][site-1] * b[:x][site-1]
        optim.D["x,y,x"][site-1] = optim.D["x,y,x"][site-2] * vxp[site-1] * optim.ops["d2y"][site-1] * b[:x][site-1]
        optim.D["y,x,y"][site-1] = optim.D["y,x,y"][site-2] * vyp[site-1] * optim.ops["d2x"][site-1] * b[:y][site-1]
        optim.D["y,y,y"][site-1] = optim.D["y,y,y"][site-2] * vyp[site-1] * optim.ops["d2y"][site-1] * b[:y][site-1]

    end
end

function buildright!(optim::QuantumFluidsOpt, v::Dict{Symbol,MPS}, a::Dict{Symbol,MPS}, b::Dict{Symbol,MPS}, site::Int, n::Int)
    vx=prime(linkinds, dag(v[:x]))
    vy=prime(linkinds, dag(v[:y]))

    vxp=prime(siteinds, vx)
    vyp=prime(siteinds, vy)
    

    if site==n-1
        optim.A[:x][site+1] = vx[site+1] * a[:x][site+1]
        optim.A[:y][site+1] = vy[site+1] * a[:y][site+1]
        
        optim.H["x,x"][site+1] = vxp[site+1] * optim.ops["d2x"][site+1] * v[:x][site+1]
        optim.H["x,y"][site+1] = vxp[site+1] * optim.ops["d1x_d1y"][site+1] * v[:y][site+1]
        optim.H["y,y"][site+1] = vyp[site+1] * optim.ops["d2y"][site+1] * v[:y][site+1]
        
        optim.D["x,x,x"][site+1] = vxp[site+1] * optim.ops["d2x"][site+1] * b[:x][site+1]
        optim.D["x,y,x"][site+1] = vxp[site+1] * optim.ops["d2y"][site+1] * b[:x][site+1]
        optim.D["y,x,y"][site+1] = vyp[site+1] * optim.ops["d2x"][site+1] * b[:y][site+1]
        optim.D["y,y,y"][site+1] = vyp[site+1] * optim.ops["d2y"][site+1] * b[:y][site+1]
    
    elseif site<n-1
        optim.A[:x][site+1] = optim.A[:x][site+2] * vx[site+1] * a[:x][site+1]
        optim.A[:y][site+1] = optim.A[:y][site+2] * vy[site+1] * a[:y][site+1]

        optim.H["x,x"][site+1] = optim.H["x,x"][site+2] * vxp[site+1] * optim.ops["d2x"][site+1] * v[:x][site+1]
        optim.H["x,y"][site+1] = optim.H["x,y"][site+2] * vxp[site+1] * optim.ops["d1x_d1y"][site+1] * v[:y][site+1]
        optim.H["y,y"][site+1] = optim.H["y,y"][site+2] * vyp[site+1] * optim.ops["d2y"][site+1] * v[:y][site+1]
        
        optim.D["x,x,x"][site+1] = optim.D["x,x,x"][site+2] * vxp[site+1] * optim.ops["d2x"][site+1] * b[:x][site+1]
        optim.D["x,y,x"][site+1] = optim.D["x,y,x"][site+2] * vxp[site+1] * optim.ops["d2y"][site+1] * b[:x][site+1]
        optim.D["y,x,y"][site+1] = optim.D["y,x,y"][site+2] * vyp[site+1] * optim.ops["d2x"][site+1] * b[:y][site+1]
        optim.D["y,y,y"][site+1] = optim.D["y,y,y"][site+2] * vyp[site+1] * optim.ops["d2y"][site+1] * b[:y][site+1]
    end
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
    L = center == firstindex(environments) ? ITensor(1.0) : environments[center-1]
    R = center == lastindex(environments) ? ITensor(1.0) : environments[center+1]
    if dagger 
        return  noprime(environment(swapprime(dag(L), 1=>0), swapprime(dag(R), 1=>0), Node, swapprime(dag(Op), 1=>0)))
    end
    return noprime(environment(L, R, Node, Op))
end



function lhs(optim, candidate, dt2)
    i=optim.center
    cx, cy = candidate[1], candidate[2]
    mu=optim.params.mu
    
    H_xx = mu*dt2*environment(optim.H["x,x"], i, cx, optim.ops["d2x"][i])
    H_xy = mu*dt2*environment(optim.H["x,y"], i, cy, optim.ops["d1x_d1y"][i])
    H_yy = mu*dt2*environment(optim.H["y,y"], i, cy, optim.ops["d2y"][i])
    H_yx = mu*dt2*environment(optim.H["x,y"], i, cx, optim.ops["d1x_d1y"][i]; dagger=true)

    Hcx=H_xx + H_xy 
    Hcy=H_yx + H_yy 
    C=[Hcx, Hcy]
    return [cx - Hcx, cy - Hcy]
end


function linoperator(optim, v, tau2, xinds, yinds, nx, ny)
    cx = ITensor(v[1:nx], xinds)
    cy = ITensor(v[nx+1:end], yinds)
    candidate=[cx, cy]
    result=lhs(optim, candidate, tau2)
    return vcat(vec.(array.(result))...)
end


function rhs(optim, a, b, tau, ni) 
    i=optim.center
    beta_x = environment(optim.A[:x], i, a[:x][i], nothing)
    beta_x -= (tau * ni * ni) * environment(optim.D["x,x,x"], i, b[:x][i], optim.ops["d2x"][i])
    beta_x -= (tau * ni * ni) * environment(optim.D["x,y,x"], i, b[:x][i], optim.ops["d2y"][i])

    beta_y = environment(optim.A[:y], i, a[:y][i], nothing)
    beta_y -= (tau * ni * ni) * environment(optim.D["y,x,y"], i, b[:y][i], optim.ops["d2x"][i])
    beta_y -= (tau * ni * ni) * environment(optim.D["y,y,y"], i, b[:y][i], optim.ops["d2y"][i])
    return [beta_x, beta_y]
end



#optimization of [vx,vy] with the RK4 methods: it takes intermediate steps [ax,ay] and [bx,by]
function optimize(optim, vx, vy, ax, ay, bx, by, tau; eps=1e-6, maxiter=100)
    v=Dict(:x=>vx, :y=>vy)
    a=Dict(:x=>ax, :y=>ay)
    b=Dict(:x=>bx, :y=>by)

    tau2=tau*tau
    ni=optim.params.ni
    dq=optim.params.dq
    N=optim.params.N
    ops=optim.ops

    function sweep!(i::Int)
        movecenter!(optim, i, v, a, b, N)

        cx = v[:x][optim.center]
        cy = v[:y][optim.center]

        xinds=inds(cx)
        yinds=inds(cy)

        nx=prod(dim.(xinds))
        ny=prod(dim.(yinds))

        operator= x -> linoperator(optim, x, tau2, xinds, yinds, nx, ny)
        
        M=FunctionMap{Float64,false}(operator, nx+ny)

        beta = vcat(vec.(array.(rhs(optim, a, b, tau, ni)))...)
        @assert length(beta) == nx + ny "Length of beta does not match the expected dimension size. Expected: $(nx + ny), got: $(length(beta))"

        cvec = vcat(vec.(array.([cx, cy]))...)
        @assert length(cvec) == nx + ny "Length of cvec does not match the expected dimension size. Expected: $(nx + ny), got: $(length(cvec))"
    
        cg!(cvec, M, beta, maxiter=maxiter, reltol=eps, abstol=eps^2, verbose=true, log=true)

        #update the MPS with the optimized values
        v[:x][optim.center] = ITensor(cvec[1:nx], xinds)
        v[:y][optim.center] = ITensor(cvec[nx+1:end], yinds)
    end

    E_0=1e-10
    E_1=2*eps

    while abs((E_1 - E_0)/E_0) > eps
        for i in 1:N
            println("updating site $i")
            sweep!(i)
            
        end

        for i in N-1:-1:1
            println("updating site $i")
            sweep!(i)
        end
        E_0 = E_1
        cvec = vcat(vec.(array.([v[:x][optim.center], v[:y][optim.center]]))...)
        E_1 = dot(cvec, cvec)
        println("Energy: $E_1")
    end
    return v[:x], v[:y]
end


function RK4(optim, ux, uy; eps=1e-6, maxiter=100)
    #RK4 method for time evolution of the MPS
    #ux, uy: MPS for the velocity field in x and y directions
    #eps: tolerance for the optimization
    #maxiter: maximum number of iterations for the conjugate gradient solver

    dt=optim.params.dt
    U1x, U1y = optimize(optim, ux, uy, ux/4, uy/4, ux, uy, dt/6; eps=eps, maxiter=maxiter)


    b = 3 .* [U1x, U1y] +  (1/4) .* [ux, uy]

    reset!(optim)
    U2x, U2y = optimize(optim, ux, uy, ux/4, uy/4, b[1], b[2], dt/3; eps=eps, maxiter=maxiter)

    b = (3/2) .* [U2x, U2y] +  (5/8) .* [ux, uy]
    reset!(optim)
    U3x, U3y = optimize(optim, ux, uy, ux/4, uy/4, b[1], b[2], dt/3; eps=eps, maxiter=maxiter)

    b = 3 .* [U3x, U3y] +  (1/4) .* [ux, uy]
    reset!(optim)
    U4x, U4y = optimize(optim, ux, uy, ux/4, uy/4, b[1], b[2], dt/6; eps=eps, maxiter=maxiter)

    result = [U1x, U1y] + [U2x, U2y] + [U3x, U3y] + [U4x, U4y] 
    return result[1], result[2]
end