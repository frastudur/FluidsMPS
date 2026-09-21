function hadamard_prod_mpo(A::MPS, deltas::MPO)
    return MPO(*(deltas, prime(siteinds, prime(siteinds, A)))[:])
end

function convective_operators(A::MPS, deltas::MPO, partial::MPO) 
    Hprod=hadamard_prod_mpo(A, deltas)
    C1=apply(Hprod, partial)
    C2=apply(partial, Hprod)
    return C1, C2
end