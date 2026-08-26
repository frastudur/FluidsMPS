kinetic_energy(ux, uy, h) = 0.5 * sum(abs2.(ux) .+ abs2.(uy)) * h^2

Reynolds_number(u0, h, viscosity) = u0 * h / viscosity

time_scale(Lbox, u0) = Lbox / u0