using TOML 

config=TOML.parsefile("time_evolve/config.toml")

# General parameters
penalty=config["general"]["penalty"]
viscosity=config["general"]["viscosity"]
nbits=config["general"]["nbits"]
maxdim=config["general"]["maxdim"]

#optimization parameters
maxiter=config["optimization"]["maxiter"]
maxsweeps=config["optimization"]["maxsweeps"]

#time evolution parameters
ttotal=config["time_evolution"]["ttotal"]

output_folder = ARGS[1] 
@show output_folder

open("$output_folder/config.toml", "w") do file
    TOML.print(file, config)
end