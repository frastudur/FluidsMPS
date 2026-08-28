using TOML 

config=TOML.parsefile("time_evolve/config.toml")

#if maxdim is provided as a command line argument, override the value in the config file
if length(ARGS) > 1
    for arg in ARGS[2:end]
        maxdim = parse(Int, arg)
        config["general"]["maxdim"] = maxdim
        
    end
end

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