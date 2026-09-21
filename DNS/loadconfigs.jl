using TOML 

config=TOML.parsefile("DNS/config.toml")

viscosity=config["general"]["viscosity"]
nbits=config["general"]["nbits"]
ttotal=config["time_evolution"]["ttotal"]

output_folder = ARGS[1] 
@show output_folder

open("$output_folder/config.toml", "w") do file
    TOML.print(file, config)
end