#!/usr/bin/env julia

using HDF5
using ITensors
using ITensorMPS
using Plots
using Printf

const DEFAULT_FPS = 8
const DEFAULT_MAX_FRAMES = 120

"""Return the newest velocity_field.h5 below time_evolve/."""
function newest_velocity_file(root::AbstractString = @__DIR__)
    search_root = joinpath(root, "time_evolve")
    isdir(search_root) || error("Could not find $search_root")
    files = String[]
    for (dir, _, names) in walkdir(search_root)
        "velocity_field.h5" in names && push!(files, joinpath(dir, "velocity_field.h5"))
    end
    isempty(files) && error("No velocity_field.h5 found below $search_root")
    return files[argmax(mtime.(files))]
end

"""Find physical times for which both ux and uy snapshots exist."""
function available_snapshots(file::HDF5.File)
    ux_suffixes, uy_suffixes = Set{String}(), Set{String}()
    for name in keys(file)
        ux_match = match(r"^ux_t(.+)$", name)
        uy_match = match(r"^uy_t(.+)$", name)
        ux_match === nothing || push!(ux_suffixes, ux_match[1])
        uy_match === nothing || push!(uy_suffixes, uy_match[1])
    end
    snapshots = NamedTuple{(:time, :suffix), Tuple{Float64, String}}[]
    for suffix in intersect(ux_suffixes, uy_suffixes)
        time = tryparse(Float64, suffix)
        time === nothing || push!(snapshots, (time = time, suffix = suffix))
    end
    sort!(snapshots; by = snapshot -> snapshot.time)
    isempty(snapshots) && error("The HDF5 file contains no matching ux_tTIME/uy_tTIME snapshots")
    return snapshots
end

"""Expand the qudit MPS representation into a matrix indexed as field[x, y]."""
function mps_to_field(state::MPS)
    sites = siteinds(state)
    nbits = length(sites)
    n = 2^nbits
    full_tensor = Array(contract(state), sites...)
    field = Matrix{Float64}(undef, n, n)
    local_states = Vector{Int}(undef, nbits)
    for x in 0:(n - 1), y in 0:(n - 1)
        for site in 1:nbits
            shift = nbits - site
            xbit = (x >> shift) & 1
            ybit = (y >> shift) & 1
            local_states[site] = 1 + xbit + 2ybit
        end
        field[x + 1, y + 1] = real(full_tensor[local_states...])
    end
    return field
end

function read_snapshot(file::HDF5.File, suffix::AbstractString)
    ux = mps_to_field(read(file, "ux_t$suffix", MPS))
    uy = mps_to_field(read(file, "uy_t$suffix", MPS))
    size(ux) == size(uy) || error("ux and uy have different grid dimensions at t=$suffix")
    return ux, uy
end

"""Select snapshots uniformly over the complete available time interval."""
function select_snapshots(snapshots::AbstractVector, max_frames::Integer)
    max_frames > 1 || error("max_frames must be greater than one")
    length(snapshots) <= max_frames && return snapshots
    indices = unique(round.(Int, range(1, length(snapshots); length = max_frames)))
    return snapshots[indices]
end

function component_plot(field::AbstractMatrix, coordinate::Symbol, time::Real, color_limit::Real)
    coordinate in (:x, :y) || error("coordinate must be :x or :y")
    nx, ny = size(field)
    x = range(0.0, 1.0; length = nx)
    y = range(0.0, 1.0; length = ny)
    limit = color_limit > 0 ? color_limit : 1.0
    label = coordinate === :x ? "uₓ" : "uᵧ"
    formatted_time = @sprintf("%.8g", time)
    return heatmap(x, y, field'; color = :balance, clims = (-limit, limit),
        aspect_ratio = :equal, xlims = (0, 1), ylims = (0, 1), xlabel = "x",
        ylabel = "y", title = "$label velocity component — t = $formatted_time",
        colorbar_title = label, size = (800, 700))
end

function render_velocity(input_file::AbstractString, output_dir::AbstractString;
                         fps::Integer = DEFAULT_FPS,
                         max_frames::Integer = DEFAULT_MAX_FRAMES)
    isfile(input_file) || error("Input file does not exist: $input_file")
    fps > 0 || error("fps must be positive")
    max_frames > 1 || error("max_frames must be greater than one")
    mkpath(output_dir)
    h5open(input_file, "r") do file
        all_snapshots = available_snapshots(file)
        snapshots = select_snapshots(all_snapshots, max_frames)
        latest = all_snapshots[end]
        latest_ux, latest_uy = read_snapshot(file, latest.suffix)
        x_limit = maximum(abs, latest_ux)
        y_limit = maximum(abs, latest_uy)
        println("Reference values at the latest time: |uₓ| ≤ $x_limit, |uᵧ| ≤ $y_limit")
        x_png = joinpath(output_dir, "latest_velocity_x_coordinate.png")
        y_png = joinpath(output_dir, "latest_velocity_y_coordinate.png")
        savefig(component_plot(latest_ux, :x, latest.time, x_limit), x_png)
        savefig(component_plot(latest_uy, :y, latest.time, y_limit), y_png)
        
        length(snapshots) > 1 || error(
            "Only one snapshot (t=$(latest.time)) is present; cannot create an evolution GIF")
        x_animation, y_animation = Animation(), Animation()
        for snapshot in snapshots
            ux, uy = snapshot.suffix == latest.suffix ? (latest_ux, latest_uy) :
                                                       read_snapshot(file, snapshot.suffix)
            frame(x_animation, component_plot(ux, :x, snapshot.time, x_limit))
            frame(y_animation, component_plot(uy, :y, snapshot.time, y_limit))
        end
        x_gif = joinpath(output_dir, "velocity_x_coordinate.gif")
        y_gif = joinpath(output_dir, "velocity_y_coordinate.gif")
        gif(x_animation, x_gif; fps)
        gif(y_animation, y_gif; fps)

        println("Available snapshots: $(length(all_snapshots))")
        println("Animation frames: $(length(snapshots)) (uniformly sampled)")
        println("Input: $(abspath(input_file))")
        println("Latest physical time: $(latest.time)")
        println("x component: $(abspath(x_png))")
        println("y component: $(abspath(y_png))")
        println("x animation: $(abspath(x_gif))")
        println("y animation: $(abspath(y_gif))")
        return x_png, y_png, x_gif, y_gif
    end
end

function usage()
    println("""
    Usage: julia --project=. plot.jl [INPUT.h5] [OUTPUT_DIR] [FPS] [MAX_FRAMES]

    Defaults:
      INPUT.h5   newest time_evolve/**/velocity_field.h5
      OUTPUT_DIR <directory containing INPUT.h5>/plots
      FPS        $DEFAULT_FPS
      MAX_FRAMES $DEFAULT_MAX_FRAMES
    """)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if any(arg -> arg in ("-h", "--help"), ARGS)
        usage()
        exit()
    end
    length(ARGS) <= 4 || (usage(); error("Expected at most four arguments"))
    input_file = abspath(get(ARGS, 1, newest_velocity_file()))
    output_dir = abspath(get(ARGS, 2, joinpath(dirname(input_file), "plots")))
    fps = parse(Int, get(ARGS, 3, string(DEFAULT_FPS)))
    max_frames = parse(Int, get(ARGS, 4, string(DEFAULT_MAX_FRAMES)))
    render_velocity(input_file, output_dir; fps, max_frames)
end
