#!/bin/bash


script="tests/runtime_one_step/script.jl"


run_id=$(date +%Y%m%d_%H%M%S)
output_folder="tests/runtime_one_step/run_${run_id}"
mkdir -p $output_folder

cp "$script" "$output_folder/script.jl"

nohup julia --project=. "$script" "$output_folder" > "$output_folder/output.log" 2>&1 &

echo "PID: $!"
echo "log file: $output_folder/output.log"