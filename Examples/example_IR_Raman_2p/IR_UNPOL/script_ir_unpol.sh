#!/bin/bash



# Set variables
dir_to_run="./tdscha_ir_0"
output="output_0.txt"

# Run the command
# COMMAND="nohup mpirun -np 8 python-jl ./run.py $dir_to_run > $output &"
COMMAND="nohup python-jl ./run.py $dir_to_run > $output &"
echo "Running command: $COMMAND"
$COMMAND

# Wait for the command to finish
wait


# Set variables
dir_to_run="./tdscha_ir_1"
output="output_1.txt"

# Run the command
# COMMAND="nohup mpirun -np 8 python-jl ./run.py $dir_to_run > $output &"
COMMAND="nohup python-jl ./run.py $dir_to_run > $output &"
echo "Running command: $COMMAND"
$COMMAND

# Wait for the command to finish
wait


# Set variables
dir_to_run="./tdscha_ir_2"
output="output_2.txt"

# Run the command
# COMMAND="nohup mpirun -np 8 python-jl ./run.py $dir_to_run > $output &"
COMMAND="nohup python-jl ./run.py $dir_to_run > $output &"
echo "Running command: $COMMAND"
$COMMAND

# Wait for the command to finish
wait
