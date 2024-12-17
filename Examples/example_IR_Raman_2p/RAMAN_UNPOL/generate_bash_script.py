import sys, os
import subprocess
import numpy as np

if __name__ == '__main__':
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    # The number of Raman polarizations to run
    np_raman = 7
    
    # Set variables
    # The directories where you saved the initial Lanczos
    dir_to_run = "./tdscha_raman_INDEX"
    # Lanczos steps to do
    steps = 4 
    # The status is saved after save_each steps
    save_each = 1
    # the output file where everything is printed
    output = "output_INDEX.txt"

    # The name of the bash file
    file = open('script_raman_unpol.sh', 'w')
    file.write("""#!/bin/bash

""")


    for i in range(np_raman):
        file.write("""

# Set variables
dir_to_run="{}"
output="{}"

# Run the command
# COMMAND="nohup mpirun -np 8 python-jl ./run.py $dir_to_run > $output &"
COMMAND="nohup python-jl ./run.py $dir_to_run > $output &"
echo "Running command: $COMMAND"
$COMMAND

# Wait for the command to finish
wait
""".format(dir_to_run.replace('INDEX','{}'.format(i)),\
          output.replace('INDEX','{}'.format(i))))
    file.close()
    
    
    os.system('chmod +x *.sh')
                   
    