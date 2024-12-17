How to run a IR calculation including 1ph and 2ph porcesses?


1) Run create_input_ir_unpol_2ph.py. This creates the input files for TDSCHA. Here you set if you want to dinclude D3 and D4

2) Run generate_bash_script.py to create the script_ir_unpol.sh file that you can use to run the TDSCHA calculations

3) Run script_ir_unpol.sh

4) Show the result with plot_ir_unpol.py