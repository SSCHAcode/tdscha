How to run a Raman calculation including 1ph and 2ph porcesses?


1) Run create_input_raman_unpol_2ph.py. This creates the input files for TDSCHA. Here you set if you want to dinclude D3 and D4

2) Run generate_bash_script.py to create the script_raman_unpol.sh file that you can use to run the TDSCHA calculations

3) Run script_raman_unpol.sh

4) Show the result with plot_raman_unpol.py