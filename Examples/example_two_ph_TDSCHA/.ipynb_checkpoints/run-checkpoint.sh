#!/bin/bash

ONETWO='1'
SYMMNOSYMM='symm'

# python3 script_2ph.py xx $ONETWO $SYMMNOSYMM  &
# python3 script_2ph.py yy $ONETWO $SYMMNOSYMM  &
# python3 script_2ph.py zz $ONETWO $SYMMNOSYMM  &
python3 script_2ph.py xy $ONETWO $SYMMNOSYMM  
# python3 script_2ph.py xz $ONETWO $SYMMNOSYMM  &
# python3 script_2ph.py yz $ONETWO $SYMMNOSYMM  &
# python3 script_2ph.py xy2 $ONETWO $SYMMNOSYMM  &
# python3 script_2ph.py xz2 $ONETWO $SYMMNOSYMM  &
# python3 script_2ph.py yz2 $ONETWO $SYMMNOSYMM  &
