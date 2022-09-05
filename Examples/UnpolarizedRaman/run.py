import numpy as np
import os
import sys

geom = ["xxxx", "xxyy", "xxzz", "yyyy", "yyzz", "zzzz", "xyxy", "xzxz", "yzyz"]

for i,iq in enumerate(geom):
    os.system("python prepare_cluster_lanczos.py %s" %(iq))
