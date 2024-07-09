import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import os
import sys
import argparse


'''adding the project directory to the
PYTHONPATH environment variables and
importing the modules script
'''

try:
    sys.path.append(os.getcwd())
    from modules.modules import *
except ModuleNotFoundError as e:
    print("You have to add the project directory to the PYTHONPATH \
           environment variable")
    

