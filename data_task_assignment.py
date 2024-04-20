import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import ttest_ind 
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi



#Load the spreadsheet. 
covid_data = pd.read_csv('owid-covid-data(2).csv', low_memory=False)

print(covid_data.head()) #print the first 5 rows of the data