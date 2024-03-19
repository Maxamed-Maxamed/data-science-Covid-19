import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Load data
covid_data = pd.read_csv('covid_data.csv', low_memory=False)

# Run frequencies on the variables of interest
print("\nFrequencies of the variables of interest:")
print(covid_data[[ 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']].describe())