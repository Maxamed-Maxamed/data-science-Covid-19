import packaging  # Import the 'packaging' module 

import pandas as pd 
  # Import the 'pandas' module
covid_data = pd.read_csv('covid_data.csv', low_memory=False)  # Load data

# Display the first 5 rows of the data
print(covid_data.head())


# Display the last 5 rows of the data
print(covid_data.tail(5))


# location in Ireland
location = 'Ireland' # Define the location
variable = 'total_vaccinations' # Define the variable
# Display the description statistics for the variable for the location n
print(covid_data[covid_data['location'] == location][[variable]].describe())
