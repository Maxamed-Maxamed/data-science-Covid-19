import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import statsmodels.api as sm
import statsmodels.formula.api as smf


# Load data
covid_data = pd.read_csv('covid_data.csv', low_memory=False)

# Run frequencies on the variables of interest
print("\nFrequencies of the variables of interest:")
print(covid_data[[ 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']].describe())




# Check for missing data in the variables of interest
print("\nMissing data in the variables of interest:")
print(covid_data[['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']].isnull().sum())

# Considered calculating vaccination coverage as a percentage of the population.
# Calculate vaccination coverage as a percentage of the population
covid_data[ 'vaccination_coverage_percent'] = (covid_data['people_vaccinated'] / covid_data['population']) * 100




# Display the first few rows of the dataset with the new variable
print("\nFirst few rows of the dataset with vaccination coverage as a percentage of the population:")
print(covid_data[['people_vaccinated', 'population', 'vaccination_coverage_percent']].head())


#Manipulate the date variable to extract year, month, and day information
covid_data['date'] = pd.to_datetime(covid_data['date']).apply (lambda x: x.date())
covid_data['year'] = covid_data['date'].apply(lambda x: x.year)
covid_data['month'] = covid_data['date'].apply(lambda x: x.month)
covid_data['day'] = covid_data['date'].apply(lambda x: x.day) 
print(covid_data.head())



# Display the first few rows of the dataset with the new variables
print("\nFirst few rows of the dataset with the new variables:")
print(covid_data[['year', 'month', 'day']].head())


# Display description statistics for vaccination data 
print("\n Description statistics for vaccination data:")
print(covid_data[covid_data['year'] == 2021][['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']].describe()) 


# Display description statistics for vaccination data Ireland. 
print("\n Description statistics for vaccination data Ireland:")
print(covid_data[covid_data['location'] == 'Ireland'][['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']].describe())


# # Create the bar chart visualization
# print("\nBar chart visualization:")

# plt.figure(figsize=(10, 6))  # Set the figure size
# continent_count_plot = sns.countplot( 
#     x='continent', 
#     data=covid_data, 
#     order=covid_data['continent'].value_counts().index, 
#     palette='viridis'
# )  # Create the count plot

# plt.title('Number of Observations by Continent')  # Set the title
# plt.xlabel('Continent')  # Set the x-axis label
# plt.ylabel('Number of Observations')  # Set the y-axis label
# plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
# plt.show()  # Show the plot

# continent_count = covid_data['continent'].value_counts()  # Count the number of cases by continent  
# print("\nNumber of Observations by continent:")
# print(continent_count)




#part 2
# print("\nBar chart visualization:")
# # Create the bar chart visualization
# plt.figure(figsize=(10, 6))  # Set the size of the figure
# continent_count_plot = sns.countplot(
#     x='continent', 
#     data=covid_data, 
#     palette='viridis',  # Choose a color palette for the chart
#     order=covid_data['continent'].value_counts().index  # Order bars by count
# )
# plt.title('Number of Observations by Continent')  # Set the title of the chart
# plt.xlabel('Continent')  # Label the x-axis
# plt.ylabel('Count')  # Label the y-axis
# plt.xticks(rotation=45)  # Rotate the labels on the x-axis for better readability
# plt.tight_layout()  # Adjust the layout to make sure everything fits well

# # Extract counts for each continent directly from the data for use in the explanation
# continent_counts = covid_data['continent'].value_counts()

# # Display the plot
# plt.show()
# # Explain the results 
# print(f"The number of observations by continent are:\n{continent_counts}") # Display the counts for each continent




print("\nBar chart visualization:")
# Create the bar chart visualization
plt.figure(figsize=(10, 6))  
continent_count_plot = sns.countplot(
    x='continent',  
    data=covid_data,  
    hue='continent', # Assign x to hue
    palette='viridis',  
    order=covid_data['continent'].value_counts().index,
    # legend=False # Hide the legend
)
plt.title('Number of Observations by Continent')  
plt.xlabel('Continent')  
plt.ylabel('Count')   
plt.xticks(rotation=45)  
plt.tight_layout()  

# Extract counts for each continent directly from the data for use in the explanation
continent_counts = covid_data['continent'].value_counts()

# Display the plot
plt.show()
# Explain the results
print(f"The number of observations by continent are:\n{continent_counts}") 









#create Histogram bar chart visualization
print("\nHistogram bar chart visualization:")   
plt.figure(figsize=(10, 6))  # Set the figure size
vaccination_coverage_percent_plot = sns.histplot(
    x='vaccination_coverage_percent',
    data=covid_data,
    bins=10,
    kde=True,
    color='blue',
    alpha=0.5
)
# Set the title
plt.title('Vaccination Coverage Percent')
# Set the x-axis label
plt.xlabel('Vaccination Coverage Percent')
# Set the y-axis label
plt.ylabel('Count')
# Show the plot
plt.show() 



# # Create a histogram to visualize the distribution of total vaccinations 
print("\nHistogram for total_vaccinations:")
plt.figure(figsize=(10, 6))
total_vaccinations_plot = sns.histplot(
    x='total_vaccinations',
    data=covid_data,
    bins=10, 
    kde=True,
    color='green',
    alpha=0.5
)
# Set the title
plt.title('Total Vaccinations')
# Set the x-axis label
plt.xlabel('Total Vaccinations')
# Set the y-axis label
plt.ylabel('Count')
# Show the plot
plt.show() 

# # Create a histogram to visualize the distribution of people vaccinated
print("\nHistogram for people_vaccinated:")
plt.figure(figsize=(10, 6))
people_vaccinated_plot = sns.histplot(
    x='people_vaccinated',
    data=covid_data,
    bins=10,
    kde=True,
    color='red',
    alpha=0.5
)
# Set the title 
plt.title('People Vaccinated') 
# Set the x-axis label
plt.xlabel('People Vaccinated')
# Set the y-axis label
plt.ylabel('Count')
# Show the plot
plt.show()
# Create a histogram to visualize the distribution of people fully vaccinated
print("\nHistogram for people_fully_vaccinated:")
plt.figure(figsize=(10, 6))
people_fully_vaccinated_plot = sns.histplot(
    x='people_fully_vaccinated',
    data=covid_data,
    bins=10,
    kde=True,
    color='blue',
    alpha=0.5
)
# Set the title
plt.title('People Fully Vaccinated')
# Set the x-axis label
plt.xlabel('People Fully Vaccinated')
# Set the y-axis label
plt.ylabel('Count')
# Show the plot
plt.show()




# For total_vaccinations
f_statistic, p_value = stats.f_oneway(
    covid_data[covid_data['continent'] == 'Africa']['total_vaccinations'].dropna(),
    covid_data[covid_data['continent'] == 'Americas']['total_vaccinations'].dropna(),
    covid_data[covid_data['continent'] == 'Asia']['total_vaccinations'].dropna(),
    covid_data[covid_data['continent'] == 'Europe']['total_vaccinations'].dropna(),
    covid_data[covid_data['continent'] == 'Oceania']['total_vaccinations'].dropna()
)
print("\nANOVA test for total_vaccinations across continents:")
print(f"F-statistic: {f_statistic}, P-value: {p_value}")

# Repeat the process for 'people_vaccinated' and 'people_fully_vaccinated'
# Remember to replace 'total_vaccinations' with the appropriate variable name in each call


# Conduct ANOVA test to compare total vaccinations, people vaccinated and people fully vaccinated between continents. 
print("\nANOVA test to compare vaccination variables between continents:") 
from scipy import stats 
stats.f_oneway(covid_data[covid_data['continent']=='Africa']['total_vaccinations'],covid_data[covid_data['continent']=='Americas']['total_vaccinations'],covid_data[covid_data['continent']=='Asia']['total_vaccinations'],covid_data[covid_data['continent']=='Europe']['total_vaccinations'],covid_data[covid_data['continent']=='Oceania']['total_vaccinations'])
stats.f_oneway(covid_data[covid_data['continent']=='Africa']['people_vaccinated'],covid_data[covid_data['continent']=='Americas']['people_vaccinated'],covid_data[covid_data['continent']=='Asia']['people_vaccinated'],covid_data[covid_data['continent']=='Europe']['people_vaccinated'],covid_data[covid_data['continent']=='Oceania']['people_vaccinated'])
stats.f_oneway(covid_data[covid_data['continent']=='Africa']['people_fully_vaccinated'],covid_data[covid_data['continent']=='Americas']['people_fully_vaccinated'],covid_data[covid_data['continent']=='Asia']['people_fully_vaccinated'],covid_data[covid_data['continent']=='Europe']['people_fully_vaccinated'],covid_data[covid_data['continent']=='Oceania']['people_fully_vaccinated'])   # Conduct ANOVA test to compare total vaccinations, people vaccinated and people fully vaccinated between continents.







# # Create a scatter plot to visualize the relationship between total vaccinations, people vaccinated and people fully vaccinated. 
print("\nScatter plot between vaccination variables:")
plt.figure(figsize=(10, 8))
plt.scatter(covid_data['total_vaccinations'], covid_data['people_vaccinated'], color='green', alpha=0.5, label='Total Vaccinations vs People Vaccinated')
plt.scatter(covid_data['total_vaccinations'], covid_data['people_fully_vaccinated'], color='blue', alpha=0.5, label='Total Vaccinations vs People Fully Vaccinated')
plt.scatter(covid_data['people_vaccinated'], covid_data['people_fully_vaccinated'], color='red', alpha=0.5, label='People Vaccinated vs People Fully Vaccinated')
plt.legend(loc='upper left')
plt.xlabel('Total Vaccinations')
plt.ylabel('People Vaccinated/Fully Vaccinated')
plt.title('Relationship between Vaccination Variables')
plt.show()

# Create a bivariate chart using jointplot to visualize the relationship between total vaccinations, people vaccinated and people fully vaccinated.
print("\nBivariate chart using jointplot:")
g = sns.jointplot(x="total_vaccinations", y="people_vaccinated", data=covid_data, color="g")
g = sns.jointplot(x="total_vaccinations", y="people_fully_vaccinated", data=covid_data, color="b")
g = sns.jointplot(x="total_vaccinations", y= "total_vaccinations", data=covid_data, color="r")
plt.show()

# # Create a pairplot to visualize the relationship between total vaccinations, people vaccinated and people fully vaccinated.
print("\nPairplot:")
g = sns.pairplot(covid_data[['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']], diag_kind="hist", plot_kws=dict(alpha=0.5))
plt.show()


# # Create a correlation matrix to visualize the relationship between total vaccinations, people vaccinated and people fully vaccinated.
print("\nCorrelation matrix:")
corr = covid_data[['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']].corr()
sns.heatmap(corr, annot=True)
plt.title('Correlation between Vaccination Variables')
plt.show()
