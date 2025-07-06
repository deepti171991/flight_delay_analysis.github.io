import pandas as pd
import numpy as np
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import mysql.connector
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score




##############   STEP 1: data reading     ############
# the excel workbook is in .csv format
# reading the file
data = pd.read_csv('Airline_Delay_Cause.csv')



#############   STEP 2: Understanding the data  ###########

# 2.1 print the first 5 records 
print(data.head())       # [5 rows x 21 columns]

# 2.2 to understand the excel sheet 
print("size of the excel sheet \n",data.shape)       # (171666, 21)



########### STEP 3 : Data Cleaning ###############
# 3.1 handle missing values
print("missing values are  \n ", data.isnull().sum()) # about 240 missing records for some columns

# 3.2 dropping null records
data_cleaned = data.dropna()
print("missing values after cleaning are  \n ", data_cleaned.isnull().sum())

# 3.3 now shape of the cleaned data
print("cleaned records\n", data_cleaned.shape)  # (171223, 21)

# 3.4 remove all duplicate values
print("duplicates in data \n ",data_cleaned.duplicated().sum()) # no duplicate records for this data

# 3.5 checking the datatypes of the data
print(data_cleaned.dtypes)



########### STEP 4: Create database and Tables in MYSQL###########

# 4.1 creating a database : Install MySQL Driver run this in terminal
# python -m pip install mysql-connector-python

# 4.2 Test MySQL Connector : import mysql.connector 

# 4.3 Create and test connection

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="Rushank17!"
)
print(mydb)


# 4.4  Creating a Database 
mycursor = mydb.cursor()   # create a pointer to the database
#mycursor.execute("CREATE DATABASE data_analysis_db") # name a database

# 4.5 Check if Database Exists
mycursor.execute("SHOW DATABASES")
# to print all the databases in my connection
for x in mycursor:
  print(x)


# 4.6 creating one table to store the .csv file in database data_analysis_db
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="Rushank17!",
  database="data_analysis_db"
)

mycursor = mydb.cursor()

# TABLE Name: flight_data_table
#mycursor.execute("""
#CREATE TABLE flight_data_table (
    #year INT,
    #month INT,
    #carrier VARCHAR(3),
    #carrier_name VARCHAR(300),
    #airport VARCHAR(3),
    #airport_name VARCHAR(300),
    #arr_flights INT,
    #arr_del15 INT,
    #carrier_ct INT,
    #weather_ct INT,
    #nas_ct INT,
    #security_ct INT,
    #late_aircraft_ct INT,
    #arr_cancelled INT,
    #arr_diverted INT,
    #arr_delay FLOAT,
    #carrier_delay FLOAT,
    #weather_delay FLOAT,
    #nas_delay FLOAT,
    #security_delay FLOAT,
    #late_aircraft_delay FLOAT,
    #PRIMARY KEY (year, month, carrier, airport)
    
#)
#""") 


# 4.7 check the table created
mycursor = mydb.cursor()
mycursor.execute("SHOW TABLES")
for table in mycursor:
    print(table)

# 4.8 to check the attributes of the table
mycursor.execute("DESCRIBE flight_data_table")
for attribute in mycursor:
    print(attribute)






############ STEP 5: inserting records from .csv file into flight_data_table
#5.1 preparing the data 
data_table=data_cleaned[['year','month','carrier','carrier_name', 'airport', 'airport_name','arr_flights',
                         'arr_del15','carrier_ct','weather_ct','nas_ct','security_ct','late_aircraft_ct',
                          'arr_cancelled','arr_diverted','arr_delay','carrier_delay','weather_delay',
                          'nas_delay','security_delay','late_aircraft_delay'
                          ]]
                                                   
print("data inserted in flight_data_table  \n", data_table.head())


# 5.2: Create connection engine 
username = 'root'
password = 'Rushank17!'
host = 'localhost'
database = 'data_analysis_db'

# include this at top: from sqlalchemy import create_engine 
# pip install pandas sqlalchemy openpyxl pymysql
engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}/{database}')


# 5.3 inserting data into table1

# data_table.to_sql('flight_data_table', con=engine, if_exists='append', index=False)
print(" Data inserted successfully into flight_data_table ")


# 5.4 Quering the flight_data_table

# Query first 5 rows
mycursor.execute("SELECT * FROM flight_data_table LIMIT 5")
rows = mycursor.fetchall()

print("First 5 rows in flight_data_table")
for row in rows:
    print(row)


#  Query flights most affected by delays

mycursor.execute("""
SELECT 
carrier_name,
SUM(arr_del15) AS total_delayed_flights
FROM flight_data_table
GROUP BY carrier_name
ORDER BY total_delayed_flights DESC;

""")

#  display the query results 

# Fetch and print the results
results = mycursor.fetchall()

print("Carrier Name | Total Delayed Flights")
for row in results:
    print(f"{row[0]} | {row[1]}")



########## STEP 6: EDA analysis##############

# 6.1 Univaraiate Analysis 

# overall distribution of the arrived flights 
plt.figure(figsize=(10,4))
sns.histplot(data_cleaned['arr_flights'],bins=10,kde=True)
plt.title("Figure 1:Overall distribution of Arrived Flights")
plt.xlabel("Number of Flights Arrived")
plt.show()


# Count Plot of Arrival Cancelled Flights
plt.figure(figsize=(10,5))
sns.countplot(x='arr_cancelled', data=data_cleaned)
plt.title("Count Plot of Arrival Cancelled Flights")
plt.xlabel("Number of Cancelled Arrivals")
plt.ylabel("Count of Records")
plt.show()
# as the data is too skewered now checking for cancellation rate in terms of percentage


# Calculate cancellation rate
data['cancel_rate'] = data_cleaned['arr_cancelled'] / (data_cleaned['arr_cancelled'] + data_cleaned['arr_flights'])*100

plt.figure(figsize=(10,6))
sns.histplot(data['cancel_rate'], bins=50, kde=True, color='orange')
plt.title(" Figure2: Distribution of Flight Cancellation Rate Percentage")
plt.xlabel("Flight Cancellation Rate in terms of %")
plt.ylabel("Frequency")
plt.show()



# to check the distribution of arrival delay in hours
data_cleaned['arr_delay_hours'] = data_cleaned['arr_delay'] / 60
sns.stripplot(x=data_cleaned['arr_delay_hours'],color='red',alpha=0.3)
plt.title('Figure3: Strip Plot of Arrival Delay')
plt.xlabel('Arrival Delay in hours')
plt.show()



# 6.2 Bivariate analysis 

# Before we begin the analysis, doing some calculations
# creating a new column total_delayed_flights, because no direct information about it.

data_cleaned['total_delayed_flights'] = data_cleaned[['carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct']].sum(axis=1)

print("check if new column created",data_cleaned.head())

# checking the difference between total delayed flights and count of flights delayed >15minutes
plt.figure(figsize=(8,6))
sns.scatterplot(x='arr_del15', y='total_delayed_flights', data=data_cleaned, alpha=0.6)
plt.title('Total Delayed Flights vs Flights Delayed > 15 Minutes')
plt.xlabel('Flights Delayed > 15 Minutes (arr_del15)')
plt.ylabel('Total Delayed Flights (total_delayed_flights)')
plt.tight_layout()
plt.show()## there is no much difference in both the values


# checking for difference 
data_cleaned['delay_diff'] = data_cleaned['total_delayed_flights'] - data_cleaned['arr_del15']

plt.figure(figsize=(8,6))
sns.histplot(data_cleaned['delay_diff'], bins=30, kde=True)
plt.title('Distribution of Difference Between Total Delayed Flights and number of flights delayed >15min')
plt.xlabel('Difference (total_delayed_flights - arr_del15)')
plt.tight_layout()
plt.show() # the difference is negligible.



# bar plot giving the sum of arrived vs delayed flights
plt.figure(figsize=(10,5))
plt.bar(
    ['arr_flights', 'total_delayed_flights'],
    [data_cleaned['arr_flights'].sum(), data_cleaned['total_delayed_flights'].sum()],
    color='yellow'
)
plt.title('Figure 4: Total Arrived Flights vs Total Delayed Flights')
plt.ylabel('Number of Flights')
plt.tight_layout()
plt.show()




# to check how the total delayed flights vs number of flights delayed due to weather reasons
plt.figure(figsize=(10,6))
sns.scatterplot(x='weather_ct', y='total_delayed_flights', data=data_cleaned, alpha=0.6, color='green')
plt.title('Figure 5 : Total delayed flights vs flights delayed due to weather reasons ')
plt.xlabel('Number of arrival flights delayed due to weather reasons')
plt.ylabel('Total delayed flights')
plt.show()


cancelled_per_year = data_cleaned.groupby('year')['arr_cancelled'].sum().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(x='year', y='arr_cancelled', data=cancelled_per_year, color='deeppink')
plt.title('Figure 6: Total Flights Cancelled per Year')
plt.xlabel('Year')
plt.ylabel('Number of Cancelled Flights')
plt.show()
# due to covid highest number of flight cancellatons in year 2020


data1 = data_cleaned.groupby(['year', 'month'], as_index=False)['arr_cancelled'].sum()

plt.figure(figsize=(10,7))
sns.barplot(x='year', y='arr_cancelled', hue='month', data=data1,palette='viridis')
plt.title('Figure 7: Total Flights Cancelled by Month and Year')
plt.xlabel('Year')
plt.ylabel('Total Cancelled Flights')
plt.legend(title='Month')
plt.show()
# again to add to figure 6 more cancellation in april 2020



variables = ['arr_flights', 'total_delayed_flights', 'arr_cancelled', 'arr_diverted']

# Create the pair plot
sns.pairplot(data_cleaned[variables],height=1.5)
plt.suptitle('Figure 8: Pairwise Relationships among 4 variables')
plt.figure(figsize=(8,6))
plt.tight_layout()  
plt.show()


# Define the 5 delay-related variables - 
delay_vars = ['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']

# Compute the correlation matrix
corr_matrix = data_cleaned[delay_vars].corr()

# Plot the heatmap
#plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f")
plt.title('Figure 9:Heatmap for Correlation Between all the Delay Types')
plt.xticks(rotation=45)  # Rotate x-axis labels
plt.yticks(rotation=0)               # Keep y-axis labels horizontal
plt.tight_layout()  
plt.show()





##### meaningful insights from above analysis

delay_vars = ['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']
delay_totals = data_cleaned[delay_vars].sum()

# Plot the pie chart 
plt.figure(figsize=(8, 8))
plt.pie(delay_totals, labels=delay_vars, autopct='%1.1f%%', startangle=140)
plt.title('Figure 10: Pie chart for different delay causes')
plt.axis('equal')  # Equal aspect ratio ensures pie is a circle.
plt.show()



# top 10 airlines with highest total delay minutes
airline_delay = data_cleaned.groupby('carrier_name')['arr_delay'].sum()

# Select top 10 airlines with highest total delay minutes
top10_airlines_delay = airline_delay.sort_values(ascending=False).head(10)

print("top10_airlines_delay by airlines",top10_airlines_delay.head())
# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=top10_airlines_delay.index, y=top10_airlines_delay.values,palette='viridis')
plt.title('Figure 11: Top 10 Airlines based on  Total Arrival Delay (Minutes)')
plt.xlabel('Airline')
plt.ylabel('Total Arrival Delay (minutes)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()




# Top 10 Airports by Total Arrival Delay (in Minutes)
# Group and sort by total delay minutes
airport_delays = data_cleaned.groupby('airport_name')['arr_delay'].sum().sort_values(ascending=False).head(10)
# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=airport_delays.index, y=airport_delays.values, palette='viridis')
plt.title('Figure 12: Top 10 Airports by Total Arrival Delay (in Minutes)')
plt.ylabel('Total Arrival Delay (minutes)')
plt.xlabel('Airport Name')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Aggregate total weather delays by airline and year
weather_delay = data_cleaned.groupby(['year', 'carrier_name'])['weather_delay'].sum().reset_index()

plt.figure(figsize=(10, 8))
sns.lineplot(data=weather_delay, x='year', y='weather_delay', hue='carrier_name', marker='o')
plt.title('Figure 13: Yearly Weather Delays for All Airlines')
plt.ylabel('Total Weather Delay (minutes)')
plt.xlabel('Year')
plt.legend(title='Airline', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



#######################################Linear regression

# Features and target
features = ['carrier_delay','late_aircraft_delay']
X = data_cleaned[features]  # Selecting these columns as input features 
y = data_cleaned['arr_delay']  # Target variable - this is to be predicted


# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression() 
model.fit(X_train, y_train)  # Train the model on the training data

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error: average squared difference between actual and predicted delays
r2 = r2_score(y_test, y_pred)  # R-squared: proportion of variance explained by the model


print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel('Actual Arrival Delay (minutes)')
plt.ylabel('Predicted Arrival Delay (minutes)')
plt.title('Figure 14: Actual vs Predicted Arrival Delay ')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line for perfect prediction
plt.show()





