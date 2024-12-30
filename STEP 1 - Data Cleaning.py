#Data Cleaning of the dataset and summary statistics before data simulation
import pandas as pd
import os
import numpy as np
import math

#Setting up the working directory
os.chdir("/Users/jasonrayen/Downloads/Jason Masters/Sem 2/Applications of Metadata in Complex Big Data Problems (AIT 582)/Project")
os.getcwd()

#Importing the dataset
cervical_cancer = pd.read_csv("cervicalcancer.csv")

#The dataset has a lot of unclean records which has to be removed/replaced. Lets have a look at the brief statistical information
#of the dataset

#Type of all columns
print(cervical_cancer.dtypes)

#All information here are objects which have to be converted to integers. Before that the dataset has to be cleaned.


#Missing values of all columns
#checking for Null Values
#Checking for null values and resolving them
cervical_cancer.isnull().sum() #It shows 0 Null values because the null values are represented by ?

#Checking the count of "?" of each column
missing_values_count = cervical_cancer.apply(lambda x: x.eq('?').sum())

print(missing_values_count)

#Removing Columns that have many missing values which serve no purpose
cervical_cancer = cervical_cancer.drop(columns = ['STDs: Time since first diagnosis','STDs: Time since last diagnosis'], axis = 1)


#Replacing the ? with NULL values
cervical_cancer = cervical_cancer.replace('?', np.nan)

#Changing all columns datatypes to integers
cervical_cancer = cervical_cancer.astype(float)

#Removing the Null values in the dataset with respect to STD column
cervical_cancer.dropna(subset=['STDs'],inplace = True)

#Num of pregnencies column with NULL values is taken as 0
cervical_cancer['Num of pregnancies'].fillna(0,inplace = True)

#The smokes column's Null values are removed
cervical_cancer.dropna(subset=['Smokes','Hormonal Contraceptives','IUD','First sexual intercourse'],inplace = True)

#Filling the "Number of Sexual Partners" column with the rounded off mean value
mean_sexualPartners = math.ceil(cervical_cancer['Number of sexual partners'].mean())
cervical_cancer['Number of sexual partners'].fillna(mean_sexualPartners,inplace = True)


#The Cleaned dataset has 720 rows and 34 columns


#SUMMARY STATISTICS OF THE CLEANED DATASET
summary_statistics = cervical_cancer.describe()


#This cleaned dataset is transfered to R for further operations.
cervical_cancer.to_csv(r'/Users/jasonrayen/Downloads/Jason Masters/Sem 2/Applications of Metadata in Complex Big Data Problems (AIT 582)/Project/cleaned_cervicalcancer.csv', index=False)



























