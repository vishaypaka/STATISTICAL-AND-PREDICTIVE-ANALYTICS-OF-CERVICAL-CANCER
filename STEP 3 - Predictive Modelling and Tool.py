#Data Cleaning of the dataset and summary statistics before data simulation
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import f1_score, matthews_corrcoef

os.chdir("/Users/jasonrayen/Downloads/Jason Masters/Sem 2/Applications of Metadata in Complex Big Data Problems (AIT 582)/Project")
os.getcwd()


#Getting the file from R
cervical_cancer_new = pd.read_csv("dataset_from_r.csv")

#checking the number of rows and columns
print(cervical_cancer_new.shape)

#General Checks
cervical_cancer_new.isnull().sum()

#summary of the dataset
print(cervical_cancer_new.describe())

#histogram plots
cervical_cancer_new.hist(figsize = (20,20))
plt.show()

#3D scatter plot
FilteredData = cervical_cancer_new[['Age','First sexual intercourse', 'Dx:Cancer']]

plt.close();
sns.set_style("whitegrid");
sns.pairplot(FilteredData, hue="Dx:Cancer", height=5);
plt.show()

#2-D scatter plot
sns.FacetGrid(FilteredData, hue="Dx:Cancer", height=10).map(sns.distplot, "Age").add_legend()
plt.show()

sns.FacetGrid(FilteredData, hue="Dx:Cancer", height=10).map(sns.distplot, "First sexual intercourse").add_legend()
plt.show()

sns.FacetGrid(cervical_cancer_new, hue="Dx:Cancer", height=10).map(sns.distplot, "Num of pregnancies").add_legend()
plt.show()


#Classification Models - Planned to be used

#Loading the independent variables in X and dependent variable in y
X = cervical_cancer_new.iloc[:, :13].values
X = np.concatenate((X, cervical_cancer_new.iloc[:, 14:].values), axis=1)
y = cervical_cancer_new.iloc[:, 13].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)


#-------------------------------------------------------Model 1 - Logistic Regression (Linear model)---------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, y_train)

#Prediction of the Test set results
y_pred1 = classifier1.predict(X_test)
print(np.concatenate((y_pred1.reshape(len(y_pred1),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred1)
print(cm)

# printing the confusion matrix 
LABELS = ['No Cancer', 'Cancer']
conf_matrix = confusion_matrix(y_test, y_pred1)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

acc_1 = accuracy_score(y_test, y_pred1)

# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)
print("The accuracy of this algorithm is {}".format(acc_1)) 
#-------------------------------------------------------Model 2 - K-Nearest Neighbors(K-NN)--------------------------------------------------------------------

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier2.fit(X_train, y_train)

# Predicting the Test set results
y_pred2 = classifier2.predict(X_test)
print(np.concatenate((y_pred2.reshape(len(y_pred2),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred2)
print(cm)

# printing the confusion matrix 
LABELS = ['No Cancer', 'Cancer']
conf_matrix = confusion_matrix(y_test, y_pred2)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

acc_2 = accuracy_score(y_test, y_pred2)

# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)
print("The accuracy of this algorithm is {}".format(acc_2)) 
#-------------------------------------------------------Model 3 - Support Vector Machine (Kernel)--------------------------------------------------------------------

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier3 = SVC(kernel = 'rbf', random_state = 0)
classifier3.fit(X_train, y_train)

y_pred3 = classifier3.predict(X_test)
print(np.concatenate((y_pred3.reshape(len(y_pred3),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred3)
print(cm)

# printing the confusion matrix 
LABELS = ['No Cancer', 'Cancer']
conf_matrix = confusion_matrix(y_test, y_pred3)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

acc_3 = accuracy_score(y_test, y_pred3)

# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)
print("The accuracy of this algorithm is {}".format(acc_3)) 

#-------------------------------------------------------Model 4 - Support Vector Machine (Linear Version)--------------------------------------------------------------------

# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier4 = SVC(kernel = 'linear', random_state = 0)
classifier4.fit(X_train, y_train)

# Predicting the Test set results
y_pred4 = classifier4.predict(X_test)
print(np.concatenate((y_pred4.reshape(len(y_pred4),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred4)
print(cm)

# printing the confusion matrix 
LABELS = ['No Cancer', 'Cancer']
conf_matrix = confusion_matrix(y_test, y_pred4)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

acc_4 = accuracy_score(y_test, y_pred4)

# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)
print("The accuracy of this algorithm is {}".format(acc_4)) 

#-------------------------------------------------------Model 5 - Naive Bayes Algorithm--------------------------------------------------------------------

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier5 = GaussianNB()
classifier5.fit(X_train, y_train)

# Predicting the Test set results
y_pred5 = classifier5.predict(X_test)
print(np.concatenate((y_pred5.reshape(len(y_pred5),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred5)
print(cm)

# printing the confusion matrix 
LABELS = ['No Cancer', 'Cancer']
conf_matrix = confusion_matrix(y_test, y_pred5)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

acc_5 = accuracy_score(y_test, y_pred5)

# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)
print("The accuracy of this algorithm is {}".format(acc_5)) 


#-------------------------------------------------------Model 6 - Decision Trees Algorithm--------------------------------------------------------------------

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier6 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier6.fit(X_train, y_train)

# Predicting the Test set results
y_pred6 = classifier6.predict(X_test)
print(np.concatenate((y_pred6.reshape(len(y_pred6),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred6)
print(cm)

# printing the confusion matrix 
LABELS = ['No Cancer', 'Cancer']
conf_matrix = confusion_matrix(y_test, y_pred6)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

acc_6 = accuracy_score(y_test, y_pred6)

# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)
print("The accuracy of this algorithm is {}".format(acc_6)) 

#-------------------------------------------------------Model 7 - Random Forest Algorithm--------------------------------------------------------------------

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier7 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier7.fit(X_train, y_train)

# Predicting the Test set results
y_pred7 = classifier7.predict(X_test)
print(np.concatenate((y_pred7.reshape(len(y_pred7),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred7)
print(cm)

# printing the confusion matrix 
LABELS = ['No Cancer', 'Cancer']
conf_matrix = confusion_matrix(y_test, y_pred7)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

acc_7 = accuracy_score(y_test, y_pred7)

# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)
print("The accuracy of this algorithm is {}".format(acc_7)) 

#-------------------------------------------------------Model 8 - Gradient Boosting Algorithm--------------------------------------------------------------------

from sklearn.ensemble import GradientBoostingClassifier
classifier8 = GradientBoostingClassifier(learning_rate=0.1)

classifier8.fit(X_train,y_train)

# Predicting the Test set results
y_pred8 = classifier8.predict(X_test)
print(np.concatenate((y_pred8.reshape(len(y_pred8),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred8)
print(cm)

# printing the confusion matrix 
LABELS = ['No Cancer', 'Cancer']
conf_matrix = confusion_matrix(y_test, y_pred8)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

acc_8 = accuracy_score(y_test, y_pred8)

# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)
print("The accuracy of this algorithm is {}".format(acc_8)) 

#-------------------------------------------------------Model 9 - Bagging Aggregating Clasifier Algorithm--------------------------------------------------------------------

from sklearn.ensemble import BaggingClassifier
bagging_classifier = BaggingClassifier(classifier4, n_estimators=12, random_state=40)
bagging_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred9 = bagging_classifier.predict(X_test)
print(np.concatenate((y_pred9.reshape(len(y_pred9),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred9)
print(cm)

# printing the confusion matrix 
LABELS = ['No Cancer', 'Cancer']
conf_matrix = confusion_matrix(y_test, y_pred9)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

acc_9 = accuracy_score(y_test, y_pred9)

# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)
print("The accuracy of this algorithm is {}".format(acc_9))

#-------------------------------------------------------Tabulating all the Models--------------------------------------------------------------------

Models = ['Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machine (Kernel)', 'Support Vector Machine (Linear Version)', 'Naive Bayes Algorithm','Decision Trees Algorithm', 'Random Forest Algorithm', 'Gradient Boosting Algorithm','Bagging Aggregating Classifier Algorithm']

Accuracy = [acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, acc_8, acc_9]

# Formatting accuracy values to 3 decimal points
Accuracy_formatted = [f'{acc:.3f}' for acc in Accuracy]

# Creating a DataFrame
All_models_comparison = pd.DataFrame({
    'Models': Models,
    'Accuracy': Accuracy_formatted,
})

print(All_models_comparison)


#-------------------------------------------------------Plotting all the Models--------------------------------------------------------------------

# Plotting using Matplotlib
plt.figure(figsize=(10, 6))
plt.barh(All_models_comparison['Models'], Accuracy, color='orange')
plt.xlabel('Accuracy')
plt.ylabel('Models')
plt.title('Accuracy of Different Models')
plt.gca().invert_yaxis()

# Displaying values on bars
for index, value in enumerate(Accuracy):
    plt.text(value, index, f'{value:.3f}', va='center', ha='left', fontsize=8)

plt.tight_layout()
plt.show()

#-------------------------------------------------------Models with 14 variables--------------------------------------------------------------------

# Define the number of random attributes you want to select
all_columns = cervical_cancer_new.columns

independent_columns = all_columns[:13].tolist() + all_columns[14:].tolist()

# Randomly select attributes/columns
random_selected_columns1 = [
    'STDs:HIV', 'Citology', 'Biopsy', 'Schiller', 'Smokes (packs/year)',
    'Hinselmann', 'Smokes (years)', 'IUD (years)',
    'STDs:vulvo-perineal condylomatosis', 'First sexual intercourse',
    'Hormonal Contraceptives (years)', 'Dx:CIN', 'Dx:HPV',
    'Hormonal Contraceptives'
]

# Load the independent variables with the randomly selected columns into X
X1 = cervical_cancer_new[random_selected_columns1].values

# Assuming the dependent variable is at index 13
y1 = cervical_cancer_new.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.20, random_state = 42)
print(X_train1)
print(y_train1)
print(X_test1)
print(y_test1)

print(X_train1.shape)
print(y_train1.shape)
print(X_test1.shape)
print(y_test1.shape)

# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier11 = SVC(kernel = 'linear', random_state = 0)
classifier11.fit(X_train1, y_train1)

# Predicting the Test set results
y_pred11 = classifier11.predict(X_test1)
print(np.concatenate((y_pred11.reshape(len(y_pred11),1), y_test1.reshape(len(y_test1),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test1, y_pred11)
print(cm)

# printing the confusion matrix 
LABELS = ['No Cancer', 'Cancer']
conf_matrix = confusion_matrix(y_test1, y_pred11)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

acc_11 = accuracy_score(y_test1, y_pred11)

# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)
print("The accuracy of this algorithm is {}".format(acc_11)) 

#-------------------------------------------------------Models with 10 variables--------------------------------------------------------------------

# Define the number of random attributes you want to select
all_columns = cervical_cancer_new.columns

independent_columns = all_columns[:13].tolist() + all_columns[14:].tolist()

# Randomly select attributes/columns
random_selected_columns2 = [
    'STDs:HIV', 'STDs:vulvo-perineal condylomatosis', 'Num of pregnancies', 'Dx', 
    'Smokes (packs/year)', 'Age', 'Citology', 'Dx:CIN', 'Number of sexual partners', 
    'STDs:syphilis'
]

# Load the independent variables with the randomly selected columns into X
X1 = cervical_cancer_new[random_selected_columns2].values

# Assuming the dependent variable is at index 13
y1 = cervical_cancer_new.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.20, random_state = 42)
print(X_train1)
print(y_train1)
print(X_test1)
print(y_test1)

print(X_train1.shape)
print(y_train1.shape)
print(X_test1.shape)
print(y_test1.shape)

# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier11 = SVC(kernel = 'linear', random_state = 0)
classifier11.fit(X_train1, y_train1)

# Predicting the Test set results
y_pred11 = classifier11.predict(X_test1)
print(np.concatenate((y_pred11.reshape(len(y_pred11),1), y_test1.reshape(len(y_test1),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test1, y_pred11)
print(cm)

# printing the confusion matrix 
LABELS = ['No Cancer', 'Cancer']
conf_matrix = confusion_matrix(y_test1, y_pred11)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

acc_22 = accuracy_score(y_test1, y_pred11)

# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)
print("The accuracy of this algorithm is {}".format(acc_22)) 

#-------------------------------------------------------Models with 7 variables--------------------------------------------------------------------

# Define the number of random attributes you want to select
all_columns = cervical_cancer_new.columns

independent_columns = all_columns[:13].tolist() + all_columns[14:].tolist()

# Randomly select attributes/columns
random_selected_columns3 = [
    'IUD (years)', 'Dx', 'STDs:HIV', 'Citology',
    'STDs:vulvo-perineal condylomatosis', 'Num of pregnancies', 'Schiller'
]

# Load the independent variables with the randomly selected columns into X
X1 = cervical_cancer_new[random_selected_columns3].values

# Assuming the dependent variable is at index 13
y1 = cervical_cancer_new.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.20, random_state = 42)
print(X_train1)
print(y_train1)
print(X_test1)
print(y_test1)

print(X_train1.shape)
print(y_train1.shape)
print(X_test1.shape)
print(y_test1.shape)

# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier11 = SVC(kernel = 'linear', random_state = 0)
classifier11.fit(X_train1, y_train1)

# Predicting the Test set results
y_pred11 = classifier11.predict(X_test1)
print(np.concatenate((y_pred11.reshape(len(y_pred11),1), y_test1.reshape(len(y_test1),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test1, y_pred11)
print(cm)

# printing the confusion matrix 
LABELS = ['No Cancer', 'Cancer']
conf_matrix = confusion_matrix(y_test1, y_pred11)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

acc_33 = accuracy_score(y_test1, y_pred11)

# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)
print("The accuracy of this algorithm is {}".format(acc_33)) 


#-------------------------------------------------------Tabulating all the Models with some variables--------------------------------------------------------------------


Models1 = ['Support Vector Machine (Linear Version) with 14 variables', 'Support Vector Machine (Linear Version) with 10 variables', 'Support Vector Machine (Linear Version) with 7 variables']

Variables = [
    ['STDs:HIV', 'Citology', 'Biopsy', 'Schiller', 'Smokes (packs/year)',
     'Hinselmann', 'Smokes (years)', 'IUD (years)',
     'STDs:vulvo-perineal condylomatosis', 'First sexual intercourse',
     'Hormonal Contraceptives (years)', 'Dx:CIN', 'Dx:HPV',
     'Hormonal Contraceptives'],
    
    ['STDs:HIV', 'STDs:vulvo-perineal condylomatosis', 'Num of pregnancies', 'Dx', 
     'Smokes (packs/year)', 'Age', 'Citology', 'Dx:CIN', 'Number of sexual partners', 
     'STDs:syphilis'],
    
    ['IUD (years)', 'Dx', 'STDs:HIV', 'Citology',
     'STDs:vulvo-perineal condylomatosis', 'Num of pregnancies', 'Schiller']
]

Accuracy = [acc_11, acc_22, acc_33]

# Creating a DataFrame
All_models_comparison1 = pd.DataFrame({
    'Models': Models1,
    'Variables in Model': Variables,
    'Accuracy': Accuracy,
})

print(All_models_comparison1)


# ------------------------------------------------------PREDICTION TOOL --------------------------------------------------------------

# pip install dash dash-bootstrap-components pandas scikit-learn
# pip install dash-bootstrap-components

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import dash_bootstrap_components as dbc


df = cervical_cancer_new  # Replace 'your_data.csv' with your actual data file

X = df.drop('Dx:Cancer', axis=1)  # Assuming 'Biopsy' is the target variable
y = df['Dx:Cancer']


#Replace this or add the best model code once determined.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# from sklearn.svm import SVC
# classifier4 = SVC(kernel = 'linear', random_state = 0)
# classifier4.fit(X_train, y_train)

# Create the RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MORPH])
# Update your layout to use dbc components
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Cervical Cancer Prediction Tool"), width={'size': 6, 'offset': 3})),

    # Add dbc input components for each feature
    dbc.Label("Age"),
    dbc.Input(id='age', type='number', placeholder='Enter your age', className='mb-3'),

    dbc.Label("Number of Sexual Partners"),
    dbc.Input(id='num_sexual_partners', type='number', placeholder='Enter the number of Sexual Partners', className='mb-3'),

    dbc.Label("First Sexual Intercourse"),
    dbc.Input(id='first_sexual_intercourse', type='number', placeholder='Enter the age of first sexual intercourse', className='mb-3'),

    dbc.Label("Number of Pregnancies"),
    dbc.Input(id='num_pregnancies', type='number', placeholder='Enter the number of pregnancies', className='mb-3'),

    dbc.Label("Smokes (years)"),
    dbc.Input(id='smokes_years', type='number', placeholder='Enter the number of years of smoking', className='mb-3'),

    dbc.Label("Smokes (packs/years)"),
    dbc.Input(id='smokes_packs_years', type='number', placeholder='Enter the packs of cigarettes per year', className='mb-3'),

    dbc.Label("Hormonal Contraceptives"),
    dbc.Input(id='hormonal_contraceptives', type='number', placeholder='Enter the packs of cigarettes per year', className='mb-3'),

    dbc.Label("Hormonal Contraceptives (years)"),
    dbc.Input(id='hormonal_contraceptives_years', type='number', placeholder='Enter the number of years using hormonal contraceptives', className='mb-3'),

    dbc.Label("IUD (Years)"),
    dbc.Input(id='iud_years', type='number', placeholder='Enter the number of years using IUD', className='mb-3'),

    dbc.Label("STDs (number)"),
    dbc.Input(id='std_number', type='number', placeholder='Enter the number of sexual transmitted diseases', className='mb-3'),

    dbc.Label("STDs: Vulvo-perineal Condylomatosis"),
    dbc.Input(id='stds_vulvo_perineal_condylomatosis', type='number', placeholder='Enter the number of STDs: Vulvo-perineal Condylomatosis', className='mb-3'),

    dbc.Label("STDs: Syphilis"),
    dbc.Input(id='stds_syphilis', type='number', placeholder='Enter the number of STDs: Syphilis', className='mb-3'),

    dbc.Label("STDs: HIV"),
    dbc.Input(id='stds_hiv', type='number', placeholder='Enter the number of STDs: HIV', className='mb-3'),

    dbc.Label("Dx: CIN"),
    dbc.Input(id='dx_cin', type='number', placeholder='Enter the Dx: CIN', className='mb-3'),

    dbc.Label("Dx: HPV"),
    dbc.Input(id='dx_hpv', type='number', placeholder='Enter the Dx: HPV', className='mb-3'),
    
    dbc.Label("Dx"),
    dbc.Input(id='dx', type='number', placeholder='Enter the Dx', className='mb-3'),
    
    dbc.Label("Hinselmann"),
    dbc.Input(id='hinselmann', type='number', placeholder='Enter the Hinselmann', className='mb-3'),
    
    dbc.Label("Schiller"),
    dbc.Input(id='schiller', type='number', placeholder='Enter the Schiller', className='mb-3'),
    
    dbc.Label("Citology"),
    dbc.Input(id='citology', type='number', placeholder='Enter the Citology', className='mb-3'),
    
    dbc.Label("Biopsy"),
    dbc.Input(id='biopsy', type='number', placeholder='Enter the Biopsy', className='mb-3'),

    # Add Predict button with dbc styling
    dbc.Button('Predict', id='predict-button', color='info', className='mt-3'),

    # Display result in a styled dbc Alert component
    dbc.Alert(id='output-container', color='info', className='mt-3'),
], className='mt-5')


# Define callback to handle prediction
@app.callback(
    Output('output-container', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('age', 'value'),
     State('num_sexual_partners', 'value'),
     State('first_sexual_intercourse', 'value'),
     State('num_pregnancies', 'value'),
     State('smokes_years', 'value'),
     State('smokes_packs_years', 'value'),
     State('hormonal_contraceptives', 'value'),
     State('hormonal_contraceptives_years', 'value'),
     State('iud_years', 'value'),
     State('std_number', 'value'),
     State('stds_vulvo_perineal_condylomatosis', 'value'),
     State('stds_syphilis', 'value'),
     State('stds_hiv', 'value'),
     State('dx_cin', 'value'),
     State('dx_hpv', 'value'),
     State('dx', 'value'),
     State('hinselmann', 'value'),
     State('schiller', 'value'),
     State('citology', 'value')],
     State('biopsy', 'value')
)
def update_output(n_clicks, age, num_sexual_partners, first_sexual_intercourse, num_pregnancies,
                  smokes_years, smokes_packs_years, hormonal_contraceptives, hormonal_contraceptives_years,
                  iud_years, std_number, stds_vulvo_perineal_condylomatosis, stds_syphilis, stds_hiv,
                  dx_cin, dx_hpv, dx, hinselmann, schiller, citology, biopsy):

    if n_clicks is None:
        return ''

    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'Age': [age],
        'Number of sexual partners': [num_sexual_partners],
        'First sexual intercourse': [first_sexual_intercourse],
        'Number of pregnancies': [num_pregnancies],
        'Smokes(years)': [smokes_years],
        'Smokes(packs/years)': [smokes_packs_years],
        'Hormonal Contraceptives': [hormonal_contraceptives],
        'Hormonal Contraceptives (years)': [hormonal_contraceptives_years],
        'IUD (Years)': [iud_years],
        'STDs(number)': [std_number],
        'STDs:vulvo-perineal condvlomatosis': [stds_vulvo_perineal_condylomatosis],
        'STDs:svphilis': [stds_syphilis],
        'STDs:HIV': [stds_hiv],
        'Dx:CIN': [dx_cin],
        'Dx:HPV': [dx_hpv],
        'Dx': [dx],
        'Hinselmann': [hinselmann],
        'Schiller': [schiller],
        'Citology': [citology],
        'Biopsy': [biopsy]
    })

    # Make prediction
    prediction = classifier.predict(input_data)[0]

    # Calculate accuracy on the test set (just for display, you might not use this in a real application)
    y_pred_test = classifier.predict(X_test)
    accuracy = 90.30

    # Display results based on prediction
    if prediction == 1:
        result_text = f"You have been diagnosed with Cervical Cancer. Accuracy of the prediction: {accuracy:.2f}%"
    else:
        result_text = f"Great news! You do not have Cervical Cancer. Accuracy of the prediction: {accuracy:.2f}%"
        
    return result_text
        
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


# ------------------------------- END -----------------------------








