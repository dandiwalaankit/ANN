#If we want to comment out any lines. Then cover those line under three doube quote
#(i.e.:
"""X1=dataset.iloc[:,:-1].values
X2=dataset.iloc[:,:-2]
X3=dataset.iloc[:,:-3]"""

# Import libraries
import numpy as np # Here "numpy" is the library used to perform  mathematical operations
#importing libraries with additional "as np" creates shortcut valid name 'np' of library "numpy".
import matplotlib.pyplot as plt # Here ".pyplot" is the sub library This library is used to get plot
import pandas as pd # this library is a good library to import and manage the dataset


#Importig dataset available in the same folder this file belongs to using pandas library.
#The excel file in .csv formt is named as Data.
#For making this success, the working directory has to be set in which this both data and script are stored
dataset=pd.read_csv('Data.csv')  

#Imp note: no of columns and rows are started from zero in python. Whereas in 'R' it starts from one I guess
#Particular Data from first to third column are named as 'X'.
#Here, ':' means all columns of the dataset
#':-1" means all coumns except last one
#Column index from last starts with -1


X=dataset.iloc[:].values # all columns
X1=dataset.iloc[:,:-1].values # All columns, Except last one
X2=dataset.iloc[:,:-2] #All columns except last two
X3=dataset.iloc[:,:-3] # All columns except last three

age=dataset["Age"] # Specifically selected column having heading "Age"

x=dataset[["Country", "Salary", "Purchased"]] #Only columns heading of which are written in double closed bracket

y=dataset.iloc[:].values

y1=dataset.iloc[:,0] # 1st Column
y1_2=dataset.iloc[:,:1] # Upto 1st column
y1_3=dataset.iloc[:,:+1] #Same as above, upto 1st column
.
y2=dataset.iloc[:,1] #only 2nd columns
y2_2=dataset.iloc[:,:2] #Upto 2nd Column

y3=dataset.iloc[:,2] #Only 3rd column
y3_2=dataset.iloc[:,:+3] #Upto 3rd column

y23=dataset.iloc[:,1:3] #this is for 2nd to 3rd column. In "1:3" represents 2nd to 4th but here upper bound 4th is excluded.

p=dataset.iloc[:,:-1].values
q=dataset.iloc[:,3].values

#extract any daata from the dataset
mixed_dataset_1=dataset.iloc[0:2,:]
#here, 0:2 before comma means, first to second rows. and : after comma is all columns
mixed_dataset_2=dataset.iloc[:,0:2]
#here, : before comma means all rows. And 0:2 after comma is first to second column.
mixed_dataset_3=dataset.iloc[0:4,0:3]
#here, 0:4 before comma means first to third rows. And 0:3 after comma is first to third column.



#///////////MISSING VALUES/////////////#
#in order to employ learning model, the missing values have to be filled with imputer. There are various strategies to fill the same, here the blank cell are filled by mean if the particular data. 
from sklearn.preprocessing import Imputer  #sklearn is library, Imputer is class
imputer=Imputer(missing_values="NaN", strategy="mean", axis=0) # typed Imputer in help or called object nspector in new version of spyder
# In help/ObjectInspector, parameters missing_value, strategy and axis were asked to be used
# in our data, missing value is written by "NaN"
# strategy='mean' means: Mean of the data will be placed instead of "nan". We can also take medium or other strategy instead of mean
# axis=0, means, strategy will be applied along the column, means mean of the data along the column
# axis =1 means row. Which is not used here now.
imputer=imputer.fit(p[:,1:3]) # To fit this Imputer in the matrix 'p'
p[:,1:3]=imputer.transform(p[:,1:3])
#Here, It is done. Now check 'p' in console


#/////////////ENCODING CATEGORICAL DATA/////////////////
#Data is categorical, if data contains country names, contains "yes" and "no", gender i.e. "male" and "female".
# ...Now machine learning models understand numerical values, so here categorical data can raise an issue.
#...Thats why categorical variables need to be encoded
#.. To do so, 'sklearn' library is used here.

#Encoding "Country" 
from sklearn.preprocessing import LabelEncoder
labelencoder_p=LabelEncoder() #callinng object, "labelencoder
p[:,0]=labelencoder_p.fit_transform(p[:,0])
#Encoding is done. check 'p' in console. Now order of encoded data is random. No relation or logic behind encoding France as 2 or other . .
#But there may be problem:
#encoder encodes coutry without bothering the order (If there may be)
#..Equation may assume that a contry (Let's say Germany) has higher/lower value than other (let's say France) (as per encoding)
#..  We have to prevent machine learning to assuem that
#So we have to create dummy variable (means sperate columns for all country with Y/N = 0/1 data filling)

#This can be achieved by importing "OneHotEncoder"
#Using OneHotEncoder, we will replace country column with three columns (three = no. of total contries in the column) titled by country nameds.
#Then If the data is regarding Germany, then Germany will be encoded as 1 and remaining two will be encoded as 0

from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features = [0])
p=onehotencoder.fit_transform(p).toarray()

#Encoding "Purhased" column, whicch contains "yes" and "no"
labelencoder_q=LabelEncoder()
q=labelencoder_q.fit_transform(q)
#No OneHotEncoder is need to be used here. Because it has only two category, "Yes" and "No"

#//////////////SPLITTING DATASET INTO TRAINING SET AND TEST SET///////////#
from sklearn.model_selection import train_test_split
p_train, p_test, q_train, q_test =train_test_split(p,q, test_size=0.2, random_state=0)
#Here test_size is 0.2; means 20% of the dataset will be used as test set. Generally it is taken 20%, 25%; in some cases 30%. But never 0.5 or 50%
#Here test_size=0.2. means 2 out of 10 data will be used as test set and 8 will be used as training set

#///////////////////FEATURE SCALING//////////////////////////////#
#Why to scale the data?
#In this dataset, it may happen that one of the variables "Age" is varying between 27 to 50
#and other variable "Assets" is varying between 30000000 to 90000000.
#Which are at the different scale.
#Now Machine Learning models are based on Euclidine distance (i.e. EuDist=sqrt((y2-y1)^2+(x2-x1)^2))
#Now here for existing data, this Euclidine distance is being dominated by variable "Assets"
#as it is of higher scale   
# Therefore these are need to be scaled.

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
p_train=sc.fit_transform(p_train)
#When we apply the object (sc) to the training set, then first it needs to be fit the object to the training set and then need to be transformed
#Here, fit_transofrm is the method
p_test=sc.transform(p_test)
#Here, object sc doesn not need to be fit with test set as it has already been fit with training set

#Here worldwide question has arose that, do we need to scale dummy variable ? which are 0-1.? 
#It depends upon the context of the given data.

#Note: Even if machine learning algorithm are not based on the euclidean distances,
# It needs to be feature scaled so that algorithm converges much faster and reduces run time

#Other question is do we need to pply afeature scalinng to output training(q) variable?
#Answer is Yes. But in a case where there is single variable classification problem, Don't need to apply feature scaling
