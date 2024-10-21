## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:
Read the given Data.
STEP 2:
Clean the Data Set using Data Cleaning Process.
STEP 3:
Apply Feature Encoding for the feature in the data set.
STEP 4:
Apply Feature Transformation for the feature in the data set.
STEP 5:
Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
````
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
````
![image](https://github.com/user-attachments/assets/2b25d89e-81d5-436b-91f4-fef10f04f67e)
````
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
````
![image](https://github.com/user-attachments/assets/298ed08e-0d10-4ef7-a502-2c05a36d74cc)
````
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
````
 ![image](https://github.com/user-attachments/assets/742e0c9b-97b2-4580-b51f-c7e2a1960164)
````
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
````
````
df2=pd.concat([df2,enc],axis=1)
df2
````
![image](https://github.com/user-attachments/assets/1d81d5b0-420b-4a6c-868d-f24d5af4634b)
````
pd.get_dummies(df2,columns=["nom_0"])
````
![image](https://github.com/user-attachments/assets/aac0bd04-7c53-4ea5-81a5-511a7dc80bc4)
````
pip install --upgrade category_encoders
````
![image](https://github.com/user-attachments/assets/53d69cdd-5dff-41de-96c7-6ca10251949c)
````
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
````
![image](https://github.com/user-attachments/assets/1f5ae11b-4cc3-4230-9320-463a57383f77)
````
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
````

     
# RESULT:
       # INCLUDE YOUR RESULT HERE

       
