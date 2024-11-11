![318692763-d2714505-ceae-48c6-b428-fc421aaa735d](https://github.com/user-attachments/assets/1362e977-38c3-40b8-895f-afa0b4e13b9b)## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM
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
NAME: PANIMALAR P
REGISTER NO:212222110031
````
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

`````
![image](https://github.com/user-attachments/assets/cfa37d90-64a4-4983-9671-88d0f775b2b7)

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
![318897767-781ddd71-1fc6-499b-9234-b83778405580](https://github.com/user-attachments/assets/009c1362-b165-4a81-94b7-87e13ae0eb52)
````
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
````

![318897871-6f1877a4-9ba9-45d6-8df2-38fdc103a0ef](https://github.com/user-attachments/assets/80a585d0-3858-47d5-b3fe-4f3172a5a058)
````
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
````
![318897982-63cbb12a-e9eb-447e-855a-e56c706bbfa9](https://github.com/user-attachments/assets/de3cacc3-4f41-44bb-a750-4b82f9cbd429)
````
df.skew()
````

![318898092-3d04bbce-76dc-4571-8c8d-5aad234c1766](https://github.com/user-attachments/assets/b39e6abd-d6f1-4357-8c11-b474506db40c)
````
np.log(df["Highly Positive Skew"])
````

![318898189-7247340c-6488-4b75-9deb-0ad3f10e03fd](https://github.com/user-attachments/assets/7128205d-fedd-4847-8b21-8e3cf5982d01)
````
np.reciprocal(df["Moderate Positive Skew"])
````
![318898261-71ae0399-a828-406a-93a6-0e36cc31e249](https://github.com/user-attachments/assets/d36f1f36-dc08-4b1b-bad0-f79a388842e1)
````
np.sqrt(df["Highly Positive Skew"])
````
![318898327-9b500fd0-9b55-4397-b1e8-364652aca983](https://github.com/user-attachments/assets/6174f9bc-56b1-41d8-a767-a8d6086249a8)

````
np.square(df["Highly Positive Skew"])
````
![318898423-d243323b-c97e-4c55-a41f-f76d176e6461](https://github.com/user-attachments/assets/42e067c5-180f-4522-b024-6a8113a3a711)

 ````
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
````
![318898509-758eaaba-b780-4fee-8487-d8242a9d6148](https://github.com/user-attachments/assets/d5ae4f38-ab5b-47e4-abfe-7d49057cf2d4)
````
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
````

![318898927-4945b8c6-e27d-4526-9032-0c0aeb9ab576](https://github.com/user-attachments/assets/b6a7d45c-fd19-4a25-bf21-78fe22b560f2)
````
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
````
![318899248-52a7553c-c1bd-4489-a0cb-b13a27684c23](https://github.com/user-attachments/assets/69443ba2-6f09-406d-aa72-9d153bb2117b)

````
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
````
![318899545-3688ed78-4920-4cd4-9e33-4420fc790b8d](https://github.com/user-attachments/assets/e0b3eb9d-4305-4fb2-9c9d-6b2395162a18)

````
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
````
![318899696-9ef5152c-d766-48e1-857c-a7dbfde4e648](https://github.com/user-attachments/assets/67d7d7ef-639b-4a86-9b82-cf1405213b1f)
````
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
````
![318899799-fde4b296-88ec-46ad-b6f3-2cf2b64a15f2](https://github.com/user-attachments/assets/41bc21b9-ecd5-4172-bf62-66e0d0ddcfbb)

````
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
````

![318899874-57bae70b-8ee0-4ab1-86bf-733d2597089d](https://github.com/user-attachments/assets/3a3db138-720c-4b12-88d7-946ad1f0919f)
````
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
````
![318900112-3987a28b-3816-41b2-9a9d-6a1cedf8382e](https://github.com/user-attachments/assets/ffa4c1b7-d4c6-4369-8416-f2404bc79121)






 
# RESULT:
     
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
