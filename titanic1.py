import pickle
import streamlit as st
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic.main import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv("F:\\MSc(TCS)\\Lectures_Sem2\\ML2021\\data\\titanic\\train.csv",
                 header=0,usecols=[2,4,5,6,7,9,1])
############# family size #############
for i, data in df.iterrows():
    df.at[i,'FamilySize'] = data['SibSp'] + data['Parch'] + 1
df['FamilySize']=df['FamilySize'].astype(int)

#####  you can use LabelEncoder class for Gender
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
df['Gender_Encode']=label_encoder.fit_transform(df["Sex"] )
######### drop the Gender column with string ########
df.drop(['Sex'],axis=1, inplace=True)
############## fill the NaN of Age #############
import numpy as np
std_age=df['Age'].std()
avg_age=df['Age'].mean()
for i,data in df.iterrows():
    if pd.isnull(data['Age']):
        r=np.random.uniform(avg_age-std_age,avg_age+std_age)
        r=np.round(r,0)
        df.at[i,'Age']=r
        print(i," ",df.at[i,'Age'])
#############one hot encoding for Pclass############################
pd.get_dummies(df['Pclass'], prefix='class')
df=pd.concat([df,pd.get_dummies(df['Pclass'], prefix='class')],axis=1)
st.dataframe(df)

################ std scaler operation on Fare and Age ####################
#st.dataframe(X_train['Fare'])
print(df['Fare'].values)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#convert Series to array then reshape and learn the distribution in sc object
# for single column data reshape is reqd and not reqd for multiple columns
df_Fare=sc.fit_transform((df['Fare'].values).reshape(-1,1))
# apply the learnt distribution on the Fare column to change the values
df_Fare1=sc.transform((df['Fare'].values).reshape(-1,1))
df['Fare_stdScaler']=pd.DataFrame(df_Fare1)
#convert Series to array then reshape and learn the distribution in sc object
df_Age=sc.fit_transform((df['Age'].values).reshape(-1,1))
# apply the learnt distribution on the Age column to change the values
df_Age1=sc.transform((df['Age'].values).reshape(-1,1))
df['Age_stdScaler']=pd.DataFrame(df_Age1)
#########################Remove Pclass Age Fare Sibsp Parch from dataframe #############
df.drop(['Pclass','Age','Fare','SibSp','Parch'],axis=1, inplace=True)
####### display the reqd dataframe #############
st.dataframe(df)
############################################
################# define the feature and target ###########
X = df.drop("Survived", axis=1)
y = df["Survived"]
##########################split to test and train###################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
df[df.isnull().any(axis=1)]  # whereever null show that row
###############################
print(X_train.columns)
# Creating FastAPI instance
app = FastAPI()
############## data struc for input request
class request_body(BaseModel):
    pclass:int
    age: float
    gender: str
    familysize = int
    fare: float

# Creating and Fitting our Model
clf = GaussianNB()
clf.fit(X_train, y_train)
pickle.dump(clf,open('model.pkl','wb')) # write binary

# load the model from disk
loaded_model = pickle.load(open('model.pkl', 'rb'))
#import streamlit as st
def UI_Page():
    st.title("ML Algorithm")
    pclass=st.selectbox("Passenger Class :",[1,2,3])
    if pclass==1:
        class_1=1
        class_2=0
        class_3=0
    elif pclass==2:
        class_1 = 0
        class_2 = 1
        class_3 = 0
    elif pclass==3:
        class_1 = 0
        class_2 = 0
        class_3 = 1
    age=st.slider("Age :",min_value=1,max_value=95)
    from numpy import asarray
    df_Age_scaled = sc.transform(asarray(age).reshape(1,-1))
    print("$$$",df_Age_scaled)
    gender=st.radio('Gender', ['Male','Female'])
    if gender=='Male':
        g=1
    else:
        g=0
    familysize=st.selectbox("Family Size",[1,2,3,5,7,8,9])
    fare=st.text_input("Fare :")

    df_Fare_scaled = sc.transform(asarray(fare).reshape(1,-1))
    print("$$$",df_Fare_scaled)
    ok=st.button("Predict the Survival")  # ok has True value when user clicks button
    try:
        if ok==True:       # if user pressed ok button then True passed
            testdata=np.array([[familysize,g,class_1,class_2,class_3,df_Fare_scaled,df_Age_scaled]])
            classindx = loaded_model.predict(testdata)[0]
            if classindx==0:
                st.error("Person did not survive")
            elif classindx==1:
                st.success("Person survived")
    except Exception as e:   # user way of writing error
         st.info(e)

@app.post('/predict')
def predict(data: request_body):
    # Making the data in a form suitable for prediction
    test_data = [[
        data.pclass,
        data.age,
        data.familysize,
        data.gender,
        data.fare,
    ]]

    # Predicting the Class
    class_idx = loaded_model.predict(test_data)[0]

    # Return the Result
    return {'survived': class_idx}

if __name__ == "__main__":
    uvicorn.run(app)
