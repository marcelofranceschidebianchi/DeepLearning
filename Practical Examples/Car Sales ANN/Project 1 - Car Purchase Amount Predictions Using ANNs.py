
# coding: utf-8

# 
# # PREDICT CAR PURCHASING DOLLAR AMOUNT USING ANNs (REGRESSION TASK)
# ![BRAIN.png](attachment:BRAIN.png)
# 
# # Marcelo Bianchi
# 

# # PROBLEM STATEMENT

# You are working as a car salesman and you would like to develop a model to predict the total dollar amount that customers are willing to pay given the following attributes: 
# - Customer Name
# - Customer e-mail
# - Country
# - Gender
# - Age
# - Annual Salary 
# - Credit Card Debt 
# - Net Worth 
# 
# The model should predict: 
# - Car Purchase Amount 

# # STEP #0: LIBRARIES IMPORT
# 

# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # STEP #1: IMPORT DATASET

# In[27]:


car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')


# In[28]:


car_df


# # STEP #2: VISUALIZE DATASET

# In[4]:


sns.pairplot(car_df)


# # STEP #3: CREATE TESTING AND TRAINING DATASET/DATA CLEANING

# In[5]:


X = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)


# In[6]:


X


# In[7]:


y = car_df['Car Purchase Amount']
y.shape


# In[8]:


from sklearn.preprocessing import MinMaxScaler

scaler_x = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)


# In[9]:


scaler_x.data_max_


# In[10]:


scaler_x.data_min_


# In[11]:


print(X_scaled)


# In[12]:


X_scaled.shape


# In[13]:


y.shape


# In[14]:


y = y.values.reshape(-1,1)


# In[15]:


y.shape


# In[16]:


y


# In[17]:


scaler_y = MinMaxScaler()

y_scaled = scaler_y.fit_transform(y)


# In[18]:


y_scaled


# # STEP#4: TRAINING THE MODEL

# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)


# In[20]:


import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

model = Sequential()
model.add(Dense(25, input_dim=5, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()


# In[21]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[22]:


epochs_hist = model.fit(X_train, y_train, epochs=20, batch_size=25,  verbose=1, validation_split=0.2)


# # STEP#5: EVALUATING THE MODEL 

# In[23]:


print(epochs_hist.history.keys())


# In[24]:


plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])


# In[25]:


# Gender, Age, Annual Salary, Credit Card Debt, Net Worth 

# ***(Note that input data must be normalized)***

X_test_sample = np.array([[0, 0.4370344,  0.53515116, 0.57836085, 0.22342985]])
#X_test_sample = np.array([[1, 0.53462305, 0.51713347, 0.46690159, 0.45198622]])

y_predict_sample = model.predict(X_test_sample)

print('Expected Purchase Amount=', y_predict_sample)
y_predict_sample_orig = scaler_y.inverse_transform(y_predict_sample)
print('Expected Purchase Amount=', y_predict_sample_orig)


# #  REGRESSION TASK FINALIZED! 
