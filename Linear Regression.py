# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv(r'diabetic_kidney_disease.csv')

 
x=data["FBG (mg/dL)"]
y=data["UACR (mg/g creatinine)"]
minnx= x.min()
miny=y.min()
maxy=y.max()
maxix=x.max()
temp1=0

for i in x:
    temp1=(x-minnx)/(maxix-minnx)

temp2=0
for i in y:
    temp2=(y-miny)/(maxy-miny)

x_train=temp1[:65]
y_train=temp2[:65]
x_test=temp1[65:]
y_test=temp2[65:]

learning_rate=0.006
itr_num=100 
j = 0
m=65
C1 = []
C2 = []
C1.insert(0 , (np.random.uniform(0.0001, 0.250)))
C2.insert(0 , (np.random.uniform(0.1, 0.9))) 
h=0
c = 0
Theta1=0
Theta2 = 0
error = []
error.insert(0, 222)

def CalculateHypo(C1,C2,x_train):
        return (C1+(C2*x_train))

while j <itr_num:
    h=CalculateHypo(C1[j],C2[j],x_train)
    summation = 0
    counter = 0
    dif=0
    theta_dif = 0
    for i in y_train.keys():
       dif = y_train[i] - h[i]
       sq_dif = dif**2
       summation += sq_dif   
       theta_dif = ((y_train[i] - h[i])*x_train[i])
       counter+= theta_dif
       
       
    error.insert(j+1,(summation /(m)))
    b =  (C1[j]-((learning_rate *1/m) * counter ))
    a =  (C2[j]-((learning_rate *1/m) * counter ))      
    C1.insert(j+1 , b)
    C2.insert(j+1 , a)  
    
    j+=1
    if itr_num == j:
        if c == 10:
            break
        else:
            c+=1
            itr_num = np.random.randint(90 , 150)
            learning_rate= np.random.uniform(0.001, 0.5)
            print ('Best fit with the min MSE of try ',c,' : ', min(error))
            ind=error.index(min(error) , 1 , j)
            print ('Best fit with Theta 1 in try ',c,' : ', C1[ind])
            print ('Best fit with Theta 2 in try ',c,' : ', C2[ind])
            print('----------------///////////////////////////////////////////////////////////--------------------')
            Theta1 = C1[ind]
            Theta2= C2[ind]
            j = 0 
            C1.clear()
            C2.clear()
            error.clear()
            C1.insert(0 , (np.random.uniform(0, 0.08)))
            C2.insert(0 , (np.random.uniform(0, 0.8))) 
            error.insert(0, 222)
            continue
            
        
print(Theta1 , ' ' , Theta2)   
h = CalculateHypo(Theta1,Theta2,x_test)
plt.xlabel("X_fbg")
plt.ylabel("Y_uacr")
plt.title("Real vs predicted values")
plt.scatter(x_train,y_train)
plt.plot(x_test,h)
plt.show()