import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# Predicting the house price based on the area
df = pd.read_csv('house_price.csv')
print(df)
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area, df.price, color='red', marker='+')
plt.show()

# linear regression:
reg = linear_model.LinearRegression()
print(reg.fit(df[['area']], df.price))
# print(reg.predict(3300))
print("This is the value of m slope : ",reg.coef_) # it will give you the value of (m)
print("This is the value of b slope : ",reg.intercept_) # it will give you the value of (b)
# Now we calculating the (y) value using (y = mx + c)
y1 = (reg.coef_) * 3300 + (reg.intercept_)
print("The price of 3300 area is : ",y1) # it will give you the price of 3300 area
y2 = (reg.coef_) * 5000 + (reg.intercept_)
print("The price of 5000 area is : ",y2) # it will give you the price of 5000 area

# Creating an linear regression graph to see the line:
plt.xlabel('area', fontsize=20)
plt.ylabel('price', fontsize=20)
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
plt.show()

# Here we have the area and we are interseting the price in the dataset:
d = pd.read_csv('area.csv')
print(d)
p = reg.predict(d)
d['price'] = p
d.to_csv("prediction.csv", index=False)

'''
# Predicting the income based on the year:
# this is for the finding the capital income for 2020 year:
df = pd.read_csv('canada_per_capita_income.csv')
print(df)
plt.xlabel('year')
plt.ylabel('per capita income')
plt.scatter(df.year, df.per_capita_income, color='red', marker='+')
# plt.show()

reg = linear_model.LinearRegression()
print(reg.fit(df[['year']], df.per_capita_income))
print("The value of (m) slope is: ", reg.coef_)
print("The value of (b) slope is: ", reg.intercept_)
y1 = (reg.coef_) * 2020 + (reg.intercept_)
print("The value of year 2020 is: ",y1)

plt.xlabel('year', fontsize=10)
plt.ylabel('per_capita_income', fontsize=10)
plt.scatter(df.year, df.per_capita_income, color='red', marker='+')
plt.plot(df.year, reg.predict(df[['year']]), color='blue')
plt.show()
'''