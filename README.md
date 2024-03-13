# 3
```ruby
class Animal:
    def __init__(self, name):
        self._name = name
    def get_name(self):
        return self._name
				
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)
        self._breed = breed
    def bark(self):
        print("woof!")
				
class Calculator:
    def add(self, a, b=None, c=None):
        if b is not None and c is not None:
            return a + b + c
        elif b is not None:
            return a + b
        else:
            return a

if __name__ == "__main__":
    animal = Animal("Generic Animal")
    print(f"Animal name: {animal.get_name()}")
    dog = Dog("Buddy", "Golden Retriever")
    print(f"Dog Name: {dog.get_name()}, Breed: {dog._breed}")
    dog.bark()
    cal = Calculator()
    r1 = cal.add(1)
    r2 = cal.add(1, 2)
    r3 = cal.add(1, 2, 3)
    print(f"Result: {r1}, Result2: {r2}, Result3: {r3}")
```
# 5a
```ruby
import numpy as np
import matplotlib.pyplot as plt
# Array manipulation
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
# Concatenation
concatenated_arr = np.concatenate((arr1, arr2))
# Reshaping
reshaped_arr = np.reshape(concatenated_arr, (3, 4))
print("Array Manipulation:")
print("Original Arrays:")
print(arr1)
print(arr2)
print("Concatenated Array:")
print(concatenated_arr)
print("Reshaped Array:")
print(reshaped_arr)
# Sorting
sorted_arr = np.sort(reshaped_arr, axis=None)
print("\nSorting:")
print("Sorted Array:")
print(sorted_arr)
# Horizontal splitting
horizontal_split = np.split(sorted_arr, 3, axis=1)
print("\nHorizontal Splitting:")
print("Split Arrays (along columns):")
for i, arr in enumerate(horizontal_split):
    print(f"Array {i + 1}: {arr}")
# Vertical splitting
vertical_split = np.split(sorted_arr, 3, axis=0)
print("\nVertical Splitting:")
print("Split Arrays (along rows):")
for i, arr in enumerate(vertical_split):
    print(f"Array {i + 1}: {arr}")
```
# 5b broadcasting------------------------
```ruby
import numpy as np
print("Arrays of same size\n")
a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
print(a + b)
print("...")
print("Add scalar 8 to an array")
print(a + 8)
print("...")
print("Add 20 to 10 Array")
m = np.ones((3, 3))
print(m)
print(".....")
print("Broadcasting of both arrays\n")
a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
print(a)
print(b)
print(a + b)
print("......")
print("Right side padding")
print(a[:, np.newaxis].shape)
print(m + a[:, np.newaxis])
print("......")
print("Computes log(exp(a)+exp(b))\n")
print(np.logaddexp(m, a[:, np.newaxis]))
print("........")
print("Centering\n")
x = np.random.random((10, 3))
print(x)
xmean = x.mean(axis=0)
print(xmean)
xcentered = x - xmean
print(xcentered)
print(xcentered.mean(axis=0))
print(".....")
```
# 5b------------------------------------------------
```ruby
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]
z = np.sin(x) * 10 + np.cos(10 + y * x) * np.cos(x)
plt.imshow(z, origin="lower", extent=[2, 10, 2, 10], cmap='viridis')
plt.colorbar()
plt.show()
```
# 6
```ruby
# line plot
from matplotlib import pyplot as plt
weekdays=[1,2,3,4,5]
subject_attendance=[65,78,89,56,86]
plt.plot(weekdays,subject_attendance)
plt.title('Attendance')
plt.xlabel('Weekday')
plt.ylabel('No. of students present')
plt.show()
# Bargraph
from matplotlib import pyplot as plt
plt.bar([3,6,9,12],[100,150,200,180],label='Grocery Sales',color='g')
plt.xlabel('month')
plt.ylabel('sales in crores')
plt.title('Sales')
plt.show()
# histogram
from matplotlib import pyplot as plt
Customer_waittime=[30,32.5,35,40,14,22,38,45,43,36,49,29.2,33]
inter=[0,5,10,15,20,25,30,35,40,45,50,55]
plt.hist(Customer_waittime,inter,histtype='bar',rwidth=0.8)
plt.xlabel('time in mins')
plt.ylabel('no. of customers')
plt.title('Histogram')
plt.show()
# Scatter plot
from matplotlib import pyplot as plt
Net_Profit=[10,18,24,24,29,37,45,56]
Sales=[100,200,250,300,380,450,500,600]
plt.scatter(Net_Profit,Sales,label='correlation',color='k')
plt.title('Scatter Plot')
plt.legend()
plt.xlabel('Profit')
plt.ylabel('Sales')
plt.show()
# Boxplot
import matplotlib.pyplot as plt
x=[[10,20,30,40,50,60],[30,20,40,45,50,70],[-30,10,20,30,40,50],[10,20,30,40,50,90]]
plt.boxplot(x,labels=["Aditi","Amar","Bhanu","Surya"],patch_artist="True",showmeans="True",meanline="True")
plt.title("Marks Stats")
plt.xlabel("Student Names")
plt.ylabel("Marks")
plt.show()
```
# 7 
```ruby
import numpy as np

m = int(input("Enter the number of rows: "))
n = int(input("Enter the number of columns: "))

arr = np.ones((m, n), dtype=int)

print(arr)
print('Array flags:')
print(arr.flags)
print('Shape of an array:')
print(arr.shape)
print('Array size:')
print(arr.size)
print('Array itemsize:')
print(arr.itemsize)
print('Array dimension:')
print(arr.ndim)
```
# 8 
```ruby
import numpy as np
import matplotlib.pyplot as plt

def estimate_coef(x, y):
    n = np.size(x)
    m_x, m_y = np.mean(x), np.mean(y)
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x
    return b_0, b_1

def plot_reg_line(x, y, b):
    plt.scatter(x, y, color="blue", marker="o", s=30)
    y_pred = b[0] + b[1] * x
    plt.plot(x, y_pred, color="green")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()

def main():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 121])
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {}\nb_1 = {}".format(b[0], b[1]))
    plot_reg_line(x, y, b)

if __name__ == "__main__":
    main()

```
# 9
```ruby
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# Load dataset
dataset = pd.read_csv(r'C:\Users\JSSMCA\Desktop\User_Data.csv')

# Prepare features and target variable
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Train logistic regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(x_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# Visualizing the training set results
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

```
# 10
```ruby
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\JSSMCA\Desktop\infosys1.csv', parse_dates=["Date"])

print(df.head())
print("Date index:")
print(df.index)

print(df.loc['2021-09-07'])

print(df.loc['2021-09'])

print("Mean of Days High of Infosys during Sept 2021:")
print(df.loc['2021-09', 'High'].mean())

print("Total volume of Infosys during Sept 2021:")
print(df.loc['2021-09', 'Volume'].sum())

print("Details of stock between dates: April 8th to 10th")
print(df.loc['2021-04-08': '2021-04-10'])

df['Close'].plot()

df.resample('M', on='Date').mean()['Close'].plot(kind='bar')
plt.show()
```
# 11
```ruby
from numpy import random 
import seaborn as sns 
import matplotlib.pyplot as plt

# Scatter plot
height = [162, 64, 69, 75, 66, 68, 65, 71, 76, 73] 
weight = [120, 136, 148, 175, 137, 165, 154, 172, 200, 187] 
print("**scatterplot**") 
sns.scatterplot(x=height, y=weight)
plt.show()

# Create a count plot
gender = ["Female", "Female", "Female", "Female", "Male", "Male", "Male", "Male", "Male", "Male"] 
print("**countplot**")
sns.countplot(x=gender)
plt.show()

# Distplot
print("**distplot**")
sns.distplot([0, 1, 2, 3, 4, 5])
plt.show()

# Normal distribution
print("**Normal distribution**")
sns.distplot(random.normal(size=1000), hist=True)
plt.show()

# Binomial distribution
print("**Binomial distribution**")
sns.distplot(random.binomial(n=10, p=0.5, size=1000), hist=True, kde=True)
plt.show()
```
