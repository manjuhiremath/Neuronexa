/3333333333333333333333333333333333333333333333333 
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
5a-------------------------------------------------------------------
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
xcentered = x - xmea
print(xcentered)
print(xcentered.mean(axis=0))
print(".....")

5b------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]
z = np.sin(x) * 10 + np.cos(10 + y * x) * np.cos(x)
plt.imshow(z, origin="lower", extent=[2, 10, 2, 10], cmap='viridis')
plt.colorbar()
plt.show()

