import numpy as np
import matplotlib.pyplot as plt

# параметри
n = 15
x1 = np.linspace(-7, 3, n)
x2 = np.linspace(-4.4, 1.7, n)

X1, X2 = np.meshgrid(x1, x2)

# цільова функція
Y = (X1**2) * np.sin(X2 - 1)

# графік
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X1, X2, Y)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
ax.set_title("Target surface")

plt.show()
