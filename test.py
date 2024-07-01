import numpy as np

data = np.loadtxt(f"readings/old.txt", delimiter=",")

print(data)

newdata = data.copy()
newdata[1:, :3] -= data[0, :3]
data = np.delete(newdata, 0, axis=0)

print(data)