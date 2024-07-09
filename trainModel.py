from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import pickle
import os
import time


def train(trainingSet):
    data = np.loadtxt(f"readings/{trainingSet}", delimiter=",")
    newdata = data.copy()
    newdata[1:, :3] -= data[0, :3]
    data = np.delete(newdata, 0, axis=0)
    X = data[:, :2]
    Y = data[:, 3:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=1000, random_state=42)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    mse = mean_absolute_error(Y_test, Y_pred)
    print(f"MSE: {mse}")
    filename, _ = os.path.splitext(trainingSet)

    with open(f"./models/model_{filename}.pkl", "wb") as toDump:
        pickle.dump(model, toDump)

if __name__ == "__main__":
    if not os.path.exists("./models"):
        os.mkdir("./models")
    trainingSet = "rbvs_20062024.txt"
    #trainingSet = input("Enter trainind dataset filename: ")
#    while True:
    train(trainingSet)
 #       continue
  #      time.sleep(30)
