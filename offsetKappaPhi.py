import pickle
from cothread.catools import *
import numpy as np

class recenter():
    def __init__(self):
        modelPath = ("models/model_rbvs_03062024.pkl")

        with open(modelPath, "rb") as f:
            self.model = pickle.load(f)

    def predictOffset(self):
        xZ = 1
        yZ = 2
        zZ = 3
        # ^^ READ IN SAVED X, Y, Z CENTRING VALUES ^^
        self.kappa = np.round(caget('BL23I-MO-GONIO-01:KAPPA'), 1)
        self.phi = np.round(caget('BL23I-MO-GONIO-01:PHI'), 1)

        x, y, z = self.model.predict([[self.kappa, self.phi]])[0]
        self.x = xZ + x
        self.y = yZ + y
        self.z = zZ + z

    def applyOffset(self):
        
        pass

if __name__ == "__main__":
    run = recenter()
    run.predictOffset()