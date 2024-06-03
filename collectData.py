import numpy as np
from cothread.catools import *
from datetime import datetime
import sys
import os

now = datetime.now()

def getZero():
    x = np.round(float(caget('BL23I-MO-GONIO-01:X')), 4)
    y = np.round(float(caget('BL23I-MO-GONIO-01:Y')), 4)
    z = np.round(float(caget('BL23I-MO-GONIO-01:Z')), 4)

    return x, y, z

def getReadbacks(xZ, yZ, zZ):
    x = np.round(float(caget('BL23I-MO-GONIO-01:X')), 4) - xZ
    y = np.round(float(caget('BL23I-MO-GONIO-01:Y')), 4) - yZ
    z = np.round(float(caget('BL23I-MO-GONIO-01:Z')), 4) - zZ
    #omega = caget('BL23I-MO-GONIO-01:OMEGA')
    kappa = np.round(caget('BL23I-MO-GONIO-01:KAPPA'), 1)
    phi = np.round(caget('BL23I-MO-GONIO-01:PHI'), 1)
    if phi == -0.0:
        phi = 0.0

    vals = [x, y, z, kappa, phi]

    return vals

if __name__ == "__main__":
    if not os.path.exists("./readings"):
        os.mkdir("./readings")
    start = input("Is the pin centered at kappa/phi 0/0? (y/n)").upper()
    if start == "Y":
        pass
    else:
        sys.exit("To start, centre the pin at kappa/phi 0/0 then restart the script")
    xZ, yZ, zZ = getZero()
    while True:
        RBVS = getReadbacks(xZ, yZ, zZ)
        print(RBVS)
        with open(f'readings/rbvs_{now.strftime("%d%m%Y")}.txt', 'a') as f:
            f.write(str(RBVS))
            f.write('\n')
        next = input("Press Enter when centered to record new position, or press Q to finish: ").upper()
        if next == "Q":
            sys.exit(0)
        else:
            pass