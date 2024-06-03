import numpy as np
from cothread.catools import *
from datetime import datetime

now = datetime.now()

def getReadbacks():
    x = caget('BL23I-MO-GONIO-01:X')
    y = caget('BL23I-MO-GONIO-01:Y')
    z = caget('BL23I-MO-GONIO-01:Z')
    omega = caget('BL23I-MO-GONIO-01:OMEGA')
    kappa = caget('BL23I-MO-GONIO-01:KAPPA')
    phi = caget('BL23I-MO-GONIO-01:PHI')
    
    vals = [np.round(float(x), 4), np.round(float(y), 4), np.round(float(z), 4), np.round(float(omega), 1), np.round(float(kappa), 1), np.round(float(phi), 1)]

    return vals

if __name__ == "__main__":
    RBVS = getReadbacks()
    print(RBVS)
    with open(f'rbvs_{now.strftime("%d%m%Y")}.txt', 'a') as f:
        f.write(str(RBVS))
        f.write('\n')