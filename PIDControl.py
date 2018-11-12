import PID
import time
import numpy as np
import matplotlib.pyplot as plt

targetT = 35
P = 1
I = 0
D = 0

pid = PID.PID(P, I, D)
pid.SetPoint = targetT
pid.setSampleTime(1)

# real = np.ones(10) #np.arange(1,11)
givenCommand = np.ones(20) * 4 #np.arange(1, 11)
recSteer = []

pid.SetPoint = 4 # = target value
tmp = 1

for i in range (0,20):
    # read CAN steering
    # pid.update(real[i])
    pid.update(tmp)
    print(pid.output)
    tmp = pid.output
    recSteer.append(pid.output)
    time.sleep(0.5)

print(recSteer)


x = np.arange(1,21)

plt.plot(x, recSteer, "b")
plt.plot(x, givenCommand, "r")
plt.show()

