import math
import numpy as np
from bayes import standardizeWithTrain, gaussianDist

tr = np.array([[216,5.68], [69, 4.78], [302,2.31], [60,3.16], [393,4.2]])
t = np.array([[242,4.56]])
t = standardizeWithTrain(t, tr.mean(axis=0), tr.std(axis=0,ddof=1))
yes_ch = np.array([0.05509059, -0.95719904, -1.01917595])
yes_avg = np.array([1.24771393, 0.56878857, -0.65327706])
no_ch = np.array([0.64731446, 1.27396994])
no_avg = np.array([-1.29448434, 0.1312589])


p_yes = 3/5
p_no = 2/5

### Answers for 2c
P_F1_GetA = p_yes * gaussianDist(yes_ch, t[:,0]) * gaussianDist(yes_avg, t[:,1])
print(str.format("{}\n", P_F1_GetA))
P_F1_NotGetA = p_no * gaussianDist(no_ch, t[:,0]) * gaussianDist(no_avg, t[:,1])
print(str.format("{}\n", P_F1_NotGetA))
""" 
A = 0.0574317804086334

not A = 0.023070889187926262

A > not A """


