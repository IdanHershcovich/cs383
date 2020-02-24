import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#constants
learn = 0.1
convergence = pow(2,(-32))
x1 = 0
x2 = 0
dj_value = 1000000
iterations = 0


#empty list to save the changes
j_iterations = []
x1_iterations = []
x2_iterations = []


x1_iterations.append(x1)
x2_iterations.append(x2)

#partial derivative of x1`
def djdx1(x1, x2):
	partial_der = ((2 * x1) + (2 * x2) - 4)
	return partial_der

#partial derivative of x2
def djdx2(x1,x2):
	partial_der = ((2 * x1) + (2 * x2) - 4)
	return partial_der

#J function given in HW
def jFunc(x1,x2):
	J = pow((x1+x2-2),2)
	return J

#Repeat until convergence!
while dj_value > convergence:

	j_old = jFunc(x1,x2)

	# calculating our new x1 and x2 values with the learning rate

	new_x1 = x1- (learn*(djdx1(x1,x2)))
	new_x2 = x2- (learn*(djdx2(x1,x2)))

	# saving to the array

	x1_iterations.append(new_x1)
	x2_iterations.append(new_x2)

	# calculating J with the new x1,x2 values
	j_new = jFunc(new_x1,new_x2)
	
	#check the diff between the new J and old J to check for convergence
	dj_value = abs(j_new - j_old)
	j_iterations.append(j_new)

	#setting the new x1,x2s as our normal values
	x1 = new_x1
	x2 = new_x2
	iterations +=1



x1_iterations = np.array(x1_iterations)
x2_iterations = np.array(x2_iterations)
j_iterations = np.array(j_iterations)


## Plotting Commented out for sake of running on tux. Images were only required in the Report.
# For saving the images when testing, just uncomment the plotting statements

# plt.plot(j_iterations)
# plt.title("Plot iteration vs J ")
# plt.ylabel("J")
# plt.xlabel("Iterations")
# plt.savefig("J_Iterations.png")


# plt.plot(x1_iterations)
# plt.title("Plot iteration vs X1")
# plt.ylabel("X1")
# plt.xlabel("Iterations")
# plt.savefig("X1_Iterations.png")


# plt.plot(x2_iterations)
# plt.title("Plot iteration vs X2")
# plt.ylabel("X2")
# plt.xlabel("Iterations")
# plt.savefig("X2_Iterations.png")



