This submission includes 3 scripts. In general, it is necessary for the csv file to be in the same directory
as the scripts, and all scripts need to be in the same directory, as one script imports from another.

partialGradientPlot.py:
	This script is a very simple script for part 2, Gradient Descent. To run it, just call the program with no arguments.
	The scripts has code for plotting the graphs, but they're commented out to avoid issues with Tux. The images are included in the submission,
	and they're in the Report, as instructed by the HW.

cflr.py:
	This is the script for part 3 of the homework. It is run without any arguments, but it needs the csv in the same directory.
	To give it another CSV, just change the name of the csv being passed as the parameter in the function inside main.

	main function: closedFormLinReg(nameOfCSVInDIr)

sFolds.py:
	This script is for computing sFolds from part 4. This script also needs a CSV in the same directory, but since
	it imports a function from cflr.py, it also needs to be in the same directory as cflr. Run with no arguments. 
	To change the value of S, just pass in the desired S as a parameter in the sFolds function. 

	main function: sFolds(nameOfCSVInDIr, S)