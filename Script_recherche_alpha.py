import matplotlib.pyplot as plt
import numpy as np


X = []
Y = []


while(len(X) != 99):
	input_X = input()
	if input_X != '':
		X.append(float(input_X))

while(len(Y) != 99):
	input_Y = input()
	if input_Y != '':
		Y.append(int(input_Y))



file1 = open("data.txt","w+")
for i in range(99):
	str = '({}, {})'.format(X[i], round(Y[i])) + '\n'
	file1.write(str)

file1.close()
	
