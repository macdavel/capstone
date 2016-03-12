import neurolab as nl
import numpy as np
import pylab as pl
import copy
import itertools
from itertools import takewhile
import scipy.stats


import xlrd

master_array = []
target_master_array = []





book = xlrd.open_workbook("capstoneMaster.xlsx",ragged_rows=True)
print "The number of worksheets is", book.nsheets
print "Worksheet name(s):", book.sheet_names()
sh = book.sheet_by_index(0)
# print sh.name, sh.nrows, sh.ncols

print sh.ncols
print sh.nrows

def calcManualSSE(actual, forecast):
	assert len(actual) == len (forecast)
	error = 0
	for value in xrange(len(actual)):
		e = actual[value]-forecast[value]
		error += e**2
	return error




def createMasterArray(row_number):
	print "The desired length: "+str(sh.row_len(row_number))
	master_array
	for rx in range(1, sh.row_len(row_number)): 
		thecell = sh.cell(rowx = row_number, colx = rx).value
		# print type(thecell)
		master_array.append(thecell)
		target_master_array.append([thecell])
	return master_array

# print master_array

# createMasterArray(18)
#using -2 because empty datapoint at the end of the excel file








# input_data = master_array[:-2]
# target = target_master_array[1:-1]


# for indx in reversed(range(len(input_data))):
# 	input_data[indx].append(input_data[indx-1][0])

# print input_data
# for index in len(input_data)-1:
# 	new_input.append(input_data[index],input_data[index+1])

# two_input_target


# target = target_master_array[1:-1]

# print input_data
# print target


# print "This is the output"
# print train_set_input


# print len(input_data)
# print len(target)


def runTrainingAlgorithms(train_set_input, train_set_output, predict_input, predict_output, number_of_lags):
# Create network with 3 layers
	data_parameters = []
	for input_lag in range(number_of_lags):
		data_parameters.append([-1,1])
	print data_parameters	
	# net = nl.net.newelm(data_parameters, [60,1], [nl.trans.TanSig(), nl.trans.PureLin()])
	net = nl.net.newff(data_parameters, [1,1])
	# Set initialized functions and init


	# net.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
	# net.layers[1].initf= nl.init.InitRand([-0.1, 0.1], 'wb')
	# net.init()

	# net = nl.net.newff([[-7, 7]],[5, 1])
	# Train network
	# print "About to print train output"
	# print len(train_set_output)
	# print len(train_set_input)
	# print train_set_output
	# print "done printing"
	# print train_set_input
	error = net.train( train_set_input, train_set_output, epochs=5000, show=1000, goal=0.000000000000000000000000000000000000000000000000000000015)
	# error = net.train(inp, tar, epochs=500, show=100, goal=0.02)
	# Simulate network
	output = net.sim(predict_input)
	msef = nl.error.SSE()
	msef2 = nl.error.SAE()

	# print "Printing the MSEF object"
	# print msef
	print "#######################"
	print "#######################"
	print "#######################"
	newoutput = list(itertools.chain(*output))
	newpredictoutp = list(itertools.chain(*predict_output))

	# print type(newoutput)
	print '##################'
	# print type(newpredictoutp)
	x = np.array([newoutput])
	y = np.array([newpredictoutp])
	# print x
	sse = msef(x, y)
	print "The Sum of Squared Errors is: "+str(sse)
	print "The Sum of Squared Errors is: "+str(calcManualSSE(newoutput,newpredictoutp))
	print "The Sum of Absolute Errors is: "+str(msef2(x, y))
	print "Spearman Correlation: "+str(scipy.stats.spearmanr(newoutput,newpredictoutp))
	print "Sign Rank: "+str(scipy.stats.wilcoxon(newoutput,newpredictoutp))
	# print output

	# output2 = net.sim(train_set_input)

	# Plot result

	# pl.subplot(211)
	# pl.plot(output2)
	# pl.plot(train_set_output)
	# pl.xlabel('Observation')
	# pl.ylabel('In sample Prediction')

	pl.subplot(212)
	pl.plot(predict_output)
	pl.plot(output)
	pl.legend(['Actual', 'Forecasted'])
	pl.show()


def generate_train_inputDataset(number_of_lags, row_number):
	input_data = []
	target = []
	master_array = createMasterArray(row_number)

	input_data = [master_array[x:x+number_of_lags] for x in range(0, len(master_array)-number_of_lags)]
	target = [master_array[x:x+1] for x in range(number_of_lags, len(master_array))]

	print '######## hey'

	# print len(input_data)
	# print len(target)


	train_set_input = input_data[:(len(input_data)/2)]
	train_set_output = target[:(len(input_data)/2)]

	predict_input = input_data[(len(input_data)/2):]
	predict_output = target[(len(input_data)/2):]



	print "Training lenghths"

	print len(train_set_input)
	print len(train_set_output)

	if len(train_set_input) != len(train_set_output):
		raise Exception("Data points do not add up!")

	runTrainingAlgorithms(train_set_input,train_set_output,predict_input,predict_output,number_of_lags)




############################################



# for i in [18, 20, 21, 22, 23]:
print "Generating Data for Row Number: "
generate_train_inputDataset(2,44)