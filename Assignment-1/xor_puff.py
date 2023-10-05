import numpy as np
from sklearn.linear_model import LogisticRegression
# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( Z_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# The first 64 columns contain the config bits
	# The next 4 columns contain the select bits for the first mux
	# The next 4 columns contain the select bits for the second mux
	# The first 64 + 4 + 4 = 72 columns constitute the challenge
	# The last column contains the response

	models = {}
	for j in range(16):
		for i in range(j):
			models[(i,j)] = LogisticRegression(C=10)
	x1 = Z_train[:,64:68]
	x2 = Z_train[:,68:72]
	num = np.array([8,4,2,1])
	x1=np.matmul(x1,num)
	x2=np.matmul(x2,num)
	x1=x1.reshape((x1.shape[0],1))
	x2=x2.reshape((x1.shape[0],1))
	X=Z_train[:,:64]
	Y=Z_train[:,-1]
	X = np.hstack((X, x1))
	X = np.hstack((X, x2))

	for j in range(16):
		for i in range(j):
			filter_arr1 = X[:,64] == i
			train = X[filter_arr1]
			ans = Y[filter_arr1]
			filter_arr2 = train[:,65] == j
			train = train[filter_arr2]
			ans = ans[filter_arr2]
			
			filter_arr1 = X[:,64] == j
			train2 = X[filter_arr1]
			ans2 = Y[filter_arr1]
			filter_arr2 = train2[:,65] == i
			train2 = train2[filter_arr2]
			ans2 = ans2[filter_arr2]
			ans2 = 1 - ans2

			train = np.concatenate((train,train2))
			ans = np.concatenate((ans,ans2))
			models[(i,j)].fit(train[:,:64],ans)

	return models					# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict( X_tst, models ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to make predictions on test challenges
	x1 = X_tst[:,64:68]
	x2 = X_tst[:,68:72]
	num = np.array([8,4,2,1])
	x1=np.matmul(x1,num).astype(int)
	x2=np.matmul(x2,num).astype(int)
	X=X_tst[:,:64]
	Y=X_tst[:,-1]
	pred = np.zeros(X_tst.shape[0])

	for k in range(X.shape[0]):
		i = x1[k]
		j = x2[k]
		
		if i<j:
			pred[k] = models[(i,j)].predict(X[k,:].reshape(1,-1))
			
		elif j<i:
			pred[k] = 1 - models[(j,i)].predict(X[k,:].reshape(1,-1))

	return pred
