import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################
    X = my_map(X_train) 
    classifier = LogisticRegression(C=100,random_state=0)
    classifier.fit(X, y_train)
    w = classifier.coef_[0]  
    b = classifier.intercept_[0]  
    return w,b
	# Use this method to train your model using training CRPs
	# X_train has 32 columns containing the challeenge bits
	# y_train contains the responses
	
	# THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
	# If you do not wish to use a bias term, set it to 0



################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################
    d = np.flip(2 * X - 1, axis=1)
    feat = np.cumprod(d, axis=1)
    mask = np.triu(np.ones((32, 32), dtype=bool), k=1)
    pairs = feat[:, :, np.newaxis] * feat[:, np.newaxis, :]
    pairs_masked = pairs[:, mask]
    X_new = np.concatenate([feat, pairs_masked], axis=1)
	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
    return X_new
