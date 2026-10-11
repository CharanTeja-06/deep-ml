
import numpy as np

def jaccard_index(y_true, y_pred):
	y_true=np.asarray(y_true)
	y_pred=np.asarray(y_pred)
	tp=np.sum((y_true==1) &(y_pred==1))
	fp=np.sum((y_true==0) &(y_pred==1))
	fn=np.sum((y_true==1) &(y_pred==0))
	result = tp/(tp+fp+fn)
	return round(result, 3)
