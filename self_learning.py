# using the implementatiin of semi sup learning

from frameworks.SelfLearning import *
from methods.scikitTSVM import *
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# from sklearn import datasets
from sklearn.svm import SVC
from sklearn.semi_supervised import label_propagation
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
# from dataset_wrapper import label_mapping
import ipdb

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../data/', help='path to the folder where prepared data is downloaded')
parser.add_argument('--cat', type=int, default=1, help='category to train for')
parser.add_argument('--num_label_sam', type=int, default=2000, help='fraction of labeled data')
parser.add_argument('--num_unlabel_sam', type=int, default=10000, help='fraction of labeled data')
parser.add_argument('--nc', type=int, default=2500, help='number of components for PCA')
parser.add_argument('--kernel', type=str, default='rbf', help='which kernel to use: {rbf, knn}')
parser.add_argument('--num_iter', type=int, default=30, help='number of iterations for label spreading')
parser.add_argument('--num_neighbors', type=int, default=7, help='number of nearest neighbors for knn kernel')
parser.add_argument('--gamma', type=float, default=20, help='gamma value for label spreading')
parser.add_argument('--exp_name', type=str, default='Self_learning', help='experiment name')
parser.add_argument('--save_path', type=str, default='../src/outputs/', help='save path')
opt = parser.parse_args()

num_label_sam = opt.num_label_sam
num_unlabel_sam = opt.num_unlabel_sam

exp_name = '{}_rbf_frac{}_nc{}_iter{}_gamma{}'.format(opt.exp_name, opt.num_label_sam, opt.nc, opt.num_iter, opt.gamma)

print('Starting : ', exp_name)
print('='*50)
for arg in vars(opt):
    print('{} : {}'.format(arg, getattr(opt, arg)))
print('='*50)

# #############################################################################
# Data loading and preparation
# #############################################################################
X_train = np.load(os.path.join(opt.data_path, 'train_input.npy'))
Y_train = np.load(os.path.join(opt.data_path, 'train_output.npy'))
X_test = np.load(os.path.join(opt.data_path, 'test_input.npy'))
Y_test = np.load(os.path.join(opt.data_path, 'test_output.npy'))


train_input_labelled = X_train[:num_label_sam,:]
train_out_labelled = Y_train[:num_label_sam,(opt.cat - 1)]
train_input_unlabelled = X_train[num_label_sam:(num_unlabel_sam + num_label_sam),:]
train_out_unlabelled = -1 * np.ones((num_unlabel_sam), dtype = int)
Y_test = Y_test[:,opt.cat - 1]
# would be used for transductive metrics! 
orig_label_train = Y_train[:(num_unlabel_sam+num_label_sam),opt.cat-1]
# ipdb.set_trace()
train_in = np.concatenate((train_input_labelled, train_input_unlabelled), axis = 0)
train_out = np.concatenate((train_out_labelled, train_out_unlabelled), axis = 0)
train_out = train_out.astype(int)
# aaply PCA and reduce the dimensions!

print('Applying PCA to select {} components.'.format(opt.nc))
pca = PCA(n_components=opt.nc)
pca_transformer = pca.fit(train_in)
train_in_red = pca_transformer.transform(train_in)
test_in_red = pca_transformer.transform(X_test)


# any_scikitlearn_classifier = SVC(probability=True)
# ssmodel = SelfLearningModel(any_scikitlearn_classifier)
ssmodel = SKTSVM()
ssmodel.fit(train_in_red, train_out)

predicted_labels = ssmodel.predict(test_in_red)

print("Test data confison matrix:")
print(confusion_matrix(Y_test, predicted_labels))






