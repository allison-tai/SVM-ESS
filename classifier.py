'''
This script perfoms the basic process for applying a machine learning
algorithm to a dataset using Python libraries.

The four steps are:
   1. Download a dataset (using pandas)
   2. Process the numeric data (using numpy)
   3. Train and evaluate learners (using scikit-learn)
   4. Plot and compare results (using matplotlib)


The data is downloaded from URL, which is defined below. As is normal
for machine learning problems, the nature of the source data affects
the entire solution. When you change URL to refer to your own data, you
will need to review the data processing steps to ensure they remain
correct.

============
Example Data
============
The example is from http://mlr.cs.umass.edu/ml/datasets/Spambase
It contains pre-processed metrics, such as the frequency of certain
words and letters, from a collection of emails. A classification for
each one indicating 'spam' or 'not spam' is in the final column.
See the linked page for full details of the data set.

This script uses three classifiers to predict the class of an email
based on the metrics. These are not representative of modern spam
detection systems.
'''

# Remember to update the script for the new data when you change this URL
# URL = "http://mlr.cs.umass.edu/ml/machine-learning-databases/spambase/spambase.data"

# Uncomment this call when using matplotlib to generate images
# rather than displaying interactive UI.
#import matplotlib
#matplotlib.use('Agg')

# from pandas import read_table

import os 
import sys
os.chdir(os.path.dirname(sys.argv[0]))
print(os.getcwd())

import json

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

k = 3
dmax = 100

def fast_replace(line, abet):
    line = list(line)
    line[:] = [abet.get(c,c) for c in line]
    return ''.join(line)

def extract_fasta(file, abet, fname):
    seqs = []
    headers = []
    with open(file) as f:
        seq = ""
        for line in f:
               if line[0] == '>':
                   headers.append(line)
                   if seq != "":
                       seqs.append(seq)              
                   seq = ""
               else:
                   line = fast_replace(line.lower(), abet)
                   seq += line
        seqs.append(seq)
    json.dump(seqs, open(fname + '_seqs.json', 'w'))
    json.dump(headers, open(fname + '_head.json', 'w'))
           
def feature_dr(seqs, dmax, data_case):
    ptab = {}
    indices = []
    indptr = [0]
    data = []
    for seq in seqs:
        idict = {}
        cur_dat = []
        cur_ind = []
        for d in range(dmax):
            if d >= len(seq):
                break
            for i in range(len(seq) - d):
                f = seq[i]
                s = seq[i + d]
                if f == 'x' or s == 'x':
                    continue
                idx = ptab.setdefault(f + s + str(d), len(ptab))
                if idx in idict:
                    cur_dat[idict[idx]] += 1
                else:
                    idict[idx] = len(cur_ind)
                    cur_dat.append(1)
                    cur_ind.append(idx)
        data += cur_dat
        indices += cur_ind
        indptr.append(len(indices))
    json.dump(ptab, open('ptab' + '_' + str(data_case) + '.json', 'w'))
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(len(seqs), len(ptab)), dtype=np.int16)
    sparse.save_npz('X_dr' + '_' + str(data_case) + '.npz', matrix)
    return matrix

def feature_sp(seqs, k, data_case, have_ktab):
    if have_ktab:
        ktab = json.load(open('ktab'+ str(k) + '_' + str(data_case) + '.json', 'r'))
    else:
        ktab = {}
    indices = []
    indptr = [0]
    data = []
    for seq in seqs:
        idict = {}
        cur_dat = []
        cur_ind = []
        for i in range(len(seq) - k):
            kmer = seq[i:i+k]
            if 'x' in kmer:
                continue
            if have_ktab:
                idx = ktab[kmer]
            else:
                idx = ktab.setdefault(kmer, len(ktab))
            if idx in idict:
                cur_dat[idict[idx]] += 1
            else:
                idict[idx] = len(cur_ind)
                cur_dat.append(1)
                cur_ind.append(idx)
        data += cur_dat
        indices += cur_ind
        indptr.append(len(indices))
    if not have_ktab:
        json.dump(ktab, open('ktab'+ str(k) + '_' + str(data_case) + '.json', 'w'))
    return sparse.csr_matrix((data, indices, indptr), shape=(len(seqs), len(ktab)), dtype=np.float64)

def split_seq(seq, L):
    if (L == 0):
        return [seq]
    return split_seq(seq[:int(np.floor(len(seq)/2))], L-1) + split_seq(seq[int(np.floor(len(seq)/2)):], L-1)

def feature_lsp(train_seqs, test_seqs, k, data_case):
    # each index corresponds to which part of the sequence they're from
    subseqs = [[], [], [], [], [], [], [], []]
    for seq in train_seqs:
        temp = split_seq(seq, 3)
        for i in range(8):
            subseqs[i].append(temp[i])

    mat3 = [feature_sp(subseqs[i], k, data_case, True) for i in range(8)] # level 3 training matrices
    mat2 = [mat3[0] + mat3[1], mat3[2] + mat3[3], mat3[4] + mat3[5], mat3[6] + mat3[7]]
    mat1 = [mat2[0] + mat2[1], mat2[2] + mat2[3]]
    mat0 = mat1[0] + mat1[1]

    lok_train = []
    lok_train.append([X.dot(X.T) for X in mat3]) # my level 3 kernel
    lok_train.append([X.dot(X.T) for X in mat2]) # my level 2 kernel
    lok_train.append([X.dot(X.T) for X in mat1]) # my level 1 kernel

    train_kernel = np.zeros(shape=(len(train_seqs),len(train_seqs)))

    for X in lok_train[0]:
        train_kernel += 0.5*X
    for X in lok_train[1]:
        train_kernel += 0.25*X
    for X in lok_train[0]:
        train_kernel += 0.125*X
    train_kernel += 0.125*mat0.dot(mat0.T)
    sparse.save_npz('lsp_train.npz', mat3)

    test_kernel = None
    test3 = None

    if test_seqs:
        subseqs = [[], [], [], [], [], [], [], []]
        for seq in test_seqs:
            temp = split_seq(seq, 3)
            for i in range(8):
                subseqs[i].append(temp[i])

        test3 = [feature_sp(subseqs[i], k, data_case, True) for i in range(8)] # level 3 test matrices
        test2 = [test3[0] + test3[1], test3[2] + test3[3], test3[4] + test3[5], test3[6] + test3[7]]
        test1 = [test2[0] + test2[1], test2[2] + test2[3]]
        test0 = test1[0] + test1[1]

        import itertools
        lok_test = []
        lok_test.append([Y.dot(X.T) for X, Y in zip(mat3, test3)]) # my level 3 kernel
        lok_test.append([Y.dot(X.T) for X, Y in zip (mat2, test2)]) # my level 2 kernel
        lok_test.append([Y.dot(X.T) for X, Y in zip(mat1, test1)]) # my level 1 kernel

        test_kernel = np.zeros(shape=(len(test_seqs),len(train_seqs)))
        for X in lok_test[0]:
            test_kernel += 0.5*X
        for X in lok_test[1]:
            test_kernel += 0.25*X
        for X in lok_test[0]:
            test_kernel += 0.125*X
        test_kernel += 0.125*test0.dot(mat0.T)
        sparse.save_npz('lsp_test.npz', test3)
    #kernel = np.zeros(shape=(15*len(seqs),15*len(seqs)))

    #for i in range(len(seqs)):
        #for j in range(len(seqs)):
            #for p in range(8):
                #kernel[i+p,j+p] = kernel3[0][i+p][j+p]
            #for p in range(4):
                #kernel[i+8+p,j+8+p] = kernel2[0][i+p][j+p]
            #for p in range(2):
                #kernel[i+10+p,j+10+p] = kernel1[1][i+p][j+p]
            #kernel[i+14,j+14] = kernel0[i,j]
    #return kernel, mat
    return train_kernel, test_kernel, mat3, test3


def feature_mm(seqs):
    indices = []
    indptr = [0]
    data = []
    for seq in seqs:
        idict = {}
        cur_dat = []
        cur_ind = []
        for i in range(len(seq) - 5):
            kmer = seq[i:i+5]
            try:
                ind = (abet[kmer[0]], abet[kmer[1]], abet[kmer[2]], abet[kmer[3]], abet[kmer[4]])
            except:
                continue
            flag = False
            my_idx = ind[4]*(20**4) + ind[3]*20**3 + ind[2]*20**2 + ind[1]*20 + ind[0]
            idxs = [[j*(20**4) + ind[3]*20**3 + ind[2]*20**2 + ind[1]*20 + ind[0],
                    ind[4]*20**4 + j*(20**3) + ind[2]*20**2 + ind[1]*20 + ind[0],
                    ind[4]*20**4 + ind[3]*20**3 + j*(20**2) + ind[1]*20 + ind[0],
                    ind[4]*20**4 + ind[3]*20**3 + ind[2]*20**2 + j*20 + ind[0],
                    ind[4]*20**4 + ind[3]*20**3 + ind[2]*20**2 + ind[1]*20 + j] for j in range(20)]
            idxs = (idx for group in idxs for idx in group)
            for idx in idxs:
                if idx == my_idx:
                    if flag:
                        continue
                    else:
                        flag = True
                if idx in idict:
                    cur_dat[idict[idx]] += 1
                else:
                    idict[idx] = len(cur_ind)
                    cur_dat.append(1)
                    cur_ind.append(idx)
        data += cur_dat
        indices += cur_ind
        indptr.append(len(indices))
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(len(seqs), 20**5), dtype=np.int16)
    sparse.save_npz('X_mm5_1.npz', matrix)
    return matrix
    

def normalized_kernel(X, Y):
    normX = np.array([sparse.linalg.norm(X, axis=1)])
    normY = np.array([sparse.linalg.norm(Y, axis=1)])
    return sparse.csr_matrix((X.dot(Y.T))/np.sqrt(normX.T.dot(normY)))

abet_repl = {
            'k':'k','e':'k','r':'k',
            's':'t','q':'t','t':'t',
            'f':'y', 'y':'y',
            'l':'l','i':'l','v':'l','m':'l',
            'a':'a', 'c':'c', 'd':'d', 'g':'g', 'h':'h', 'n':'n', 'p':'p', 'w':'w',
            'x': 'x',
            '\n': ''
        }

abet_null = {'\n': ''}

abet = {'a':0, 'r':1, 'n':2, 'd':3, 'c':4, 'q':5, 'e':6, 'g':7, 'h':8,
       'i':9, 'l':10, 'k':11, 'm':12, 'f':13, 'p':14, 's':15, 't':16, 'w':17, 'y':18, 'v':19}

def train_classifier(feature_type, data_case): # have option for which to train
    #if (data_case == '2'):
    #    extract_fasta('postrain.fa', abet_repl, 'pos2')
    #    extract_fasta('negtrain2.fa', abet_repl, 'neg2')
    #    pos_seqs, neg_seqs = json.load(open('pos2_seqs.json', 'r')), json.load(open('neg2_seqs.json', 'r'))
    #elif (data_case == '1'):
    #    extract_fasta('postrain.fa', abet_null, 'pos')
    #    extract_fasta('negtrain.fa', abet_null, 'neg')
    #    extract_fasta('postrain.fa', abet_repl, 'pos_red')
    #    extract_fasta('negtrain.fa', abet_repl, 'neg_red')
    pos_seqs, neg_seqs = json.load(open('pos_seqs.json', 'r')), json.load(open('neg_seqs.json', 'r'))
    pos_red, neg_red = json.load(open('pos_red_seqs.json', 'r')), json.load(open('neg_red_seqs.json', 'r'))

    n_pos, n_neg = len(pos_seqs), len(neg_seqs)
    y = np.append(np.ones(n_pos), np.zeros(n_neg))

    from sklearn.model_selection import train_test_split
    seq_train, seq_test, y_train, y_test = train_test_split(pos_seqs + neg_seqs, y, test_size=0.2, stratify = y, random_state=56)
    red_train, red_test, y_train, y_test = train_test_split(pos_red + neg_red, y, test_size=0.2, stratify = y, random_state = 56)
    y = np.append(y_train, y_test)
    n_train, n_test = len(seq_train), len(seq_test)

    from sklearn.svm import SVC
    from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, auc
   
    if feature_type == 'sp' or feature_type == 'all':
        X_sp = feature_sp(seq_train + seq_test, k, data_case, False)
        X_sp_red = feature_sp(red_train + red_test, k, data_case, False)
        X_train, X_test = X_sp[:n_train,:], X_sp[n_train:,:]
        X_red_train, X_red_test = X_sp_red[:n_train,:], X_sp_red[n_train:,:]
        classifier = SVC(C=1.0, kernel='linear', class_weight = 'balanced')
        if check == True:
            classifier.fit(X_train, y_train)
            score_sp = f1_score(y_test, classifier.predict(X_test))
            y_prob = classifier.decision_function(X_test)
            # Generate the P-R and ROC curves
            p_sp, r_sp, thresh_pr_sp = precision_recall_curve(y_test, y_prob)
            fpr_sp, tpr_sp, thresh_roc_sp = roc_curve(y_test, y_prob, pos_label=1)
            auc_sp = auc(fpr_sp, tpr_sp)

            classifier.fit(X_red_train, y_train)
            y_red_prob = classifier.decision_function(X_red_test)
            fpr_sp_red, tpr_sp_red, thresh_roc_sp_red = roc_curve(y_test, y_red_prob, pos_label=1)
            auc_sp_red = auc(fpr_sp_red, tpr_sp_red)
        else:
            classifier.fit(X_sp, y)

        support = classifier.support_
        alpha = classifier.dual_coef_.toarray()
        beta = float(classifier.intercept_)

    if feature_type == 'dr' or feature_type == 'all':
        #X_dr = feature_dr(seq_train + seq_test, dmax, data_case)
        #X_dr_red = feature_dr(red_train + red_test, dmax, data_case)
        X_dr = sparse.load_npz('X_dr' + '_' + str(data_case) + '.npz')
        X_train, X_test = X_dr[:n_train,:], X_dr[n_train:,:]
        #X_red_train, X_red_test = X_dr_red[:n_train,:], X_dr_red[n_train:,:]
        if feature_type == 'dr':
            classifier = SVC(C=1.0, kernel='linear', class_weight = 'balanced')
        if check == True:
            classifier.fit(X_train, y_train)
            score_dr = f1_score(y_test, classifier.predict(X_test))
            y_prob = classifier.decision_function(X_test)
            
            # Generate the P-R and ROC curves
            p_dr, r_dr, thresh_pr_dr = precision_recall_curve(y_test, y_prob)
            fpr_dr, tpr_dr, thresh_roc_dr = roc_curve(y_test, y_prob, pos_label=1)
            auc_dr = auc(fpr_dr, tpr_dr)

            #classifier.fit(X_red_train, y_train)
            #y_red_prob = classifier.decision_function(X_red_test)
            #fpr_dr_red, tpr_dr_red, thresh_roc_dr_red = roc_curve(y_test, y_red_prob, pos_label=1)
            #auc_dr_red = auc(fpr_dr_red, tpr_dr_red)
        else:
            classifier.fit(X_dr, y)

        support = classifier.support_
        alpha = classifier.dual_coef_.toarray()
        beta = float(classifier.intercept_)


    if feature_type == 'mm' or feature_type == 'all':
        #X_mm = feature_mm(seq_train + seq_test, False)
        X_mm = sparse.load_npz('X_mm5_1.npz')
        #X_mm_red = feature_mm(red_train + red_test, True)
        X_train, X_test = X_mm[:n_train,:], X_mm[n_train:,:]
        #X_red_train, X_red_test = X_mm_red[:n_train,:], X_mm_red[n_train:,:]
        if feature_type == 'mm':
            classifier = SVC(C=1.0, kernel='linear', class_weight = 'balanced')
        if check == True:
            classifier.fit(X_train, y_train)
            score_mm = f1_score(y_test, classifier.predict(X_test))
            y_prob = classifier.decision_function(X_test)
            # Generate the P-R and ROC curves
            p_mm, r_mm, thresh_pr_mm = precision_recall_curve(y_test, y_prob)
            fpr_mm, tpr_mm, thresh_roc_mm = roc_curve(y_test, y_prob, pos_label=1)
            auc_mm = auc(fpr_mm, tpr_mm)

         #   classifier.fit(X_red_train, y_train)
          #  y_red_prob = classifier.decision_function(X_red_test)
           # fpr_mm_red, tpr_mm_red, thresh_roc_mm_red = roc_curve(y_test, y_red_prob, pos_label=1)
        else:
            classifier.fit(X_mm, y)

        support = classifier.support_
        alpha = classifier.dual_coef_.toarray()
        beta = float(classifier.intercept_)

    #if feature_type == 'lsp' or feature_type == 'all':
        #classifier_lsp = SVC(C=1.0, kernel='precomputed', class_weight = 'balanced')
        #if check == True:
            #train_kernel, test_kernel, Xtrain, Xtest = feature_lsp(seq_train, seq_test, k, data_case)
            #classifier_lsp.fit(train_kernel, y_train)
            #score_lsp = f1_score(y_test, classifier_lsp.predict(test_kernel))
            #y_prob = classifier_lsp.decision_function(test_kernel)
            #auc_mm = roc_auc_score(y_test, y_prob, average='weighted')
            # Generate the P-R and ROC curves
            #p_lsp, r_lsp, thresh_pr_lsp = precision_recall_curve(y_test, y_prob)
            #fpr_lsp, tpr_lsp, thresh_roc_lsp = roc_curve(y_test, y_prob, pos_label=1)
        #else:
            #kernel, empty1, X, empty2 = feature_lsp(seq_train + seq_test, [], k, data_case)
            #classifier_lsp.fit(kernel, y)
    
    # Include the score in the title
    #result = 'SVC (F1 score={:.3f})'.format(score), precision, recall
    #plt.title('Receiver Operating Characteristic')
    #plt.plot(fpr_sp, tpr_sp, 'b', label = 'sp, AUC = %0.2f'% auc_sp)
    #plt.plot(fpr_sp_red, tpr_sp_red, 'r', label = 'sp, reduced, AUC = %0.2f'% auc_sp_red)
    #plt.plot(fpr_dr, tpr_dr, 'green', label = 'dr, AUC = %0.2f'% auc_dr)
    #plt.plot(fpr_dr_red, tpr_dr_red, 'purple', label = 'dr, reduced, AUC = %0.2f'% auc_dr_red)
    #plt.plot(fpr_mm, tpr_mm, 'orange', label = 'mm, AUC = %0.2f'% auc_mm)
    #plt.xlim([-0.1,1.1])
    #plt.ylim([-0.1, 1.1])
    #plt.savefig('auc_graph.png')
    #plt.show
    return support, alpha, beta, X_dr


def get_histograms_dr(seq, weights, alpha, X, data_case):
    ptab = json.load(open('ptab' + '_' + str(data_case) + '.json', 'r'))
    pos_wts, neg_wts = [], []
    pos_ind, neg_ind = [0], [0]
    for i in range(len(seq)):
        for d in range(dmax):
            if i + d < len(seq):
                f = seq[i]
                s = seq[i + d]
                pkey = f + s + str(d)
                if pkey in ptab:
                    idx = ptab[pkey] # get index of feature
                    if idx not in weights:
                        weights[idx] = np.sum(np.multiply(X[:,idx].toarray(), alpha.T))
                    weight = weights[idx]
                    if (weight > 0):
                        pos_wts.append(weight) # add individual wcm
                    else:
                        neg_wts.append(weight) # same, but negative
        pos_ind.append(len(pos_wts)) # tell us which weights belong to which index
        neg_ind.append(len(neg_wts)) # same, but negative
    json.dump(weights, open('weights_dr' + '_' + str(data_case) + '.json', 'w'))
    return (pos_wts, neg_wts, pos_ind, neg_ind)

def get_histograms_sp(seq, weights, alpha, X, data_case):
    ktab = json.load(open('ktab' + str(k) + '_' + str(data_case) + '.json', 'r'))
    pos_wts, neg_wts = [], []
    pos_ind, neg_ind = [0], [0]
    for i in range(len(seq) - k):
        kmer = seq[i:i+k]
        if kmer in ktab:
            idx = ktab[kmer]
            if idx not in weights:
                weights[idx] = np.sum(np.multiply(X[:,idx].toarray(), alpha.T))
            weight = weights[idx]
            if (weight > 0):
                pos_wts.append(weight)
            else:
                neg_wts.append(weight)
        pos_ind.append(len(pos_wts))
        neg_ind.append(len(neg_wts))
    json.dump(weights, open('weights_sp'+ str(k)  + '_' + str(data_case) + '.json', 'w'))
    return (pos_wts, neg_wts, pos_ind, neg_ind)

def get_histograms_lsp(seq, weights, alpha, X, data_case):
    subseqs = [[], [], [], [], [], [], [], []]
    for seq in seqs:
        temp = split_seq(seq, 3)
        for i in range(8):
            subseqs[i].append(temp[i])

def get_histograms_mm(seq, weights, alpha, X):
    pos_wts, neg_wts = [], []
    pos_ind, neg_ind = [0], [0]
    for i in range(len(seq) - 5):
        kmer = seq[i:i+5]
        try:
            ind = (abet[kmer[0]], abet[kmer[1]], abet[kmer[2]], abet[kmer[3]], abet[kmer[4]])
        except:
            continue
        flag = False
        my_idx = ind[4]*(20**4) + ind[3]*20**3 + ind[2]*20**2 + ind[1]*20 + ind[0]
        idxs = [[j*(20**4) + ind[3]*20**3 + ind[2]*20**2 + ind[1]*20 + ind[0],
                    ind[4]*20**4 + j*(20**3) + ind[2]*20**2 + ind[1]*20 + ind[0],
                    ind[4]*20**4 + ind[3]*20**3 + j*(20**2) + ind[1]*20 + ind[0],
                    ind[4]*20**4 + ind[3]*20**3 + ind[2]*20**2 + j*20 + ind[0],
                    ind[4]*20**4 + ind[3]*20**3 + ind[2]*20**2 + ind[1]*20 + j] for j in range(20)]
        idxs = (idx for group in idxs for idx in group)
        for idx in idxs:
            if idx == my_idx:
                if flag:
                    continue
                else:
                    flag = True
            if idx not in weights:
                weights[idx] = np.sum(np.multiply(X[:,idx].toarray(),alpha.T))
            weight = weights[idx]
            if (weight > 0):
                pos_wts.append(weight)
            else:
                neg_wts.append(weight)
        pos_ind.append(len(pos_wts))
        neg_ind.append(len(neg_wts))
    return (pos_wts, neg_wts, pos_ind, neg_ind)

def compute_score(roi, seq_info):
    lmin, lmax, rmin, rmax = roi
    pos_wts, neg_wts, pos_ind, neg_ind = seq_info
	# where seq_info is a tuple containing pos_wts, neg_wts, pos_ind, neg_ind
    return -(np.sum(pos_wts[pos_ind[lmin]:pos_ind[rmax]]) + np.sum(neg_wts[neg_ind[lmax]:neg_ind[rmin]]))

def split(roi):
	if (roi[1] - roi[0] > roi[3] - roi[2]):
		roi1 = (roi[0], int(np.floor((roi[0] + roi[1])/2)), roi[2], roi[3])
		roi2 = (int(np.floor((roi[0] + roi[1])/2) + 1), roi[1], roi[2], roi[3])
	else:
		roi1 = (roi[0], roi[1], int(np.floor((roi[2] + roi[3])/2) + 1), roi[3])
		roi2 = (roi[0], roi[1], roi[2], int(np.floor((roi[2] + roi[3])/2)))
	return roi1, roi2

def legal(roi):
	return (roi[0] <= roi[3]) and (roi[3] - roi[0] >= 150) and (roi[2] - roi[1] <= 300)

def finished(roi):
	return roi[1] == roi[0] and roi[3] == roi[2]

def ess(seq, weights, alpha, X, feature_type, data_case):
	import queue
	q = queue.PriorityQueue(0)
	if (feature_type == 'sp'):
		seq_info = get_histograms_sp(seq, weights, alpha, X, data_case)
		roi = (0, len(seq)-k, 0, len(seq)-k)
	elif (feature_type == 'dr'):
		seq_info = get_histograms_dr(seq, weights, alpha, X, data_case)
		roi = (0, len(seq), 0, len(seq))
	elif (feature_type == 'mm'):
		seq_info = get_histograms_mm(seq, weights, alpha, X)
		roi = (0, len(seq)-5, 0, len(seq)-5)
	elif (feature_type == 'lsp'):
		seq_info = get_histograms_lsp(seq, weights, alpha, X)
		roi = (0, len(seq)-k, 0, len(seq)-k)
	while True:
		roi1, roi2 = split(roi)
		if legal(roi1):
			q.put((compute_score(roi1, seq_info), roi1))
		if legal(roi2):
			q.put((compute_score(roi2, seq_info), roi2))
		score, roi = q.get()
		if finished(roi):
			return (score, roi[0], roi[2])

def find_all(seqs, support, alpha, beta, X, feature_type, data_case):
    #weights = json.load(open('weights_sp' + str(k) + '_red.json'))
    #weights = json.load(open('weights_dr.json'))
    weights = {}
    guesses = []
    if (feature_type != 'lsp'):
        X = X[support,:].tocsc()
    for seq in seqs:
        (score, start, end) = ess(seq, weights, alpha, X, feature_type, data_case)
        best_guess = seq[start:end]
        guesses.append((best_guess, start, end, -score + beta))
    return guesses

feature_type = 'dr'
data_case = '1'
check = True

extract_fasta('fullseq.fa', abet_repl, 'full')
full_seqs = json.load(open('full_seqs.json', 'r'))
support, alpha, beta, X = train_classifier(feature_type, data_case)
guesses = find_all(full_seqs, support, alpha, beta, X, feature_type, data_case)