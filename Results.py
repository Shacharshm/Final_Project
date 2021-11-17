import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from collections import defaultdict



def _get_t_acc(y_hat, y, k_time):
    '''
    accuracy as f(time)
    '''
    #print('y_hat:', len(y_hat))
    #print('y:', len(y))
    a = np.zeros(k_time)
    for ii in range(k_time):
        y_i = y[ii::k_time]
        #print('y_i:', len(y_i))
        y_hat_i = y_hat[ii::k_time]
        correct = [1 for p, q in zip(y_i, y_hat_i) if p==q]
        a[ii] = sum(correct)/len(y_i)
        
    return a

def _get_confusion_matrix(y, predicted):
    '''
    confusion matrix per class
    '''
    y, p = y, predicted

    return confusion_matrix(y, p)

def _gru_test_acc(model, X, y, clip_time, k_sub):
    '''
    masked accuracy for gru
    '''
    # mask to ignore padding
    mask = model.layers[0].compute_mask(X)

    # predicted labels
    #y_hat = model.predict_classes(X)
    predict_y_hat=model.predict(X) 
    y_hat=np.argmax(predict_y_hat,axis=2)

    # remove padded values
    # converts matrix to vec
    y_hat = y_hat[mask==True]
    y = y[mask==True]
    y = y.numpy()

    a = np.zeros(k_sub)
    sub_size = len(y_hat)//k_sub
    for s in range(k_sub):
        # group based on k_sub
        y_hat_s = y_hat[s*sub_size:(s+1)*sub_size]
        y_s = y[s*sub_size:(s+1)*sub_size]
        # accuracy for each group
        correct = (y_hat_s==y_s).sum().item()
        a[s] = correct/len(y_s)

    # accuracy as a function of t
    k_class = len(clip_time)
    a_t = {}
    for ii in range(k_class):
        y_i = y[y==ii]
        y_hat_i = y_hat[y==ii]
        k_time = clip_time[ii]
        a_t[ii] = np.zeros((k_sub, k_time))
        sub_size = len(y_hat_i)//k_sub
        for s in range(k_sub):
            # group based on k_sub
            y_hat_s = y_hat_i[s*sub_size:(s+1)*sub_size]
            y_s = y_i[s*sub_size:(s+1)*sub_size]
            # accuracy for each group
            a_t[ii][s] = _get_t_acc(y_hat_s, y_s, k_time)

    c_mtx = _get_confusion_matrix(y, y_hat)
    
    return a, a_t, c_mtx

def _get_true_class_prob(y,y_probs,seq_len):
    
    y_prob_true = defaultdict(list)
    
    for i in range(y.shape[0]):
        if int(y[i,0]) not in y_prob_true:
            y_prob_true[int(y[i,0])] = []
            
        y_prob_true[int(y[i,0])].append(y_probs[i,:seq_len[i],int(y[i,0])])
    return y_prob_true

def Get_Results(args, model, X_train, y_train, train_list, train_len, X_test, y_test, test_list, test_len, clip_time):
    # results dict init
    results = {}

    # mean accuracy across time
    results['train'] = np.zeros(len(test_list))
    results['val'] = np.zeros(len(test_list))

    # per class temporal accuracy
    results['t_train'] = {}
    results['t_test'] = {}
    for ii in range(args.k_class):
        results['t_train'][ii] = np.zeros(
            (len(test_list), clip_time[ii]))
        results['t_test'][ii] = np.zeros(
            (len(test_list), clip_time[ii]))


    results_prob = {}    
    for method in 'train test'.split():
        results_prob[method] = {}
        for measure in 'acc t_prob'.split():
            results_prob[method][measure] = {}
    
    '''
    results on train data
    '''
    a, a_t, c_mtx = _gru_test_acc(model, X_train, y_train, clip_time, len(train_list))
    results['train'] = a
    print('tacc = %0.3f' %np.mean(a))
    for ii in range(args.k_class):
        results['t_train'][ii] = a_t[ii]
    results['train_conf_mtx'] = c_mtx

    # train temporal probs
    results_prob['train']['acc'] = model.evaluate(X_train,y_train)[1]
    X_train_probs= model.predict(X_train)
    results_prob['train']['t_prob'] = _get_true_class_prob(y_train, X_train_probs, train_len)

    '''
    results on test data
    '''
    a, a_t, c_mtx = _gru_test_acc(model, X_test, y_test,
                                  clip_time, len(test_list))
    results['test'] = a
    print('sacc = %0.3f' %np.mean(a))
    for ii in range(args.k_class):
        results['t_test'][ii] = a_t[ii]
    results['test_conf_mtx'] = c_mtx

    # test temporal probs
    results_prob['test']['acc'] = model.evaluate(X_test,y_test)[1]
    X_test_probs= model.predict(X_test)
    results_prob['test']['t_prob'] = _get_true_class_prob(y_test, X_test_probs, test_len)

    return results, results_prob