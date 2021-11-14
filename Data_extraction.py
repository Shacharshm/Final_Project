import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
import pickle
import time
import os
import pywt
import tensorflow as tf
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_memory_growth(gpus[1], True)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

K_SEED = 330
K_RUNS = 3

def _info(s):
    print('---')
    print(s)
    print('---')

def _get_clip_labels():
    '''
    assign all clips within runs a label
    use 0 for testretest
    '''
    # where are the clips within the run?
    timing_file = pd.read_csv('data/videoclip_tr_lookup.csv')

    clips = []
    for run in range(K_RUNS):
        run_name = 'MOVIE%d' %(run+1) #MOVIEx_7T_yz
        timing_df = timing_file[timing_file['run'].str.contains(run_name)]  
        timing_df = timing_df.reset_index(drop=True)

        for jj, row in timing_df.iterrows():
            clips.append(row['clip_name'])
            
    clip_y = {}
    jj = 1
    for clip in clips:
        if 'testretest' in clip:
            clip_y[clip] = 0
        else:
            clip_y[clip] = jj
            jj += 1

    return clip_y

def permute(coeffs):
    '''
    Shuffles the wavelet transform coefficients of a timeseries
    
    Parameters:
    coeffs: list of the wavelet coefficients. Example (cA, cD1, cD2 ...cDn)
    
    Returns:
    perm_coeffs: list of the wavelet coefficiets in which only detail (cDn)
                    coefficients are permuted
    '''
    permuted_coeffs = []
    for coeff in coeffs:
        coeff_copy = coeff.copy()
        np.random.shuffle(coeff_copy)
        permuted_coeffs.append(coeff_copy)
    return permuted_coeffs

def shuffle_ts(X, X_len):
    '''
    Shuffling clip time series
    
    Parameters:
    X: tensor (batch x time x feature)
    X_len: timeseries length tensor (batch x 1)
    
    Returns:
    X_copy: shuffled timeseries tensor (batch x time x feature)
    '''
    
    X_copy = tf.identity(X)
    X_copy = X_copy.numpy()
    # create mask to ignore padding
    mask = X_copy == 0.0
    # Go thru every example in the batch
    for ii in range(X_copy.shape[0]):
        unpadded_ts = X_copy[ii,:X_len[ii],:]
        # Take wavelet transform
        coeffs = pywt.wavedec(unpadded_ts,'db2',level=2,axis=0)
        # reconstruct the ts and assert if it is close to orig ts
        recon_ts = pywt.waverec(coeffs, 'db2', axis = 0)
        tf.debugging.assert_near(unpadded_ts,recon_ts[:X_len[ii]])
        # get perm coeffs
        perm_coeffs = permute(coeffs)
        # construct shuffled timeseries
        perm_ts = pywt.waverec(perm_coeffs, 'db2', axis = 0)
        X_copy[ii,:X_len[ii],:] = perm_ts[:X_len[ii],:]
        
    return tf.convert_to_tensor(X_copy,dtype='float32')

def _clip_class_df(args):
    '''
    data for 15-way clip classification

    args.roi: number of ROIs
    args.net: number of subnetworks (7 or 17)
    args.subnet: subnetwork; 'wb' if all subnetworks
    args.invert_flag: all-but-one subnetwork
    args.r_roi: number of random ROIs to pick
    args.r_seed: random seed for picking ROIs

    save each timepoint as feature vector
    append class label based on clip

    return:
    pandas df
    '''
    
    load_path = (args.input_data + '/data_MOVIE_runs_%s' %(args.roi_name) +
        '_%d_net_%d_ts.pkl' %(args.roi, args.net))

    with open(load_path, 'rb') as f:
        data = pickle.load(f)
        
    # where are the clips within the run?
    timing_file = pd.read_csv('data/videoclip_tr_lookup.csv')
    
    '''
    main
    '''
    clip_y = _get_clip_labels()
    
    table = []
    for run in range(K_RUNS):
        
        print('loading run %d/%d' %(run+1, K_RUNS))
        run_name = 'MOVIE%d' %(run+1) #MOVIEx_7T_yz

        # timing file for run
        timing_df = timing_file[
            timing_file['run'].str.contains(run_name)]  
        timing_df = timing_df.reset_index(drop=True)

        for subject in data:

            # get subject data (time x roi x run)
            vox_ts = data[subject][:, :, run]

            for jj, clip in timing_df.iterrows():

                start = int(np.floor(clip['start_tr']))
                stop = int(np.ceil(clip['stop_tr']))
                clip_length = stop - start
                
                # assign label to clip
                y = clip_y[clip['clip_name']]

                for t in range(clip_length):
                    act = vox_ts[t + start, :]
                    t_data = {}
                    t_data['Subject'] = subject
                    t_data['timepoint'] = t
                    for feat in range(vox_ts.shape[1]):
                        t_data['feat_%d' %(feat)] = act[feat]
                    t_data['y'] = y
                    table.append(t_data)

    df = pd.DataFrame(table)
    df['Subject'] = df['Subject'].astype(int)
        
    return df

def _get_clip_seq(df, subject_list, args):
    '''
    return:
    X: input seq (batch_size x time x feat_size)
    y: label seq (batch_size x time)
    X_len: len of each seq (batch_size x 1)
    batch_size <-> number of sequences
    time <-> max length after padding
    '''
    features = [ii for ii in df.columns if 'feat' in ii]
    
    X = []
    y = []
    for subject in subject_list:
        for i_class in range(args.k_class):
            
            if i_class==0: # split test-retest into 4
                seqs = df[(df['Subject']==subject) & 
                    (df['y'] == 0)][features].values
                label_seqs = df[(df['Subject']==subject) & 
                    (df['y'] == 0)]['y'].values

                k_time = int(seqs.shape[0]/K_RUNS)
                for i_run in range(K_RUNS):
                    seq = seqs[i_run*k_time:(i_run+1)*k_time, :]
                    label_seq = label_seqs[i_run*k_time:(i_run+1)*k_time]
                    if args.zscore:
                        # zscore each seq that goes into model
                        seq = (1/np.std(seq))*(seq - np.mean(seq))

                    X.append(torch.FloatTensor(seq))
                    y.append(torch.LongTensor(label_seq))
            else:
                seq = df[(df['Subject']==subject) & 
                    (df['y'] == i_class)][features].values
                label_seq = df[(df['Subject']==subject) & 
                    (df['y'] == i_class)]['y'].values
                if args.zscore:
                    # zscore each seq that goes into model
                    seq = (1/np.std(seq))*(seq - np.mean(seq))
                
                X.append(torch.FloatTensor(seq))
                y.append(torch.LongTensor(label_seq))
            
    X_len = torch.LongTensor([len(seq) for seq in X])

    # pad sequences
    X = pad_sequence(X, batch_first=True, padding_value=0)
    y = pad_sequence(y, batch_first=True, padding_value=-100)
            
    return X.to(args.device), X_len.to(args.device), y.to(args.device)

def Get_Data(args):
    '''
    X_train: type: EagerTensor shape: [number of examples, time, features(voxels)]
    train_len: type: EagerTensor shape: [number of examples]
    y_train: type: EagerTensor shape: [number of exmples, time]
    X_test: type: EagerTensor shape: [number of examples, time, features(voxels)]
    test_len: type: EagerTensor shape: [number of examples]
    y_test: type: EagerTensor shape: [number of exmples, time]
    '''
    
    _info(args.roi_name)
    # Get all combinations of the parameter grid
    param_grid = {'k_hidden':args.k_hidden,'k_layers':args.k_layers}
    param_grid = [comb for comb in ParameterGrid(param_grid)]

    print(len(param_grid))
    print(len(args.k_layers))

    _info('Number of hyperparameter combinations: '+str(len(param_grid)))
    _info(args.roi_name)

    start = time.clock()
    df = _clip_class_df(args)
    print('data loading time: %.2f seconds' %(time.clock()-start))

    # get X-y from df
    subject_list = df['Subject'].unique()
    train_list = subject_list[:args.train_size]
    test_list = subject_list[args.train_size:]

    print('number of subjects = %d' %(len(subject_list)))
    features = [ii for ii in df.columns if 'feat' in ii]
    k_feat = len(features)
    print('number of features = %d' %(k_feat))
    args.k_class = len(np.unique(df['y']))
    print('number of classes = %d' %(args.k_class))

    # length of each clip
    clip_time = np.zeros(args.k_class)
    for ii in range(args.k_class):
        class_df = df[df['y']==ii]
        clip_time[ii] = np.max(np.unique(class_df['timepoint'])) + 1
    clip_time = clip_time.astype(int) # df saves float
    print('seq lengths = %s' %clip_time)

    # get train, test sequences
    X_train, train_len, y_train = _get_clip_seq(df, 
        train_list, args)
    X_test, test_len, y_test = _get_clip_seq(df, 
        test_list, args)
    X_train = shuffle_ts(X_train, train_len)

    return X_train, train_len, y_train, X_test, test_len, y_test
