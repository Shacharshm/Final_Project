import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from sklearn.metrics import r2_score

from functools import partial

def GRUClassifier(X, k_layers=1, k_hidden=32, k_class=15,
                  l2=0.001, dropout=1e-6, lr=0.006, seed=42):
    
    """
    Parameters
    ---------
    X: tensor (batch x time x feat)
    k_layers: int, number of hidden layers
    k_hidden: int, number of units
    k_class: int, number of classes
    
    Returns
    -------
    model: complied model
    """
    
    tf.random.set_seed(seed)
    regularizer = keras.regularizers.l2(l2)
    CustomGRU = partial(keras.layers.GRU,
                            kernel_regularizer=regularizer,
                            dropout=dropout,
                            recurrent_dropout=dropout
                           )
    '''
    For masking, refer: 
        https://www.tensorflow.org/guide/keras/masking_and_padding
        https://gist.github.com/ragulpr/601486471549cfa26fe4af36a1fade21
    '''
    input_layers = [layers.Masking(mask_value=0.0, 
                                   input_shape = [None, X.shape[-1]])]
    
    hidden_layers = []
    for ii in range(k_layers):
        hidden_layers.append(CustomGRU(k_hidden,return_sequences=True))
        
    output_layer = [layers.TimeDistributed(layers.Dense(k_class,activation='softmax'))]
    
    optimizer = keras.optimizers.Adam(lr=lr)
    
    model = keras.models.Sequential(input_layers+hidden_layers+output_layer)
    
    model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,metrics=['sparse_categorical_accuracy'])
    
    return model