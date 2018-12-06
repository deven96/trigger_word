import os
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv1D, BatchNormalization, \
                             Activation, Dropout, GRU, TimeDistributed
from keras.optimizers import Adam


TRAINED_WEIGHT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tr_model.h5")
Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram

def create_model(input_shape):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    
    X_input = Input(shape = input_shape)
    
    # Step 1: CONV layer
    X = Conv1D(196, kernel_size=15, strides=4)(X_input)   # CONV1D
    X = BatchNormalization()(X)                           # Batch normalization
    X = Activation('relu')(X)                             # ReLu activation
    X = Dropout(0.8)(X)                                   # dropout (use 0.8)

    # Step 2: First GRU Layer
    X = GRU(units = 128, return_sequences = True)(X)      # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                   # dropout (use 0.8)
    X = BatchNormalization()(X)                           # Batch normalization
    
    # Step 3: Second GRU Layer
    X = GRU(units = 128, return_sequences = True)(X)      # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                   # dropout (use 0.8)
    X = BatchNormalization()(X)                           # Batch normalization
    X = Dropout(0.8)(X)                                   # dropout (use 0.8)
    
    # Step 4: Time-distributed dense layer
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)
    model = Model(inputs = X_input, outputs = X)
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    return model  

def process_model():
    model = load_model(TRAINED_WEIGHT)

    return model