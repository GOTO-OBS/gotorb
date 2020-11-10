from tensorflow import keras

def vgg6(input_shape, n_classes=1):
    """
    VGG6 basic architecture, using [16, 32] conv-pool-ReLU blocks. Uses spatial dropout on convolutional layers
    for more effective regularisation.

    :param input_shape: shape of incoming image tensor
    :param n_classes: number of prediction classes
    """
    model = keras.models.Sequential(name='VGG6')
    model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, name='conv1',
                                  kernel_regularizer=keras.regularizers.l2(0.0001)))
    model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', name='conv2',
                                  kernel_regularizer=keras.regularizers.l2(0.0001)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(keras.layers.SpatialDropout2D(0.25))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv3',
                                  kernel_regularizer=keras.regularizers.l2(0.0001)))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv4',
                                  kernel_regularizer=keras.regularizers.l2(0.0001)))
    model.add(keras.layers.MaxPooling2D(pool_size=(4, 4), padding='valid'))
    model.add(keras.layers.SpatialDropout2D(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu', name='fc_1',
                                 activity_regularizer=keras.regularizers.l2(0.0001)))
    model.add(keras.layers.Dropout(0.5))
    # output layer
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model.add(keras.layers.Dense(n_classes, activation=activation, name='fc_out'))

    return model

def bayesian_vgg6(input_shape, dropout_prob=0.01, n_classes=1):
    """
    Bayesian version of vgg6, using Monte Carlo dropout throughout for posterior estimation.
    Level of dropout is tunable, to set the right level of regularisation. As before, spatial dropout is used on
    convolutional layers for better regularisation.

    :param dropout_prob: dropout probability applied to weight matrix.
    :param input_shape: shape of incoming image tensor
    :param n_classes: number of prediction classes
    """
    model = keras.models.Sequential(name='VGG6_MCdropout_heinit')
    model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, name='conv1',
                                  kernel_regularizer=keras.regularizers.l2(0.0001),
                                  kernel_initializer="he_uniform"))
    model.add(keras.layers.SpatialDropout2D(dropout_prob))
    model.add(keras.layers.Conv2D(16, (3, 3), name='conv2',
                                  kernel_regularizer=keras.regularizers.l2(0.0001),
                                  kernel_initializer="he_uniform"))
    model.add(keras.layers.SpatialDropout2D(dropout_prob))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv3',
                                  kernel_regularizer=keras.regularizers.l2(0.0001),
                                  kernel_initializer="he_uniform"))
    model.add(keras.layers.SpatialDropout2D(dropout_prob))
    model.add(keras.layers.Conv2D(32, (3, 3), name='conv4',
                                  kernel_regularizer=keras.regularizers.l2(0.0001),
                                  kernel_initializer="he_uniform"))
    model.add(keras.layers.MaxPooling2D(pool_size=(4, 4), padding='valid'))
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(dropout_prob)) # use regular dropout after flatten to reduce complexity.

    model.add(keras.layers.Dense(256, activation='relu', name='fc_1',
                                 activity_regularizer=keras.regularizers.l2(0.0001),
                                 kernel_initializer="he_uniform"))
    model.add(keras.layers.Dropout(dropout_prob))
    # output layer
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model.add(keras.layers.Dense(n_classes, activation=activation, name='fc_out'))

    return model

def bayesian_sepconvnet(input_shape, dropout_prob=0.01, n_classes=1):
    """
    Experimental net with separable convolutions. For larger models this could provide a fairly sizeable speedup,
    especially when deployed on GPU. Kept in for future development.

    :param input_shape: shape of input image tensor
    :param dropout_prob: probability of dropout between layers
    :param n_classes: number of classes to predict
    :return: initialised Keras model
    """
    model = keras.models.Sequential(name='sepconvnet_MCdropout')

    # block 1 -  extract features
    model.add(keras.layers.SeparableConv2D(16, (3, 3), input_shape=input_shape, name='sepconv1',
                                           pointwise_regularizer=keras.regularizers.l2(1e-4),
                                           pointwise_initializer='he_uniform',
                                           depthwise_initializer='he_uniform'))
    model.add(keras.layers.ReLU())
    model.add(keras.layers.SpatialDropout2D(dropout_prob))

    # block 2 - extract features and downsample
    model.add(keras.layers.SeparableConv2D(16, (3, 3), name='sepconv2',
                                           pointwise_regularizer=keras.regularizers.l2(1e-4),
                                           pointwise_initializer='he_uniform',
                                           depthwise_initializer='he_uniform'))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.ReLU())
    model.add(keras.layers.SpatialDropout2D(dropout_prob))

    # block 3 - extract features
    model.add(keras.layers.SeparableConv2D(32, (3, 3), name='sepconv3',
                                           pointwise_regularizer=keras.regularizers.l2(1e-4),
                                           pointwise_initializer='he_uniform',
                                           depthwise_initializer='he_uniform'))
    model.add(keras.layers.ReLU())
    model.add(keras.layers.SpatialDropout2D(dropout_prob))

    # block 4 - extract features and stride down 4x4
    model.add(keras.layers.SeparableConv2D(32, (3, 3), name='sepconv4',
                                           pointwise_regularizer=keras.regularizers.l2(1e-4),
                                           pointwise_initializer='he_uniform',
                                           depthwise_initializer='he_uniform'))
    model.add(keras.layers.MaxPooling2D((4, 4)))
    model.add(keras.layers.ReLU())

    # flatten and use regular dropout
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(dropout_prob))

    # Dense layer
    model.add(keras.layers.Dense(256, name='dense1', 
                                 kernel_initializer='he_uniform'))
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Dropout(dropout_prob))
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model.add(keras.layers.Dense(n_classes, activation=activation, name='dense_out'))

    return model

def bayesian_vgg6_hyperparameterised(input_shape, dropout_prob, block1_size,
                                     block2_size, fc_size, initialiser,
                                     kern_size, regulariser_strength, activation):
    """
    Bayesian version of vgg6, using Monte Carlo dropout throughout for posterior estimation.
    This version is directly parameterisable, which when combined with the `hyperparameter_tuning` module enables
    grid searches.

    :param input_shape: shape of input stamps, format (height, width, channels)
    :param dropout_prob: probability of dropout for a given neuron
    :param block1_size: number of convolutional filters in block 1
    :param block2_size: number of convolutional filters in block 2
    :param fc_size: number of neurons in the fully-connected final layer
    :param initialiser: weight matrix initialiser, specified as either string or Initializer instance
    :param kern_size: kernel size for each convolution operation
    :param regulariser_strength: strength of L2 loss penalty
    :param activation: activation function, specified as string.
    :return: Uncompiled `keras` model instance.
    """

    model = keras.models.Sequential(name='bayesian_vgg6_hyperparam')
    model.add(keras.layers.Conv2D(block1_size, (kern_size, kern_size), input_shape=input_shape, name='conv1',
                                  kernel_regularizer=keras.regularizers.l2(regulariser_strength),
                                  kernel_initializer=initialiser))
    model.add(parse_activ_function(activation))
    model.add(keras.layers.SpatialDropout2D(dropout_prob))
    model.add(keras.layers.Conv2D(block1_size, (kern_size, kern_size), name='conv2',
                                  kernel_regularizer=keras.regularizers.l2(regulariser_strength),
                                  kernel_initializer=initialiser))
    model.add(keras.layers.SpatialDropout2D(dropout_prob))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(parse_activ_function(activation))

    model.add(keras.layers.Conv2D(block2_size, (kern_size, kern_size), name='conv3',
                                  kernel_regularizer=keras.regularizers.l2(regulariser_strength),
                                  kernel_initializer=initialiser))
    model.add(parse_activ_function(activation))
    model.add(keras.layers.SpatialDropout2D(dropout_prob))
    model.add(keras.layers.Conv2D(block2_size, (kern_size, kern_size), name='conv4',
                                  kernel_regularizer=keras.regularizers.l2(regulariser_strength),
                                  kernel_initializer=initialiser))
    model.add(keras.layers.MaxPooling2D(pool_size=(4, 4), padding='valid'))
    model.add(parse_activ_function(activation))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(dropout_prob)) # use regular dropout after flatten to reduce complexity.

    model.add(keras.layers.Dense(fc_size, name='fc_1',
                                 activity_regularizer=keras.regularizers.l2(regulariser_strength),
                                 kernel_initializer=initialiser))
    model.add(parse_activ_function(activation))
    model.add(keras.layers.Dropout(dropout_prob))

    model.add(keras.layers.Dense(1, activation='sigmoid', name='fc_out'))

    return model

def compat_build_hp_model(dropout_prob, block1_size, block2_size, fc_size, initialiser, kern_size, regulariser_strength,
                   activation):
    """
    Build an uninitialised model that can be passed to `classifier.train()` based on a set of hyperparameters
    derived from a parameter search. Included as a convenient way to match the call signature inside the training
    script, as a compatibility layer. When called by train, n_classes is a dummy arg and anything other than
    n_classes == 1 is incompatible. Be careful!

    :return: lambda that returns a keras sequential when called with input_shape and n_classes.
    """

    proto = lambda input_shape, n_classes: bayesian_vgg6_hyperparameterised(input_shape, dropout_prob=dropout_prob, block1_size=block1_size,
                                                          block2_size=block2_size, fc_size=fc_size,
                                                          initialiser=initialiser, kern_size=kern_size,
                                                          regulariser_strength=regulariser_strength,
                                                          activation=activation)

    return proto

def parse_activ_function(activation_str):
    """
    Generate an activation layer based on the activation string provided.
    Necessary to support the advanced activations that can't be passed to other layers

    :param activation_str: string descriptor for activation function (ReLU, LeakyReLU, ELU)
    :return: `keras.layers` instance to pass to `model.add()`
    """

    # Parse activation function
    if activation_str == "ReLU":
        activation = keras.layers.ReLU()
    elif activation_str == "LeakyReLU":
        activation = keras.layers.LeakyReLU()
    elif activation_str == "ELU":
        activation = keras.layers.ELU()
    else:
        raise ValueError("Activation not specified")

    return activation