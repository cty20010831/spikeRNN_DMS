import numpy as np
import tensorflow as tf
import scipy.io

'''
CONTINUOUS FIRING-RATE RNN CLASS
'''

class FR_RNN_dale:
    def __init__(self, N, P_inh, P_rec, P_inter, w_in, som_N, w_dist, gain, apply_dale, w_out, num_layers=3):
        self.N = N
        self.P_inh = P_inh
        self.P_rec = P_rec
        self.P_inter = P_inter
        self.w_in = w_in
        self.som_N = som_N
        self.w_dist = w_dist
        self.gain = gain
        self.apply_dale = apply_dale
        self.w_out = w_out
        self.num_layers = num_layers

        # Layer-specific variables
        self.inh = []
        self.exc = []
        self.NI = []
        self.NE = []
        self.som_inh = []
        self.W = [] 
        self.mask_recur = []
        self.som_mask_recur = []
        self.W_inter = []

        # Initialize layers
        for layer in range(self.num_layers):
            inh, exc, NI, NE, som_inh = self.assign_exc_inh()
            self.inh.append(inh)
            self.exc.append(exc)
            self.NI.append(NI)
            self.NE.append(NE)

            W_recur, mask_recur, som_mask_recur = self.initialize_W_recur()
            self.W.append(W_recur)
            self.mask_recur.append(mask_recur)
            self.som_mask_recur.append(som_mask_recur)

            # Initialize inter-layer weights and masks for non-input layers
            if layer != 0:
                W_inter = self.initialize_W_inter()
                self.W_inter.append(W_inter)

    def assign_exc_inh(self):
        if self.apply_dale:
            inh = np.random.rand(self.N, 1) < self.P_inh
            exc = ~inh
            NI = len(np.where(inh == True)[0])
            NE = self.N - NI
        else:
            inh = np.random.rand(self.N, 1) < 0
            exc = ~inh
            NI = len(np.where(inh == True)[0])
            NE = self.N - NI

        som_inh = np.where(inh == True)[0][:self.som_N] if self.som_N > 0 else []
        return inh, exc, NI, NE, som_inh

    def initialize_W_recur(self):
        """
        Method to initialize recurrent weight matrices
        """
        w = np.zeros((self.N, self.N), dtype=np.float32)
        idx = np.where(np.random.rand(self.N, self.N) < self.P_rec)
        
        if self.w_dist.lower() == 'gamma':
            w[idx[0], idx[1]] = np.random.gamma(2, 0.003, len(idx[0]))
        elif self.w_dist.lower() == 'gaus':
            w[idx[0], idx[1]] = np.random.normal(0, 1.0, len(idx[0]))
            w = w / np.sqrt(self.N * self.P_rec) * self.gain

        if self.apply_dale:
            w = np.abs(w)

        mask = np.ones((self.N, self.N), dtype=np.float32)
        mask[np.where(self.inh == True)[0], :] = -1  # Negative for inhibitory neurons
        mask[np.where(self.exc == True)[0], :] = 1   # Positive for excitatory neurons

        som_mask = np.ones((self.N, self.N), dtype=np.float32)
        if self.som_N > 0:
            for i in self.som_inh:
                som_mask[i, np.where(self.inh == True)[0]] = 0

        return w, mask, som_mask
    
    def initialize_W_inter(self):
        """
        Method to initialize inter-layer weight matrices
        """
        w = np.zeros((self.N, self.N), dtype=np.float32)
        idx = np.where(np.random.rand(self.N, self.N) < self.P_inter)

        w[idx[0], idx[1]] = np.float32(np.random.randn(self.N, self.N))

        return w

    def load_net(self, model_dir):
        """
        Method to load pre-configured network settings
        """
        settings = scipy.io.loadmat(model_dir)
        self.N = settings['N'][0][0]
        self.som_N = settings['som_N'][0][0]
        self.num_layers = settings['num_layers'][0][0]

        # Load layer-specific data
        self.inh = [settings[f'inh_{i}'] == 1 for i in range(self.num_layers)]
        self.exc = [settings[f'exc_{i}'] == 1 for i in range(self.num_layers)]
        self.NI = [len(np.where(self.inh[i] == True)[0]) for i in range(self.num_layers)]
        self.NE = [self.N - ni for ni in self.NI]

        self.W = [settings[f'w_{i}'] for i in range(self.num_layers)]
        self.mask_recur = [settings[f'm_{i}'] for i in range(self.num_layers)]
        self.som_mask_recur = [settings[f'som_m_{i}'] for i in range(self.num_layers)]

        # Load inter-layer weights and masks
        self.W_inter = [settings[f'w_inter_{i+1}_{i+2}'] for i in range(self.num_layers - 1)]

        # Load input and output weights
        self.w_in = settings['w_in']
        self.w_out = settings['w_out']
        self.b_out = settings['b_out'][0][0]

        return self
    
    def display(self):
        """
        Method to print the network setup
        """
        print('Network Settings')
        print('====================================')
        print('Number of Units: ', self.N)
        print('Number of Layers: ', self.num_layers)
        
        for layer in range(self.num_layers):
            print(f'\nLayer {layer+1} Settings:')
            print('\t Number of Excitatory Units: ', self.NE[layer])
            print('\t Number of Inhibitory Units: ', self.NI[layer])
            
            # Display weight matrix statistics for the current layer
            full_w = self.W[layer] * self.mask_recur[layer]
            zero_w = len(np.where(full_w == 0)[0])
            pos_w = len(np.where(full_w > 0)[0])
            neg_w = len(np.where(full_w < 0)[0])
            print('\t Weight Matrix, W:')
            print(f'\t\t Zero Weights: {zero_w / (self.N * self.N) * 100:.2f} %')
            print(f'\t\t Positive Weights: {pos_w / (self.N * self.N) * 100:.2f} %')
            print(f'\t\t Negative Weights: {neg_w / (self.N * self.N) * 100:.2f} %')

        # Display inter-layer weights and masks
        if self.num_layers > 1:
            for layer in range(self.num_layers - 1):
                print(f'\nInter-Layer Connections: Layer {layer+1} to Layer {layer+2}')
                inter_w = self.W_inter[layer] * self.mask_inter[layer]
                inter_zero_w = len(np.where(inter_w == 0)[0])
                print('\t Inter-Layer Weight Matrix:')
                print(f'\t\t Zero Weights: {inter_zero_w / (self.N * self.N) * 100:.2f} %')

        print('====================================')
        print('Input Weight Matrix, w_in:')
        print(self.w_in)
        print('Output Weight Matrix, w_out:')
        print(self.w_out)
        print('Output Bias, b_out:', self.b_out)


'''
Task-specific input signals
'''
def generate_input_stim_go_nogo(settings):
    """
    Method to generate the input stimulus matrix for the
    Go-NoGo task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            taus: time-constants (in steps)
            DeltaT: sampling rate
    OUTPUT
        u: 1xT stimulus matrix
        label: either +1 (Go trial) or 0 (NoGo trial) 
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']

    u = np.zeros((1, T)) #+ np.random.randn(1, T)
    u_lab = np.zeros((2, 1))
    if np.random.rand() <= 0.50:
        u[0, stim_on:stim_on+stim_dur] = 1
        label = 1
    else:
        label = 0 

    return u, label

def generate_input_stim_xor(settings):
    """
    Method to generate the input stimulus matrix (u)
    for the XOR task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            delay: delay b/w two stimuli (in steps)
            taus: time-constants (in steps)
            DeltaT: sampling rate
    OUTPUT
        u: 2xT stimulus matrix
        label: 'same' or 'diff'
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    delay = settings['delay']

    # Initialize u
    u = np.zeros((2, T))

    # XOR task
    labs = []
    if np.random.rand() < 0.50:
        u[0, stim_on:stim_on+stim_dur] = 1
        labs.append(1)
    else:
        u[0, stim_on:stim_on+stim_dur] = -1
        labs.append(-1)

    if np.random.rand() < 0.50:
        u[1, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay] = 1
        labs.append(1)
    else:
        u[1, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay] = -1
        labs.append(-1)

    if np.prod(labs) == 1:
        label = 'same'
    else:
        label = 'diff'

    return u, label

def generate_input_stim_mante(settings):
    """
    Method to generate the input stimulus matrix for the
    mante task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            taus: time-constants (in steps)
            DeltaT: sampling rate
    OUTPUT
        u: 4xT stimulus matrix (first 2 rows for motion/color and the second
        2 rows for context
        label: either +1 or -1
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']

    # Color/motion sensory inputs
    u = np.zeros((2, T))
    u_lab = np.zeros((2, 1))
    if np.random.rand() <= 0.50:
        u[0, stim_on:stim_on+stim_dur] = np.random.randn(1, stim_dur) + 0.5
        u_lab[0, 0] = 1
    else:
        u[0, stim_on:stim_on+stim_dur] = np.random.randn(1, stim_dur) - 0.5
        u_lab[0, 0] = -1

    if np.random.rand() <= 0.50:
        u[1, stim_on:stim_on+stim_dur] = np.random.randn(1, stim_dur) + 0.5
        u_lab[1, 0] = 1
    else:
        u[1, stim_on:stim_on+stim_dur] = np.random.randn(1, stim_dur) - 0.5
        u_lab[1, 0] = -1

    # Context input
    c = np.zeros((2, T))
    label = 0
    if np.random.rand() <= 0.50:
        c[0, :] = 1

        if u_lab[0, 0] == 1:
            label = 1
        elif u_lab[0, 0] == -1:
            label = -1
    else:
        c[1, :] = 1

        if u_lab[1, 0] == 1:
            label = 1
        elif u_lab[1, 0] == -1:
            label = -1

    return np.vstack((u, c)), label


'''
Task-specific target signals
'''
def generate_target_continuous_go_nogo(settings, label):
    """
    Method to generate a continuous target signal (z) 
    for the Go-NoGo task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            taus: time-constants (in steps)
            DeltaT: sampling rate
        label: either +1 or -1
    OUTPUT
        z: 1xT target signal
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']

    z = np.zeros((1, T))
    if label == 1:
        z[0, stim_on+stim_dur:] = 1
    # elif label == 0:
        # z[0, stim_on+stim_dur:] = -1

    return np.squeeze(z)

def generate_target_continuous_xor(settings, label):
    """
    Method to generate a continuous target signal (z) 
    for the XOR task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            delay: delay b/w two stimuli (in steps)
            taus: time-constants (in steps)
            DeltaT: sampling rate
        label: string value (either 'same' or 'diff')
    OUTPUT
        z: 1xT target signal
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    delay = settings['delay']
    task_end_T = stim_on+2*stim_dur + delay

    z = np.zeros((1, T))
    # Adjust the decision period for Herbert's task
    # Here, we choose to fix the decision period right 
    # between the end of second stimulus and end of trial
    if label == 'same':
        z[0, task_end_T:task_end_T+25] = 1
    elif label == 'diff':
        z[0, task_end_T:task_end_T+25] = -1

    return np.squeeze(z)

def generate_target_continuous_mante(settings, label):
    """
    Method to generate a continuous target signal (z) 
    for the MANTE task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            taus: time-constants (in steps)
            DeltaT: sampling rate
        label: either +1 or -1
    OUTPUT
        z: 1xT target signal
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']

    z = np.zeros((1, T))
    if label == 1:
        z[0, stim_on+stim_dur:] = 1
    else:
        z[0, stim_on+stim_dur:] = -1

    return np.squeeze(z)

'''
CONSTRUCT TF GRAPH FOR TRAINING
'''
def construct_tf(fr_rnn, settings, training_params):
    T = settings['T']
    taus = settings['taus']
    DeltaT = settings['DeltaT']
    task = settings['task']
    learning_rate = training_params['learning_rate']

    # Initialize lists for synaptic currents and firing rates for each layer
    x = [[] for _ in range(fr_rnn.num_layers)]  # Synaptic currents
    r = [[] for _ in range(fr_rnn.num_layers)]  # Firing rates

    # Input stimulus placeholder
    if task == 'xor':
        stim = tf.placeholder(tf.float32, [2, T], name='u')
    elif task == 'mante':
        stim = tf.placeholder(tf.float32, [4, T], name='u')
    elif task == 'go-nogo':
        stim = tf.placeholder(tf.float32, [1, T], name='u')

    # Target output placeholder
    z = tf.placeholder(tf.float32, [T,], name='target')

    # Initialize synaptic time constants (decay) using a sigmoid
    if len(taus) > 1:
        taus_gaus = tf.Variable(tf.random_normal([fr_rnn.N, 1]), dtype=tf.float32, 
                                name='taus_gaus', trainable=True)
    else:
        taus_gaus = tf.Variable(tf.random_normal([fr_rnn.N, 1]), dtype=tf.float32, 
                                name='taus_gaus', trainable=False)
        print('Synaptic decay time-constants will not get updated!')

    # Initialize synaptic currents for the first time step
    for layer in range(fr_rnn.num_layers):
        x[layer].append(tf.random_normal([fr_rnn.N, 1], dtype=tf.float32) / 100)
        # Initialize the first firing rate for each layer
        if training_params['activation'] == 'sigmoid':
            r[layer].append(tf.sigmoid(x[layer][0]))
        elif training_params['activation'] == 'clipped_relu':
            r[layer].append(tf.clip_by_value(tf.nn.relu(x[layer][0]), 0, 20))
        elif training_params['activation'] == 'softplus':
            r[layer].append(tf.clip_by_value(tf.nn.softplus(x[layer][0]), 0, 20))

    # Initialize weight matrices and masks for each layer
    W = [tf.get_variable(f'w_{layer}', initializer=fr_rnn.W[layer], dtype=tf.float32, trainable=True)
         for layer in range(fr_rnn.num_layers)]
    masks = [tf.get_variable(f'm_{layer}', initializer=fr_rnn.mask_recur[layer], dtype=tf.float32, trainable=False)
             for layer in range(fr_rnn.num_layers)]
    som_masks = [tf.get_variable(f'som_m_{layer}', initializer=fr_rnn.som_mask_recur[layer], dtype=tf.float32, trainable=False)
                 for layer in range(fr_rnn.num_layers)]

    w_in = tf.get_variable('w_in', initializer=fr_rnn.w_in, dtype=tf.float32, trainable=False)
    
    # Initialize inter-layer weights and masks with explicit shapes
    W_inter = [tf.get_variable(f'w_inter_{layer+1}_{layer+2}', 
                               shape=fr_rnn.W_inter[layer].shape,  # Explicitly define the shape
                               dtype=tf.float32, 
                               initializer=tf.constant_initializer(fr_rnn.W_inter[layer]),
                               trainable=True)
               for layer in range(fr_rnn.num_layers - 1)]

    w_out = tf.get_variable('w_out', initializer=fr_rnn.w_out, dtype=tf.float32, trainable=True)
    b_out = tf.Variable(0, dtype=tf.float32, name='b_out', trainable=True)

    # Pass the synaptic time constants through the sigmoid function
    if len(taus) > 1:
        taus_sig = tf.sigmoid(taus_gaus) * (taus[1] - taus[0]) + taus[0]
    else:
        taus_sig = taus[0]

    # Forward pass computation
    o = []
    for t in range(1, T):
        for layer in range(fr_rnn.num_layers):
            # Enforce Dale's principle on the recurrent weight matrix
            if fr_rnn.apply_dale:
                W[layer] = tf.nn.relu(W[layer])

            # Apply the recurrent mask and SOM mask
            ww = tf.matmul(W[layer], masks[layer])
            ww = tf.multiply(ww, som_masks[layer])

            # Determine input current based on the layer
            if layer == 0:
                # First layer: input includes the external stimulus
                input_current = tf.matmul(w_in, tf.expand_dims(stim[:, t-1], 1))
            else:
                # Subsequent layers: input comes from the previous layer's firing rates
                input_current = tf.matmul(W_inter[layer-1], r[layer-1][t-1])

            # Synaptic update for the current layer
            next_x = tf.multiply((1 - DeltaT / taus_sig), x[layer][t-1]) + \
                     tf.multiply((DeltaT / taus_sig), (tf.matmul(ww, r[layer][t-1]) + input_current)) + \
                     tf.random_normal([fr_rnn.N, 1], dtype=tf.float32) / 10
            x[layer].append(next_x)

            # Update firing rates based on the current layer's synaptic current
            if training_params['activation'] == 'sigmoid':
                r[layer].append(tf.sigmoid(next_x))
            elif training_params['activation'] == 'clipped_relu':
                r[layer].append(tf.clip_by_value(tf.nn.relu(next_x), 0, 20))
            elif training_params['activation'] == 'softplus':
                r[layer].append(tf.clip_by_value(tf.nn.softplus(next_x), 0, 20))

        # Compute the output from the last layer
        next_o = tf.matmul(w_out, r[-1][t]) + b_out
        o.append(next_o)

    return stim, z, x, r, o, W, w_in, W_inter, masks, som_masks, w_out, b_out, taus_gaus


'''
DEFINE LOSS AND OPTIMIZER
'''
def loss_op(o, z, training_params):
    """
    Method to define loss and optimizer for ONLY ONE target signal
    INPUT
        o: list of output values
        z: target values
        training_params: dictionary containing training parameters
            learning_rate: learning rate

    OUTPUT
        loss: loss function
        training_op: optimizer
    """
    # Loss function
    loss = tf.zeros(1)
    loss_fn = training_params['loss_fn']
    for i in range(0, len(o)):
        if loss_fn.lower() == 'l1':
            loss += tf.norm(o[i] - z[i])
        elif loss_fn.lower() == 'l2':
            loss += tf.square(o[i] - z[i])
    if loss_fn.lower() == 'l2':
        loss = tf.sqrt(loss)

    # Optimizer function
    with tf.name_scope('ADAM'):
        optimizer = tf.train.AdamOptimizer(learning_rate = training_params['learning_rate']) 

    training_op = optimizer.minimize(loss) 

    return loss, training_op

'''
EVALUATE THE TRAINED MODEL
NOTE: NEED TO BE UPDATED!!
'''
def eval_tf(model_dir, settings, u):
    """
    Method to evaluate a trained TF graph
    INPUT
        model_dir: full path to the saved model .mat file
        stim_params: dictionary containig the following keys
        u: 12xT stimulus matrix
            NOTE: There are 12 rows (one per dot pattern): 6 cues and 6 probes.
    OUTPUT
        o: 1xT output vector
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    delay = settings['delay']
    DeltaT = settings['DeltaT']

    # Load the trained mat file
    var = scipy.io.loadmat(model_dir)

    # Get some additional params
    N = var['N'][0][0]
    exc_ind = [np.bool(i) for i in var['exc']]

    # Get the delays
    taus_gaus = var['taus_gaus']
    taus = var['taus'][0] # tau [min, max]
    taus_sig = (1/(1+np.exp(-taus_gaus))*(taus[1] - taus[0])) + taus[0] 

    # Synaptic currents and firing-rates
    x = np.zeros((N, T)) # synaptic currents
    r = np.zeros((N, T)) # firing-rates
    x[:, 0] = np.random.randn(N, )/100
    r[:, 0] = 1/(1 + np.exp(-x[:, 0]))
    # r[:, 0] = np.minimum(np.maximum(x[:, 0], 0), 1) #clipped relu
    # r[:, 0] = np.clip(np.minimum(np.maximum(x[:, 0], 0), 1), None, 10) #clipped relu
    # r[:, 0] = np.clip(np.log(np.exp(x[:, 0])+1), None, 10) # softplus
    # r[:, 0] = np.minimum(np.maximum(x[:, 0], 0), 6)/6 #clipped relu6


    # Output
    o = np.zeros((T, ))
    o_counter = 0

    # Recurrent weights and masks
    # w = var['w0'] #!!!!!!!!!!!!
    w = var['w']

    m = var['m']
    som_m = var['som_m']
    som_N = var['som_N'][0][0]

    # Identify excitatory/inhibitory neurons
    exc = var['exc']
    exc_ind = np.where(exc == 1)[0]
    inh = var['inh']
    inh_ind = np.where(inh == 1)[0]
    som_inh_ind = inh_ind[:som_N]

    for t in range(1, T):
        # next_x is [N x 1]
        ww = np.matmul(w, m)
        ww = np.multiply(ww, som_m)

        # next_x = (1 - DeltaT/tau)*x[:, t-1] + \
                # (DeltaT/tau)*(np.matmul(ww, r[:, t-1]) + \
                # np.matmul(var['w_in'], u[:, t-1])) + \
                # np.random.randn(N, )/10

        next_x = np.multiply((1 - DeltaT/taus_sig), np.expand_dims(x[:, t-1], 1)) + \
                np.multiply((DeltaT/taus_sig), ((np.matmul(ww, np.expand_dims(r[:, t-1], 1)))\
                + np.matmul(var['w_in'], np.expand_dims(u[:, t-1], 1)))) +\
                np.random.randn(N, 1)/10

        x[:, t] = np.squeeze(next_x)
        r[:, t] = 1/(1 + np.exp(-x[:, t]))
        # r[:, t] = np.minimum(np.maximum(x[:, t], 0), 1)
        # r[:, t] = np.clip(np.minimum(np.maximum(x[:, t], 0), 1), None, 10)
        # r[:, t] = np.clip(np.log(np.exp(x[:, t])+1), None, 10) # softplus
        # r[:, t] = np.minimum(np.maximum(x[:, t], 0), 6)/6


        wout = var['w_out']
        wout_exc = wout[0, exc_ind]
        wout_inh = wout[0, inh_ind]
        r_exc = r[exc_ind, :]
        r_inh = r[inh_ind, :]

        o[o_counter] = np.matmul(wout, r[:, t]) + var['b_out']
        # o[o_counter] = np.matmul(wout_exc, r[exc_ind, t]) + var['b_out'] # excitatory output
        # o[o_counter] = np.matmul(wout_inh, r[inh_ind, t]) + var['b_out'] # inhibitory output
        o_counter += 1
    return x, r, o

