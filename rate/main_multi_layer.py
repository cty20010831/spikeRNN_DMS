import os
import time
import scipy.io
import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

# Import utility functions
from utils import set_gpu
from utils import restricted_float
from utils import str2bool

# Import the continuous rate model
from model_multi_layer import FR_RNN_dale

# Import the tasks
from model_multi_layer import generate_input_stim_xor, generate_target_continuous_xor

from model_multi_layer import generate_input_stim_mante, generate_target_continuous_mante

from model_multi_layer import generate_input_stim_go_nogo, generate_target_continuous_go_nogo

# Import functions for training and evaluating
from model_multi_layer import construct_tf, loss_op, eval_tf

# Parse input arguments
parser = argparse.ArgumentParser(description='Training rate RNNs')
parser.add_argument('--gpu', required=False,
        default='0', help="Which gpu to use")
parser.add_argument("--gpu_frac", required=False,
        type=restricted_float, default=0.4,
        help="Fraction of GPU mem to use")
parser.add_argument("--n_trials", required=True,
        type=int, default=200, help="Number of epochs")
parser.add_argument("--mode", required=True,
        type=str, default='Train', help="Train or Eval")
parser.add_argument("--output_dir", required=True,
        type=str, help="Model output path")
parser.add_argument("--N", required=True,
        type=int, help="Number of neurons")
parser.add_argument("--n_layers", required=True,
        type=int, help="Number of layers")
parser.add_argument("--gain", required=False,
        type=float, default = 1.5, help="Gain for the connectivity weight initialization")
parser.add_argument("--P_inh", required=False,
        type=restricted_float, default = 0.20,
        help="Proportion of inhibitory neurons")
parser.add_argument("--P_rec", required=False,
        type=restricted_float, default = 0.20,
        help="Connectivity probability")
parser.add_argument("--P_inter", required=False,
        type=restricted_float, default = 0.80,
        help="Inter-layer connection probability")
parser.add_argument("--som_N", required=True,
        type=int, default = 0, help="Number of SST neurons")
parser.add_argument("--task", required=True,
        type=str, help="Task (XOR, sine, etc...)")
parser.add_argument("--act", required=True,
        type=str, default='sigmoid', help="Activation function (sigmoid, clipped_relu)")
parser.add_argument("--loss_fn", required=True,
        type=str, default='l2', help="Loss function (either L1 or L2)")
parser.add_argument("--apply_dale", required=True,
        type=str2bool, default='True', help="Apply Dale's principle?")
parser.add_argument("--decay_taus", required=True,
        nargs='+', type=float,
        help="Synaptic decay time-constants (in time-steps). If only one number is given, then all\
        time-constants set to that value (i.e. not trainable). Otherwise specify two numbers (min, max).")
args = parser.parse_args()

# Set up the output dir where the output model will be saved
out_dir = os.path.join(args.output_dir, 'models', args.task.lower())
if args.apply_dale == False:
    out_dir = os.path.join(out_dir, 'NoDale')
if len(args.decay_taus) > 1:
    out_dir = os.path.join(out_dir, 'P_rec_' + str(args.P_rec) + '_Taus_' + str(args.decay_taus[0]) + '_' + str(args.decay_taus[1]))
else:
    out_dir = os.path.join(out_dir, 'P_rec_' + str(args.P_rec) + '_Tau_' + str(args.decay_taus[0]))

if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)

# Number of units/neurons
N = args.N
som_N = args.som_N; # number of SST neurons 

# Define the training parameters (learning rate, training termination criteria, etc...)
training_params = {
        'learning_rate': 0.01, # learning rate
        'loss_threshold': 7, # loss threshold (when to stop training)
        'eval_freq': 100, # how often to evaluate task perf
        'eval_tr': 100, # number of trials for eval
        'eval_amp_threh': 0.7, # amplitude threshold during response window
        'activation': args.act.lower(), # activation function
        'loss_fn': args.loss_fn.lower(), # loss function ('L1' or 'L2')
        'P_rec': args.P_rec
        }

# Define task-specific parameters
# NOTE: Each time step is 5 ms
if args.task.lower() == 'go-nogo':
    # GO-NoGo task
    settings = {
            'T': 200, # trial duration (in steps)
            'stim_on': 50, # input stim onset (in steps)
            'stim_dur': 25, # input stim duration (in steps)
            'DeltaT': 1, # sampling rate
            'taus': args.decay_taus, # decay time-constants (in steps)
            'task': args.task.lower(), # task name
            }
elif args.task.lower() == 'xor':
    # XOR task 
    # For the DMS task in our case, the time period is between 0 to 4s, with dt = 0.02s.
    # The first stimulus shows up between 1 and 1.5s, 
    # and the second stimulus shows up between 3 and 3.5s. 
    settings = {
            'T': 200, # trial duration (in steps)
            'stim_on': 50, # input stim onset (in steps)
            'stim_dur': 25, # input stim duration (in steps)
            'delay': 75, # delay b/w the two stimuli (in steps)
            'DeltaT': 1, # sampling rate
            'taus': args.decay_taus, # decay time-constants (in steps)
            'task': args.task.lower(), # task name
            }
elif args.task.lower() == 'mante':
    # Sensory integration task
    settings = {
            'T': 500, # trial duration (in steps)
            'stim_on': 50, # input stim onset (in steps)
            'stim_dur': 200, # input stim duration (in steps)
            'DeltaT': 1, # sampling rate
            'taus': args.decay_taus, # decay time-constants (in steps)
            'task': args.task.lower(), # task name
            }

# Define the name for the saved file (removed the time component in the original code)
if len(settings['taus']) > 1:
    format_values = args.task.lower(), N, settings['taus'][0], settings['taus'][1], training_params['activation'], args.n_layers
    fname = 'Task_{}_N_{}_Taus_{}_{}_Act_{}_{}_layers.mat'.format(*format_values)
    plot_name = 'Task_{}_N_{}_Taus_{}_{}_Act_{}_{}_layers.png'.format(*format_values)
elif len(settings['taus']) == 1:
    format_values = args.task.lower(), N, settings['taus'][0], training_params['activation']
    fname = 'Task_{}_N_{}_Tau_{}_Act_{}_{}_layers.mat'.format(*format_values)
    plot_name = 'Task_{}_N_{}_Taus_{}_Act_{}_{}_layers.png'.format(*format_values)
    
'''
Define a helper function for plotting
'''
def plotting(eval_perf_over_time, eval_loss_over_time, t_x, eval_target, eval_o):
    plt.clf()

    # Plot Evaluation Performance
    plt.subplot(411)
    plt.plot(eval_perf_over_time)
    plt.title('Evaluation Performance')
    plt.xlabel('Trial')
    plt.ylabel('Performance')

    # Plot Evaluation Loss
    plt.subplot(412)
    plt.plot(eval_loss_over_time)
    plt.title('Evaluation Loss')
    plt.xlabel('Trial')
    plt.ylabel('Loss')

    # Convert t_x to NumPy array for easier manipulation
    num_layers = len(t_x)  # Number of layers
    t_x_array = [np.array(layer_x) for layer_x in t_x]  # Convert each layer's data to a NumPy array

    # Plot Neural Activity for Each Layer
    for layer_idx in range(num_layers):
        plt.subplot(4, num_layers, 3 + layer_idx + 1)  # Create subplots dynamically
        layer_x_array = t_x_array[layer_idx]  # Extract data for the current layer
        
        # Plot synaptic activity for all neurons in the current layer over time
        for neuron_idx in range(layer_x_array.shape[1]):
            plt.plot(layer_x_array[:, neuron_idx])
        
        plt.title(f'Neural Activity (Layer {layer_idx + 1})')
        plt.xlabel('Time')
        plt.ylabel('Activity')

    # Overlay z and o in the same plot
    plt.subplot(414)
    plt.plot(eval_target, label='Target Signal (z)', linestyle='--')
    plt.plot(np.squeeze(eval_o), label='Output Signal (o)')
    plt.title('Overlay of Target and Output Signals')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.pause(0.01)
    plt.draw()

'''
Train the model
'''
if args.mode.lower() == 'train':
    '''
    Initialize the input and output weight matrices
    '''
    # Go-Nogo task
    if args.task.lower() == 'go-nogo':
        w_in = np.float32(np.random.randn(N, 1))
        w_out = np.float32(np.random.randn(1, N)/100)

    # XOR task
    elif args.task.lower() == 'xor':
        w_in = np.float32(np.random.randn(N, 2))
        w_out = np.float32(np.random.randn(1, N)/100)

    # Sensory integration task
    elif args.task.lower() == 'mante':
        w_in = np.float32(np.random.randn(N, 4))
        w_out = np.float32(np.random.randn(1, N)/100)

    '''
    Initialize the continuous rate model
    '''
    P_inh = args.P_inh # inhibitory neuron proportion
    P_rec = args.P_rec # initial connectivity probability (i.e. sparsity degree)
    P_inter = args.P_inter # initial inter connectivity probability
    print('P_rec set to ' + str(P_rec))
    print('P_inter set to ' + str(P_inter))

    w_dist = 'gaus' # recurrent weight distribution (Gaussian or Gamma)
    num_layers = args.n_layers # number of layers 
    net = FR_RNN_dale(N, P_inh, P_rec, P_inter, w_in, som_N, w_dist, args.gain, args.apply_dale, w_out, num_layers=num_layers)
    print(f'Intialized the {str(num_layers)}-layer network...')

    '''
    Construct the TF graph for training
    '''
    input_node, z, x, r, o, w, w_in, w_inter, inter_masks, m, som_m, w_out, b_out, taus = construct_tf(net, settings, training_params)
    print('Constructed the TF graph...')

    # Loss function and optimizer
    loss, training_op = loss_op(o, z, training_params)

    '''
    Start the TF session and train the network
    '''
    sess = tf.Session(config=tf.ConfigProto(gpu_options=set_gpu(args.gpu, args.gpu_frac)))
    init = tf.global_variables_initializer()

    start_time = time.time()
    with tf.Session() as sess:
        print('Training started...')
        init.run()
        training_success = False

        if args.task.lower() == 'go-nogo':
            # Go-NoGo task
            u, label = generate_input_stim_go_nogo(settings)
            target = generate_target_continuous_go_nogo(settings, label)
            x0, r0, w0, w_in0, taus_gaus0 = \
                    sess.run([x, r, w, w_in, taus], feed_dict={input_node: u, z: target})

        elif args.task.lower() == 'xor':
            # XOR task
            u, label = generate_input_stim_xor(settings)
            target = generate_target_continuous_xor(settings, label)
            x0, r0, w0, w_in0, taus_gaus0 = \
                    sess.run([x, r, w, w_in, taus], feed_dict={input_node: u, z: target})

        elif args.task.lower() == 'mante':
            # Sensory integration task
            u, label = generate_input_stim_mante(settings)
            target = generate_target_continuous_mante(settings, label)
            x0, r0, w0, w_in0, taus_gaus0 = \
                    sess.run([x, r, w, w_in, taus], feed_dict={input_node: u, z: target})

        # For storing all the loss vals
        losses = np.zeros((args.n_trials,))

        # Initialize lists to store performance and loss values over time
        eval_perf_over_time = []
        eval_loss_over_time = []

        for tr in range(args.n_trials):
            # Generate a task-specific input signal
            if args.task.lower() == 'go-nogo':
                u, label = generate_input_stim_go_nogo(settings)
                target = generate_target_continuous_go_nogo(settings, label)
            elif args.task.lower() == 'xor':
                u, label = generate_input_stim_xor(settings)
                target = generate_target_continuous_xor(settings, label)
            elif args.task.lower() == 'mante':
                u, label = generate_input_stim_mante(settings)
                target = generate_target_continuous_mante(settings, label)

            print("Trial " + str(tr) + ': ' + str(label))

            # Train using backprop
            _, t_loss, t_w, t_o, t_w_out, t_x, t_r, t_m, t_som_m, t_w_in, t_w_inter, t_b_out, t_taus_gaus = \
                sess.run([training_op, loss, w, o, w_out, x, r, m, som_m, w_in, w_inter, b_out, taus],
                        feed_dict={input_node: u, z: target})


            print('Loss: ', t_loss)
            losses[tr] = t_loss

            '''
            Evaluate the model and determine if the training termination criteria are met
            '''
            # Go-NoGo task
            if args.task.lower() == 'go-nogo':
                resp_onset = settings['stim_on'] + settings['stim_dur']
                if (tr-1)%training_params['eval_freq'] == 0:
                    eval_perf = np.zeros((1, training_params['eval_tr']))
                    eval_losses = np.zeros((1, training_params['eval_tr']))
                    eval_os = np.zeros((training_params['eval_tr'], settings['T']-1))
                    eval_labels = np.zeros((training_params['eval_tr'], ))
                    for ii in range(eval_perf.shape[-1]):
                        eval_u, eval_label = generate_input_stim_go_nogo(settings)
                        eval_target = generate_target_continuous_go_nogo(settings, eval_label)
                        eval_o, eval_l = sess.run([o, loss], feed_dict = \
                                {input_node: eval_u, z: eval_target})
                        eval_losses[0, ii] = eval_l
                        eval_os[ii, :] = eval_o
                        eval_labels[ii, ] = eval_label
                        if eval_label == 1:
                            if np.max(eval_o[resp_onset:]) > training_params['eval_amp_threh']:
                                eval_perf[0, ii] = 1
                        else:
                            if np.max(np.abs(eval_o[resp_onset:])) < 0.3:
                                eval_perf[0, ii] = 1

                    eval_perf_mean = np.nanmean(eval_perf, 1)
                    eval_loss_mean = np.nanmean(eval_losses, 1)
                    print("Perf: %.2f, Loss: %.2f"%(eval_perf_mean, eval_loss_mean))

                    if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.95 and tr > 1500:
                        # For this task, the minimum number of trials required is set to 1500 to 
                        # ensure that the trained rate model is stable.
                        training_success = True
                        break

            # XOR task
            elif args.task.lower() == 'xor':
                if (tr-1)%training_params['eval_freq'] == 0:
                    eval_perf = np.zeros((1, training_params['eval_tr']))
                    eval_losses = np.zeros((1, training_params['eval_tr']))
                    eval_os = np.zeros((training_params['eval_tr'], settings['T']-1))
                    eval_labels = []
                    for ii in range(eval_perf.shape[-1]):
                        eval_u, eval_label = generate_input_stim_xor(settings)
                        eval_target = generate_target_continuous_xor(settings, eval_label)
                        eval_o, eval_l = sess.run([o, loss], feed_dict = \
                                {input_node: eval_u, z: eval_target})
                        eval_losses[0, ii] = eval_l
                        eval_os[ii, :] = eval_o
                        eval_labels.append(eval_label)
                        # Adjust the decision period from the original code
                        task_end_T = settings['stim_on']+2*settings['stim_dur'] + settings['delay']
                        if eval_label == 'same':
                            if np.max(eval_o[task_end_T:]) > training_params['eval_amp_threh']:
                                eval_perf[0, ii] = 1
                        else:
                            if np.min(eval_o[task_end_T:]) < -training_params['eval_amp_threh']:
                                eval_perf[0, ii] = 1

                    eval_perf_mean = np.nanmean(eval_perf, 1)
                    eval_loss_mean = np.nanmean(eval_losses, 1)
                    print("Perf: %.2f, Loss: %.2f"%(eval_perf_mean, eval_loss_mean))

                    eval_perf_over_time.append(eval_perf_mean)
                    eval_loss_over_time.append(eval_loss_mean)

                    # Plotting section
                    plotting(eval_perf_over_time, eval_loss_over_time, t_x, eval_target, eval_o)

                    if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.95:
                        training_success = True
                        break
                
    
            # Sensory integration task
            elif args.task.lower() == 'mante':
                if (tr-1)%training_params['eval_freq'] == 0:
                    eval_perf = np.zeros((1, training_params['eval_tr']))
                    eval_losses = np.zeros((1, training_params['eval_tr']))
                    eval_os = np.zeros((training_params['eval_tr'], settings['T']-1))
                    eval_labels = np.zeros((training_params['eval_tr'], ))
                    for ii in range(eval_perf.shape[-1]):
                        eval_u, eval_label = generate_input_stim_mante(settings)
                        eval_target = generate_target_continuous_mante(settings, eval_label)
                        eval_o, eval_l = sess.run([o, loss], feed_dict = \
                                {input_node: eval_u, z: eval_target})
                        eval_losses[0, ii] = eval_l
                        eval_os[ii, :] = eval_o
                        eval_labels[ii, ] = eval_label
                        if eval_label == 1:
                            if np.max(eval_o[-200:]) > training_params['eval_amp_threh']:
                                eval_perf[0, ii] = 1
                        else:
                            if np.min(eval_o[-200:]) < -training_params['eval_amp_threh']:
                                eval_perf[0, ii] = 1

                    eval_perf_mean = np.nanmean(eval_perf, 1)
                    eval_loss_mean = np.nanmean(eval_losses, 1)
                    print("Perf: %.2f, Loss: %.2f"%(eval_perf_mean, eval_loss_mean))

                    if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.95:
                        training_success = True
                        break

        print("\nFinished training")
        end_time = time.time()
        print("Training time: {} seconds".format(end_time - start_time))

        # Save the training plot under `plots` directory
        plt.tight_layout()
        plot_out_dir = out_dir.replace("models", "plots")
        if not os.path.exists(plot_out_dir):
            os.makedirs(plot_out_dir, exist_ok=True)  # Ensure the directory exists
        plt.savefig(os.path.join(plot_out_dir, plot_name))
        plt.close()

        # Save the trained params in a .mat file
        var = {}
        var['x0'] = x0
        var['r0'] = r0
        var['w0'] = w0
        var['taus_gaus0'] = taus_gaus0
        var['w_in0'] = w_in0
        var['u'] = u
        var['o'] = t_o
        var['w'] = t_w
        var['x'] = t_x
        var['target'] = target
        var['w_out'] = t_w_out
        var['r'] = t_r
        var['m'] = t_m
        var['som_m'] = t_som_m
        var['N'] = N
        var['exc'] = net.exc
        var['inh'] = net.inh
        var['w_in'] = t_w_in
        var['W_inter'] = t_w_inter
        var['b_out'] = t_b_out
        var['som_N'] = som_N
        var['losses'] = losses
        var['taus'] = settings['taus']
        var['eval_perf_mean'] = eval_perf_mean
        var['eval_loss_mean'] = eval_loss_mean
        var['eval_os'] = eval_os
        var['eval_labels'] = eval_labels
        var['taus_gaus'] = t_taus_gaus
        var['tr'] = tr
        var['activation'] = training_params['activation']
        scipy.io.savemat(os.path.join(out_dir, fname), var)


'''
Evaluate the trained model
This part need to be updated after eval_tf function is updated! 
'''
# if args.mode.lower() == 'eval':
#     model_dir = os.path.join(out_dir, fname)

#     # Make sure the trained model has been saved as .mat file in the output directory
#     if os.path.exists(model_dir):
#         print(f"Model file {model_dir} found.")
#         x, r, o = eval_tf(model_dir, settings, np.zeros((12, settings['T'])))
#     else: 
#         raise FileNotFoundError(f"Model file {model_dir} not found. Please ensure the .mat file is in the correct output directory.")