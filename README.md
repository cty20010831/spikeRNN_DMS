# Functional Spiking RNNs

This forked repository is used to build a recurrent neural network following Dale's principle for the Delayed Match to Sample (DMS) task (see this [paper](https://www.sciencedirect.com/science/article/pii/S0896627320300611) for task instruction). 

## Requirements

### Continuous Rate RNN
The code for constructing and training **continuous-variable rate recurrent neural network (RNN) models** is implemented in Python (tested in Python 3.6.9). The code also requires TensorFlow (tested in TensorFlow 1.5.0 and 1.10.0). 

- tensorflow 1.5.0 or 1.10.0
- tensorflow-gpu 1.5.0 or 1.10.0 (if a GPU card is available, this can speed up training time for large models)
- numpy 1.16.4
- scipy 1.3.1

### Spiking RNN
The code for constructing **spiking RNN models** is implemented in MATLAB (tested in R2016a and R2016b). The code implements leaky integrate-and-fire (LIF) networks, and is a modified version of the code developed by [Nicola et al., 2016](https://senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=190565&file=/NicolaClopath2017/#tabs-1).

### Building the virtual environment
Here, I created the virtual environment in [Python 3.6.8](https://www.python.org/downloads/release/python-368/) since I cannot find Mac `pkg` for Python 3.6.9.
```bash
# Install virtualenv
python3.6 -m virtualenv venv

# Create virtual environment (here, I used python 3.7)
python3.6 -m virtualenv venv

# Activate the virtual environment
source venv/bin/activate

# Install required packages (from the `requirements.txt`)
pip install -r requirements.txt
```

## Usage
A rate RNN model is trained first, and the trained model is then mapped to a LIF spiking RNN. The code for training rate models is located in `rate/`, while the code for mapping and constructing LIF models is in `spiking/`.

### Training Continuous Rate RNN

The main file (`rate/main.py`) takes the following input arguments:

- `gpu` (optional; default `0`): specifies which gpu to use (applicable for a system with a GPU)
- `gpu_frac` (optional; default `0.4`): fraction of available vRAM to allocate
- `n_trials` (required; default `200`): maximum number of trials for training
- `mode` (required; default `train`): either `train` or `eval` (case *insensitive*)
- `output_dir` (required): output directory path where the trained model will be saved. The model will be saved under `<output_dir>/models/<task>`.
- `N` (required): RNN size (i.e. total number of units)
- `gain` (optional; default `1.5`): gain term for the initial connectivity weights
- `P_inh` (optional; default `0.20`): *proportion* of the *N* units that will be inhibitory
- `som_N` (optional; default `0`): *number* of units that will be "somatostatin-expressing" units. Refer to the preprint for more info.
- `apply_dale` (required; default `true`): apply Dale's principle
- `task` (required): task the rate RNN will be trained to perform. Available options are `go-nogo` (Go-NoGo task), `mante` (context-dependent sensory integration task), or `xor` (temporal exclusive OR task). 
- `act` (required; default `sigmoid`): activation function. Available options are `sigmoid`, `clipped_relu`, or `softplus`.
- `loss_fn` (required; default `l2`): loss function (L1, L2, etc...). Case *insensitive*.
- `decay_taus` (required): synaptic decay time-constants (either `a b` for min a and max b or `a` for homogeneous time-constants). Multiply these numbers by 5 to convert to ms. For example, `4 20` means the min and max time constants are 20 ms and 100 ms, respectively.

The following example trains a rate model to perform the Go-NoGo task. The network contains 200 units (20% of the units are inhibitory). The training will stop if the termination criteria are met within the first 5000 trials (`n_trials`). No additional connectivity constraints are used (i.e. `som_N` is set to 0). The trained model will be saved as a MATLAB-formatted file (`.mat`) in the output directory (`../models/go-nogo/P_rec_0.20_Taus_4.0_20.0`).

```bash
# Change the directory
cd rate/

# Run the sample Go-NoGo task
python main.py --gpu 0 --gpu_frac 0.20 --n_trials 1000 --mode train --output_dir ../ --N 80 --P_inh 0.20 --som_N 0 --apply_dale True --gain 1.5 --task xor --act sigmoid --loss_fn l2 --decay_taus 4 20

# Change back to main directory
cd ..
```

The name of the output `.mat` file conforms to the following convention:

```
Task_<Task Name>_N_<N>_Taus_<min_tau>_<max_tau>_Act_<act>_<YYYY_MM_DD_TIME>.mat
```

### Mapping and Constructing LIF RNN
Trained rate RNNs are used to construct LIF RNNs. The mapping and LIF simulations are performed in MATLAB.
Given a trained rate model, the first step is to perform the grid search to determine the optimal scaling factor (lambda). This is done by `lambdad_grid_search.m`. Once the optimal scaling factor is determined, a LIF RNN can be constructed using the function `LIF_network_fnc.m`. All the required functions/scripts are located in `spiking/`.

An example script for evaluating a Go-NoGo LIF network (`eval_go_nogo.m`) is also included. The script constructs a LIF RNN trained to perform the Go-NoGo task and plots network responses. The script can be modified to evaluate models trained to perform other tasks.

## DMS Task in Herbert's Project
DMS, also known as the XOR task, is fortunarely included as a task supported by the `FR_RNN_dale` class. Hence, the most important thing is to adjust current task configuration (i.e., task-specific parameters for xor task in `rate/main.py`) to the experimental design settings in Herbert's project (see this [file](https://github.com/cty20010831/Bio_Neuro_Net_Premotor_Circuit/blob/main/original_code/softmax_recur_updated.py) for reference on Herbert's experimental design), including trial duration, input stim onset, input stim duration, delay between the two stimuli (in steps), and sampling rate. In addition, some changes with respect to the decision period following the end of presenting the second stimulus have been made for the `generate_target_continuous_xor` function in `rate/model.py` and the evalutation part in `rate/main.py`. 

```bash
# Change the directory
cd rate/

# Train the RNN model for the DMS task
python main.py --gpu 0 --gpu_frac 0.20 --n_trials 5000 --mode train --output_dir ../ --N 200 --P_inh 0.20 --som_N 0 --apply_dale True --gain 1.5 --task go-nogo --act clipped_relu --loss_fn l2 --decay_taus 5

# Change back to main directory
cd ..
```

Another thing is to include more functionality for both the ongoning progress (report) during model fitting and the evaluation of the trained model. For the progress (report) during model fitting, I added subplots of model performance, loss, neural activity, and predicted outcome for each 100 trials. 

The final thing is to examine whether the `FR_RNN_dale` class can be further developed into multi-layer RNN following Dale's principle. 

## Citation
```
@article{Kim_2019,
    Author = {Kim, Robert and Li, Yinghao and Sejnowski, Terrence J.},
    Doi = {10.1073/pnas.1905926116},
    Journal = {Proceedings of the National Academy of Sciences},
    Number = {45},
    Pages = {22811--22820},
    Publisher = {National Academy of Sciences},
    Title = {Simple framework for constructing functional spiking recurrent neural networks},
    Volume = {116},
    Year = {2019}}
```

