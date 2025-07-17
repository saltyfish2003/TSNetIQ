### TSNetIQ : A model for single-source 1D DOA Estimation

#### 1.Introduction of files

create_wav.py -> create 5s, 2KHz sine wav used as the audio source

create_regression_trainset.py -> create regression trainsets

IQ_CNN_2d.py -> CNN model as a baseline

ResNetIQ_2d.py -> ResNetIQ model as a baseline (Reproduced from Deep Learning-Based DOA Estimation)

IQ_SE_CNN_TR_2d.py -> TSNetIQ model

#### 2.Training Process

Create 5s, 2KHz sine wav used as the audio source, create regression trainsets,training 3 models...

#### 3.Params

The default number of array elements in the provided code is 4, if you change the array elements(N_MIC) in create_regression_trainset.py, you should change the kernel_size of the first conv.

##### Adjustable Params

###### create_regression_trainset.py(The default value is used in the code.)

N_MIC

d0

SNR

room_dim(The room simulated in this paper is a 3D anechoic room,actually change this makes no sense)

mic_center

source_position(room.add_source(source_position, signal=signal))

###### IQ_SE_CNN_TR_2d.py(The default value is used in the code.)

embed_dim

num_heads

ff_dim

num_layers

max_len

delta

epochs

lr

weight_decay

factor

patience

min_lr

batch_size

num_workers

#### 4.Computing Platform

AutoDL GeForce RTX4090 GPU

#### 5.Disclaimer

Due to differences in computational resources, simulation parameter settings, and model evaluation criteria, discrepancies in the results are to be expected and considered normal.