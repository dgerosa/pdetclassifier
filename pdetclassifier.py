'''
Gravitational-wave selection effects using neural-network classifiers
Davide Gerosa, Birmingham UK, 2020.

We are very happy if you want to use this code for your research.
Please cite our paper: arXiv:2007.06585.
'''

__author__='Davide Gerosa'
__email__='d.gerosa@bham.ac.uk'
__version__=0.2
__license__='MIT'

import sys,os,time,copy
from tqdm import tqdm

import numpy as np
import scipy.integrate

import astropy.cosmology

import pycbc.psd
import pycbc.detector
import pycbc.waveform
import pycbc.filter

import tensorflow as tf
from tensorflow import keras

import uuid
import deepdish


def sperical_to_cartesian(mag,theta,phi):
    '''
    Convert spherical to cartesian coordinates
    '''

    coordx = mag * np.cos(phi) * np.sin(theta)
    coordy = mag * np.sin(phi) * np.sin(theta)
    coordz = mag * np.cos(theta)

    return coordx,coordy,coordz


massvar = ['mtot','q','z']
spinvar = ['chi1x','chi1y','chi1z','chi2x','chi2y','chi2z']
intrvar=massvar+spinvar
extrvar = ['iota','ra','dec','psi']
train_variables=intrvar+extrvar

def lookup_limits():
    '''
    Define the limits in all variables. If you want to change this, please check generate_binaries() and pdet() as well.
    '''

    limits={
        'mtot'  : [2,1000],
        'q'     : [0.1,1],
        'z'     : [1e-4,4],
        'chi1x'  : [-1,1],
        'chi1y'  : [-1,1],
        'chi1z'  : [-1,1],
        'chi2x'  : [-1,1],
        'chi2y'  : [-1,1],
        'chi2z'  : [-1,1],
        'iota'  : [0,np.pi],
        'ra'    : [-np.pi,np.pi],
        'dec'   : [-np.pi/2,np.pi/2],
        'psi'   : [0,np.pi]
        }

    return limits


def generate_binaries(N):
    '''
    Generate a sample of N binaries. Edit here to specify a different traning/validation distribution.
    '''

    N=int(N)
    limits=lookup_limits()

    binaries={}
    binaries['N']=N

    for var in ['mtot','q','z']:
        binaries[var]=np.random.uniform(min(limits[var]),max(limits[var]),N)
    binaries['iota']= np.arccos(np.random.uniform(-1,1,N))
    binaries['ra']= np.pi*np.random.uniform(-1,1,N)
    binaries['psi']= np.pi*np.random.uniform(0,1,N)
    binaries['dec']= np.arccos(np.random.uniform(-1,1,N))- np.pi/2

    mag = np.random.uniform(0,1,N)
    theta = np.arccos(np.random.uniform(-1,1,N))
    phi = np.pi*np.random.uniform(-1,1,N)
    binaries['chi1x'],binaries['chi1y'],binaries['chi1z'] = sperical_to_cartesian(mag,theta,phi)

    mag = np.random.uniform(0,1,N)
    theta = np.arccos(np.random.uniform(-1,1,N))
    phi = np.pi*np.random.uniform(-1,1,N)
    binaries['chi2x'],binaries['chi2y'],binaries['chi2z'] = sperical_to_cartesian(mag,theta,phi)

    return binaries


def evaluate_binaries(inbinaries, approximant='IMRPhenomXPHM', noisecurve='design', SNRthreshold=12):
    '''
    Compute the SNRs of a set of binaries
    '''

    binaries = copy.deepcopy(inbinaries)

    ifos=['H1','L1','V1']
    flow=20.0
    fhigh=2048.
    geocent_time=0.
    delta_f=1/64.
    flen=int(fhigh / delta_f) + 1
    phi_c=0.

    dets=[]
    psds=[]
    for ifo in ifos:
        dets.append(pycbc.detector.Detector(ifo))


        if noisecurve=="design" or noisecurve=="Design":
            if ifo == 'V1':
               psds.append(pycbc.psd.AdVDesignSensitivityP1200087(flen, delta_f, flow) )
            else:
               psds.append(pycbc.psd.aLIGODesignSensitivityP1200087(flen, delta_f, flow) )

        elif noisecurve=="O1O2":
            if ifo == 'V1':
               psds.append(pycbc.psd.AdVEarlyHighSensitivityP1200087(flen, delta_f, flow) )
            else:
               psds.append(pycbc.psd.aLIGOEarlyHighSensitivityP1200087(flen, delta_f, flow) )

        elif noisecurve=="O3":
            if ifo == 'H1':
               psds.append(pycbc.psd.from_txt('./T2000012_aligo_O3actual_H1.txt', flen, delta_f,flow, is_asd_file=True) )
            elif ifo=='L1':
               psds.append(pycbc.psd.from_txt('./T2000012_aligo_O3actual_L1.txt', flen, delta_f,flow, is_asd_file=True) )
            elif ifo=='V1':
               psds.append(pycbc.psd.from_txt('./T2000012_avirgo_O3actual.txt', flen, delta_f,flow, is_asd_file=True) )

        elif noisecurve=="O4":
            if ifo == 'V1':
                psds.append(pycbc.psd.from_txt('./T2000012_avirgo_O4high_NEW.txt', flen, delta_f,flow, is_asd_file=True) )
            else:
                psds.append(pycbc.psd.from_txt('./T2000012_aligo_O4high.txt', flen, delta_f,flow, is_asd_file=True) )

        else:
            raise ValueError
    # Some derived quantities
    m1z = binaries['mtot']/(1+binaries['q'])
    m2z = binaries['q']*m1z
    lumdist = astropy.cosmology.Planck15.luminosity_distance(binaries['z']).value # Mpc

    binaries['snr']=[]
    for i in tqdm(range(binaries['N'])):
        # Waveform generator
        hp, hc = pycbc.waveform.get_fd_waveform(approximant = approximant,
                            mass1       = m1z[i],
                            mass2       = m2z[i],
                            spin1x      = binaries['chi1x'][i],
                            spin1y      = binaries['chi1y'][i],
                            spin1z      = binaries['chi1z'][i],
                            spin2x      = binaries['chi2x'][i],
                            spin2y      = binaries['chi2y'][i],
                            spin2z      = binaries['chi2z'][i],
                            inclination = binaries['iota'][i],
                            coa_phase   = phi_c,
                            delta_f     = delta_f,
                            f_lower     = flow,
                            distance    = lumdist[i]
                            )

        # Compute SNR for each specified detector
        snrs = []
        for det,psd in zip(dets,psds):

            f_plus, f_cross = det.antenna_pattern(binaries['ra'][i],binaries['dec'][i],binaries['psi'][i],geocent_time)
            template = f_plus * hp + f_cross * hc
            dt = det.time_delay_from_earth_center(binaries['ra'][i],binaries['dec'][i],geocent_time)
            template = template.cyclic_time_shift(dt)
            template.resize(len(hp) // 2 + 1)
            snr_opt = pycbc.filter.matched_filter(template, template,
                    psd = psd,
                    low_frequency_cutoff  = flow,
                    high_frequency_cutoff = fhigh - 0.5)
            maxsnr, _ = snr_opt.abs_max_loc()
            snrs.append(maxsnr)

        # Sum in quarature for the network SNR
        binaries['snr'].append(np.linalg.norm(snrs))

    binaries['snr']=np.array(binaries['snr'])

    # Detectability: 1 means "detected", 0 means "not detected"
    binaries['det']= np.where(binaries['snr']>SNRthreshold , 1,0 )

    return binaries

def store_binaries(filename, N, approximant='IMRPhenomXPHM', noisecurve='design', SNRthreshold=12):
    ''' Generate binaries, compute SNR, and store'''

    inbinaries = generate_binaries(N)
    outbinaries = evaluate_binaries(inbinaries, approximant, noisecurve, SNRthreshold)

    deepdish.io.save(filename,outbinaries)
    return filename

def readsample(filename='sample.h5'):
    '''
    Read a validation sample that already exists
    '''
    return deepdish.io.load(filename)

def splittwo(binaries):
    '''
    Split sample into two subsamples of equal size
    '''

    one={}
    two={}
    for k in train_variables+['snr','det']:
        one[k],two[k] = np.split(binaries[k],2)
    one['N'],two['N']= len(one['mtot']),len(two['mtot'])

    return one,two


def rescale(x,var):
    '''
    Rescale variable sample x of variable var between -1 and 1
    '''

    limits=lookup_limits()
    if var not in limits:
        raise ValueError

    return 1-2*(np.array(x)-min(limits[var]))/(max(limits[var])-min(limits[var]))


def nnet_in(binaries):
    '''
    Prepare neural network inputs.
    '''

    return np.array([rescale(binaries[k],k) for k in train_variables]).T

def nnet_out(binaries, which='detnetwork'):
    '''
    Prepare neural network outputs.
    '''

    return binaries['det']


def trainnetwork(train_binaries,test_binaries,filename='trained.h5'):

    if not os.path.isfile(filename):

        train_in  = nnet_in(train_binaries)
        train_out = nnet_out(train_binaries)
        test_in  = nnet_in(test_binaries)
        test_out = nnet_out(test_binaries)

        # Kernel initializer
        my_init = keras.initializers.glorot_uniform(seed=1)
        # Define neural network architecture
        model = keras.Sequential([
            # Input layer, do not change
            tf.keras.layers.InputLayer(input_shape=np.shape(train_in[0])),
            # Inner layers, can add/change
            keras.layers.Dense(32,  activation='tanh',kernel_initializer=my_init),
            #keras.layers.Dense(16,  activation='tanh',kernel_initializer=my_init),
            #keras.layers.Dense(8,  activation='tanh',kernel_initializer=my_init),
            # Output layer, do not change
            keras.layers.Dense(1, activation='sigmoid',kernel_initializer=my_init)])

        model.compile(
            # Optimization algorithm, specify learning rate
            optimizer=keras.optimizers.Adam(learning_rate=1e-2),
            # Loss function for a binary classifier
            loss='binary_crossentropy',
            # Diagnostic quantities
            metrics=['accuracy'])

        # Decrease the learning rate exponentially after the first 10 epochs
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.05)

        # Actual Training
        history = model.fit(
            # Training inputs
            train_in,
            # Training outputs
            train_out,
            # Evaluate test set at each epoch
            validation_data=(test_in, test_out),
            # Batch size, default is 32
            #batch_size=32,
            # Number of epochs
            epochs=150,
            # Store the model with the best validation accuracy
            callbacks = [
                # Drecrease learning rate
                tf.keras.callbacks.LearningRateScheduler(scheduler),
                # Store the model with the best validation accuracy
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=filename,
                    save_weights_only=False,
                    monitor='val_accuracy',
                    mode='max',
                    save_best_only=True),
                # Save logfiles for tensorboard
                tf.keras.callbacks.TensorBoard(log_dir="logs"+filename.split('.h5')[0], histogram_freq=1)],
            # Shuffle data at each epoch
            shuffle=True)

        # Store the last (not necessarily the best) iteration
        #model.save(filename)

    model = loadnetwork(filename)
    return model


def loadnetwork(filename,verbose=False):
    '''
    Load a trained neural network
    '''

    model = tf.keras.models.load_model(filename)
    if verbose:
        model.summary()

    return model


def testnetwork(model,binaries):
   '''
   Test network on a series of binaries
   '''
   test_in  = nnet_in(binaries)
   test_out = nnet_out(binaries)
   model.evaluate(test_in,  test_out, verbose=2)


def predictnetwork(model, binaries):
    '''
    Use a network to predict the detectability of a set of binaries.
    '''
    # Return the class (0 or 1) that is preferred
    predictions = np.squeeze((model.predict(nnet_in(binaries)) > 0.5).astype("int32"))
    return predictions


def pdet(model,binaries, Nmc = 10000):
    '''
    Numberical marginalization over the extrinsic parameters. Nmc is the nubmer of Monte Carlo samples used to estimate the integral.
    '''

    limits = lookup_limits()
    # Number of binaries
    N=binaries['N']

    # Resample the extrinsic variables from isotropic distribution
    extrinsic={}
    extrinsic['iota'] =  np.arccos(np.random.uniform(-1,1,Nmc))
    extrinsic['ra']   =  np.pi*np.random.uniform(-1,1,Nmc)
    extrinsic['psi']  =  np.pi*np.random.uniform(-1,1,Nmc)
    extrinsic['dec']  =  np.arccos(np.random.uniform(-1,1,Nmc))- np.pi/2

    # Inflate the array with the intrisinc variables
    intersection  = [value for value in intrvar if value in binaries]
    intr = np.repeat([rescale(binaries[k],k) for k in intersection], Nmc, axis=1)
    # Inflate the array with the extrinsic variables
    extr = np.reshape(np.repeat([rescale(extrinsic[k],k) for k in extrvar],N,axis=0), (len(extrvar),N*Nmc))
    # Pair
    both = np.concatenate((intr,extr)).T
    # Apply network
    predictions =  np.reshape( np.squeeze(( model.predict(both)> 0.5).astype("int32")), (N,Nmc) )
    # Approximante integral with monte carlo sum
    pdet = np.sum(predictions,axis=1)/Nmc

    return pdet



if __name__ == '__main__':

    if False:

        # Load sample
        binaries= readsample('sample_2e7_design_precessing_higherordermodes_3detectors.h5')
        # Split test/training
        train_binaries,test_binaries=splittwo(binaries)
        # Load trained network
        model = loadnetwork('trained_2e7_design_precessing_higherordermodes_3detectors.h5')
        # Evaluate performances on training sample
        testnetwork(model,train_binaries)
        # Evaluate performances on test sample
        testnetwork(model,test_binaries)
        # Predict on new sample
        newbinaries = generate_binaries(100)
        predictions = predictnetwork(model, newbinaries)
        print(predictions)
        # Regenerate the extrinsic angles and marginalize over them
        pdets = pdet(model,newbinaries, Nmc=1000)
        print(pdets)


    if False:

        # Generate and store a sample
        store_binaries('sample.h5',1e3,approximant='IMRPhenomXPHM',noisecurve='design',SNRthreshold=12)
        # Load sample
        binaries= readsample('sample.h5')
        # Split test/training
        train_binaries,test_binaries=splittwo(binaries)
        # Train a neural network
        trainnetwork(train_binaries,test_binaries,filename='trained.h5')
        # Load trained network
        model = loadnetwork('trained.h5')
        # Evaluate performances on training sample
        testnetwork(model,train_binaries)
        # Evaluate performances on test sample
        testnetwork(model,test_binaries)
        # Predict on new sample
        newbinaries = generate_binaries(100)
        predictions = predictnetwork(model, newbinaries)
        print(predictions)
        # Regenerate the extrinsic angles and marginalize over them
        pdets = pdet(model,newbinaries, Nmc=1000)
        print(pdets)



    if True:

        #
        # Initialize
        binaries={}
        N = int(1e3)
        binaries = generate_binaries(N)
        # Populate with your distribution
        binaries['mtot'] = np.random.normal(30, 3, N )
        binaries['q'] = np.random.uniform(0.1,1,N)
        binaries['z'] = np.random.normal(0.2, 0.01, N )
        binaries['chi1x'] = np.random.uniform(0, 0.1, N )
        binaries['chi1y'] = np.random.uniform(0, 0.1, N )
        binaries['chi1z'] = np.random.uniform(0, 0.1, N )
        binaries['chi2x'] = np.random.uniform(0, 0.1, N )
        binaries['chi2y'] = np.random.uniform(0, 0.1, N )
        binaries['chi2z'] = np.random.uniform(0, 0.1, N )
        # Load trained network
        model = loadnetwork('trained_2e7_design_precessing_higherordermodes_3detectors.h5')
        # Compute detectability averaged over extrinsic parameters
        pdets = pdet(model,binaries, Nmc=1000)
        print(pdets)
        # Integrate over entire population
        predictions = predictnetwork(model, binaries)
        integral = np.sum(predictions)/N
        print(integral)
