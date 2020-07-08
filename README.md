# pdetclassifier

### Gravitational-wave detectability with neural-network classifiers

> We present a novel machine-learning approach to estimate  selection biases in gravitational-wave observations. Using techniques similar to those commonly employed in image classification and pattern recognition, we train a series of neural-network classifiers to predict the LIGO/Virgo detectability of gravitational-wave signals from compact-binary mergers. We include the effect of spin precession, higher-order modes, and multiple detectors and show that their omission, as it is common in large population studies, tends to overestimate the inferred merger rate. Although here we train our classifiers using a simple signal-to-noise ratio threshold, our approach is ready to be used in conjunction with full pipeline injections, thus paving the way toward including empirical distributions of  astrophysical and noise triggers into gravitational-wave population analyses.


This repository contains models supporting [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX). We are very happy if you find this useful for your research; please cite our paper. 

For a DOI pointing to this repository: ZENODO BADGE

## Data products

We provide three kind of data products:

- Our code: `pdetclassifier.py`. See below for a short description.
- Pre-trained TensorFlow neural networks: `trained_*.h5`.
- Training/validation samples: `sample_*.h5`.These can be downloaded from the [github release page][https://github.com/dgerosa/pdetclassifier/releases].

Models were trained on samples of N=2e7 binaries. This sample is divided in two chunks of 1e7 sources each used for training and validation. 

The following models are those described in [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX). 
- `trained_2e7_design_nonspinning_quadrupole_1detector.h5`
- `trained_2e7_design_precessing_higherordermodes_1detector.h5`
- `trained_2e7_design_precessing_higherordermodes_3detectors.h5`
They are computed assuming LIGO/Virgo noise curves `aLIGODesignSensitivityP1200087`, `AdVDesignSensitivityP1200087` from `lal`. 

The following additional models use representative noise curves for LIGO/Virgo O1+O2, O3, and O4. The training distributions and the network setup is the same as described in the paper. 
- `trained_2e7_O1O2_precessing_higherordermodes_3detectors.h5`
- `trained_2e7_O3_precessing_higherordermodes_3detectors.h5`
- `trained_2e7_O4_precessing_higherordermodes_3detectors.h5`
For O1+O3 we use the `aLIGOEarlyHighSensitivityP1200087` and `AdVEarlyHighSensitivityP1200087` noise curves from `lal`. For O3 and O4 we use the txt files from [LIGO-T2000012][https://dcc.ligo.org/LIGO-T2000012/public].


## Code and examples

First, install the following python packages: `tensorflow`, `astropy`, `lalsuite`, `pycbc`, `tqdm`, and `deepdish`.

*Note*: if used as it is, the `pdetclassifier.py` script assumes precessing systems, higher-order modes, and a three-detector network. If you want to do something different, you'll need to hack it a little bit.  

## Example 1: use a precomputed model

Here is a code snippet to use a precomputed model:

```
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
newbinaries = generate_binaries(10)
predictions = predictnetwork(model, newbinaries)
print(predictions)
# Regenerate the extrinsic angles and marginalize over them
pdets = pdet(newbinaries, Nmc=1000)
print(pdets)
```

The `binaries` object is a python dictionary with keys 
- `mtot`: detector-frame total mass
- `q`: mass ratio
- `z`: redshift
- `chi1x`,`chi1y`,`chi1z`: dimensionless spin components of the primary
- `chi2x`,`chi2y`,`chi2z`: dimensionless spin components of the secondary
- `iota`: inclidation
- `ra`,`dec`: sky location
- `psi`: polarization.
- `snr`: the SNR
- `det`: detectability, equal to 1 if detectable or 0 if not detectable.
The frame of the spins is defined such that z is along L at 20 Hz (as in `lal`).

The `predictions` one gets at the end is a list of 0s and 1s, encoding the predicted detectability. One can then marginalize over the extrinsic angles to compute the detection probability pdet (by default the `pdet` function assumes isotropic inclination, sky-location and polarization).


## Example 2: train your own neural network

Here is an example where we generate a small training set of 1000 binaries, train a neural network, and evaluate the performances. 

```
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
newbinaries = generate_binaries(10)
predictions = predictnetwork(model, newbinaries)
print(predictions)
# Regenerate the extrinsic angles and marginalize over them
pdets = pdet(newbinaries, Nmc=1000)
print(pdets)
```









