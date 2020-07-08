# pdetclassifier

### Gravitational-wave detectability with neural-network classifiers

> We present a novel machine-learning approach to estimate  selection biases in gravitational-wave observations. Using techniques similar to those commonly employed in image classification and pattern recognition, we train a series of neural-network classifiers to predict the LIGO/Virgo detectability of gravitational-wave signals from compact-binary mergers. We include the effect of spin precession, higher-order modes, and multiple detectors and show that their omission, as it is common in large population studies, tends to overestimate the inferred merger rate. Although here we train our classifiers using a simple signal-to-noise ratio threshold, our approach is ready to be used in conjunction with full pipeline injections, thus paving the way toward including empirical distributions of  astrophysical and noise triggers into gravitational-wave population analyses.


This repository contains models supporting [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX). We are very happy if you find this useful for your research; please cite our paper. 

For DOI pointing to this repository: 

## Data products

We provide three kind of data products:

- Our code is `pdetclassifier.py`, see below for a short description.
- Pre-trained TensorFlow neural networks are called `trained_*.h5`.
- Training/validation sample are called `sample_*.h5` and can be downloaded from the github release page.

Models were trained on samples of N=2e7 binaries. This sample is divided in two chunks of 1e7 sources each used for training and validation. The following models are described carefully in [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX). 
- trained_2e7_design_nonspinning_quadrupole_1detector.h5
- trained_2e7_design_precessing_higherordermodes_1detector.h5
- trained_2e7_design_precessing_higherordermodes_3detectors.h5
They are computed assuming LIGO/Virgo at design sensitivity. In particular, we use the `aLIGODesignSensitivityP1200087`, `AdVDesignSensitivityP1200087` noise curves from here. 

The following additional models used representative noise curves for LIGO/Virgo O1+O2, O3, and O4. The training distributions and the network setup is the same we describe in the paper. 
- trained_2e7_O1O2_precessing_higherordermodes_3detectors.h5
- trained_2e7_O3_precessing_higherordermodes_3detectors.h5
- trained_2e7_O4_precessing_higherordermodes_3detectors.h5
For O1+O3 we use the `aLIGOEarlyHighSensitivityP1200087` and `AdVEarlyHighSensitivityP1200087` noise curves from here. For O3 and O4 we use the txt file provided here.


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
```

The `binaries` object is a python dictionary with keys `['mtot','q','z','chi1x','chi1y','chi1z','chi2x','chi2y','chi2z','iota','ra','dec','psi']'


## Example 2: train your own neural network












