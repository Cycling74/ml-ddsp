# ml-ddsp

A collection of differentiable Max objects based on [DDSP](https://github.com/magenta/ddsp) and [DDSP PyTorch](https://github.com/acids-ircam/ddsp_pytorch) using LibTorch with min-devkit.

## Structure

* `min-package-template` is a minimal template version of the Min-DevKit package to get started following current best-practices for package creation.
* `min-api` is a folder within the devkit containing all of the support files you will need to compile an external object written in modern C++.  This folder you will include in your own package's source folder.
* `min-lib` contains building blocks, helper classes, and unit generators that may be useful in authoring  C++ code for audio, video, and data processing.


## Prerequisites

To build externals, you will need some form of compiler support on your system. 

* On the Mac this means **Xcode 10 or higher** (you can get from the App Store for free). 
* On Windows this means **Visual Studio 2017 or 2019**  (you can download a free version from Microsoft). The installer for Visual Studio 2017 offers an option to install Git, which you should choose to do.

You will also need to install a recent version of [CMake](https://cmake.org/download/) (version 3.19 or higher).


## Building
**Note: this repository has only been tested on Mac x86_64 (Intel) and M1 using Rosetta.**

1. Clone the repository **into Max's Packages folder**. If you clone it elsewhere you will need to make an alias to it in your Packages folder.
   The *Packages* folder can be found inside of your *Max 8* folder which is inside of your user's *Documents* folder.
   Make sure you clone recursively so that all sub-modules are properly initiated : `git clone <your repository> --recursive`
2. Download [LibTorch C++](https://pytorch.org/get-started/locally/) (tested with the following version of LibTorch: Stable 1.12.0 > Mac > LibTorch > C++ > Default), unzip and move the folder into the *source* folder. In the Terminal or Console app of your choice, change directories (cd) into *source/libtorch/lib* and run `xattr -d -r com.apple.quarantine .` to avoid the *"\*.dylib cannot be opened because the developer cannot be verified"* error.
3. Change directories (cd) into the *ml-ddsp* root folder.
4. Run `mkdir build` to create a folder with your various build files.
5. Run `cd build` to put yourself into that folder.
6. Now you can generate the projects for your choosen build environment:

### Mac 

*only tested on Mac x86_64 (Intel) and M1 using Rosetta*

Run `cmake -G Xcode ..`

Next run `cmake --build .` or open the Xcode project from this "build" folder and use the GUI.

Note: you can add the `-j4` option where "4" is the number of cores to use.  This can help to speed up your builds, though sometimes the error output is interleaved in such a way as to make troubleshooting more difficult.

### Windows

*not tested yet*

You can run `cmake --help` to get a list of the options available.  Assuming some version of Visual Studio 2019, the commands to generate the projects will look like this:

`cmake -G "Visual Studio 16 2019" ..`

Or using Visual Studio 2017 it will look like this:

`cmake -G "Visual Studio 15 2017 Win64" ..`

Having generated the projects, you can now build by opening the .sln file in the build folder with the Visual Studio app (just double-click the .sln file) or you can build on the command line like this:

`cmake --build . --config Release`

### Third-party Dependencies

Apart from LibTorch, there are further third-party Max packages to be installed in order to run the example patches:

* `sigmund~` is used for pitch and loudness tracking, and can be retrieved [here](https://github.com/v7b1/sigmund_64bit-version).
* `hirt.convolver~` from the [HISSTools Impulse Response Toolbox](https://github.com/HISSTools/HISSTools_Impulse_Response_Toolbox) is used for convolution and parametric reverb and can be installed using the package manager within Max.


## Overview

The package contains the following objects, which are all differentiable and can be trained end-to-end:

### Autoencoder Objects

* `ddsp.audio-decoder~`, basic neural decoder that takes pitch and loudness, runs inference on a pre-trained model and outputs an audio signal
* `ddsp.control-decoder~`, multichannel neural decoder that takes pitch and loudness, runs inference on a pre-trained model and outputs control parameters for additive + subtractive synthesis, i.e. fundamental frequency, harmonic amplitudes and filter magnitudes
* `ddsp.latent-decoder~`, multichannel neural decoder that takes pitch and loudness, and latent parameters based on MFCCs, runs inference on a pre-trained model and outputs control parameters for additive + subtractive synthesis, i.e. fundamental frequency, harmonic amplitudes and filter magnitudes

### Synthesis Objects

* `ddsp.mc-harmonic-oscillator~`, multichannel additive synthesizer that takes fundamental frequency and harmonic amplitudes and outputs harmonic signals
* `ddsp.harmonic-oscillator~`, additive synthesizer that takes fundamental frequency and harmonic amplitudes and outputs a mono audio signal
* `ddsp.filtered-noise~`, subtractive synthesizer that takes filter magnitudes and outputs a mono audio signal

### Reverb

The `hirt.convolver~` is used to convolve the output signal with learned impulse responses which can be further processed and altered (decay, size, pre-delay, gain, eq) within the tool.

### Pre-trained Models

Download pre-trained models here: [Download](https://ml-ddsp-resources.s3.amazonaws.com/models.zip) 

Each model contains three files:
* `model.ts`, model in the torschscript file format, containing the architecture and weights of the neural network
* `impulse.wav`, learned impulse response for (de-)reverberation
* `config.yaml`, summary of all model parameters and additional configuration

The following models are currently available:

* cello
* doublebass
* flute
* saxophone
* trumpet
* violin

Compatible versions of these models for the audio, control and latent decoder can be found in the respective subfolders:

* `models/audio_decoder`
* `models/control_decoder`
* `models/latent_decoder`

These pre-trained models have been trained on monophonic recordings of acoustic instruments in the [URMP dataset](http://www2.ece.rochester.edu/projects/air/projects/URMP.html). Further explorations of trainings on non-acoustic sound sources will be explored and the collection of pre-trained models will be gradually extended.

### Example Patches

Example patches which use the differentiable Max objects are located in the `patches/` folder. 
In order to load a model and perform inference, a `load` message has to be sent to the decoder object which opens the file browser to load a torchscript model (`*.ts` file). Make sure to load a model that has been trained on the according decoder architecture.

**Note: the buffer size / signal vector size should not exceed the maximum of 1024. A buffer size between 128 and 512 has been found to work best.**

* The `audio_decoder_example.maxpat` shows how to combine the basic *ddsp.audio-decoder~* with reverb for parameter space exploration, timbre transfer and MIDI control. 
* The `control_decoder_example.maxpat` shows how to combine the *ddsp.control-decoder~*, *ddsp.mc-harmonic_oscillator~* and *ddsp.filtered_noise~* multichannel objects with reverb for parameter space exploration, timbre transfer and MIDI control.
* The `latent_decoder_example.maxpat` shows how to combine the *ddsp.latent-decoder~*, *ddsp.mc-harmonic_oscillator~* and *ddsp.filtered_noise~* multichannel objects with reverb for parameter space and latent space exploration.


## Training

*currently in a separate repository*

For training of custom models, follow the instructions in the [training repository](https://github.com/rotterbein/ml-ddsp-training/tree/mfcc-autoencoder).


## Unit Testing

On the command line you can run all unit tests using Cmake:

* on debug builds: `ctest -C Debug .`
* on release builds: `ctest -C Release .`

Or you can run an individual test, which is simply a command line program:

* `cd ..`
* `cd tests`
* mac example: `./test_dcblocker_tilde -s`
* win example: `test_dcblocker_tilde.exe -s`

Or you can run them with your IDE's debugger by selecting the "RUN_TESTS" target.


## Continuous Integration

Continuous Integration (CI) is a process by which each code check-in is verified by an automated build and automated tests to allow developers to detect problems early and distribute software easily.

The min-starter project models CI using [Github Actions](https://docs.github.com/en/actions).


## Additional Documentation

* [Min Documentation Hub](http://cycling74.github.io/min-devkit/) For guides, references, and resources
* [Min Wiki](https://github.com/Cycling74/min-devkit/wiki) For additional topics, advanced configuration, and user submissions
* [How to Create a New Object](./HowTo-NewObject.md)
* [How to Update the underlying Max API](./HowTo-UpdateTheAPI.md)


## Contributors / Acknowledgements

* See the [GitHub Contributor Graph](https://github.com/Cycling74/min-api/graphs/contributors) for the API

## Support

For support, please use the developer forums at:
http://cycling74.com/forums/
