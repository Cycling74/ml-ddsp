# ml-ddsp

A collection of differentiable Max objects based on DDSP using the min-devkit.

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
**Note: this repository has only been tested on Mac.**

1. Clone the repository **into Max's Packages folder**. If you clone it elsewhere you will need to make an alias to it in your Packages folder.
   The *Packages* folder can be found inside of your *Max 8* folder which is inside of your user's *Documents* folder.
   Make sure you clone recursively so that all sub-modules are properly initiated : `git clone <your repository> --recursive`
2. Download [LibTorch C++](https://pytorch.org/get-started/locally/), unzip and move the folder into the *source* folder.
2. In the Terminal or Console app of your choice, change directories (cd) into the min-starter folder you cloned/installed in step 0.
3. `mkdir build` to create a folder with your various build files
4. `cd build` to put yourself into that folder
5. Now you can generate the projects for your choosen build environment:

### Mac 

Run `cmake -G Xcode ..`

Next run `cmake --build .` or open the Xcode project from this "build" folder and use the GUI.

Note: you can add the `-j4` option where "4" is the number of cores to use.  This can help to speed up your builds, though sometimes the error output is interleaved in such a way as to make troubleshooting more difficult.

### Windows

You can run `cmake --help` to get a list of the options available.  Assuming some version of Visual Studio 2019, the commands to generate the projects will look like this: 

`cmake -G "Visual Studio 16 2019" ..`

Or using Visual Studio 2017 it will look like this:

`cmake -G "Visual Studio 15 2017 Win64" ..`

Having generated the projects, you can now build by opening the .sln file in the build folder with the Visual Studio app (just double-click the .sln file) or you can build on the command line like this:

`cmake --build . --config Release`


## Real-time Usage

Example patches which use the DDSP Max objects are located in the *patches* folder. The *ddsp_decoder_controls_tilde_example.maxpat* shows how to combine the *decoder_controls~*, *harmonic_oscillator~* and *filtered_noise~* MC objects for latent space exploration and timbre transfer. Two pre-trained models that are compatible with the *decoder_controls~* object are available in the according *models/decoder_controls_models* folder, which were trained on violin and saxophone data using the [URMP dataset](http://www2.ece.rochester.edu/projects/air/projects/URMP.html). In order to load a model and perform inference, the absolute path to the model needs to be set as argument in *decoder_controls~*.


## Training

*to be added*


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
