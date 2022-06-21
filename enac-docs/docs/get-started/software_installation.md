---
layout: default
title: 1. Software Installation
parent: Get Started
nav_order: 1
permalink: /docs/get-started/software_installation
---

# Installation of required software 

This tutorial has the purpose of informing the reader how to setup the software needed to use the `ICCP` to its full extent, including Python packages, EnergyPlus, EnergyPlusFMU and C++ compilers.

## Disclaimer

This tutorial has been tested on the following setup:
- Windows 10
- Python 3.8.12
- conda 4.11.0
- EnergyPlus 9.4.0-998c4b761e
- EnergyPlusToFMU 3.1.0


## Python packages

- Install [Anaconda](https://www.anaconda.com/).
- Open Anaconda prompt (you may type in the search bar to find it)
- For good practices, one should create a conda environment for each different Python project:
    - Navigate to the root of the `ICCP` project using the `cd` command.
    - Run `conda env create --file requirements.yml` to create an environment for this project.
    - Run `conda activate iccp` whenever you wish to work on this project. A little "(iccp)" should appear on the left side of your terminal.
    - After running `conda activate iccp`, please install Pytorch in this environment by following the [Pytorch installation page](https://pytorch.org/get-started/locally/).



## EnergyPlus 

- Install [EnergyPlus 9.4.0](https://github.com/NREL/EnergyPlus/releases/tag/v9.4.0) by downloading `EnergyPlus-9.4.0-998c4b761e-Windows-x86_64.exe`. 
- Double-click the .exe file to run the installation and choose an installation folder. 
- Add the installation folder path to the PATH environment variable. You may follow this [tutorial](https://helpdeskgeek.com/windows-10/add-windows-path-environment-variable/) to do it. The default installation folder path is `C:\EnergyPlus-9.4.0`  if you're wondering what an installation folder path looks like.


## EnergyPlus co-simulation in Python by FMU 

Acknowledgments to this [tutorial](https://www.youtube.com/watch?v=2CE7FGBxSeM) for helping with the setup. The tutorial is written in English but spoken in Mandarin and it's fairly long. If you encounter any problem with the summary below, please refer back to this tutorial as every download step is recorded.

### Installing C++ linkers/compilers


- Download [Visual Studio Community](https://visualstudio.microsoft.com/vs/community/). I refer to 30:50 in the tutorial video. During the installation:
	- Adding workloads is not necessary, you can continue without it
	- Open Visual Studio Installer:
		- Click the `Modify` Button on your current Visual Studip Version
		- Select `Desktop development with C++` and in the Optional settings, select:
			- MSVC v143 - VS 2022 C++ x64/x86 build tools (or the ARM64 if your system is ARM)
			- Windows 10 SDK (10.0.19041.0)
			- C++ profiling tools
			- C++ CMake tools for Windows
			- C++ ATL for latest v143 build tools (x86 & ....)
			- C++ AddressSanitizer
		- If more options than the one listed above are selected, it should not be a problem.


### EnergyPlusToFMU

- Download the [EnergyPlusToFMU Release 3.1.0](https://simulationresearch.lbl.gov/fmu/EnergyPlus/export/userGuide/download.html).
    - Extract the zip in your preferred folder.
    - To test your installation, we will follow the process described [here](https://simulationresearch.lbl.gov/fmu/EnergyPlus/export/userGuide/installation.html).
    	- Go to installation folder and go to this relative path `<path to installation folder>\Scripts\win`, it will most likely look be this absolute path `<somewhere in your folders>\EnergyPlusToFMU-v3.1.0\Scripts\win`.
    	- Open a terminal by typing `Command Prompt` 
    		- Type `cd <path to installation folder>\Scripts\win` to move to the correct folder. An example with my installation folder path is : `cd C:\Users\Harold\Downloads\EnergyPlusToFMU-v3.1.0\Scripts\win` 
    		- Type `test-c-exe.bat`
    		- The output should be : 
                ```
                ===== Removing old output files =====
                ===== Running compiler =====
                get-address-size.c
                ===== Running linker =====
                ===== Running output executable =====
                == The address size, e.g., 32 or 64, should appear below ==
                64
                == The address size should appear above ==
                ===== Cleaning up =====
                ```

    - Type `compile-c.bat  ..\..\SourceCode\utility\get-address-size.c`
    - Type `link-c-exe.bat  test.exe  get-address-size.obj`
    - Type `test.exe`
    - The output should be "64" or "32"


### Conclusion

If you have followed the steps successfully, you should be able to use `ICCP` successfully!
