---
layout: default
title: How to make EnergyPlus FMUs
parent: How-to guides
nav_order: 3
permalink: /docs/how-to/make_fmu
---

# How to make EnergyPlus FMUs

This tutorial assumes that you have followed the [software installation tutorial](../../../enac-docs/docs/get-started/software_installation) previously.

## Acknowledgements 

Acknowledgments to this [tutorial](https://www.youtube.com/watch?v=2CE7FGBxSeM) for helping with the setup. The tutorial is written in English but spoken in Mandarin and it's fairly long. If you encounter any problem with the summary below, please refer back to this tutorial as every step is recorded.

I haven't had the occasion to build a FMU myself as I was given one by the ICE Lab team (specifically Arnab Chatterjee) but I will try to give guidance to the reader. I will refer to timestamps in the mentioned video above.

## Setting up the idf file

- Have an .idf file and a corresponding .epw weather file for the wanted environment

- (1:03:50), in the .idf file, create an ExternalInterface of type Functional MockupUnit Export

-  (1:04:21), create ExternalInterfaceFunctionalMockupUnit Export To Schedule with a given schedule name e.g. HTGSTEP_OPTIMUM2, FMU variable name "Thsetpoint", and give some initial value e.g. 16

- After that, please watch the tutorial closely until 1:18:00 to know how setup the idf file.

# Converting into FMU

- (1:18:00), the idf file is converted into an FMU

- To create an FMU [(tutorial)](https://simulationresearch.lbl.gov/fmu/EnergyPlus/export/userGuide/build.html#command-line-use), run:`python <path-to-EnergyPlusToFMU>\Scripts\EnergyPlusToFMU.py  -i <path-to-EnergyPlus>\Energy+.idd -w <path-to-weather-file> -a 2 <path-to-idf-file>`  

- An example for the paths can be found here: `python C:\Users\Harold\Downloads\EnergyPlusToFMU-v3.1.0\Scripts\EnergyPlusToFMU.py  -i C:\EnergyPlusV9-3-0\Energy+.idd -w C:\Users\Harold\Desktop\ENAC-Semester-Project\DIET_Controller\custom_gym\Eplus_simulation\CHE_GE_Geneva_2004-2018.epw -a 2 C:\Users\Harold\Desktop\ENAC-Semester-Project\DIET_Controller\custom_gym\Eplus_simulation\CELLS_v1.idf`



