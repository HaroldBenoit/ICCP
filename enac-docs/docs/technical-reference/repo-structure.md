---
layout: default
title: Project structure
parent: Technical reference
nav_order: 1
permalink: /docs/technical-reference/repo-structure
---

# Project structure

Here, one can find the detailed folder structure of the project.

📦iCCP

 ┣ 📂EnergyPlus_simulations  *(where the EnergyPlus FMU are stored)*
 ┃ ┗ 📂simple_simulation *(the simulation used throughout the documentation)*
 ┣ 📂logs 
 ┃ ┗ 📂simple_simulation
 ┣ 📂src *(where the source code can be found)*
 ┃ ┣ 📂agent
 ┃ ┃ ┣ 📜Agent.py
 ┃ ┃ ┣ 📜DDPG_Agent.py
 ┃ ┃ ┣ 📜DQN_Agent.py
 ┃ ┃ ┣ 📜OnOff_Agent.py
 ┃ ┃ ┗ 📜SpinUp_DDPG.py
 ┃ ┣ 📂environment
 ┃ ┃ ┣ 📜ContinuousEnvironment.py
 ┃ ┃ ┣ 📜DiscreteEnvironment.py
 ┃ ┃ ┣ 📜Environment.py
 ┃ ┃ ┗ 📜SimpleEnvironment.py
 ┃ ┣ 📂logger
 ┃ ┃ ┣ 📜Logger.py
 ┃ ┃ ┗ 📜SimpleLogger.py
 ┃ ┣ 📜DDPG_notebook.ipynb
 ┃ ┣ 📜DQN_notebook.ipynb
 ┃ ┣ 📜Performance.py
 ┃ ┗ 📜__init__.py
 ┣ 📜.gitignore
 ┣ 📜LICENSE.txt
 ┣ 📜README.md
 ┗ 📜requirements.txt