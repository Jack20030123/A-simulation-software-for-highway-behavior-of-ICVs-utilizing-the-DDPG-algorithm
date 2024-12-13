# A-simulation-software-for-highway-behavior-of-ICVs-utilizing-the-DDPG-algorithm

1. Software Functionality
   
After launching the software, the user is first guided to an initial interface designed for selecting the total number of ICVs and HDVs involved in the simulation scenario. After inputting these values and clicking "Submit," the user proceeds to the ICV parameter configuration interface. Here, users can set detailed parameters for each vehicle, including the reward function values and initial position. Once all necessary ICV parameters are configured and submitted, the software moves on to the training parameter settings, which cover the number of training episodes, learning rates, hidden layers, initial memory size for training, replay buffer size, minimum and maximum exploration probabilities, exploration decay rate, simulation frequency, and policy update frequency, among others.

After the training phase is completed, the user can choose to use any of the previously trained results, and the software will generate various images related to the training process. These include charts of different rewards, average speed, collisions, and survival times in the simulation, helping users visualize and analyze the outcomes. Additionally, the software creates Excel spreadsheets that record reward values, speed, headway distance, and other data.

The main functions of the software are:

(1) Users can select different numbers of ICVs and HDVs according to real-world scenarios or research objectives, as well as configure various roadway environments and reward values.

(2) Using the Deep Deterministic Policy Gradient (DDPG) model and the third-party highway-env environment, the software provides a visual simulation of multi-ICV intelligent cooperative lane-changing.

(3) The training results can be visualized, showing graphs of metrics like average speed and reward values for an intuitive understanding of the training outcomes and model performance. Moreover, the corresponding Excel files record the data used for plotting, and they also include the average headway distance data for each vehicle during each training session.

2. Operating Environment

A mainstream commercial computer running a 64-bit Windows OS, macOS, or Linux can operate the software. The lane-changing decision software for ICVs requires Python 3.6 or later and must be configured with tkinter, PyTorch, numpy, matplotlib, pandas, and highway_env, among other third-party libraries.

In addition, Microsoft Office or WPS must be installed on the computer for convenient output processing of simulation data.

3. Basic Software Interfaces
   
3.1 Initial Interface

Open the "start_ui_ddpg.py" file in Python and run it to display the software menu interface, as shown in Figure 1.

<div align="center">
  <img src="https://github.com/user-attachments/assets/bd62bfa1-3dc6-4d70-b24e-0dc7beff0d97" alt="Image Description" width="300">
</div>
<p align="center">
  Figure 1: Software Menu Interface
</p>

3.2 Vehicle Parameter Configuration

After successfully entering the number of ICVs and HDVs, the user will proceed to the parameter configuration interface, as shown in Figure 2. In this interface, users can adjust the reward functions and initial positions of each ICV. The number of vehicle parameter configuration interfaces displayed corresponds to the number of ICVs entered in the previous interface. After inputting these settings, click "Submit" to advance to the training parameter configuration interface.

<div align="center">
  <img src="https://github.com/user-attachments/assets/c6d46976-aed7-4b7d-8ed1-897705864706" alt="Image Description" width="600">
</div>
<p align="center">
  Figure 2: Vehicle Parameter Configuration Interface
</p>

3.3 Training Parameter Settings

After completing the vehicle parameter configuration and clicking "Submit," the software transitions to the training parameter settings interface, as shown in Figure 3. Once parameters like the number of training episodes, learning rates, and hidden layers are fully set, the simulation interface can be entered.

<div align="center">
  <img src="https://github.com/user-attachments/assets/e12a66fb-c491-4ccd-86c2-b017177c7e8b" alt="Image Description" width="800">
</div>
<p align="center">
  Figure 3: Training Parameter Settings Interface
</p>

3.4 Simulation Interface

Once the training parameters have been set and submitted, the software automatically initiates model training. It launches highway-env to conduct the model training and evaluation in a three-lane highway scenario.
The overall simulation interface is shown in Figure 4, where yellow represents ICVs and blue represents HDVs.


<div align="center">
  <img src="https://github.com/user-attachments/assets/77e6401a-b747-49b7-be9b-ec4758fb3e71" alt="Image Description" width="1000">
</div>
<p align="center">
  Figure 4: Simulation Interface Before Training
</p>




<div align="center">
  <img src="https://github.com/user-attachments/assets/77e6401a-b747-49b7-be9b-ec4758fb3e71" alt="Image Description" width="1000">
</div>

<p align="center">
  Figure 5: Simulation Interface After Training
</p>

4. Main Functionality of the Software

After the simulation training is completed, the evaluation images and Excel spreadsheets will be saved locally in the "train" folder. The results mainly include: images of speed variations for each ICV during training, total cumulative reward function graphs for all vehicles in each training session, cumulative reward function graphs for each ICV in every training session, survival time graphs for each training session, and headway distances between each vehicle and the one in front. Additionally, Excel spreadsheets are generated to store the corresponding data, including the plotting data and the average headway distance for each vehicle in each training session.

<div align="center">
  <img src="https://github.com/user-attachments/assets/bbdefe46-a2e7-4dc5-829d-5d7e4c281d00" alt="Image Description" width="1000">
</div>

<p align="center">
  Figure 6: Speed Variation Graph of an ICV During Training
</p>
