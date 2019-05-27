# Network Packet Routing with Deep Reinforcement Learning


IMPORTANT NOTE: Programs were only tested on Linux, compatibility with Windows is not expected.


Python 3.7 and above is recommended.


It is recommended to install all libraries and run the Python program within an env.


This can be done by running python3 -m venv <env_name>


To enable this environment run source <env_directory>/bin/activate


### Libraries required for Python 3 RL implementations:

Numpy
Torch
Tensorflow
gym

### Libraries for the Python Map program:

gmplot3
bs4

In order to run our reinforcement learning implementations, run

python RunRL.py -h
It takes command line arguments to select the network, but defaults to germany50 so if python RunPolicyGradients.py is ran it will by default run germany50 and not render

to run another network you would run with the -n flag for example:
python RunRL.py -n newyork
to render it you would run with the -r flag:
python RunRL.py -n atlanta -r

 
