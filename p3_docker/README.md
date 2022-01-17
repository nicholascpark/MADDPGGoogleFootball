# Reinforcement Learning and Decision Making
This repo contains useful code for RLDM's P3.

***This Docker image is not compatible with M1 Macs.***

## Set up your system
  1. Install [Docker](https://docs.docker.com/get-docker/)
  2. Make sure the "Hello, World!" container runs properly:
  `docker run hello-world`
  3. (Optional) Make sure Docker containers can use your GPUs:
  `docker run --gpus all nvidia/cuda:11.4.0-runtime-ubuntu20.04 nvidia-smi`

## Getting the code ready
  1. Clone this repo:
     - If you are on the [GT VPN](https://faq.oit.gatech.edu/content/how-do-i-get-started-campus-vpn): `git clone git@github.gatech.edu:rldm/p3_docker.git && cd p3_docker`
     - Otherwise, use the https: `git clone https://github.gatech.edu/rldm/p3_docker.git && cd p3_docker`
  2. Pull the rldm image with:
  `docker pull mimoralea/rldm:latest`
  3. Spin up a container:
     - With GPUs:
         - On Mac or Linux:
         `docker run -it --rm --gpus all -p 8888:8888 -p 6006:6006 -p 8265:8265 -v "$PWD":/mnt mimoralea/rldm:latest`
         - On Windows:
         `docker run -it --rm --gpus all -p 8888:8888 -p 6006:6006 -p 8265:8265 -v %CD%:/mnt mimoralea/rldm:latest`
     - Without GPUs:
         - On Mac or Linux:
         `docker run -it --rm -p 8888:8888 -p 6006:6006 -p 8265:8265 -v "$PWD":/mnt mimoralea/rldm:latest`
         - On Windows:
         `docker run -it --rm -p 8888:8888 -p 6006:6006 -p 8265:8265 -v %CD%:/mnt mimoralea/rldm:latest`

## Test the container
  1. Open Jupyter: http://localhost:8888 (password: `rldm`)
  2. Go to the `notebooks` folder and open the Notebook called: `test-setup.ipynb`.
  3. From the menu, select "Cell," then "Run All".

## Run the test script
  1. `docker container ls # find the "CONTAINER ID"`
  2. `docker exec -it <Place the "CONTAINER ID" here> bash # e.g.: docker exec -it 4f96cc437ac9 bash`
  3. `python -m rldm.scripts.train_agents # this trains three policies with working hyperparameters. For more info, pass -h to the script, and separately -r`

## Monitor training runs
  1. Open Tensorboard: http://localhost:6006, and look at the running trials
  2. Open Ray Dashboard: http://localhost:8265, and look at the resources utilized
  3. Have fun!!!

## Evaluate checkpoints
  1. `docker container ls # find the "CONTAINER ID"`
  2. `docker exec -it <Place the "CONTAINER ID" here> bash # e.g.: docker exec -it 4f96cc437ac9 bash`
  3. `python -m rldm.scripts.evaluate_checkpoint -c /mnt/logs/baselines/baseline_1/checkpoint_0/checkpoint-0 # this evaluates one of the provided checkpoints. For more info, pass -h to the script`

## Render episodes from checkpoints
  1. Open Jupyter: http://localhost:8888 (password: `rldm`)
  2. Go to the `notebooks` folder and open the Notebook called: `visualize-episodes.ipynb`.
  3. From the menu, select "Cell," then "Run All". Search for the variable "checkpoint" to visualize your own agent.
