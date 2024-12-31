# Advantage Actor-Critic with Optuna

The performance of online reinforcement algorithms such as A2C is very sensitive to hyperparameters. This project utilizes [Optuna](https://optuna.org/), a hyperparameter search framework, to train the best-performing policy to solve the Gym Mujoco environments. 

## Local Setup

1. Create the conda environment

`conda env create -f environment.yml --name a2c-optuna`

2. Activate the conda environment:

 `conda activate a2c-optuna`

3. Install python dependencies

`pip install -r requirements.txt`

4. Install the root package

`pip install -e .`

5. Login to wandb

`wandb login`

6. Spin up a MySQL instance. Optuna requires it to manage studies and trials. The simplest way to do it is through a free, managed cloud service like [Aiven](https://aiven.io/free-mysql-database)

## Train the Policy

1. Create an Optuna study

`python a2c_optuna/scripts/create_study.py --study_storage {mysql_connection_url} --env_name Ant-v4 --study {study_name}`

2. Launch Optuna-managed training

`python a2c_optuna/scripts/run.py --wandb_project {wandb_project} --study_name Ant-v4-{study_name} --study_storage {mysql_connection_url}`
