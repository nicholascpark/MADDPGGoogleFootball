Project 3 readme

To view the checkpoints/hyperparameters - go to this directory:

./p3_docker/my_self_tuned_maddpg_checkpoint/checkpoint_009200/checkpoint-9200


1. To reproduce the graphs shown in the report, run:

	cd /mnt/rldm/scripts
	python graphing.py

They will be generated in /mnt/plots directory. These are generated by the .npy files that are generated by running the evaluation script. These .npy files are located in /mnt/rldm/scripts directory.


2. To evaluate the tuned and trained MADDPG checkpoint, run:

	cd /mnt/rldm/scripts
	python -m rldm.scripts.evaluate_checkpoint_maddpg -c /mnt/my_self_tuned_maddpg_checkpoint/checkpoint_009200/checkpoint-9200


Doing this will overwrite the .npy files in the /mnt/rldm/scripts directory that are used to generate the graphs in #1.


3. To train the agents using the tuned MADDPG hyperparameters, run:

	cd /mnt/rldm/scripts
	python -m rldm.scripts.train_agents_maddpg_self -b -t <num_steps>

The hyperparameters are set as the default config in the script train_agents_maddpg_self.py above