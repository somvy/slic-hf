dataset:
	python dataset/main.py

train:
	python 2-dpo/train.py

compare:
	python 2-dpo/compare.py

install:
	poetry install
	poetry shell
#login to wandb, hf
