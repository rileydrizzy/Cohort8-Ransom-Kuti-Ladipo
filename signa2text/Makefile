.DEFAULT_GOAL := help

help:
	@echo "    setup                Set up the environment with the required dependencies and environment variables"
	@echo "    freeze_reqs          Save the dependencies onto the requirements text file"
	@echo "    precommit            Runs precommit on all files"
	@echo "    download_data        Download "
	@echo "    run_train            Training"
	
setup:
	@echo "Installing and setting up dependencies..."
	. ./run_setup.sh 

	@echo "Setting Enviroment Variables"
	. ./set_environment_variables.sh
	
precommit:
	@echo "Running precommit on all files"
	python pre-commit run --all-files

freeze_reqs:
	@echo "Exporting dependencies to requirements file"
	pip freeze > requirements.txt

download_data:
	@echo "Downloading Development Data"
	python src/download_dev_data.py

run_train:
	. ./train_model.sh