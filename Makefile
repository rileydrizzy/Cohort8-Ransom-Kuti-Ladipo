.DEFAULT_GOAL := help

help:
	@echo "    setup                Set up the environment with the required dependencies and environment variables"
	@echo "    export_               save the dependencies onto the requirements text file"
	@echo "    precommit            runs precommit on all files"

setup:
	@echo "Installing and setting up dependencies..."
	chmod +x run_setup.sh
	. ./run_setup.sh
	
	@echo "Setting Enviroment Variables"
	chmod +x set_environment_variables.sh
	. ./set_environment_variables.sh
	
precommit:
	@echo "Running precommit on all files"
	python pre-commit run --all-files

export_:
	@echo "Exporting dependencies to requirements file"
	poetry export --without-hashes -f requirements.txt --output requirements.txt

run_container:
	@echo "Running Docker Container"
	run_container
