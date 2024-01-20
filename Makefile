.DEFAULT_GOAL := help

help:
	@echo "    setup                Set up the environment with the required dependencies"
	@echo "    export               save the dependencies onto the requirements txt file"
	@echo "    precommit            runs precommit on all files"

setup:
	@echo "Running setup..."
	. run_setup.sh
	
precommit:
	@echo "Running precommit on all files"
	python pre-commit run --all-files

export_:
	@echo "Exporting dependencies to requirements file"
	poetry export --without-hashes -f requirements.txt --output requirements.txt

run_container:
	@echo "Running Docker Contain"
	run_container
