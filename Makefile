.DEFAULT_GOAL := help

help:
	@echo "    setup                Set up the environment with the required dependencies"
	@echo "    export               save the dependencies onto the requirements txt file"
	@echo "    precommit            runs precommit on all files"

setup:
	@echo "Installing..."
	curl -sSL https://install.python-poetry.org | python -
	@echo "Activating virtual environment"
	poetry shell
	poetry install
	poetry add pre-commit
	@echo "Environment setup complete"	
	
precommit:
	@echo "Running precommit on all files"
	pre-commit run --all-files

export:
	@echo "Exporting dependencies to requirements file"
	poetry export --without-hashes -f requirements.txt --output requirements.txt

backup: # To push to Github without running precommit
	git commit --no-verify -m "updates"
	git push origin main
