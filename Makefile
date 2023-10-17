.DEFAULT_GOAL := help

help:
	@echo "    prepare              desc of the command prepare"
	@echo "    install              desc of the command install"
	@echo "    activate             activate virtual environment"
	@echo "    setup                perform both install ad activate"
	@echo "    export               "

install:
	@echo "Installing..."
	curl -sSL https://install.python-poetry.org | python -
	poetry init
	poetry install
	poetry run pre-commit install
	
activate:
	@echo "Activating virtual environment"
	poetry shell

setup: install activate

precommit:
	@echo "Running precommit on all files"
	pre-commit run --all-files

export:
	@echo "Exporting dependencies to requirements file"
	poetry export --without-hashes -f requirements.txt --output requirements.txt

backup: # To push to Github without running precommit
	git commit --no-verify -m "backup"
	git push origin main
