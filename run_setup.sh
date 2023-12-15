echo "Installing..."
curl -sSL https://install.python-poetry.org | python -
echo "Activating virtual environment"
poetry install
poetry shell
python pre-commit install
echo "Environment setup complete"
