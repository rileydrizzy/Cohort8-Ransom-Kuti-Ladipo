echo "Installing..."
curl -sSL https://install.python-poetry.org | python -
echo "Activating virtual environment"
poetry install
poetry shell
echo "Environment setup complete"
