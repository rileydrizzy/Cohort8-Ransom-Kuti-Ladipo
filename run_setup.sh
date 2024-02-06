#
echo "Installing..."
curl -sSL https://install.python-poetry.org | python -
echo "Activating virtual environment"
poetry install
echo "Environment setup complete"

echo "Setting Environment Keys and Variables"
. set_environment_variables.sh
