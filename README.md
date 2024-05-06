# Versatilis

* Make sure that python version is 3.12 or later
* Run ```python -m venv venv``` to create a virtual env
* Activate env - ```. venv/bin/activate```
* Install pip tools - ```pip install pip-tools```
* Copy variables from `.env.example` to `.env` and provide values for API keys
* Install postgresql - ```sudo apt-get install postgresql``` (or ```brew install postgresql``` for MacOS)
* Run ```./sync_dependencies.sh``` to install dependencies
* Run ```pre-commit install``` to install pre-commit hooks for git (this will install `black` and `pylint`)
* ```uvicorn versatilis.asgi:application```
