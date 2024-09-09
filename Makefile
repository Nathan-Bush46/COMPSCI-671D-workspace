install:
	pip install -r requirements.txt
test:
	python -m pytest -v ./src/tests/*test*.py
lint:
	pylint --disable=R,C ./src/main_workspace/*.py
format:
	black ./src/main_workspace/*.py
	black ./src/tests/*.py


all: install test lint format