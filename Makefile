.PHONY: env lint test data clean dashboard

env:
	mamba env create -f environment.yml || conda env create -f environment.yml

lint:
	ruff check src/ app/ tests/

test:
	pytest tests/ -v

data:
	python -m app.cli download-data

dashboard:
	streamlit run app/dashboard.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .ruff_cache