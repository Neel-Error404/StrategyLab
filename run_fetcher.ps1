$env:PYTHONPATH = '.'
& ".\.venv\Scripts\python.exe" -m src.core.etl.data_fetcher --mode update --pool-path 'data/pools/2022-01-01_to_2025-08-31/'
