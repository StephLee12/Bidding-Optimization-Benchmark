## train

nohup python benchmark_forecast_train.py --state_name VIC --market_year 2021 --device cuda:

nohup python benchmark_forecast_train.py --state_name NSW --market_year 2021 --device cuda:

nohup python benchmark_forecast_train.py --state_name QLD --market_year 2021 --device cuda:

nohup python benchmark_forecast_train.py --state_name SA --market_year 2021 --device cuda:

nohup python benchmark_forecast_train.py --state_name TAS --market_year 2021 --device cuda:


## predict 

nohup python benchmark_forecast_predict.py --state_name VIC --market_year 2021 --device cuda:

nohup python benchmark_forecast_predict.py --state_name NSW --market_year 2021 --device cuda:

nohup python benchmark_forecast_predict.py --state_name QLD --market_year 2021 --device cuda:

nohup python benchmark_forecast_predict.py --state_name SA --market_year 2021 --device cuda:

nohup python benchmark_forecast_predict.py --state_name TAS --market_year 2021 --device cuda:


## optim solve 

nohup python benchmark_main.py --state_name VIC --market_year 2021

nohup python benchmark_main.py --state_name NSW --market_year 2021

nohup python benchmark_main.py --state_name QLD --market_year 2021

nohup python benchmark_main.py --state_name SA --market_year 2021

nohup python benchmark_main.py --state_name TAS --market_year 2021
