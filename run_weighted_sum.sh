#!/usr/bin/sh

python src/timeseries/moo/experiments/run_gd_experiment.py ./output/exp_large_n99_normFalse_e3_nw/ --start 0 --end 20
python src/timeseries/moo/experiments/run_gd_experiment.py ./output/exp_large_n99_normFalse_e3_nw/ --start 20 --end 40
python src/timeseries/moo/experiments/run_gd_experiment.py ./output/exp_large_n99_normFalse_e3_nw/ --start 40 --end 60
python src/timeseries/moo/experiments/run_gd_experiment.py ./output/exp_large_n99_normFalse_e3_nw/ --start 60 --end 80
python src/timeseries/moo/experiments/run_gd_experiment.py ./output/exp_large_n99_normFalse_e3_nw/ --start 80 

python src/timeseries/moo/experiments/run_gd_experiment.py ./output/exp_medium_n99_normFalse_e3_nw/ --start 0 --end 20
python src/timeseries/moo/experiments/run_gd_experiment.py ./output/exp_medium_n99_normFalse_e3_nw/ --start 20 --end 40
python src/timeseries/moo/experiments/run_gd_experiment.py ./output/exp_medium_n99_normFalse_e3_nw/ --start 40 --end 60
python src/timeseries/moo/experiments/run_gd_experiment.py ./output/exp_medium_n99_normFalse_e3_nw/ --start 60 --end 80
python src/timeseries/moo/experiments/run_gd_experiment.py ./output/exp_medium_n99_normFalse_e3_nw/ --start 80 

python src/timeseries/moo/experiments/run_gd_experiment.py ./output/exp_small_n99_normFalse_e3_nw/ --start 0 --end 20
python src/timeseries/moo/experiments/run_gd_experiment.py ./output/exp_small_n99_normFalse_e3_nw/ --start 20 --end 40
python src/timeseries/moo/experiments/run_gd_experiment.py ./output/exp_small_n99_normFalse_e3_nw/ --start 40 --end 60
python src/timeseries/moo/experiments/run_gd_experiment.py ./output/exp_small_n99_normFalse_e3_nw/ --start 60 --end 80
python src/timeseries/moo/experiments/run_gd_experiment.py ./output/exp_small_n99_normFalse_e3_nw/ --start 80 
