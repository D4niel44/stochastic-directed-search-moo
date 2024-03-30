import argparse
import glob
import os
import joblib

jpgFilenamesList = glob.glob('145592*.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare the Pareto front from multiple algorithms.')
    parser.add_argument('path', help='path to the config of the experiment')
    args = parser.parse_args()

    result_files = glob.glob(os.path.join(args.path, 'results_*'))
    results = []
    for file in result_files:
        res = joblib.load(file)
        results = results + res

    joblib.dump(
        results,
        os.path.join(args.path, f'results.z'),
        compress=3)
