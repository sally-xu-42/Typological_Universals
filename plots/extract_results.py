import pandas as pd
import argparse
import fnmatch
import math
import os
import json
from collections import defaultdict

def get_files(eval_type, model_type, lang, do_checkpoints):
    matched_files = []
    path = f'./logs/evaluations/{model_type}/{eval_type}'

    if do_checkpoints:
        for dir in os.listdir(path):
            dir_path = os.path.join(path, dir)
            if os.path.isdir(dir_path) and f'-{lang}-' in dir:
                pattern = f'checkpoint-*.out'
                for name in os.listdir(dir_path):
                    if fnmatch.fnmatch(name, pattern):
                        matched_files.append(os.path.join(dir_path, name))
    else:
        pattern = f'evaluate_{eval_type}_{model_type}-*-{lang}*.out'
        for name in os.listdir(path):
            if fnmatch.fnmatch(name, pattern):
                matched_files.append(os.path.join(path, name))

    return matched_files

def remap_checkpoints(results_dict):
    # sort and map checkpoints to epochs
    sorted_dict = dict(sorted(results_dict.items(),
                              key=lambda x: ('-'.join(x[0].split('-')[:-1]), int(x[0].split('-')[-1]))))

    print(sorted_dict.keys())
    new_dict = {}
    for i, (key, value) in enumerate(sorted_dict.items()):
        new_key = '-'.join(key.split('-')[:-1]) + f'-{i % 50 + 1}'
        new_dict[new_key] = value

    return new_dict

def extract_blimp_results(files, model_type, lang, do_checkpoints):
    results_dict = {}

    # we extract the results directly from the output logs
    for file in sorted(files):
        split = file.split('-')
        # format: VO-pair1
        name = f'{split[1]}-{split[3]}'
        # format: VO-pair1-4097
        if do_checkpoints:
            name += f'-{split[-1].split("_")[0]}'

        results_dict[name] = {}

        with open(file) as f:
            lines = f.readlines()

            assert lines[-18] == "Scores:\n"
            results = lines[-17:]

            for res in results:
                task = res.split(':')[0]
                score = float(res.split(':')[1].strip()[:-1])

                results_dict[name][task] = score

    name = f'blimp_{model_type}_{lang}'
    if do_checkpoints:
        name += '_checkpoints'
        results_dict = remap_checkpoints(results_dict)

    results_df = pd.DataFrame(results_dict)

    results_df.to_csv(f'./plots/{name}.csv', index_label='task')

def extract_ppl_results(checkpoint_dir):
    # extract the results directly from checkpoints
    results_dict = defaultdict(list)

    for dir in os.listdir(checkpoint_dir): # Iterate through each directory
        epochs = []
        dir_path = os.path.join(checkpoint_dir, dir)
        for subdir in os.listdir(dir_path): # for every ckpt-xxx subdir
            subdir_path = os.path.join(dir_path, subdir)
            if os.path.isdir(subdir_path):               
                trainer_state_path = os.path.join(subdir_path, 'trainer_state.json') # find the file that contains PPLs
                if os.path.isfile(trainer_state_path):
                    with open(trainer_state_path, 'r') as f:
                        trainer_state = json.load(f)
                        for log in trainer_state.get('log_history', []):
                            # Check if the log contains eval_loss and extract it; Note that it repeats previous log
                            if 'eval_loss' in log and log['epoch'] not in epochs:
                                epochs.append(log['epoch']) # prevent appending same epoch PPL again

                                results_dict["perplexities"].append(math.exp(log['eval_loss']))
                                results_dict["epoch"].append(log['epoch'])

                                keys = ["model", "lang", "pair", "seed"]
                                values = dir.split('-')
                                for key, value in zip(keys, values):
                                    results_dict[key].append(value)

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(f'./plots/perplexity.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval_type", type=str)
    parser.add_argument("-m", "--model_type", type=str)
    parser.add_argument("-l", "--lang", type=str)
    parser.add_argument("-c", "--checkpoints", action='store_true', default=False,
                        help="Extract results from the checkpoints evaluation")
    parser.add_argument("-p", "--dir", default='./checkpoints',
                        help="Directory of model checkpoints for extracting perplexity")
    args = parser.parse_args()

    files = get_files(args.eval_type, args.model_type, args.lang, args.checkpoints)

    if args.eval_type == 'blimp':
        extract_blimp_results(files, args.model_type, args.lang, args.checkpoints)
    elif args.eval_type == 'perplexity':
        extract_ppl_results(args.dir)
    else:
        raise NotImplementedError