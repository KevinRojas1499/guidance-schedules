import os
import click
import json
import subprocess
import numpy as np
from tqdm import tqdm
import csv

@click.group()
def grid_search():
    pass

@grid_search.command()
@click.option('--prop', type=click.Choice(['ring_count', 'qed']), required=True)
@click.option('--model', type=click.Choice(['udlm', 'ar', 'mdlm']), required=True)
@click.option('--out_dir', type=str, default='grid_search')
def qm9(out_dir, prop, model):
    os.makedirs(out_dir, exist_ok=True)
    schedules = ['constant', 'linear', 'inv-linear', 'cosine', 'sine', 'V', 'inv-V']
    schedules = ['constant']
    methods = ['cfg', 'ours']
    methods = ['ours']
    strengths = np.linspace(1.0, 5.0, 5)
    pbar = tqdm(schedules)
    stats = {}
    for schedule in pbar:
        for method in methods:
            stats = {}
            for strength in strengths:
                pbar.set_description(f'Guidance-schedule: {schedule}, strength: {strength : .2f}, method: {method}')
                cmd = f'cd scripts/ && \
                        MODEL={model} \
                        PROP={prop} \
                        GUIDANCE={method} \
                        GUID_SCHEDULE={schedule} \
                        GAMMA={strength} \
                        bash eval_qm9_guidance.sh &> logs_{strength}.txt'
                subprocess.run(cmd, shell=True, capture_output=True, text=True)
                output_file = f'outputs/qm9/udlm_{prop}/qm9-eval-{method}_{prop}_T-32_gamma-{strength:.2f}_seed-1.csv'
                with open(output_file, 'r') as file:
                    csv_reader = csv.DictReader(file)
                    data = list(csv_reader)
                    for key, value in data[-1].items():
                        if key not in stats:
                            stats[key] = [] 
                        stats[key].append(value)
                
                json.dump(stats, open(f'{out_dir}/grid_search_results_{schedule}_{method}_{strength}.json', 'w'))

            
    
if __name__ == "__main__":
    
    grid_search()
