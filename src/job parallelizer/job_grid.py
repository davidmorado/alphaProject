import os
import yaml

from utils import get_grid, find_file
from argparse import ArgumentParser
from getpass import getuser
from xarray import Dataset


def get_id(s: str):
    # Assume file format: ID_bla_bla_bla.ext
    ID = s.split("_")[0]
    try:
        return int(ID)
    except:
        return 0


def next_id(path):
    dirs = os.listdir(path)
    ids  = [get_id(d) for d in dirs]
    return 0 if ids == [] else max(ids)+1


def cfg2args(cfg : dict):
    return " ".join([F"--{key} {val}" for key, val in cfg.items()])


def make_sbatch(slurm_cfg, cfg, ID):
    script  = F"#!/usr/bin/env bash\n"
    script += F"#SBATCH --output={cfg['path']}/logs/{ID}.out\n"
    script += F"#SBATCH --error={cfg['path']}/logs/{ID}.err\n"
    for key, val in slurm_cfg.items():
        script += F"#SBATCH --{key}={val}\n"

    script += F"set -e\n"
    script += F"cd {cfg['path']}\n"
    script += F"source activate {cfg['env']}\n"
    script += F"python {cfg['script']} --ID {ID}\n"
    return script


PATH = os.getcwd()
parser = ArgumentParser(description='Runs experiment with parallel HP optimization')
parser.add_argument('--config', '-c', help='relative path of config file', type=str, default=None)
args = vars(parser.parse_args())    # convert to dict

with open(F"{PATH}/{args['config']}", 'r') as cfg_file:
    config = {'name': 'test', 'env': 'ADDA'}
    config.update(yaml.safe_load(cfg_file))

user = getuser()
path = F"{PATH}/experiments/{config['name']}"
if not os.path.exists(path):
    os.makedirs(path)

RUNID = next_id(path)
path = F"{path}/{RUNID}"
if not os.path.exists(path):
    os.makedirs(path)

config['path'] = path
config['script'] = F"{PATH}/{config['script']}"
config['data_path'] = F"{find_file(config['data'], 'data')}"

if not os.path.exists(F"{path}/logs"):
    os.makedirs(F"{path}/logs")

slurm_config = {
    'job-name'  : F"{config['name']}-{RUNID}",
    'partition' : 'TEST',
    'mail-type' : 'NONE',
    'mail-user' : F"{user}@ismll.de"}
slurm_config.update(config['slurm_config'])
config['slurm_config'].update(slurm_config)

coords, cfgs, _ = get_grid(config)
config['_cfgs'] = cfgs
config['_coords'] = coords

with open(F"{path}/config.yaml", 'w') as cfg_file:
    # TODO: flow style
    yaml.dump(config, cfg_file)

# start all jobs
for ID in cfgs.keys():
    # Write and Execute SBATCH
    with open(F"{path}/sbatch.sh", "w") as file:
        file.write(make_sbatch(slurm_config, config, ID))
    os.system(F"sbatch {path}/sbatch.sh")
