import os
from argparse import ArgumentParser
from getpass import getuser
import yaml


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


def write_sbatch(slurm_cfg, cfg):
    script = F"#!/usr/bin/env bash\n"

    for key, val in slurm_cfg.items():
        script += F"#SBATCH --{key}={val}\n"

    script += F"set -e\n"
    script += F"cd {cfg['path']}\n"
    script += F"source activate {cfg['env']}\n"
    script += F"python {cfg['script_path']} {cfg2args(cfg['script_config'])}\n"
    return script


PATH = os.getcwd()
parser = ArgumentParser(description='Runs experiment with parallel HP optimization')
parser.add_argument('--config', '-c', help='relative path of config file', type=str, default=None)
args = vars(parser.parse_args())    # convert to dict

with open(F"{PATH}/{args['config']}", 'r') as cfg_file:
    config = {'name': 'test', 'env': 'ADDA'}
    config.update(yaml.load(cfg_file))

user = getuser()
path = F"{PATH}/experiments/{config['name']}"
if not os.path.exists(path):
    os.makedirs(path)

ID = next_id(path)
path = F"{path}/{ID}"
if not os.path.exists(path):
    os.makedirs(path)

config['script_config']['path'] = path

slurm_config = {
    'job-name'  : F"{config['name']}-{ID}",
    'output'    : F"{path}/log.out",
    'error'     : F"{path}/log.err",
    'partition' : 'TEST',
    'mail-type' : 'NONE',
    'mail-user' : F"{user}@ismll.de"}
slurm_config.update(config['slurm_config'])
config['slurm_config'].update(slurm_config)

with open(F"{path}/config.yml", 'w') as cfg_file:
    yaml.dump(config, cfg_file)

# Write and Execute SBATCH
sbatch_script = write_sbatch(slurm_config, config)
with open(F"{path}/sbatch.sh", "w") as file:
    file.write(sbatch_script)

os.system(F"sbatch {path}/sbatch.sh")
