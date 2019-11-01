import os

def assertfolders(folders = ['models', 'gridresults', 'tb_logs', 'errs', 'logs']):
    # creates folders
    for f in folders:
        try:
            os.makedirs(f)
        except OSError:
            pass