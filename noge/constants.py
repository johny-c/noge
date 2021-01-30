from pathlib import Path

# paths
PROJ_DIR = Path(__file__).parent.parent
DATA_DIR = PROJ_DIR / 'data'
SCRIPTS_DIR = PROJ_DIR / 'scripts'
CONFIGS_DIR = PROJ_DIR / 'configs'
EVAL_DIR = PROJ_DIR / 'mlruns'


BASELINES = ['random', 'bfs', 'dfs', 'nn']
ALGORITHM = 'noge'

SYNTHETIC_DATASETS = [
    'barabasi',
    'ladder',
    'tree',
    'grid',
    'caveman',
    'maze'
]


PLACES = {
    'MUC': {
        'city': 'Munich',
        'state': 'Bavaria',
        'country': 'Germany'
    },
    'OXF': {
        'city': 'Oxford',
        'country': 'United Kingdom'
    },
    'SFO': {
        'city': 'San Francisco',
        'state': 'California',
        'country': 'USA'
    }
}


REAL_DATASETS = list(PLACES.keys())
