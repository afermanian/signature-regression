import os
import json
import pandas as pd
from sacred.observers import FileStorageObserver
from sklearn.model_selection import ParameterGrid


def gridsearch(ex, config_grid, niter=10, dirname='my_runs'):
    """Loops over all the experiments in a configuration grid.

    Parameters
    ----------
        ex: object
            Instance of sacred.Experiment()

        config_grid: dict
            Dictionary of parameters of the experiment.

        niter: int, default=10
            Number of iterations of each experiment

        dirname: str, default='my_runs'
            Location of the directory where the experiments outputs are stored.
    """
    ex.observers.append(FileStorageObserver('results/' + dirname))
    param_grid = list(ParameterGrid(config_grid))
    for i in range(niter):
        for params in param_grid:
            ex.run(config_updates=params, info={})


def load_json(path):
    """Loads a json object
    Parameters
    ----------
    path: str
        Location of the json file.
    """
    with open(path) as file:
        return json.load(file)


def extract_config(loc):
    """ Extracts the metrics from the directory."""
    config = load_json(loc + '/config.json')
    return config


def extract_metrics(loc):
    """ Extracts the metrics from the directory. """
    metrics = load_json(loc + '/metrics.json')

    # Strip of non-necessary entries
    metrics = {key: value['values'] for key, value in metrics.items()}

    return metrics


def get_ex_results(dirname):
    """Extract all result of a configuration grid.

    Parameters
    ----------
    dirname: str
        Name of the directory where the experiments are stored.

    Returns
    -------
    df: pandas DataFrame
        Dataframe with all the experiments results
    """
    not_in = ['_sources', '.DS_Store']
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dirname = dir_path + '/' + dirname
    run_nums = [x for x in os.listdir(dirname) if x not in not_in]

    frames = []
    for run_num in run_nums:
        loc = dirname + '/' + run_num
        try:
            config = extract_config(loc)
        except Exception as e:
            print('Could not load config at: {}. Failed with error:\n\t"{}"'.format(loc, e))
        try:
            metrics = extract_metrics(loc)
        except Exception as e:
            print('Could not load metrics at: {}. Failed with error:\n\t"{}"'.format(loc, e))

        # Create a config and metrics frame and concat them
        config = {str(k): str(v) for k, v in config.items()}    # Some dicts break for some reason
        df_config = pd.DataFrame.from_dict(config, orient='index').T
        df_metrics = pd.DataFrame.from_dict(metrics, orient='index').T
        df = pd.concat([df_config, df_metrics], axis=1)
        df.index = [int(run_num)]
        frames.append(df)

    # Concat for a full frame
    df = pd.concat(frames, axis=0, sort=True)
    df.sort_index(inplace=True)

    return df


def move_legend(ax, new_loc, **kws):
    """Solves problem when moving legends with seaborn and histplot"""
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, bbox_to_anchor=(1.05, 1), **kws)

