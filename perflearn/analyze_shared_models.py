import os
import json
import pandas as pd
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

WINDOW_SIZE = 100
PLOTDIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    'plots',
)


for env in [
        'LunarLanderFixedPos-v2',
        'LunarLanderFixed-v2',
        'LunarLander-v2',
        'LunarLanderFixedHalfY-v2',
]:
    dirprefix = os.path.join('shared_models', env)
    files = os.listdir(dirprefix)
    logs = [f for f in files if 'log' in f]
    for log in logs:
        df = pd.read_csv(os.path.join(dirprefix, log))
        for key in [
                'td_max',
                'td_mean',
                '100ep_r_mean',
                '100ep_r_mean_discounted',
                '100ep_v_mean',
        ]:
            if key not in df.columns:
                continue
            if '100ep' in key:
                # Already should be smooth.
                rolling_key = key
            else:
                rolling_key = key + 'rolling'
                df[rolling_key] = df[key].rolling(WINDOW_SIZE).mean()
            plot = sns.lineplot(x='t', y=rolling_key, data=df)
            plot.get_figure().savefig(os.path.join(
                PLOTDIR,
                'shared_models-{}-{}-{}.png'.format(env, log, key)
            ))
            plt.clf()

