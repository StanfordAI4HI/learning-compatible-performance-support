import os
import collections
import argparse
import json
from json.decoder import JSONDecodeError
import numpy as np
import pandas as pd
import seaborn as sns
from multiprocessing import Pool
from matplotlib import pyplot as plt

# TODO: Fix relative import
#from utils import NumpyEncoder

CUM_T_INTERVAL = 100


def format_name(name):
    return name.lstrip('_')

def load_file_to_dataframe(fname, gamma=0.99):
    print('loading {}'.format(fname))
    with open(fname, 'r') as f:
        #d = pickle.load(f)
        try:
            d = json.load(f)
        except JSONDecodeError:
            return None, {}
        run = d.get('run')
        if run is None:
            print('getting run from filename instead of dict')
            import re
            run = int(re.match('.*?([0-9]+)\.json$', fname).group(1))
        data_lst_assisted, info = ResultsPlotter._to_state_dataframe(
            d['evals'],
            args=d['args'],
            run_id=run,
            gamma=gamma,
        )
        data = pd.DataFrame(data_lst_assisted)
        data['mode'] = 'assisted'
        data_lst_unassisted, _ = ResultsPlotter._to_state_dataframe(
            d['evals_unassisted'],
            args=d['args'],
            run_id=run,
            gamma=gamma,
            cum_ts=info['cum_ts'],
        )
        data_unassisted = pd.DataFrame(data_lst_unassisted)
        data_unassisted['mode'] = 'unassisted'

        data = data.append(data_unassisted)

        # TODO: This isn't great. Fix!
        named = False
        for learner_assumption in [
                'transition',
                'interruption',
                'disruption',
                'transition-disrupt',
        ]:
            if d['args']['name'].endswith(learner_assumption):
                data['learner_assumption'] = learner_assumption
                named = True
        if not named:
            data['learner_assumption'] = 'sees_true'

        if d['args']['learner_support'] == 'reddy_rss':
            policy = '{}_{}'.format(
                d['args']['learner_support'],
                d['args']['q_threshold'],
            )
        elif d['args']['learner_support'] == 'bumpers':
            policy = '{}_{}_{}'.format(
                d['args']['learner_support'],
                int(d['args']['bumper_threshold']),
                d['args']['trajectory_distance'],
            )
        elif d['args']['learner_support'] == 'q_bumpers':
            policy = '{}_{}boltz_v{}_{}Target_{}Alph'.format(
                d['args']['learner_support'],
                d['args']['q_bumper_boltzmann'],
                d['args'].get('q_bumper_version', ''),
                d['args'].get('q_bumper_target_r', ''),
                d['args'].get('q_bumper_alpha', ''),
            )
            if d['args'].get('q_bumper_length_normalized'):
                policy += '_lenNormalized'
        else:
            policy = d['args']['learner_support']

        is_undoing = d['args'].get('undoing')
        is_override_next_best = d['args'].get('override_next_best')
        data['p_override'] = d['args'].get('p_override')
        data['undoing'] = is_undoing
        data['override_next_best'] = is_override_next_best

        if is_undoing:
            policy += '_undoing'
        if is_override_next_best:
            policy += '_nextBest'
        data['policy'] = policy

    return data, info


class ResultsPlotter(object):
    def __init__(self, filenames, names, lower_reward_bound=None):
        filenames = filenames
        names = names
        df = pd.DataFrame()
        n_cpus = os.environ.get('SLURM_CPUS_ON_NODE')
        if n_cpus is not None:
            n_cpus = int(n_cpus)
        with Pool(n_cpus) as pool:
            dataframes_with_infos = pool.map(
                load_file_to_dataframe,
                filenames
            )
        dataframes, infos = zip(*[
            (d, info) for d, info in dataframes_with_infos if d is not None
        ])
        self.is_cliff = infos[0]['is_cliff']
        for d, name in zip(dataframes, names):
            d['name'] = format_name(name)
        self.df = pd.concat(dataframes, ignore_index=True)
        self.lower_reward_bound = lower_reward_bound


    def plot_all(self, outdir, heatmaps=False):
        # New small multiples
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        #for category in ['r_discounted', 'r', 'override_action', 'length']:
        #for category in ['r', 'override_action', 'length']:
        if self.is_cliff:
            categories = ['r', 'override_action', 'length']
        else:
            categories = ['r', 'success', 'crash', 'override_action', 'length']

        # TODO: Comment this.
        for x in [0, int(0.5 * self.df['ep'].max()), self.df['ep'].max() - 1]:
            self.plot_small_multiples(
                self.df[self.df['ep'] == x],
                #_df,
                'cum_overrides',
                outdir,
                episode=self.df['ep'].max(),
                col='mode',
                hue='policy',
                x='t',
                name_ending='SINGLE{}ep'.format(x),
            )
        #import sys; sys.exit()

        for category in categories:
            for x in ['ep', 'cum_t']:
                for rolling in [True, False]:
                    if (
                            rolling
                            and not category in [
                                'r', 'success', 'crash', 'override_action',
                            ]
                    ):
                        continue
                    """
                    self.plot_small_multiples(
                        self.df,
                        category,
                        outdir,
                        episode=self.df['ep'].max(),
                        row='policy',
                        x=x,
                        rolling=rolling,
                    )
                    """

                    # Plot single row with unassisted / assisted columns.
                    #for undoing, _df in self.df.groupby('undoing'):
                    self.plot_small_multiples(
                        self.df,
                        #_df,
                        category,
                        outdir,
                        episode=self.df['ep'].max(),
                        col='mode',
                        hue='policy',
                        #row='undoing',
                        #name_ending='_undoing' if undoing else None,
                        x=x,
                        rolling=rolling,
                    )

            if category == 'r' and self.lower_reward_bound is not None:
                self.plot_small_multiples(
                    self.df,
                    #_df,
                    'n_lower_bounds_failed',
                    outdir,
                    episode=self.df['ep'].max(),
                    col='mode',
                    hue='policy',
                    #row='undoing',
                    #name_ending='_undoing' if undoing else None,
                    lower_reward_bound=self.lower_reward_bound,
                    x=x,
                )

            if category == 'r_discounted' and self.lower_reward_bound is not None:
                self.plot_small_multiples(
                    self.df,
                    #_df,
                    'n_lower_bounds_failed_discounted',
                    outdir,
                    episode=self.df['ep'].max(),
                    col='mode',
                    hue='policy',
                    #row='undoing',
                    #name_ending='_undoing' if undoing else None,
                    lower_reward_bound=self.lower_reward_bound,
                )

        # NEW 2/25: Plot cumulative episode length
        self.plot_small_multiples(
            self.df,
            #_df,
            'length_cum',
            outdir,
            episode=self.df['ep'].max(),
            col='mode',
            hue='policy',
            x='ep',
        )

        # NEW: Plot cumulative number of overrides by t
        self.plot_small_multiples(
            self.df,
            #_df,
            'cum_overrides',
            outdir,
            episode=self.df['ep'].max(),
            col='mode',
            hue='policy',
            x='t',
        )

        # NEW: Plot best one can hope for by t
        self.plot_small_multiples(
            self.df,
            #_df,
            'best_hope_for',
            outdir,
            episode=self.df['ep'].max(),
            col='mode',
            hue='policy',
            x='t',
        )

        # NEW2: Plot x and y as function of t.
        #for key in ['cum_r', 'cum_r_discounted', 'y', 'x']:
        for key in ['cum_r_discounted', 'y', 'x']:
            self.plot_small_multiples(
                self.df,
                #_df,
                key,
                outdir,
                episode=self.df['ep'].max(),
                col='mode',
                hue='policy',
                x='t',
            )

        # TODO: Uncomment to go back from small multiples
        for mode, df in self.df.groupby('mode'):
            subdir = os.path.join(outdir, mode)
            if not os.path.exists(subdir):
                os.makedirs(subdir)

            """
            for key in ['length', 'override_action', 'r']:
                self.plot_episode_mean(df, key, os.path.join(subdir, key))
                self.plot_episode_mean(
                    df,
                    key,
                    os.path.join(subdir, key),
                    episode=df['ep'].max(),
                )
            """

            if heatmaps:
                self.plot_heatmaps(df, subdir)

                # Plot first 3 runs individually.
                for run in range(3):
                    self.plot_heatmaps(df, subdir, run=run)

    def plot_small_multiples(
            self,
            df,
            category,
            outdir,
            lims=None,
            episode=None,
            col='undoing',  # was learner_assumption
            hue='mode',
            row=None,
            name_ending=None,
            lower_reward_bound=None,
            # if x ='t', aggregate over episodes instead
            # if x ='cum_t', use cumulative timesteps instead of episodes as x-axis.
            x='ep',
            rolling=False,
            rolling_window=None,
    ):
        if rolling_window is None and self.is_cliff:
            rolling_window = 1
        elif rolling_window is None:
            rolling_window = 100

        df = df.dropna(subset=['a'])  # Exclude terminal state.
        if lims is None:
            if category == 'override_action':
                lims = (0, None)
            elif category == 'length':
                lims = (0, None)
            else:
                lims = (None, None)

        group_bys = [
            'run',
            'ep',
            'learner_assumption',
            'policy',
            'mode',
            'p_override',
            'undoing',
        ]
        if category == 'length':
            assert x in ['ep', 'cum_t']
            cat_name = 't'
            df2 = df.groupby(group_bys)['t'].max().reset_index()
        elif category == 'success':
            assert x in ['ep', 'cum_t']
            cat_name = 'success'
            df2 = df
            df2['success'] = df2['r'] == 100
            df2 = df2.groupby(group_bys)['success'].any().reset_index()
        elif category == 'crash':
            assert x in ['ep', 'cum_t']
            cat_name = 'crash'
            df2 = df
            df2['crash'] = df2['r'] == -100
            df2 = df2.groupby(group_bys)['crash'].any().reset_index()
        elif category == 'length_cum':
            cat_name = category
            df2 = df.groupby(group_bys)['t'].max().reset_index()
            df2 = df2.sort_values([c for c in group_bys if c != 'ep'] + ['ep'])
            df2[category] = df2.groupby([
                c for c in group_bys if c != 'ep'
            ])['t'].cumsum()
        elif 'n_lower_bounds_failed' in category:
            assert x == 'ep'
            if 'discounted' in category:
                key = 'r_discounted'
            else:
                key = 'r'
            cat_name = category
            df2 = df.groupby(group_bys)[key].sum().reset_index()
            df2[category] = df2[key] < lower_reward_bound
            df2[category] = df2[category].astype(int)
            df2[category] = df2.groupby([
                c for c in group_bys if c != 'ep'
            ])[category].cumsum()
        elif category == 'cum_overrides':
            assert x == 't'
            cat_name = category
            df2 = df
            df2['override_action'] = df2['override_action'].astype(int)
            df2[category] = df2.groupby(group_bys)['override_action'].cumsum()
        elif x == 't':
            cat_name = category
            df2 = df
            df2[cat_name] = df2[cat_name].astype(float)
        else:
            df2 = df
            if category == 'override_action':
                df2[category] = df2[category].astype(int)
            cat_name = category
            df2 = df2.groupby(group_bys)[category].sum().reset_index()
            if rolling:
                group_bys_other_than_ep = [c for c in group_bys if c != 'ep']
                df2 = df2.sort_values(by=group_bys_other_than_ep + ['ep']).reset_index(drop=True)
                df3 = df2\
                    .groupby(group_bys_other_than_ep)[category]\
                    .rolling(rolling_window)\
                    .mean()\
                    .reset_index(drop=True)
                df2[category] = df3

        if x == 'cum_t':
            assisted = df[df['mode'] == 'assisted'][group_bys + ['cum_t']]
            assisted = assisted[assisted.cum_t % CUM_T_INTERVAL == 0]

            # Split out unassisted by cum_t
            unassisted = df[df['mode'] == 'unassisted']
            unassisted = unassisted.groupby(group_bys)['cum_t'].first().reset_index()
            unassisted = unassisted.set_index(group_bys)['cum_t']\
                .apply(pd.Series).stack().reset_index()
            unassisted.columns = group_bys + ['cum_t_n', 'cum_t']
            df1 = pd.concat([assisted, unassisted], ignore_index=True)
            df1 = df1[df1.cum_t % CUM_T_INTERVAL == 0]

            df1 = df1.groupby(group_bys + ['cum_t']).last().reset_index()[group_bys + ['cum_t']]
            df2 = df1.merge(df2, on=group_bys, how='left')

        plot = sns.relplot(
            col=col,
            row=row,
            kind='line',
            x=x,
            y=cat_name,
            data=df2,
            hue=hue,
            hue_order=sorted(df2[hue].unique()),
            #style=hue,
            #style='p_override',
            #style_order=[1, 0.75, 0.5, 0.25],
            facet_kws={
                'margin_titles': True,
            },
        )

        #plot.set(ylim=lims[v])
        name = os.path.join(
            outdir,
            '{0}_{1}{2}{3}{4}{5}'.format(
                category,
                x,
                '' if episode is None else '_ep{}'.format(episode),
                '' if row is None else '_{}Rows'.format(row),
                '' if not rolling else '_rolling',
                name_ending or '',
            ),
        )

        plt.savefig(name + '.png')
        plt.clf()
        if row is None:
            sortbys = [col, x, hue]
        else:
            sortbys = [row, col, x, hue]
        df2.groupby(sortbys) \
            .mean()[cat_name] \
            .reset_index() \
            .sort_values(by=sortbys) \
            .to_csv(name + '.csv', index=False)
        if x == 'ep':
            df2.sort_values(by=['run'] + sortbys) \
                .to_csv(name + '_raw.csv', index=False)

            if not rolling and cat_name in ['r', 'success', 'crash']:
                df3 = df2[group_bys + [cat_name]]
                df3 = df3 \
                    .groupby([c for c in group_bys if c != 'ep']) \
                    .mean()[cat_name] \
                    .reset_index()

                df3.groupby([c for c in group_bys if c not in ['ep', 'run']]) \
                    .mean()[cat_name] \
                    .reset_index() \
                    .to_csv(name + '_mean.csv')
                df3 \
                    .groupby([c for c in group_bys if c not in ['ep', 'run']]) \
                    .sem()[cat_name] \
                    .reset_index() \
                    .to_csv(name + '_sem.csv')

                # Save last rolling_window means
                df3 = df2[df2['ep'] > episode - rolling_window][group_bys + [cat_name]]
                df3 = df3 \
                    .groupby([c for c in group_bys if c != 'ep']) \
                    .mean()[cat_name] \
                    .reset_index()

                df3.groupby([c for c in group_bys if c not in ['ep', 'run']]) \
                    .mean()[cat_name] \
                    .reset_index() \
                    .to_csv(name + '_mean_last{}.csv'.format(rolling_window))
                df3 \
                    .groupby([c for c in group_bys if c not in ['ep', 'run']]) \
                    .sem()[cat_name] \
                    .reset_index() \
                    .to_csv(name + '_sem_last{}.csv'.format(rolling_window))

    @staticmethod
    def plot_episode_mean(df, category, out_basename, lims=None, episode=None):
        df = df.dropna(subset=['a'])  # Exclude terminal state.
        if lims is None:
            if category == 'override_action':
                lims = (0, None)
            elif category == 'length':
                lims = (0, None)
            else:
                lims = (None, None)

        if category != 'length':
            if category == 'override_action':
                df[category] = df[category].astype(int)
            cat_name = category
            df = df.groupby(['run', 'ep', 'name', 'support'])[category].sum().reset_index()
        else:
            cat_name = 't'
            df = df.groupby(['run', 'ep', 'name', 'support'])['t'].max().reset_index()

        if episode is None:
            plot = sns.lineplot(
                x='ep',
                y=cat_name,
                data=df,
                hue='name',
                #hue='freedom',
                #hue_norm=(0, 1),
                style='support',
                style_order=[
                    'bumpers',
                    'reddy_rss',
                    'unassisted',
                ],
            )
        else:
            plot = sns.barplot(
                x=cat_name,
                y='name',
                data=df[df['ep'] == episode],
            )

        # Move legend outside figure
        lgd = plot.legend(
            loc='center right',
            bbox_to_anchor=(1.4, 0.5),
        )

        #plot.set(ylim=lims[v])
        plot.figure.savefig(
            '{0}{1}.png'.format(
                out_basename,
                '' if episode is None else '_ep{}'.format(episode),
            ),
            bbox_extra_artists=(lgd,) if episode is None else None,
            bbox_inches='tight',
        )
        plt.clf()
        if episode is None:
            df.groupby(['ep', 'name']) \
                .mean()[cat_name] \
                .reset_index() \
                .sort_values(by=['ep', 'name']) \
                .to_csv(out_basename + '.csv', index=False)

    @staticmethod
    def plot_heatmaps(df, outdir, run=None):
        """"""
        # TODO: slices of exps, filenames
        for name, df_name in df.groupby('name'):
            for start, end in [
                    (0, 10),
                    (df_name['ep'].max() - 10, df_name['ep'].max())
            ]:
                df_slice = df_name[(df_name['ep'] >= start) & (df_name['ep'] < end)]
                if run is not None:
                    df_slice = df_slice[df_slice['run'] == run]
                finaldir = os.path.join(outdir, name, str(end).zfill(3))
                if not os.path.exists(finaldir):
                    os.makedirs(finaldir)

                normalizer = len(df_slice.groupby(['run', 'ep']).first())
                print('normalizer: {}'.format(normalizer))

                # All states visited
                bins = np.zeros(dims)
                for _, row in df_slice.iterrows():
                    bins[row['y']][row['x']] += 1
                bins = bins / normalizer

                sns.heatmap(bins, vmin=0, vmax=None, cmap='Blues')
                plt.savefig(os.path.join(finaldir, 'state_heatmap.png'))
                plt.clf()

                # States with corrections
                df_slice = df_slice.dropna(subset=['a'])
                bins = np.zeros(dims)
                for _, row in df_slice[df_slice.override_action].iterrows():
                    bins[row['y']][row['x']] += 1
                bins = bins / normalizer


                sns.heatmap(bins, vmin=0, vmax=None, cmap='Blues')
                plt.savefig(os.path.join(
                    finaldir,
                    'override_heatmap{}.png'.format(
                        '' if run is None
                        else '_run{}'.format(run)
                    ),
                ))
                plt.clf()




    @staticmethod
    def _to_state_dataframe(d, args=None, run_id=None, gamma=0.99, cum_ts=None):
        """Get a dataframe on a state level."""

        is_cliff = False

        support = None
        env_name = ''
        if args is not None:
            env_name = args['env_name']

            support = args['learner_support']

            # Try to assign amount of "freedom" to numerical vals.
            if support == 'bumpers':
                freedom = args['bumper_threshold'] / 6
            elif support == 'reddy_rss':
                freedom = args['q_threshold']
            else:
                freedom = 1

        #try:
        #    for e in d:
        #        pass
        #except TypeError:
        #    d = [d]

        data = []
        #print(len(d))
        #import sys; sys.exit()
        cum_t = 0
        cum_ts_in_episode = []
        for ep, batch in enumerate(d):
            cum_ts_in_episode.append([])
            for i, rollout in enumerate(batch['rollouts']):
                #for t, (s0, a, r, s1, done, task_idx, info) in enumerate(rollout):
                if i > 0:
                    raise Exception('Need to change cum_t')
                cum_r = 0
                cum_r_discounted = 0
                for t, res in enumerate(rollout):
                    if cum_t % CUM_T_INTERVAL == 0 or t == 0:
                        cum_ts_in_episode[-1].append(cum_t)
                    s0 = res['prev_obs']
                    a = res['action']
                    r = res['r']
                    r_discounted = res['r'] * gamma ** t
                    cum_r += r
                    cum_r_discounted += r_discounted
                    s1 = res['obs']
                    info = res['info']

                    x = None
                    y = None
                    if 'cliff' in env_name.lower():
                        is_cliff = True
                        dims = (4, 12)
                        x = np.unravel_index(s0, dims)[1]
                        y = np.unravel_index(s0, dims)[0]
                    elif 'lunar' in env_name.lower():
                        x = s0[0]
                        y = s0[1]

                    data.append({
                        #'run': run if run_id is None else run_id,
                        'run': run_id,
                        'ep': ep,
                        'i': i,
                        't': t,
                        'cum_t': cum_t if cum_ts is None else cum_ts[ep],
                        's0': s0,
                        'x': x,
                        'y': y,
                        's1': s1,
                        'r': r,
                        'r_discounted': r_discounted,
                        'cum_r': cum_r,
                        'cum_r_discounted': cum_r_discounted,
                        'a': a,
                        'override_action': info['override_action'],
                        'best_hope_for': info.get('support', {}).get('best_hope_for'),
                        'support': support,
                        'freedom': freedom,
                    })
                    if t == len(rollout) - 1:
                        data.append({
                            'run': run_id,
                            'ep': ep,
                            'i': i,
                            't': t + 1,
                            'cum_t': cum_t + 1 if cum_ts is None else cum_ts[ep],
                            's0': s1,
                            'r': None,
                            'a': None,
                            'override_action': None,
                            'best_hope_for': None,
                            'support': support,
                            'freedom': freedom,
                        })
                    cum_t += 1
        return data, {
            'cum_ts': cum_ts_in_episode,
            'is_cliff': is_cliff,
        }


"""
def to_dataframe(d, name=None, args=None):
    data = []
    for run, evals in enumerate(d):
        for ep, batch in enumerate(evals):
            for i, rollout in enumerate(batch['rollouts']):
                override_actions = sum(x[6]['override_action']
                                       for x in rollout)
                length = len(rollout)
                try:
                    succ = batch['succ'][i]
                except TypeError:
                    succ = None
                try:
                    crash = batch['crash'][i]
                except TypeError:
                    crash = None
                rew = batch['rew'][i]

                support = None
                freedom = None
                if args is not None:
                    support = args['learner_support']

                    # Try to assign amount of "freedom" to numerical vals.
                    if support == 'bumpers':
                        freedom = args['bumper_threshold'] / 6
                    elif support == 'reddy_rss':
                        freedom = args['q_threshold']
                    else:
                        freedom = 1

                data.append({
                    'run': run,
                    'ep': ep,
                    'i': i,
                    'succ': succ,
                    'crash': crash,
                    'rew': rew,
                    'length': length,
                    'override_actions': override_actions,
                    'name': format_name(name),
                    'support': support,
                    'freedom': freedom,
                })
    return data
"""


def to_dataframe_run_ep_only(lst, f=None, name=None):
    normalized = []
    for run, eps in enumerate(lst):
        for ep, val in enumerate(eps):
            normalized.append({
                'run': run,
                'ep': ep,
                'v': val if f is None else f(val),
                'name': format_name(name),
            })
    return normalized


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirnames', nargs='+', type=str)
    parser.add_argument('--names', nargs='+', type=str)
    parser.add_argument('--plotdir', type=str)
    parser.add_argument('--heatmaps', action='store_true')
    parser.add_argument('--outdirname', type=str)
    parser.add_argument('--lower_reward_bound', type=float)
    args = parser.parse_args()

    plotdir = args.plotdir
    if plotdir is None:
        plotdir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'plots',
        )

    #base_paths = [os.path.basename(n) for n in args.data]
    #bases = [os.path.splitext(p)[0] for p in base_paths]
    bases = [
        n.split('/')[-1]
        for n in args.dirnames
    ]
    #if not args.names:
    outdir = os.path.join(
        plotdir,
        '_'.join(bases) if args.outdirname is None else args.outdirname,
    )
    #else:
    #    outdir = os.path.join(
    #        plotdir,
    #        '_'.join(args.names),
    #    )
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    names = args.names
    if not names:
        names = bases

    data_names = []
    data_files = []
    for d, name in zip(args.dirnames, names):
        for f in os.listdir(d):
            if f.startswith('data') and f.endswith('.json'):
                data_files.append(os.path.join(d, f))
                data_names.append(name)
    plotter = ResultsPlotter(
        data_files,
        data_names,
        lower_reward_bound=args.lower_reward_bound,
    )
    plotter.plot_all(outdir, heatmaps=args.heatmaps)

    """
    q_values = []
    for data, name in zip(args.data, names):
        with open(data, 'rb') as f:
            d = pickle.load(f)
            try:
                q_vs = to_dataframe_run_ep_only(
                    d['q_values'],
                    f=lambda x: np.linalg.norm(x, ord=2),
                    name=name,
                )
                q_vs = pd.DataFrame(q_vs)
                q_values.append(q_vs)
            except KeyError:
                pass

    if q_values:
        df = pd.concat(q_values)
        plot = sns.lineplot(x='ep', y='v', data=df, hue='name')
        #plot.set(ylim=lims[v])
        plt.ylabel('L2 norm of Q values')
        plot.figure.savefig(os.path.join(
            outdir,
            'q_values_l2_norm.png',
        ))
        plt.clf()
    """



