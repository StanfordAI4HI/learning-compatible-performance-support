#import tracemalloc
import resource
import subprocess
import json
import functools
from multiprocessing import Pool
from configparser import ConfigParser
import datetime
from copy import deepcopy as copy
import argparse
import random
import numpy as np
import os
import tensorflow as tf
from gym.wrappers.monitor import Monitor

from baselines.common.schedules import LinearSchedule

from . import envs
from . import policies
from . import inverse_softq
from .utils import NumpyEncoder
from . import utils


N_VAL_EVAL_ROLLOUTS = 10
SEED = 1

univ_scramble = True
data_dir = os.path.join(
    'data', '1.1-tabular-ime', 'univ_scramble' if univ_scramble else 'scramble')

newton_act_labels, aristotle_act_labels = envs.make_act_labels()

all_goals, train_goals = envs.make_goals(seed=SEED)

#g = envs.GridWorldNav(goal=(0, 0), act_labels=newton_act_labels)

train_newton_envs = [
    envs.GridWorldNav(goal=g, act_labels=newton_act_labels)
    for g in train_goals
]
train_aristotle_envs = [
    envs.GridWorldNav(goal=g, act_labels=aristotle_act_labels)
    for g in train_goals
]

Q = np.stack(
    [policies.tabsoftq_learn(e, verbose=False) for e in train_aristotle_envs],
    axis=0
)

aristotle_pilot_policies = [
    policies.make_tabsoftq_policy(Q[i], use_gumbel=True)
    for i in range(len(train_goals))
]


# Generate rollouts
def make_rollouts(policy, env, n, task_idx, render=False, display=None):
    return [
        envs.run_ep(
            policy,
            env,
            render=render,
            task_idx=task_idx,
            display=display,
        )
        for _ in range(n)
    ]


def randargmax(b):
    return np.random.choice(np.flatnonzero(b == b.max()))

def get_learner_policy(
        s,
        #test_goal, train_act_labels, test_act_labels,
        #n_act_dim,
        env,
        sess=None, model=None, Q=None,
        exploration_fraction=None,
        exploration_final_eps=None,
        exploration_final_lr=None,
        total_episodes=None,
        run=None,
):
    """Learner policy"""
    if s == 'random':
        from .agents import RandomAgent
        _policy = RandomAgent(env=env, seed=run)
    elif s == 'aristotle':
        raise NotImplementedError
        _policy = policies.make_perfect_pilot_policy(
            goal=test_goal,
            act_labels=train_act_labels,
        )
    elif s == 'inverse_softq':
        raise NotImplementedError
        # TODO: Don't assume first goal?
        _policy = lambda st: np.argmax(sess.run(model.q_t)[0], axis=1)[st]
    elif s == 'inverse_softq_learn':
        raise NotImplementedError
        # Learn from dynamics
        _T = model.get_internal_model(sess)
        obs_tp1 = np.argmax(_T, axis=2)
        __env = policies.make_val_assisted_env(
            obs_tp1=obs_tp1,
            act_labels=test_act_labels,
            n_act_dim=n_act_dim,
            goal=test_goal,
            threshold=float('inf'),
        )
        _Q = policies.tabsoftq_learn(__env, _T)
        _policy = policies.make_tabsoftq_policy(_Q, use_gumbel=False)
    elif s == 'q':
        from .agents import TabularQLearningAgent
        _policy = TabularQLearningAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            eps_schedule=LinearSchedule(
                schedule_timesteps=int(exploration_fraction * total_episodes),
                initial_p=1.0,
                final_p=exploration_final_eps,
            ),
            lr_schedule=LinearSchedule(
                schedule_timesteps=int(exploration_fraction * total_episodes),
                initial_p=1.0,
                final_p=exploration_final_lr,
            ),
            seed=run,
        )
    elif s == 'dqn':
        from .agents import DQNLearningAgent
        _policy = DQNLearningAgent(
            env=env,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            total_episodes=total_episodes,
            seed=run,
        )
    else:
        raise ValueError

    return _policy


def get_support_env(s, goal, test_act_labels, n_act_dim, threshold=None,
                    model=None, sess=None, test_env=None, q_threshold=None,
                    q_bumper_boltzmann=None,
                    q_bumper_version=None,
                    q_bumper_target_r=None,
                    q_bumper_length_normalized=False,
                    q_bumper_logistic_upper_prob=None,
                    q_bumper_alpha=None,
                    start_pos=None,
                    env_name=None,
                    trajectory_distance=None,
                    dirname=None,
                    p_override=None,
                    undoing=None,
                    p_suboptimal_override=None,
                    override_next_best=None,
                    optimal_agent_training_timesteps=None,
                    optimal_agent_smoothing_timesteps=None,
                    gamma=None,
                    ):
    if s == 'dyn_assisted':
        _T = model.get_internal_model(sess)
        obs_tp1 = np.argmax(_T, axis=2)
        _env = policies.make_val_assisted_env(
            act_labels=test_act_labels,
            n_act_dim=n_act_dim,
            obs_tp1=obs_tp1,
            goal=goal,
            threshold=None,
            dyn_transfer=True,
            start_pos=start_pos,
            env_name=env_name,
            p_override=p_override,
            undoing=undoing,
            p_suboptimal_override=p_suboptimal_override,
            override_next_best=override_next_best,
            gamma=gamma,
        )
    elif s == 'bumpers':
        assert(threshold is not None)
        """
        _env = policies.make_val_assisted_env(
            act_labels=test_act_labels,
            n_act_dim=n_act_dim,
            goal=goal,
            threshold=threshold,
            dyn_transfer=False,
            start_pos=start_pos,
            env_name=env_name,
        )
        """
        _env = policies.make_env(
            env_name=env_name,
            support_name=s,
            threshold=threshold,
            trajectory_distance=trajectory_distance,
            dirname=dirname,
            p_override=p_override,
            undoing=undoing,
            p_suboptimal_override=p_suboptimal_override,
            override_next_best=override_next_best,
            optimal_agent_training_timesteps=optimal_agent_training_timesteps,
            optimal_agent_smoothing_timesteps=optimal_agent_smoothing_timesteps,
            gamma=gamma,
        )
    elif s == 'q_bumpers':
        assert(q_bumper_boltzmann is not None)
        """
        _env = policies.make_val_assisted_env(
            act_labels=test_act_labels,
            n_act_dim=n_act_dim,
            goal=goal,
            threshold=threshold,
            dyn_transfer=False,
            start_pos=start_pos,
            env_name=env_name,
        )
        """
        _env = policies.make_env(
            env_name=env_name,
            support_name=s,
            q_bumper_boltzmann=q_bumper_boltzmann,
            q_bumper_version=q_bumper_version,
            dirname=dirname,
            p_override=p_override,
            undoing=undoing,
            p_suboptimal_override=p_suboptimal_override,
            override_next_best=override_next_best,
            q_bumper_target_r=q_bumper_target_r,
            optimal_agent_training_timesteps=optimal_agent_training_timesteps,
            optimal_agent_smoothing_timesteps=optimal_agent_smoothing_timesteps,
            q_bumper_length_normalized=q_bumper_length_normalized,
            q_bumper_logistic_upper_prob=q_bumper_logistic_upper_prob,
            q_bumper_alpha=q_bumper_alpha,
            gamma=gamma,
        )
    elif s == 'unassisted':
        """
        _env = policies.make_val_assisted_env(
            act_labels=test_act_labels,
            n_act_dim=n_act_dim,
            goal=goal,
            threshold=None,
            start_pos=start_pos,
            env_name=env_name,
        )
        """
        _env = policies.make_env(
            env_name=env_name,
            support_name=s,
            dirname=dirname,
            gamma=gamma,
        )
    elif s == 'random_pushes':
        _env = policies.make_env(
            env_name=env_name,
            support_name=s,
            dirname=dirname,
            gamma=gamma,
            optimal_agent_training_timesteps=optimal_agent_training_timesteps,
            optimal_agent_smoothing_timesteps=optimal_agent_smoothing_timesteps,
        )
    elif s == 'reddy_rss':
        # TODO: Fix so don't assume first task
        assert(q_threshold is not None)
        """
        Q = policies.tabsoftq_learn(test_env, verbose=False)
        _env = policies.make_val_assisted_env(
            act_labels=test_act_labels,
            n_act_dim=n_act_dim,
            goal=goal,
            threshold=None,
            q_threshold=q_threshold,
            optimal_Q=Q,
            dyn_transfer=False,
            start_pos=start_pos,
            env_name=env_name,
        )
        """
        _env = policies.make_env(
            env_name=env_name,
            support_name=s,
            q_threshold=q_threshold,
            dirname=dirname,
            p_override=p_override,
            undoing=undoing,
            p_suboptimal_override=p_suboptimal_override,
            override_next_best=override_next_best,
            optimal_agent_training_timesteps=optimal_agent_training_timesteps,
            optimal_agent_smoothing_timesteps=optimal_agent_smoothing_timesteps,
            gamma=gamma,
        )
    else:
        raise NotImplementedError

    return _env



def compute_assisted_perf(
        test_env,
        policy,
        goal,
        sess=None,
        model=None,
        seed=None,
        n_eval_rollouts=N_VAL_EVAL_ROLLOUTS,
        task_idx=0,
        verbose=False,
        policy_update=False,
        policy_explore=False,
        user_action=None,
        override_penalty=0,
        override_penalty_only=False,
):
    """

    learn_softq: Use estimated internal model and known rewards
        to train an agent using inverse soft q learning.

    Args:
        policy (Optional): Defaults to learned policy for first task.

    """
    # TODO: Move to gridworld dir?
    # TODO: currently, extracting policy for first task by default.

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    rollouts = []
    for i in range(n_eval_rollouts):
        rollouts.append(envs.run_ep(
            policy=policy,
            env=test_env,
            render=False,
            task_idx=task_idx,
            seed=seed + [i] if seed is not None else None,
            policy_explore=policy_explore,
            policy_update=policy_update,
            user_action=user_action,
            override_penalty=override_penalty,
            override_penalty_only=override_penalty_only,
        ))
        if verbose:
            #all_states = [x[0] for x in rollouts[-1]] + [rollouts[-1][-1][3]]
            #print([list(policies.featurize_state(x)) for x in all_states])
            print('start: {}'.format(policies.featurize_state(rollouts[-1][0]['prev_obs'])))

    perf = {}
    rew = [sum(x['r'] for x in r) for r in rollouts]
    try:
        succ = [
            1 if test_env.is_succ(r) else 0
            for r in rollouts
        ]
        crash = [
            1 if test_env.is_crash(r) else 0
            for r in rollouts
        ]
    except:
        succ = None
        crash = None
    perf.update({
        'rew': rew,
        'succ': succ,
        'crash': crash,
        'rollouts': rollouts,
    })

    return perf


"""
snapshot = None
tracemalloc.start()
"""

def get_learner_assumption_kwargs(args):
    """Get learner assumptions from command line arguments"""
    assumptions = {
        'override_penalty': args.teacher_correction_penalty,
        'override_penalty_only': args.teacher_correction_penalty_only,
    }
    if args.perceive_teacher_actions in ['all_own', 'own_only']:
        assumptions['user_action'] = args.perceive_teacher_actions
    return assumptions


def do_run(run, dirname, args):
    """
    global snapshot
    snapshot2 = tracemalloc.take_snapshot()
    print(('MEMORY', run, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if snapshot is not None:
        top_stats = snapshot2.compare_to(snapshot, 'lineno')
        print("[ Top 10 differences ]")
        for stat in top_stats[:10]:
            print(stat)
        print()
    snapshot = snapshot2
    """
    with tf.Graph().as_default():
        learner_assumptions = get_learner_assumption_kwargs(args)

        # Each run has a different random seed equal to the run id.
        np.random.seed(run)
        random.seed(run)

        is_gridworld = not 'lunar' in args.env_name.lower()

        # TODO: Reset test goal inside here? Or use environment instead?
        rollouts = [[]]
        # Initialize model with wrong transition model based on aristotle learner.
        rollouts[0] += make_rollouts(
            #policy=aristotle_pilot_policies[0],  # Was from a noisy policy.
            policy=policies.make_perfect_pilot_policy(
                goal=test_goal,
                act_labels=train_act_labels,
            ),
            env=test_env,
            n=args.n_initial_rollouts,
            task_idx=task_idx,
        )
        assert(len(rollouts[0]) == args.n_initial_rollouts)
        rollouts[0] += make_rollouts(
            #policy=aristotle_pilot_policies[0],  # Was from a noisy policy.
            policy=policies.make_perfect_pilot_policy(
                goal=test_goal,
                act_labels=train_act_labels,
            ),
            env=wrong_train_env,
            n=args.n_initial_wrong_rollouts,
            task_idx=task_idx,
        )

        model = None
        Q = None
        start_pos = None

        logs = []
        evals = []
        evals_unassisted = []
        learner_q_values = []
        with tf.Session() as sess:
            if needs_model:
                model = inverse_softq.InverseSoftQModel(
                    train_envs=[test_env]
                )

            # NOTE: Used to be inside episode loop!
            # TODO: Check if this broke anything!
            support_env = get_support_env(
                s=args.learner_support,
                model=model,
                sess=sess,
                goal=test_goal,
                test_act_labels=test_act_labels,
                n_act_dim=n_act_dim,
                threshold=args.bumper_threshold,
                q_bumper_boltzmann=args.q_bumper_boltzmann,
                q_bumper_version=args.q_bumper_version,
                q_bumper_target_r=args.q_bumper_target_r,
                q_bumper_length_normalized=args.q_bumper_length_normalized,
                q_bumper_logistic_upper_prob=args.q_bumper_logistic_upper_prob,
                q_bumper_alpha=args.q_bumper_alpha,
                q_threshold=args.q_threshold,
                test_env=test_env,
                env_name=args.env_name,
                start_pos=start_pos,
                trajectory_distance=args.trajectory_distance,
                dirname=dirname,
                p_override=args.p_override,
                undoing=args.undoing,
                p_suboptimal_override=args.p_suboptimal_override,
                override_next_best=args.override_next_best,
                optimal_agent_training_timesteps=args.optimal_agent_training_timesteps,
                optimal_agent_smoothing_timesteps=args.optimal_agent_smoothing_timesteps,
                gamma=args.gamma,
            )
            policy = get_learner_policy(
                s=args.learner_policy,
                #model=model,
                #sess=sess,
                #test_goal=test_goal,
                #train_act_labels=train_act_labels,
                #test_act_labels=test_act_labels,
                #n_act_dim=n_act_dim,
                #Q=Q,
                env=support_env,
                exploration_fraction=args.exploration_fraction,
                exploration_final_eps=args.exploration_final_eps,
                exploration_final_lr=args.exploration_final_lr,
                total_episodes=args.n_episodes,
                run=run,
            )


            for ep in range(args.n_episodes):
                #print('Rn: {} Ep: {}'.format(run, ep), flush=True)
                support_env_with_monitor = Monitor(
                    support_env,
                    directory=os.path.join(
                        dirname,
                        'assisted',
                        str(run).zfill(3),
                        str(ep).zfill(3),
                    ),
                    force=True,
                    video_callable=lambda e: True if is_gridworld or utils.IS_LOCAL else False,
                    #video_callable=(lambda e: True) if is_gridworld else None,
                )
                # Simulate human learning
                """
                if args.learner_policy == 'q':
                    assert(args.n_learn_rollouts > 0)
                    Q = policies.q_learning(
                        rollouts if ep == 0 else [rollouts[0][-args.n_learn_rollouts:]],
                        n_obs_dim=n_obs_dim,
                        n_act_dim=n_act_dim,
                        user_action=args.think_all_actions_own,
                        Q_init=Q,
                        learning_rate=args.q_learning_rate,
                    )
                """

                _logs = None
                if needs_model:
                    _logs = inverse_softq.run_learning(
                        model=model,
                        sess=sess,
                        # train_tasks=train_aristotle_envs[:1],
                        rollouts=rollouts,
                        test_goal=test_goal,
                        test_act_labels=test_act_labels,
                        train_act_labels=train_act_labels,
                        n_iters=args.n_softq_train_iters,
                        train_frac=0.9,  # TODO: Change to 1
                        **learner_assumptions
                    )

                # Test
                #episode_seed = [run, ep]

                perf = compute_assisted_perf(
                    model=model,
                    sess=sess,
                    #test_act_labels=test_act_labels,
                    #train_act_labels=train_act_labels,
                    test_env=support_env_with_monitor,
                    policy=policy,
                    goal=test_goal,
                    #seed=episode_seed,
                    n_eval_rollouts=args.n_eval_rollouts,
                    policy_explore=True,
                    policy_update=True,
                    **learner_assumptions
                )

                unassisted_perf = None
                if args.n_eval_unassisted_rollouts is not None:
                    unassisted_support_env = get_support_env(
                        s='unassisted',
                        goal=test_goal,
                        test_act_labels=test_act_labels,
                        n_act_dim=n_act_dim,
                        test_env=test_env,
                        env_name=args.env_name,
                        start_pos=start_pos,
                        trajectory_distance=args.trajectory_distance,
                        dirname=dirname,
                    )
                    unassisted_support_env_with_monitor = Monitor(
                        unassisted_support_env,
                        directory=os.path.join(
                            dirname,
                            'unassisted',
                            str(run).zfill(3),
                            str(ep).zfill(3),
                        ),
                        force=True,
                        video_callable=lambda e: True if is_gridworld or utils.IS_LOCAL else False,
                        #video_callable=(lambda e: True) if is_gridworld else None,
                    )
                    unassisted_perf = compute_assisted_perf(
                        model=model,
                        sess=sess,
                        #test_act_labels=test_act_labels,
                        #train_act_labels=train_act_labels,
                        test_env=unassisted_support_env_with_monitor,
                        policy=policy,
                        goal=test_goal,
                        #seed=episode_seed,
                        n_eval_rollouts=args.n_eval_unassisted_rollouts,
                        policy_explore=False,
                        policy_update=False,
                    )
                    unassisted_support_env_with_monitor.close()
                    unassisted_support_env.close()

                new_rollouts = perf['rollouts']
                rollouts[task_idx] += new_rollouts[:args.n_learn_rollouts]
                if _logs is not None:
                    logs.append(_logs)
                evals.append(perf)
                evals_unassisted.append(unassisted_perf)
                if args.learner_policy == 'q':
                    learner_q_values.append(copy(policy.Q))

                support_env_with_monitor.close()

        support_env.close()
        policy.close()

        out_d = {
                'logs': logs,
                'evals': evals,
                'evals_unassisted': (
                    evals_unassisted
                    if args.n_eval_unassisted_rollouts is not None
                    else None
                ),
                'q_values': learner_q_values,
                'args': vars(args),
                'run': run,
                'support_details': support_env.get_support_details(),
        }
        with open(
                os.path.join(dirname, 'data{}.json'.format(str(run).zfill(3))),
                'w',
        ) as f:
            json.dump(out_d, f, cls=NumpyEncoder)



if __name__ == '__main__':
    conf_parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False
    )
    conf_parser.add_argument('-c', '--conf_file', metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()

    arg_defaults = {
        'perceive_teacher_actions': 'all',
        'teacher_correction_penalty': 0,
        'teacher_correction_penalty_only': False,
        'p_override': 1,
        'p_suboptimal_override': 1,
        'undoing': False,
        'override_next_best': False,
        'n_softq_train_iters': 10000,
        'n_eval_rollouts': 1,  # Used to be 100 (faster if retraining is expensive)
        'n_eval_unassisted_rollouts': 1,
        'n_learn_rollouts': 1,  # DEPRECATED
        'n_runs_start_index': 0,
        'n_runs': 100,  # Used to be 1 (faster if retraining is expensive)
        'n_initial_rollouts': 0,
        'n_initial_wrong_rollouts': 0,
        'n_episodes': 100,
        'learner_policy': 'q',
        'learner_support': 'unassisted',
        'bumper_threshold': 0,
        'q_bumper_boltzmann': 1,  # No exploration.
        'q_bumper_version': 0,
        'q_bumper_target_r': None,
        'q_bumper_logistic_upper_prob': None,
        'q_bumper_length_normalized': False,
        'q_bumper_alpha': 1,  # Rescale advantage normalization by alpha
        'trajectory_distance': 'timestep',
        'q_threshold': 0.1,  # Have to be at least 90% of optimal
        'exploration_fraction': 0.9,
        'exploration_final_eps': 0.02,
        'exploration_final_lr': 0.02,
        'env_name': 'CliffWalking-treasure100-v0',
        'max_processes': None,
        'optimal_agent_training_timesteps': int(4e5),
        #'optimal_agent_smoothing_timesteps': int(4e5),
        'optimal_agent_smoothing_timesteps': None,
        'gamma': None,
    }

    if args.conf_file:
        config = ConfigParser.SafeConfigParser()
        config.read([args.conf_file])
        arg_defaults.update(dict(config.items("Defaults")))

    parser = argparse.ArgumentParser(
        parents=[conf_parser]
    )
    parser.set_defaults(**arg_defaults)
    parser.add_argument(
        '--env_name',
        choices=[
            'CliffWalking-treasure100-v0',
            'CliffWalking-nocliff-treasure100-v0',
            'CliffWalking-nocliff-treasure100-v0',
            'LunarLander-v2',
            'LunarLanderFixed-v2',
            'LunarLanderFixedHalfY-v2',
            #'GridWorld',  # Original Reddy environment. Not implemented.
        ]
    )
    parser.add_argument(
        '--learner_policy',
        choices=[
            'q',
            'dqn',
            'inverse_softq',
            'inverse_softq_learn',
            'aristotle',
            'random',
        ]
    )
    parser.add_argument(
        '--trajectory_distance',
        choices=[
            'timestep',
            'nearest',
            'wait',
        ],
        help='Method for computing distance to optimal trajectory',
    )
    parser.add_argument(
        '--learner_support',
        choices=[
            'dyn_assisted',
            'unassisted',
            'bumpers',
            'reddy_rss',
            'q_bumpers',
            'random_pushes',
        ]
    )
    parser.add_argument(
        '--bumper_threshold',
        type=float,
    )
    parser.add_argument(
        '--q_bumper_boltzmann',
        type=float,
    )
    parser.add_argument(
        '--q_bumper_version',
        type=int,
    )
    parser.add_argument(
        '--q_bumper_length_normalized',
        action='store_true',
    )
    parser.add_argument(
        '--q_bumper_target_r',
        type=float,
    )
    parser.add_argument(
        '--q_bumper_logistic_upper_prob',
        type=float,
    )
    parser.add_argument(
        '--q_bumper_alpha',
        type=float,
    )
    parser.add_argument(
        '--q_threshold',
        type=float,
    )
    parser.add_argument(
        '--gamma',
        type=float,
        help='Gamma for support, which is separate from environment',
    )
    parser.add_argument(
        '--exploration_fraction',
        type=float,
        help='Exploration fraction for learner',
    )
    parser.add_argument(
        '--exploration_final_eps',
        type=float,
        help='Final epsilon for q learner',
    )
    parser.add_argument(
        '--exploration_final_lr',
        type=float,
        help='Final learning rate for q learner',
    )
    parser.add_argument(
        '--perceive_teacher_actions',
        choices=[
            'all_own',
            'own_only',
            'all',
        ],
        help='How the learner perceives the teacher actions: all_own (thinks all teacher actions are own), own_only (ignores all teacher actions), or all (perceives all teacher actions)',
    )
    parser.add_argument(
        '--teacher_correction_penalty',
        type=float,
        help='Penalty to add to the reward learner observes for corrective teacher action',
    )
    parser.add_argument(
        '--teacher_correction_penalty_only',
        action='store_true',
        help='If true, ignore true reward for corrective teacher action and only use penalty',
    )
    parser.add_argument(
        '--undoing',
        action='store_true',
        help='Try to undo actions when overriding',
    )
    parser.add_argument(
        '--override_next_best',
        action='store_true',
        help='When overriding, select next best action',
    )
    parser.add_argument(
        '--p_suboptimal_override',
        type=float,
        help='Probability of taking suboptimal overriding action (either args.undoing or args.override_next_best.',
    )
    parser.add_argument('--optimal_agent_training_timesteps', type=int)
    parser.add_argument('--optimal_agent_smoothing_timesteps', type=int)
    parser.add_argument('--p_override', type=float, help='Probability of support overriding.')
    parser.add_argument('--n_softq_train_iters', type=int)
    parser.add_argument('--n_eval_rollouts', type=int)
    parser.add_argument('--n_eval_unassisted_rollouts', type=int)
    parser.add_argument('--n_learn_rollouts', type=int)  # DEPRECATED
    parser.add_argument('--n_runs', type=int)
    parser.add_argument('--n_runs_start_index', type=int)
    parser.add_argument('--max_processes', type=int)
    parser.add_argument('--n_initial_rollouts', type=int)
    parser.add_argument('--n_initial_wrong_rollouts', type=int)
    parser.add_argument('--n_episodes', type=int)
    parser.add_argument('--name', type=str)

    args = parser.parse_args(remaining_argv)
    if args.undoing and args.override_next_best:
        raise ValueError('Cannot define multiple suboptimal action overrides')

    print(vars(args))
    #import sys; sys.exit()


    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        'data',
    )
    if args.name:
        dirname = os.path.join(data_dir, args.name)
    else:
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        dirname = os.path.join(data_dir, suffix)


    test_goal = train_goals[0]
    print('TEST_GOAL: {}'.format(test_goal))

    train_act_labels = aristotle_act_labels
    test_act_labels = newton_act_labels

    task_idx = 0

    wrong_train_env = train_aristotle_envs[0]
    test_env = train_newton_envs[0]
    n_obs_dim = test_env.unwrapped.R.shape[0]
    n_act_dim = test_env.unwrapped.R.shape[1]



    test_goal = train_goals[0]

    needs_model = (
        (
            args.learner_policy in [
                'inverse_softq',
                'inverse_softq_learn',
            ]
        )
        or (
            args.learner_support == 'dyn_assisted'
        )
    )


    #logs = []
    #evals = []
    #evals_unassisted = []
    #q_values = []

    n_cpus = utils.num_cpu()
    if n_cpus is not None:
        n_cpus = int(n_cpus)
    if args.max_processes is not None:
        n_cpus = min(n_cpus or os.cpu_count(), args.max_processes)

    #print(('MEMORY', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if n_cpus is None or n_cpus > 1:
        with Pool(n_cpus, maxtasksperchild=1) as pool:
            pool.map(
                functools.partial(do_run, dirname=dirname, args=args),
                range(
                    args.n_runs_start_index,
                    args.n_runs_start_index + args.n_runs,
                ),
            )
    else:
        for i in range(args.n_runs_start_index,
                       args.n_runs_start_index + args.n_runs):
            do_run(i, dirname=dirname, args=args)

    #print(('MEMORY', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    #for run in range(args.n_runs):
    #    do_run(run, dirname=dirname, args=args)

    """
    with open(os.path.join(dirname, 'data.pkl'), 'wb') as f:
        pickle.dump({
            'logs': logs,
            'evals': evals,
            'evals_unassisted': (
                evals_unassisted
                if args.n_eval_unassisted_rollouts is not None
                else None
            ),
            'q_values': q_values,
            'args': vars(args)
        }, f, pickle.HIGHEST_PROTOCOL)
    """


    """
    # Compress detailed logs
    subprocess.call([
        'tar',
        '-czf',
        os.path.join(dirname, 'extras.tar.gz'),
        '-C',
        '.',
        os.path.join(dirname, 'unassisted'),
        os.path.join(dirname, 'assisted'),
    ])
    subprocess.call([
        'rm',
        '-r',
        os.path.join(dirname, 'unassisted'),
        os.path.join(dirname, 'assisted'),
    ])

    # Record status
    with open(os.path.join(dirname, 'status'), 'w') as f:
        f.write('compressed\n')
    """
