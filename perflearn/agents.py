import os
import json
import tempfile
import joblib

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

from gym.utils import seeding

import baselines.common.tf_util as U
#from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

#from baselines.common.tf_util import get_session
from baselines.common.tf_util import make_session
from baselines.deepq.models import build_q_func

from baselines.deepq.deepq import ActWrapper

from . import utils
from .utils import randargmax
from .utils import IS_LOCAL


def get_session(config=None, num_cpu=None):
    """Get default session or create one with a given config"""
    if num_cpu is None:
        num_cpu = utils.num_cpu()

    sess = tf.get_default_session()
    if sess is None:
        sess = make_session(config=config, num_cpu=num_cpu, make_default=True)
    return sess


def save_variables(save_path, variables=None, sess=None, scope=None):
    sess = sess or get_session()
    variables = variables or tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope,
    )

    ps = sess.run(variables)
    save_dict = {v.name: value for v, value in zip(variables, ps)}
    dirname = os.path.dirname(save_path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(save_dict, save_path)


def load_variables(load_path, variables=None, sess=None, scope=None):
    sess = sess or get_session()
    variables = variables or tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope,
    )

    loaded_params = joblib.load(os.path.expanduser(load_path))
    restores = []
    if isinstance(loaded_params, list):
        assert len(loaded_params) == len(
            variables), 'number of variables loaded mismatches len(variables)'
        for d, v in zip(loaded_params, variables):
            restores.append(v.assign(d))
    else:
        for v in variables:
            restores.append(v.assign(loaded_params[v.name]))

    sess.run(restores)


class Agent(object):
    def __init__(self, env, seed=None):
        self.env = env
        self.np_random, _ = seeding.np_random(seed)

    @property
    def n_states(self):
        return self.env.observation_space.n

    @property
    def n_actions(self):
        return self.env.action_space.n

    def act(self, s, explore):
        raise NotImplementedError

    def update(self, s, a, s1, r, done, verbose=False):
        pass

    def close(self):
        pass


def make_obs_ph(name):
    return ObservationInput(observation_space, name=name)


class DQNLearningAgent(Agent):
    def __init__(
        self,
        env,
        # observation_space,
        # action_space,
        network=None,
        scope='deepq',
        seed=None,
        lr=None,  # Was 5e-4
        lr_mc=5e-4,
        total_episodes=None,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=None,  # was 0.02
        train_freq=1,
        train_log_freq=100,
        batch_size=32,
        print_freq=100,
        checkpoint_freq=10000,
        # checkpoint_path=None,
        learning_starts=1000,
        gamma=None,
        target_network_update_freq=500,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6,
        save_path=None,
        load_path=None,
        save_reward_threshold=None,
        **network_kwargs
    ):
        super().__init__(env, seed)
        if train_log_freq % train_freq != 0:
            raise ValueError(
                'Train log frequency should be a multiple of train frequency')
        elif checkpoint_freq % train_log_freq != 0:
            raise ValueError(
                'Checkpoint freq should be a multiple of train log frequency, or model saving will not be logged properly')
        print('init dqnlearningagent')
        self.train_log_freq = train_log_freq
        self.scope = scope
        self.learning_starts = learning_starts
        self.save_reward_threshold = save_reward_threshold
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.total_episodes = total_episodes
        self.total_timesteps = total_timesteps
        # TODO: scope not doing anything.
        if network is None and 'lunar' in env.unwrapped.spec.id.lower():
            if lr is None:
                lr = 1e-3
            if exploration_final_eps is None:
                exploration_final_eps = 0.02
            #exploration_fraction = 0.1
            #exploration_final_eps = 0.02
            target_network_update_freq = 1500
            #print_freq = 100
            # num_cpu = 5
            if gamma is None:
                gamma = 0.99

            network = 'mlp'
            network_kwargs = {
                'num_layers': 2,
                'num_hidden': 64,
            }

        self.target_network_update_freq = target_network_update_freq
        self.gamma = gamma

        get_session()
        # set_global_seeds(seed)
        # TODO: Check whether below is ok to substitue for set_global_seeds.
        try:
            import tensorflow as tf
            tf.set_random_seed(seed)
        except ImportError:
            pass

        self.q_func = build_q_func(network, **network_kwargs)

        # capture the shape outside the closure so that the env object is not serialized
        # by cloudpickle when serializing make_obs_ph

        def make_obs_ph(name):
            return ObservationInput(env.observation_space, name=name)

        act, self.train, self.train_mc, self.update_target, debug = deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=self.q_func,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr),
            optimizer_mc=tf.train.AdamOptimizer(learning_rate=lr_mc),
            gamma=gamma,
            grad_norm_clipping=10,
            param_noise=False,
            scope=scope,
            # reuse=reuse,
        )

        act_params = {
            'make_obs_ph': make_obs_ph,
            'q_func': self.q_func,
            'num_actions': env.action_space.n,
        }

        self._act = ActWrapper(act, act_params)

        self.print_freq = print_freq
        self.checkpoint_freq = checkpoint_freq
        # Create the replay buffer
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps

        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_size,
                alpha=prioritized_replay_alpha,
            )
            if prioritized_replay_beta_iters is None:
                if total_episodes is not None:
                    raise NotImplementedError(
                        'Need to check how to set exploration based on episodes')
                prioritized_replay_beta_iters = total_timesteps
            self.beta_schedule = LinearSchedule(
                prioritized_replay_beta_iters,
                initial_p=prioritized_replay_beta0,
                final_p=1.0,
            )
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)
            self.replay_buffer_mc = ReplayBuffer(buffer_size)
            self.beta_schedule = None
        # Create the schedule for exploration starting from 1.
        self.exploration = LinearSchedule(
            schedule_timesteps=int(
                exploration_fraction * total_timesteps if total_episodes is None else total_episodes),
            initial_p=1.0,
            final_p=exploration_final_eps,
        )

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        self.update_target()

        self.episode_lengths = [0]
        self.episode_rewards = [0.0]
        self.discounted_episode_rewards = [0.0]
        self.start_values = [None]
        self.lunar_crashes = [0]
        self.lunar_goals = [0]
        self.saved_mean_reward = None

        self.td = None
        if save_path is None:
            self.td = tempfile.mkdtemp()
            outdir = self.td
            self.model_file = os.path.join(outdir, "model")
        else:
            outdir = os.path.dirname(save_path)
            os.makedirs(outdir, exist_ok=True)
            self.model_file = save_path
        print('DQN agent saving to:', self.model_file)
        self.model_saved = False

        if tf.train.latest_checkpoint(outdir) is not None:
            # TODO: Check scope addition
            load_variables(self.model_file, scope=self.scope)
            # load_variables(self.model_file)
            logger.log('Loaded model from {}'.format(self.model_file))
            self.model_saved = True
            raise Exception('Check that we want to load previous model')
        elif load_path is not None:
            # TODO: Check scope addition
            load_variables(load_path, scope=self.scope)
            # load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))

        self.train_log_file = None
        if save_path and load_path is None:
            self.train_log_file = self.model_file + '.log.csv'
            with open(self.train_log_file, 'w') as f:
                cols = [
                    'episode',
                    't',
                    'td_max',
                    'td_mean',
                    '100ep_r_mean',
                    '100ep_r_mean_discounted',
                    '100ep_v_mean',
                    '100ep_n_crashes_mean',
                    '100ep_n_goals_mean',
                    'saved_model',
                    'smoothing',
                ]
                f.write(','.join(cols) + '\n')

        self.training_episode = 0
        self.t = 0
        self.episode_t = 0

        """
        n = observation_space.n
        m = action_space.n
        self.Q = np.zeros((n, m))

        self._lr_schedule = lr_schedule
        self._eps_schedule = eps_schedule
        self._boltzmann_schedule = boltzmann_schedule
        """

        # Make placeholder for Q values
        self.q_values = debug['q_values']

    def _log_training_details(
            self,
            episode=None,
            t=None,
            td_max=None,
            td_mean=None,
            r_mean=None,
            r_mean_discounted=None,
            v_mean=None,
            n_crashes_mean=None,
            n_goals_mean=None,
            saved_model=False,
            smoothing=False,
    ):
        if self.train_log_file is not None:
            with open(self.train_log_file, 'a+') as f:
                f.write('{}\n'.format(','.join([
                    str(episode),
                    str(t),
                    '{:.5f}'.format(td_max) if td_max is not None else '',
                    '{:.5f}'.format(td_mean) if td_mean is not None else '',
                    '{:.1f}'.format(r_mean) if r_mean is not None else '',
                    '{:.1f}'.format(
                        r_mean_discounted) if r_mean_discounted is not None else '',
                    '{:.1f}'.format(v_mean) if v_mean is not None else '',
                    '{:.1f}'.format(
                        n_crashes_mean) if n_crashes_mean is not None else '',
                    '{:.1f}'.format(
                        n_goals_mean) if n_goals_mean is not None else '',
                    str(int(saved_model)),
                    str(int(smoothing)),
                ])))

    def get_q_values(self, s):
        return self.q_values(s)[0]
        """
        q_t = self.q_func(
            self.obs_t_input.get(),
            self.n_actions,
            scope='q_func',
            reuse=True,  # reuse parameters from act
        )
            Q = sess.run(
                Q_values,
                feed_dict={Q_obs: np.array(states)}
            )

        raise NotImplementedError
        """

    def act(self, s, explore, explore_eps=None):
        # Take action and update exploration to the newest value
        # get_session()
        obs = s
        if explore and explore_eps is None:
            update_eps = self.exploration.value(
                self.t if self.total_episodes is None
                else self.training_episode
            )
        elif explore:
            update_eps = explore_eps
        else:
            update_eps = 0
        return self._act(
            np.array(obs)[None],
            update_eps=update_eps,
        )[0]

    def smooth(
            self,
            behavior_policy,
            evaluation_timesteps,
            max_k_random_actions=50,
    ):
        """Sample episodes to use for monte-carlo rollouts."""
        obs = self.env.reset()
        ep = 0
        episode_rewards = []
        episode_states = []
        episode_actions = []
        # TODO: Don't hard-code, and bias towards smaller.

        def get_random_k_t():
            k_random = self.np_random.randint(0, max_k_random_actions)
            random_t = self.np_random.randint(k_random, 200)
            return k_random, random_t
        k_random_actions, random_t = get_random_k_t()
        for t in range(evaluation_timesteps):
            episode_t = len(episode_actions)
            if IS_LOCAL and episode_t >= random_t:
                self.env.render()
            if episode_t < k_random_actions or episode_t == random_t:
                next_action = behavior_policy.act(
                    obs,
                    explore=True,
                    explore_eps=1,
                )
            else:
                next_action = behavior_policy.act(obs, explore=False)
            obs1, reward, done, _ = self.env.step(next_action)
            episode_rewards.append(reward)
            episode_states.append(obs)
            episode_actions.append(next_action)
            obs = obs1
            if done:
                for i, (o, a) in enumerate(
                        zip(episode_states[random_t:],
                            episode_actions[random_t:])
                ):
                    weighted_rewards = [
                        r * self.gamma ** j
                        for j, r in enumerate(episode_rewards[random_t + i:])
                    ]
                    reward_to_go = sum(weighted_rewards)
                    self.replay_buffer_mc.add(
                        o, a, reward_to_go, None, None,
                    )

                    # Update model.
                    obses_t, actions, rewards, _, _ = self.replay_buffer_mc.sample(
                        self.batch_size
                    )
                    weights = np.ones_like(rewards)
                    td_errors = self.train_mc(
                        obses_t, actions, rewards, weights)
                    # print(rewards)
                    # print(td_errors)
                    #print(self.get_q_values(o)[a], reward_to_go)
                    # print('----')
                    simulated_t = t - len(episode_rewards) + random_t + i
                    if simulated_t % self.train_log_freq == 0:
                        self._log_training_details(
                            episode=ep,
                            t=simulated_t,
                            td_max=np.max(np.abs(td_errors)),
                            td_mean=np.mean(np.abs(td_errors)),
                            smoothing=True,
                        )

                    # Save model
                    if (self.checkpoint_freq is not None and simulated_t % self.checkpoint_freq == 0):
                        if self.print_freq is not None:
                            logger.log("Saving model due to smoothing")
                        # TODO: Check scope addition
                        save_variables(self.model_file, scope=self.scope)
                        # save_variables(self.model_file)
                        self.model_saved = True

                obs = self.env.reset()
                episode_rewards = []
                episode_states = []
                episode_actions = []
                ep += 1
                k_random_actions, random_t = get_random_k_t()

            """
            # Finish
            obs = obs1
            self.t += 1
            if done:
                self.episode_rewards.append(0.0)
                self.training_episode += 1
                obs = self.env.reset()
            """
        # TODO: Check that model isn't getting worse?
        # TODO: Reload last best saved model like in self.end_learning?

    @property
    def mean_100ep_reward(self):
        return round(np.mean(self.episode_rewards[-101:-1]), 1)

    @property
    def mean_100ep_discounted_reward(self):
        return round(np.mean(self.discounted_episode_rewards[-101:-1]), 1)

    @property
    def mean_100ep_start_value(self):
        return round(np.mean(self.start_values[-100:]), 1)

    @property
    def mean_100ep_lunar_crashes(self):
        return round(np.mean(self.lunar_crashes[-100:]), 1)

    @property
    def mean_100ep_lunar_goals(self):
        return round(np.mean(self.lunar_goals[-100:]), 1)

    @property
    def mean_100ep_length(self):
        return round(np.mean(self.episode_lengths[-100:]), 1)

    def update(self, s, a, s1, r, done, verbose=False, freeze_buffer=False):
        # get_session()
        obs = s
        new_obs = s1
        action = a
        rew = r
        # Store transition in the replay buffer.
        if not freeze_buffer:
            self.replay_buffer.add(obs, action, rew, new_obs, float(done))
        obs = new_obs

        self.episode_rewards[-1] += rew
        self.episode_lengths[-1] += 1
        self.discounted_episode_rewards[-1] += rew * \
            self.gamma ** self.episode_t
        if self.start_values[-1] is None:
            self.start_values[-1] = max(self.get_q_values(s))
        if rew == -100:
            self.lunar_crashes[-1] = 1
        elif rew == 100:
            self.lunar_goals[-1] = 1

        mean_100ep_reward = self.mean_100ep_reward

        td_errors = None
        if self.t > self.learning_starts and self.t % self.train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if self.prioritized_replay:
                experience = self.replay_buffer.sample(
                    self.batch_size,
                    beta=self.beta_schedule.value(t),
                )
                (obses_t, actions, rewards, obses_tp1,
                 dones, weights, batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(
                    self.batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
            td_errors = self.train(
                obses_t, actions, rewards, obses_tp1, dones, weights)
            if self.prioritized_replay:
                new_priorities = np.abs(td_errors) + \
                    self.prioritized_replay_eps
                self.replay_buffer.update_priorities(
                    batch_idxes, new_priorities)

        if self.t > self.learning_starts and self.t % self.target_network_update_freq == 0:
            # Update target network periodically.
            self.update_target()

        saved = False
        if (self.checkpoint_freq is not None and self.t > self.learning_starts and
                self.training_episode > 100 and self.t % self.checkpoint_freq == 0):
            if (
                    self.saved_mean_reward is None
                    or mean_100ep_reward > self.saved_mean_reward
                    or (
                        self.save_reward_threshold is not None
                        and mean_100ep_reward >= self.save_reward_threshold
                    )
            ):
                saved = True
                if self.print_freq is not None:
                    logger.log("Saving model due to mean reward increase (or mean reward above {}): {} -> {}".format(
                        self.save_reward_threshold if self.save_reward_threshold is not None else 'NULL',
                        self.saved_mean_reward,
                        mean_100ep_reward
                    ))
                # TODO: Check scope addition
                save_variables(self.model_file, scope=self.scope)
                # save_variables(self.model_file)
                self.model_saved = True
                self.saved_mean_reward = mean_100ep_reward

        if self.t > self.learning_starts and self.t % self.train_log_freq == 0:
            self._log_training_details(
                episode=self.training_episode,
                t=self.t,
                td_max=np.max(np.abs(td_errors)),
                td_mean=np.mean(np.abs(td_errors)),
                r_mean=mean_100ep_reward,
                r_mean_discounted=self.mean_100ep_discounted_reward,
                v_mean=self.mean_100ep_start_value,
                n_crashes_mean=self.mean_100ep_lunar_crashes,
                n_goals_mean=self.mean_100ep_lunar_goals,
                saved_model=saved,
            )

        self.t += 1
        self.episode_t += 1
        if done:
            self.start_values.append(None)
            self.episode_rewards.append(0.0)
            self.episode_lengths.append(0)
            self.lunar_crashes.append(0)
            self.lunar_goals.append(0)
            self.discounted_episode_rewards.append(0.0)
            self.training_episode += 1
            self.episode_t = 0

    def end_learning(self):
        if self.model_saved:
            if self.print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(
                    self.saved_mean_reward))
            # TODO: Check scope addition
            load_variables(self.model_file, scope=self.scope)
            # load_variables(self.model_file)

    def close(self):
        if self.td is not None:
            import shutil
            shutil.rmtree(self.td)
        # get_session().close()


class RandomAgent(Agent):
    def __init__(self, env, seed=None):
        print('initializing random agent')
        super().__init__(env, seed)

    def act(self, s, explore):
        # Take action and update exploration to the newest value
        return self.np_random.choice(self.n_actions)


class TabularQLearningAgent(Agent):
    # TODO: Force Q values at terminal states to be 0 to handle unmodified
    # cliffwalking environment.
    def __init__(
            self,
            action_space,
            observation_space,
            lr_schedule,
            eps_schedule=None,
            boltzmann_schedule=None,
            seed=None,
    ):
        super().__init__(env=None, seed=seed)
        n = observation_space.n
        m = action_space.n
        self.Q = np.zeros((n, m))

        self.training_episode = 0
        self._lr_schedule = lr_schedule
        self._eps_schedule = eps_schedule
        self._boltzmann_schedule = boltzmann_schedule

    def act(self, s, explore):
        # Break ties randomly.
        best_a = randargmax(self.Q[s, :], self.np_random)
        if not explore:
            return best_a

        # Boltzmann exploration.
        if self._boltzmann_schedule is not None:
            p = np.exp(
                self._boltzmann_schedule.value(self.training_episode)
                * self.Q[s, :]
            )
            p /= sum(p)
            n_actions = len(self.Q[s, :])
            return self.np_random.choice(
                n_actions,
                p=p,
            )

        # Epsilon-greedy exploration.
        if (
            self.np_random.random_sample()
            < self._eps_schedule.value(self.training_episode)
        ):
            return self.np_random.choice(range(len(self.Q[s, :])))
        else:
            return best_a

    def get_q_values(self, s):
        return self.Q[s, :]

    def update(self, s, a, s1, r, done, verbose=False):
        if verbose:
            print(a)
            print({
                'before': self.Q[s, :],
            })
        self.Q[s, a] += self._lr_schedule.value(self.training_episode) \
            * (r + np.max(self.Q[s1, :]) - self.Q[s, a])
        if verbose:
            print({
                'after': self.Q[s, :],
            })

        if done:
            self.training_episode += 1
