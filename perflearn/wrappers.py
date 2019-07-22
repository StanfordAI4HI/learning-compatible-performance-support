import os
import random
import numpy as np
from copy import deepcopy as copy
from gym import Wrapper

from baselines.common.schedules import LinearSchedule

#from .policies import state_dist
from . import utils as ut
from .utils import randargmax


DEFAULT_TIMESTEPS = int(4e6)

MODELS_DIR = os.environ.get(
    'PERFLEARN_MODELS_DIR',
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        'shared_models',
    )
)


OPTIMAL_TRAJECTORIES = {
    'CliffWalking-treasure100-v0': [
        (3, 0),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (2, 7),
        (2, 8),
        (2, 9),
        (2, 10),
        (2, 11),
        (3, 11),
    ],
    'CliffWalking-nocliff-treasure100-v0': [
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
        (3, 5),
        (3, 6),
        (3, 7),
        (3, 8),
        (3, 9),
        (3, 10),
        (3, 11),
    ],
}


class UtilityWrapper(Wrapper):
    def __init__(self, env, gamma=1.0):
        super().__init__(env)
        self.gamma = gamma
        self.t = 0
        self.last_s = None

        self.real_dyn = dict()
        self.real_reward = dict()
        self.real_terminal = dict()
        try:
            for s in self.env.unwrapped.P:
                for a in self.env.unwrapped.P[s]:
                    next_states = self.env.unwrapped.P[s][a]
                    if len(next_states) != 1:
                        raise NotImplementedError('Stochastic environment')
                    _, s1, r, is_done = next_states[0]
                    self.real_dyn[s, a] = s1
                    self.real_reward[s, a] = r
                    self.real_terminal[s, a] = is_done
        except AttributeError:
            # Won't work for Lunar Lander
            pass

    @property
    def is_gridworld(self):
        return 'cliff' in self.env.unwrapped.spec.id.lower()

    @property
    def env_name(self):
        return self.env.unwrapped.spec.id

    def reset(self):
        s = self.env.reset()
        self.last_s = s
        self.r = 0
        self.t = 0
        return s

    def step(self, action):
        s, r, d, i = self.env.step(action)
        self.last_s = s
        self.r += r * self.gamma ** self.t
        self.t += 1
        return (s, r, d, i)

    def state_dist(self, s, s1, norm=1):
        if not self.is_gridworld:
            raise NotImplementedError('State distance may not work')
        return self._state_dist(
            s, s1, dims=self.env.unwrapped.shape, norm=norm,
        )

    @staticmethod
    def _state_dist(s, s2, dims, norm):
        return np.linalg.norm(
            np.array(np.unravel_index(s, dims))
            - np.array(np.unravel_index(s2, dims)),
            norm,
        )

    def _get_optimal_q_agent_deep(
            self,
            reuse=True,
            verbose=True,
            n_test_episodes=100,
            optimal_agent_training_timesteps=DEFAULT_TIMESTEPS,
            optimal_agent_smoothing_timesteps=None,
            k_initial_random_actions=None,
            random_actions_episode_freq=2,
            random_actions_start_t=None,
    ):
        if k_initial_random_actions is None:
            # TODO: Change this to maximum k random actions,
            # TODO: like in perflearn.agents?
            #k_random_actions_max = 100
            if 'half' not in self.env_name.lower():
                # Regular lunar lander
                #k_random_actions = 50
                k_random_actions = 0
                if random_actions_start_t is None:
                    random_actions_start_t = optimal_agent_training_timesteps / 2
            else:
                k_random_actions = 0

        def test_agent(agent):
            cum_rewards = []
            for _ in range(n_test_episodes):
                obs = self.env.reset()
                done = False
                cum_reward = 0
                while not done:
                    next_action = agent.act(obs, explore=False)
                    obs1, reward, done, _ = self.env.step(next_action)
                    cum_reward += reward
                    obs = obs1
                cum_rewards.append(cum_reward)
            return cum_rewards

        # TODO: Name models based on smoothing?
        print('RESOLVING optimal q!!!!!!')
        final_training_rewards = None
        final_smoothed_rewards = None
        from .agents import DQNLearningAgent
        path = os.path.join(
            MODELS_DIR,
            self.env_name,
            'model{:.1f}M'.format(optimal_agent_training_timesteps / 1e6),
        )
        if optimal_agent_smoothing_timesteps:
            scope_name = 'optimal_agent_smoothed'
            final_path = path + '_smoothed'
        else:
            scope_name = 'optimal_pilot'
            final_path = path

        if os.path.exists(final_path) and reuse:
            agent = DQNLearningAgent(
                env=self.env,
                load_path=final_path,
                scope=scope_name,
            )
        else:
            if 'lunar' in self.env_name.lower():
                #save_reward_threshold = 200
                save_reward_threshold = float('-inf')
                #freeze_buffer_timesteps = 0.75 * optimal_agent_training_timesteps
                freeze_buffer_timesteps = float('inf')
                lr = 5e-4
            # First, obtain trained agent
            if os.path.exists(path):
                agent = DQNLearningAgent(
                    env=self.env,
                    load_path=path,
                    scope='optimal_pilot',
                )
            else:
                agent = DQNLearningAgent(
                    env=self.env,
                    save_path=path,
                    total_timesteps=optimal_agent_training_timesteps,
                    exploration_fraction=0.1,  # TODO: Increase?
                    # exploration_final_eps=0.10,
                    scope='optimal_pilot',
                    #buffer_size=round(0.5 * optimal_agent_training_timesteps),
                    save_reward_threshold=save_reward_threshold,
                    lr=lr,
                )
                cum_reward = 0
                ep = 0
                episode_t = 0
                #k_random_actions = None
                obs = self.env.reset()
                for t in range(optimal_agent_training_timesteps):
                    take_random_action = episode_t < k_random_actions
                    if take_random_action:
                        explore_eps = 1
                        # print('random action (t={}, episode_t={})'.format(
                        #    t, episode_t,
                        # ))
                    else:
                        explore_eps = None
                    should_take_random_actions = (
                        random_actions_start_t is not None
                        and t > random_actions_start_t
                        and ep % random_actions_episode_freq == 0
                    )
                    if explore_eps is None and should_take_random_actions:
                        explore_eps = 0.20  # TODO: Experiment with param
                    if episode_t == 0 and should_take_random_actions:
                        print('exploring on ep {} with {:.2f}'.format(
                            ep, explore_eps
                        ))
                    next_action = agent.act(
                        obs,
                        explore=True,
                        explore_eps=explore_eps,
                    )
                    obs1, reward, done, _ = self.env.step(next_action)
                    cum_reward += reward
                    if not take_random_action:
                        agent.update(
                            s=obs,
                            a=next_action,
                            s1=obs1,
                            r=reward,
                            done=done,
                            freeze_buffer=(t >= freeze_buffer_timesteps),
                        )
                    obs = obs1
                    episode_t += 1
                    if done:
                        if verbose:
                            print({
                                'ep': ep,
                                'cum_reward': cum_reward,
                            })
                        obs = self.env.reset()
                        cum_reward = 0
                        ep += 1
                        episode_t = 0
                agent.end_learning()  # Restores last saved model.

            # Then do policy evaluation, optionally

            final_training_rewards = test_agent(agent)
            if optimal_agent_smoothing_timesteps:
                behavior_policy = agent
                agent = DQNLearningAgent(
                    env=self.env,
                    save_path=final_path,
                    scope=scope_name,
                    lr_mc=lr,
                    # lr_mc=5e-5,
                    buffer_size=float('inf'),
                )
                agent.smooth(behavior_policy,
                             optimal_agent_smoothing_timesteps)
                final_smoothed_rewards = test_agent(agent)
                # TODO: agent.end_learning()?

        return (agent, {
            'test_rewards': final_training_rewards,
            'smoothed_rewards': final_smoothed_rewards,
        })

    def _get_optimal_q_agent(
            self,
            q_learning_episodes=10000,  # For non-deep only
            n_test_episodes=100,
            max_tries=10,
            verbose=False,
            logging=True,
            optimal_agent_training_timesteps=None,
            optimal_agent_smoothing_timesteps=None,
    ):
        """Returns Q values for an optimal* agent.

        Learning retries until either the policy never scores
        below the (known) optimal reward or max_tries is reached.

        """
        if optimal_agent_training_timesteps is None:
            optimal_agent_training_timesteps = DEFAULT_TIMESTEPS
        if not self.is_gridworld:
            return self._get_optimal_q_agent_deep(
                n_test_episodes=n_test_episodes,
                optimal_agent_training_timesteps=optimal_agent_training_timesteps,
                optimal_agent_smoothing_timesteps=optimal_agent_smoothing_timesteps,
            )
        # Solve using q learning for now.
        # TODO: Calculate exact q values?
        print('RESOLVING optimal q!!!!!!')
        from .agents import TabularQLearningAgent

        if self.env_name == 'CliffWalking-treasure100-v0':
            goal_reward = 87
        elif self.env_name == 'CliffWalking-nocliff-treasure100-v0':
            goal_reward = 89
        else:
            raise NotImplementedError

        test_reward = float('-inf')
        n_tries = 0
        while test_reward < goal_reward and n_tries < max_tries:
            n_tries += 1
            agent = TabularQLearningAgent(
                action_space=self.env.action_space,
                observation_space=self.env.observation_space,
                eps_schedule=LinearSchedule(
                    schedule_timesteps=int(0.9 * q_learning_episodes),
                    initial_p=1.0,
                    final_p=0.02,
                ),
                lr_schedule=LinearSchedule(
                    schedule_timesteps=int(0.9 * q_learning_episodes),
                    initial_p=1.0,
                    final_p=0.02,
                ),
            )
            for ep in range(q_learning_episodes):
                obs = self.env.reset()
                done = False
                cum_reward = 0
                while not done:
                    next_action = agent.act(obs, explore=True)
                    obs1, reward, done, _ = self.env.step(next_action)
                    cum_reward += reward
                    agent.update(
                        s=obs,
                        a=next_action,
                        s1=obs1,
                        r=reward,
                        done=done,
                    )
                    obs = obs1
                if verbose:
                    print({
                        'ep': ep,
                        'lr': agent._lr,
                        'eps': agent._eps,
                        'q_norm': np.linalg.norm(agent.Q),
                        'cum_reward': cum_reward,
                    })

            # Test learned agent.
            cum_rewards = []
            for _ in range(n_test_episodes):
                obs = self.env.reset()
                done = False
                cum_reward = 0
                while not done:
                    next_action = agent.act(obs, explore=False)
                    obs1, reward, done, _ = self.env.step(next_action)
                    cum_reward += reward
                    obs = obs1
                cum_rewards.append(cum_reward)
            test_reward = min(cum_rewards)

        print('{} tries'.format(n_tries))

        return (agent, {
            'tries': n_tries,
            'test_rewards': cum_rewards,
        })

    def _get_boltzmann_q_policy_evaluation(
            self,
            boltzmann_parameter,
            optimal_q,
            terminal_q_difference=0.01,
            max_iterations=10000,
    ):
        """Compute Q values for a boltzmann policy using policy evaluation.

        Args:
            terminal_q_difference (Optional[float]): Maximum difference in
                Q values allowed when terminating. Defaults to 0.01.
            max_iterations (Optional[int]): Maximum number of iterations.
                Defaults to 10,000.

        """
        if not self.is_gridworld:
            raise NotImplementedError('Stochastic policies not implemented')
        boltzmann_pi = np.exp(boltzmann_parameter * np.array(optimal_q))
        boltzmann_pi = boltzmann_pi / \
            np.reshape(boltzmann_pi.sum(axis=1), (-1, 1))

        q_difference = float('inf')
        i = 0
        new_q = np.zeros(boltzmann_pi.shape)
        while q_difference > terminal_q_difference and i < max_iterations:
            q_difference = 0
            for s, row in enumerate(boltzmann_pi):
                for a, p in enumerate(row):
                    s1 = self.real_dyn[s, a]
                    r = self.real_reward[s, a]
                    is_terminal = self.real_terminal[s, a]
                    q = new_q[s, a]
                    if is_terminal:
                        q1 = r
                    else:
                        q1 = r + np.dot(boltzmann_pi[s1, :], new_q[s1, :])
                    new_q[s, a] = q1
                    q_difference = max(q_difference, np.abs(q1 - q))
            i += 1
        return new_q

    def _get_boltzmann_q(
            self,
            final_boltzmann_parameter,
            optimal_q=None,
            q_learning_episodes=10000,
            n_test_episodes=100,
            max_tries=10,
            verbose=False,
            logging=True,
    ):
        """Compute Q values for a boltzmann policy.

        Args:
            optimal_q (Optional[[[float]]]): Q values to use for computing
                boltzmann Q values using policy evaluation. If None,
                learn Q values using boltzmann exploration instead.

        """
        if optimal_q is not None:
            return self._get_boltzmann_q_policy_evaluation(
                boltzmann_parameter=final_boltzmann_parameter,
                optimal_q=optimal_q,
            )

        print('RESOLVING!!!!!!')

        from .policies import TabularQLearningAgent

        agent = TabularQLearningAgent(
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            eps_schedule=LinearSchedule(
                schedule_timesteps=int(0.9 * q_learning_episodes),
                initial_p=0,
                final_p=final_boltzmann_parameter,
            ),
            lr_schedule=LinearSchedule(
                schedule_timesteps=int(0.9 * q_learning_episodes),
                initial_p=1.0,
                final_p=0.02,
            ),
        )
        for ep in range(q_learning_episodes):
            obs = self.env.reset()
            done = False
            cum_reward = 0
            while not done:
                next_action = agent.act(obs, explore=True)
                obs1, reward, done, _ = self.env.step(next_action)
                cum_reward += reward
                agent.update(
                    s=obs,
                    a=next_action,
                    s1=obs1,
                    r=reward,
                    done=done,
                )
                obs = obs1
            if verbose:
                print({
                    'ep': ep,
                    'lr': agent._lr,
                    'eps': agent._eps,
                    'q_norm': np.linalg.norm(agent.Q),
                    'cum_reward': cum_reward,
                })

        # Test learned agent.
        cum_rewards = []
        for _ in range(n_test_episodes):
            obs = self.env.reset()
            done = False
            cum_reward = 0
            while not done:
                next_action = agent.act(obs, explore=False)
                obs1, reward, done, _ = self.env.step(next_action)
                cum_reward += reward
                obs = obs1
            cum_rewards.append(cum_reward)
        test_reward = np.mean(cum_rewards)

        print('Mean Boltzmann reward: {}'.format(test_reward))

        return copy(agent.Q)

    def _get_undoing_action(self, q=None):
        """Returns one-step or two-step undoing action, else None.

        Args:
            s: Current state
            q: Q values for the two-step lookahead.

        Tries the following in order:
        - Get back to the current state.
        - Get back to the previous state.
        - Get to a state s.t. the best action (first, not random) from
        that state (as defined by the provided q function) would bring one
        back to the current state.

        """
        s = self.last_s
        # One-step
        next_states = [self.real_dyn[s, a] for a in self.env.unwrapped.P[s]]
        no_ops = [s_prime == s for s_prime in next_states]
        if any(no_ops):
            return randargmax(np.array(no_ops))

        # Two-step

        # First check if any action will get back to previous state.
        two_step_noops = [s_prime == self.last_s for s_prime in next_states]
        if any(two_step_noops):
            return randargmax(np.array(two_step_noops))

        if q is None:
            return None
        two_step_noops = [
            self.real_dyn[s_prime, np.argmax(q[s_prime, :])] == s
            for s_prime in next_states
        ]
        if any(two_step_noops):
            return randargmax(np.array(two_step_noops))
        # TODO: What if multiple states where best action goes to s,
        # but one intermediate state is in an undesirable area,
        # e.g., outside the bumpers.
        return None

    def _get_next_best_action(self, q):
        """Get next best action.

        Args:
            q [float]: Local q.

        """
        # TODO: Debug this.
        max_q = np.max(q)
        try:
            next_max_q = np.max([x for x in q if x != max_q])
            p = [1 if x == next_max_q else 0 for x in q]
            p = np.array(p) / sum(p)
            return np.random.choice(
                len(q),
                p=p,
            )
        except ValueError:
            return None

    def _get_next_action_and_info(
            self,
            user_action,
            recommended_action=None,
    ):
        """Returns next action and info about how decision was made."""
        p_suboptimal_override = self.p_suboptimal_override
        override_next_best = self.override_next_best
        undoing = self.undoing
        optimal_q = self.optimal_q_agent.get_q_values(self.last_s)
        p_override = self.p_override

        should_override = False
        undoing_act = None
        random_no_override = None
        random_no_suboptimal_override = None
        next_action = user_action
        if recommended_action is not None:
            should_override = True
            next_action = recommended_action
            if undoing:
                undoing_act = self._get_undoing_action(q=optimal_q)
                if (
                        undoing_act is not None
                        and random.random() <= p_suboptimal_override
                ):
                    next_action = undoing_act
                elif undoing_act is not None:
                    random_no_suboptimal_override = True
            elif override_next_best:
                next_best_act = self._get_next_best_action(q=optimal_q)
                if (
                        next_best_act is not None
                        and random.random() <= p_suboptimal_override
                ):
                    next_action = next_best_act
                elif next_best_act is not None:
                    random_no_suboptimal_override = True

            random_no_override = False
            if not undoing and optimal_q[next_action] <= optimal_q[user_action]:
                # Don't override with actions that aren't better than
                # user action.
                next_action = user_action
            elif random.random() > p_override:
                # Don't override sometimes.
                next_action = user_action
                random_no_override = True

        info = {
            'final_action': next_action,
            'override_action': next_action != user_action,
            'random_no_override': random_no_override,
            'random_no_suboptimal_override': random_no_suboptimal_override,
            'recommended_act': recommended_action,
            'should_override': should_override,
            'undoing_action': undoing_act,
        }
        return next_action, info

    def get_support_details(self):
        return {}

    def close(self):
        super().close()
        try:
            self.optimal_q_agent.close()
        except AttributeError:
            pass


class Unassisted(UtilityWrapper):
    def step(self, action):
        observation, reward, done, info = super().step(action)
        # TODO: Use _get_next_action_and_info()?
        info['final_action'] = action
        info['override_action'] = False
        return observation, reward, done, info


class Bumpers(UtilityWrapper):
    def __init__(
            self,
            env,
            trajectory_distance,
            threshold=None,
            p_override=1,
            undoing=False,
            p_suboptimal_override=1,
            override_next_best=False,
            optimal_agent_training_timesteps=None,
            optimal_agent_smoothing_timesteps=None,
    ):
        super().__init__(env)

        self.p_override = p_override
        self.undoing = undoing
        self.p_suboptimal_override = p_suboptimal_override
        self.override_next_best = override_next_best

        self.threshold = threshold
        self.trajectory_distance = trajectory_distance
        self.optimal_trajectory = [
            np.ravel_multi_index(x, self.env.unwrapped.shape)
            for x in OPTIMAL_TRAJECTORIES[env.unwrapped.spec.id]
        ]
        self.trajectory_t = None

        self.optimal_q_agent, self.optimal_q_agent_details = \
            self._get_optimal_q_agent(
                optimal_agent_training_timesteps=optimal_agent_training_timesteps,
                optimal_agent_smoothing_timesteps=optimal_agent_smoothing_timesteps,
            )

    def reset(self):
        if self.env_name not in OPTIMAL_TRAJECTORIES:
            raise NotImplementedError(
                'May need to recompute trajectory if start state changes'
            )
        self.trajectory_t = 0
        return super().reset()

    def _move_forward_in_optimal_trajectory(self):
        if (
                self.last_s is None
                or self.trajectory_t >= len(self.optimal_trajectory) - 2
        ):
            return False

        if self.trajectory_distance == 'wait':
            previous_dists = [
                self.state_dist(
                    self.optimal_trajectory[self.trajectory_t],
                    self.last_s,
                ),
                self.state_dist(
                    self.optimal_trajectory[self.trajectory_t + 1],
                    self.last_s,
                )
            ]
            current_dists = [
                self.state_dist(
                    self.optimal_trajectory[self.trajectory_t],
                    self.last_s,
                ),
                self.state_dist(
                    self.optimal_trajectory[self.trajectory_t + 1],
                    self.last_s,
                )
            ]
            return (
                previous_dists[0] <= current_dists[0]
                and previous_dists[1] > current_dists[1]
            )

        return True

    def step(self, action):
        if self._move_forward_in_optimal_trajectory():
            self.trajectory_t += 1

        # TODO: Change, if stochastic environment.
        next_states = [
            self.real_dyn[self.last_s, a]
            for a in range(self.action_space.n)
        ]

        # For each action, how close will it get to each part of optimal
        # trajectory
        dist_matrix = np.array([
            [
                self.state_dist(target_state, sp)
                for target_state in self.optimal_trajectory
            ]
            for sp in next_states
        ])

        dists_from_optimal_state = dist_matrix[:, self.trajectory_t + 1]

        if self.trajectory_distance in ['timestep', 'wait']:
            """Aim for time-dependent position in trajectory"""
            dists_from_allowed = dists_from_optimal_state
        elif self.trajectory_distance == 'nearest':
            """Aim for closest place in trajectory"""
            dists_from_allowed = np.min(
                dist_matrix,
                axis=1,
            )

        best_action = None
        should_override = (
            self.threshold is not None
            and dists_from_allowed[action] > self.threshold
        )
        if should_override:
            if self.trajectory_distance in ['timestep', 'wait']:
                best_action = randargmax(-dists_from_allowed)
            elif self.trajectory_distance == 'nearest':
                actions_within_bumpers = [
                    d <= self.threshold for d in dists_from_allowed
                ]
                if any(actions_within_bumpers):
                    best_action = randargmax(np.array([
                        -d if is_within_bumpers else float('-inf')
                        for d, is_within_bumpers in zip(
                            dists_from_optimal_state,
                            actions_within_bumpers,
                        )
                    ]))
                else:
                    # TODO: Consider moving towards optimal state instead.
                    # TODO: If there is an action that resets to start state,
                    # like falling off a cliff, this could commit suicide.
                    best_action = randargmax(-dists_from_allowed)
            else:
                raise NotImplementedError('Unknown trajectory distance')

            #v_best = -dists[best_action]
            #v_action = -dists[action]

        next_act, additional_info = self._get_next_action_and_info(
            user_action=action,
            recommended_action=best_action,
        )
        observation, reward, done, info = super().step(next_act)
        info.update(additional_info)
        info['support'] = {
            't': self.trajectory_t,
            # 'v_best': v_best,
            # 'v_action': v_action,
            # 'dists': dists,
        }
        return observation, reward, done, info

    def get_support_details(self):
        return {
            'target_trajectory': self.optimal_trajectory,
            'optimal_q_agent_details': self.optimal_q_agent_details,
        }


class QThreshold(UtilityWrapper):
    """Sidd Reddy support."""

    def __init__(
            self,
            env,
            q_threshold,
            p_override=1,
            undoing=False,
            p_suboptimal_override=1,
            override_next_best=False,
            optimal_agent_training_timesteps=None,
            optimal_agent_smoothing_timesteps=None,
    ):
        super(QThreshold, self).__init__(env)

        self.optimal_q_agent, self.optimal_q_agent_details = \
            self._get_optimal_q_agent(
                optimal_agent_training_timesteps=optimal_agent_training_timesteps,
                optimal_agent_smoothing_timesteps=optimal_agent_smoothing_timesteps,
            )

        self.q_threshold = q_threshold
        self.p_override = p_override
        self.undoing = undoing
        self.p_suboptimal_override = p_suboptimal_override
        self.override_next_best = override_next_best

    def step(self, action):
        s = self.last_s
        optimal_q = self.optimal_q_agent.get_q_values(s)
        q_normalized = optimal_q - np.min(optimal_q)
        recommended_act = None
        if q_normalized[action] < (
                (1 - self.q_threshold) * np.max(q_normalized)
        ):
           #next_act = np.argmax(q_normalized)
            recommended_act = randargmax(q_normalized)
            # TODO: Doesn't select action most similar to user action.

        next_action, additional_info = self._get_next_action_and_info(
            user_action=action,
            recommended_action=recommended_act,
        )
        observation, reward, done, info = super().step(next_action)
        info.update(additional_info)
        info['support'] = {
            'q_values': optimal_q,
            'best_hope_for': self.r + np.max(optimal_q),
        }
        # , info['support']['final_action'])
        return observation, reward, done, info

    def get_support_details(self):
        d = {'optimal_q_agent_details': self.optimal_q_agent_details}
        if self.is_gridworld:
            d.update({
                'optimal_q': copy(self.optimal_q_agent.Q),
            })
        return d


class QBumpers(UtilityWrapper):
    def __init__(
            self,
            env,
            boltzmann_parameter,
            p_override=1,
            undoing=False,
            p_suboptimal_override=1,
            override_next_best=False,
            version=0,
            target_r=None,
            optimal_agent_training_timesteps=None,
            optimal_agent_smoothing_timesteps=None,
            length_normalized=False,
            logistic_upper_prob=None,
            gamma=1.0,
            alpha=1.0,
    ):
        super().__init__(env, gamma=gamma)

        if length_normalized and not self.is_gridworld:
            self.length_normalized = 170  # TODO: Check length estimate.
        else:
            self.length_normalized = None
        self.logistic_upper_prob = logistic_upper_prob
        self.prespecified_target_r = target_r
        self.optimal_q_agent, self.optimal_q_agent_details = \
            self._get_optimal_q_agent(
                optimal_agent_training_timesteps=optimal_agent_training_timesteps,
                optimal_agent_smoothing_timesteps=optimal_agent_smoothing_timesteps,
            )
        self.boltzmann_q = None
        if self.is_gridworld:
            self.boltzmann_q = self._get_boltzmann_q(
                final_boltzmann_parameter=boltzmann_parameter,
                optimal_q=self.optimal_q_agent.Q,
            )
        self.p_override = p_override
        self.undoing = undoing
        self.p_suboptimal_override = p_suboptimal_override
        self.override_next_best = override_next_best

        self.version = version
        self.alpha = alpha

    def reset(self):
        observation = super().reset()
        if self.prespecified_target_r is not None:
            self.target_r = self.prespecified_target_r
        else:
            try:
                self.target_r = np.max(self.boltzmann_q[observation, :])
            except TypeError:
                self.target_r = None
        optimal_q = self.optimal_q_agent.get_q_values(observation)
        self.upper_bound_r = np.max(optimal_q)
        print('Initializing dqn at {0} with optimal q {1} and lower bound {2}'.format(
            observation,
            optimal_q,
            self.target_r,
        ))
        self.advantages = []
        if self.target_r is not None and self.target_r >= self.upper_bound_r:
            raise ValueError(
                'The lower bound on performance must be lower than the best reward the support can provide.')
        return observation

    def step(self, action):
        """Take a step.

        Versions:
        - 0: Override if target is no longer feasible by following optimal
            policy. Lower target if target no longer feasible.
        - 1: Override if target is no longer feasible by following boltzmann
            policy. Lower target if target no longer feasible.
        - 2: Override with probability proportional to fraction of distance
            of future rewards that are lost.
        - 3: Override with probability proportional to fraction of increased
            value (risk) of changing to second-best action.
            Never lower target.
            Use new metric that is fraction of Pi* - Pi_tau above lower bound
            of P_tau(s_0).
        - 4: Override with probability proportional to fraction of increased
            value (risk) of user action.
            Never lower target.
            Use new metric that is fraction of Pi* - Pi_tau above lower bound
            of P_tau(s_0).
        - 5: Override with probability proportional to fraction of increased
            value of changing to second-best action.
            Never lower target.
            Don't use new risk metric.
        - 6: Override with probability proportional to fraction of optimal
            value lost with changing to user action.
            Never lower target.
            Don't use new risk metric.

        """
        should_override = False
        recommended_act = None
        next_act = action
        s = self.last_s
        discount = self.gamma ** (self.t + 1)

        adjusted_expectations = None
        optimal_q = self.optimal_q_agent.get_q_values(s)

        def rescale(v):
            if 'lunar' in self.env_name.lower():
                # -150 is lower bound on sum of rewards
                v = v / (self.upper_bound_r - (-150))
            elif 'cliff' in self.env_name.lower():
                # Theoretically, one could get -100 * 100 rewards?
                v = v / (self.upper_bound_r - (-150))
            else:
                raise NotImplementedError
            return v
        total_advantage = sum(
            self.gamma ** i * advantage
            for i, advantage in enumerate(self.advantages)
        )
        total_rescaled_advantage = sum(
            self.gamma ** i * rescale(advantage)
            for i, advantage in enumerate(self.advantages)
        )
        next_advantages = [
            optimal_q[a] - np.max(optimal_q)
            for a in range(self.action_space.n)
        ]
        next_rescaled_advantages = [rescale(adv) for adv in next_advantages]

        if self.version in [0, 1]:
            adjusted_expectations = False
            best_can_hope_for = self.r + discount * np.max(optimal_q)
            prev_target_r = self.target_r
            if (best_can_hope_for <= self.target_r):
                # TODO: Consider adjusting expectations at other times.
                # Lower expectations to prevent being pulled along.
                self.target_r = self.r + discount * \
                    np.max(self.boltzmann_q[s, :])
                adjusted_expectations = True

            already_optimal = (
                optimal_q[action] == np.max(optimal_q)
            )
            recommended_act = None
            if self.version == 0:
                override_condition_met = (
                    self.r + discount * optimal_q[action] < self.target_r
                    and not already_optimal
                )
            else:
                override_condition_met = (
                    self.r + discount *
                    self.boltzmann_q[s, action] < self.target_r
                    and not already_optimal
                )
            if override_condition_met:
                recommended_act = randargmax(optimal_q)
            support_info = {
                'best_hope_for': best_can_hope_for,
                'adjusted_expectations': adjusted_expectations,
                'prev_target_r': prev_target_r,
                'target_r': self.target_r,
                'best_going_with_user_action': (
                    self.r + discount * optimal_q[action]
                ),
                'best_going_with_override_action': (
                    self.r + discount * optimal_q[recommended_act]
                    if recommended_act is not None else None
                ),
            }
        elif self.version == 2:
            best_to_go = np.max(optimal_q)
            best_boltzmann = np.max(self.boltzmann_q[s, :])
            best_user_action = optimal_q[action]

            boltzmann_gap = best_to_go - best_boltzmann
            user_action_frac_boltzmann_gap = (
                (best_to_go - best_user_action) / boltzmann_gap
            )

            if boltzmann_gap == 0 and best_user_action == best_to_go:
                pass
            if boltzmann_gap == 0 and best_user_action < best_to_go:
                recommended_act = randargmax(optimal_q)
            elif random.random() <= user_action_frac_boltzmann_gap:
                recommended_act = randargmax(optimal_q)
            else:
                pass
            support_info = {
                'best_to_go': best_to_go,
                'best_boltzmann': best_boltzmann,
                'boltzmann_gap': boltzmann_gap,
                'user_action_frac_boltzmann_gap': user_action_frac_boltzmann_gap,
                'best_going_with_user_action': (
                    optimal_q[action]
                ),
                'best_going_with_override_action': (
                    optimal_q[recommended_act]
                    if recommended_act is not None else None
                ),
                'best_hope_for': self.r + discount * np.max(optimal_q),
            }
        elif self.version >= 3:
            adjusted_expectations = False
            recommended_act = None
            upper = self.upper_bound_r
            lower = self.target_r
            if lower is not None:
                lower = upper - (upper - lower) * self.alpha

            def get_inverse_risk(a):
                denominator = optimal_q[a] - self.boltzmann_q[s, a]
                numerator = max(self.r + discount * optimal_q[a] - lower, 0)
                if denominator == 0 and numerator >= 0:
                    return 1
                elif denominator == 0:
                    return 0
                return numerator / denominator
            next_best_action = self._get_next_best_action(optimal_q)
            best_action = randargmax(optimal_q)
            if self.version in [3, 4]:
                scores = [
                    get_inverse_risk(a) for a in range(self.action_space.n)
                ]
            elif self.version in [5, 6, 11]:
                scores = [
                    max((self.r + discount * optimal_q[a] - lower) /
                        (upper - lower),
                        0)
                    for a in range(self.action_space.n)
                ]
            elif self.version in [7, 8]:
                # Advantages with fixed lower bound.
                scores = [
                    max((upper + (total_advantage + discount * advantage) - lower) /
                        (upper - lower),
                        0)
                    for advantage in next_advantages
                ]
            elif self.version in [9, 10]:
                scores = [
                    max(
                        1 - (total_rescaled_advantage + discount * advantage)
                        / -self.alpha,
                        0
                    )
                    for advantage in next_rescaled_advantages
                ]
            else:  # next version starts at 12
                raise NotImplementedError
            p_override = None
            threshold = None
            if (
                    self.override_next_best
                    and self.version == 11
                    and scores[action] <= 0
            ):
                # TODO: Make this an option rather than a version?
                best_action_to_consider = best_action
            elif self.override_next_best:
                best_action_to_consider = next_best_action
            else:
                best_action_to_consider = best_action

            already_optimal = False
            if optimal_q[action] >= optimal_q[best_action_to_consider]:
                # Don't override with an action that isn't better
                # than user action.
                already_optimal = True
                pass
            elif scores[best_action_to_consider] == 0:
                p_override = 1
                recommended_act = best_action
            else:
                if self.version in [3, 5, 7, 9]:
                    # TODO: Can't distinguish between different actions
                    # if they are all below lower.
                    # TODO: What if scores[next_best_action] == 0
                    p_override = (
                        scores[best_action_to_consider] - scores[action]
                    ) / scores[best_action_to_consider]
                else:
                    p_override = 1 - scores[action]
                p_override = min(max(p_override, 0), 1)
                if (
                        self.length_normalized is not None
                        and p_override > 0 and p_override < 1
                ):
                    threshold = 1 - (
                        (1 - p_override) / max(
                            1,
                            self.length_normalized - self.t
                        )
                    )
                elif (
                        self.logistic_upper_prob is not None
                        and p_override > 0 and p_override < 1
                ):
                    threshold = ut.sigmoid(
                        x=p_override,
                        lower=0,
                        upper=1,
                        p=self.logistic_upper_prob,
                    )
                else:
                    threshold = p_override
                if random.random() <= threshold:
                    recommended_act = best_action_to_consider

            support_info = {
                'q_values': optimal_q,
                'scores': scores,
                'upper_bound_r': self.upper_bound_r,
                'target_r': self.target_r,
                'scores_going_with_user_action': (
                    scores[action]
                ),
                'scores_going_with_override_action': (
                    scores[recommended_act] if recommended_act is not None
                    else None
                ),
                'p_override': p_override,
                'p_override_scaled': threshold,
                'already_optimal': already_optimal,
                'best_hope_for': self.r + discount * np.max(optimal_q),
                'total_advantage': total_advantage,
                'total_rescaled_advantage': total_rescaled_advantage,
                'next_advantages': next_advantages,
                'next_rescaled_advantages': next_rescaled_advantages,
            }
        else:
            raise NotImplementedError

        next_act, additional_info = self._get_next_action_and_info(
            user_action=action,
            recommended_action=recommended_act,
        )
        # TODO: Ignoring recommendations for v3-6. Reorg.
        if self.version >= 3:
            next_act = action if recommended_act is None else recommended_act
            additional_info.update({
                'final_action': next_act,
                'override_action': next_act != action,
            })
        self.advantages.append(next_advantages[next_act])
        observation, reward, done, info = super().step(next_act)
        info.update(additional_info)
        info['support'] = support_info
        return observation, reward, done, info

    def get_support_details(self):
        d = {
            'boltzmann_q': self.boltzmann_q,
            'optimal_q_agent_details': self.optimal_q_agent_details,
        }
        if self.is_gridworld:
            d.update({
                'optimal_q': copy(self.optimal_q_agent.Q),
            })
        return d


class RandomPushes(UtilityWrapper):
    def __init__(
            self,
            env,
            optimal_agent_training_timesteps=None,
            optimal_agent_smoothing_timesteps=None,
            p_start_random=1,
            k_random=50,
    ):
        super().__init__(env)
        self.p_start_random = p_start_random
        self.k_random = k_random
        self.optimal_q_agent, self.optimal_q_agent_details = \
            self._get_optimal_q_agent(
                optimal_agent_training_timesteps=optimal_agent_training_timesteps,
                optimal_agent_smoothing_timesteps=optimal_agent_smoothing_timesteps,
            )

    def reset(self):
        self.random_actions_started_t = None
        return super().reset()

    def step(self, action):
        if (
                self.random_actions_started_t is None
                and random.random() < self.p_start_random
        ):
            self.random_actions_started_t = self.t
        if (
                self.random_actions_started_t is not None
                and self.t < self.random_actions_started_t + self.k_random
        ):
            action = random.randrange(self.action_space.n)
        else:
            s = self.last_s
            optimal_q = self.optimal_q_agent.get_q_values(s)
            action = randargmax(optimal_q)

        observation, reward, done, info = super().step(action)
        info['final_action'] = action
        return observation, reward, done, info
