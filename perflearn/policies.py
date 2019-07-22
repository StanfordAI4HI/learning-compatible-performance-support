import numpy as np
import types
from copy import deepcopy as copy
from scipy.special import logsumexp
import gym
from . import envs



def q_learning(
        rollouts,
        n_obs_dim,
        n_act_dim,
        user_action,
        Q_init=None,
        learning_rate=0.8,
        ftol=0,
        gamma=envs.GAMMA,
        verbose=False,
):
    """Tabular q learning.

    Args:
        user_action (boolean): Assumes all actions are own.

    """
    raise Exception('Deprecated. Use agents.TabularQLearningAgent instead.')
    n = n_obs_dim
    m = n_act_dim
    Q = np.zeros((n, m)) if Q_init is None else copy(Q_init)

    # TODO: Don't assume first task in rollouts?
    for rollout in rollouts[0]:
        for x in rollout:
            s, a_input, r, s1, _, _, info = x
            a_final = info['final_action']
            if user_action:
                a = a_input
            else:
                a = a_final
            if verbose:
                print('true: {}, thinks: {}'.format(a_final, a))
            Q[s, a] += learning_rate * (r + np.max(Q[s1, :]) - Q[s, a])
    return Q





def make_perfect_pilot_policy(
        goal,
        act_labels,
        seed=None,
        gw_size=envs.GW_SIZE,
        verbose=False,
):
    """Simple policy that moves toward goal, no trained Q values"""
    gx, gy = goal
    if seed is not None:
        np.random.seed(seed)

    def pilot_policy(obs):
        x = obs // gw_size
        y = obs % gw_size
        if verbose:
            print('({}, {}) -> ({}, {})'.format(x, y, gx, gy))
        up = gx < x
        down = gx > x
        left = gy < y
        right = gy > y
        lr = left or right
        ud = up or down
        if lr and (not ud or np.random.random() < 0.5):
            if left:
                if verbose:
                    print('left')
                return act_labels[obs, 0]
            elif right:
                if verbose:
                    print('right')
                return act_labels[obs, 1]
        elif ud:
            if up:
                if verbose:
                    print('up')
                return act_labels[obs, 2]
            elif down:
                if verbose:
                    print('down')
                return act_labels[obs, 3]
        if verbose:
            print('left')
        return act_labels[obs, 0]
    return pilot_policy


def tabsoftq_iter(
        R,
        T,
        maxiter=1000,
        verbose=True,
        Q_init=None,
        learning_rate=1,
        ftol=0,
        gamma=envs.GAMMA,
):
    n, m = R.shape[:2]  # n is number of states, m is number of actions
    Q = np.zeros((n, m)) if Q_init is None else copy(Q_init)
    prevQ = copy(Q)
    if verbose:
        diffs = []
    for iter_idx in range(maxiter):
        V = logsumexp(prevQ, axis=1)
        V_broad = V.reshape((1, 1, n))
        Q = np.sum(T * (R + gamma * V_broad), axis=2)
        Q = (1 - learning_rate) * prevQ + learning_rate * Q
        diff = np.mean((Q - prevQ)**2)/(np.std(Q)**2)
        if verbose:
            diffs.append(diff)
        if diff < ftol:
            break
        prevQ = copy(Q)
    if verbose:
        from matplotlib import pyplot as plt
        plt.xlabel('Number of Iterations')
        plt.ylabel('Avg. Squared Bellman Error')
        plt.title('Soft Q Iteration')
        plt.plot(diffs)
        plt.yscale('log')
        plt.show()
    return Q


def tabsoftq_learn(env, T=None, verbose=True):
    R = env.unwrapped.R
    if T is None:
        T = env.unwrapped.T
    return tabsoftq_iter(R, T, verbose=verbose)


aristotle_softq_pilot_temp = 1


def make_tabsoftq_policy(Q, use_gumbel=True, n_act_dim=envs.N_ACT_DIM):
    """
    https://www.wolframalpha.com/input/?i=gumbel+distribution+(0,+1)
    """
    def tabsoftq_policy(obs):
        v = aristotle_softq_pilot_temp*Q[obs, :]
        if use_gumbel:
            v += np.random.gumbel(0, 1, n_act_dim)
        return np.argmax(v)
    return tabsoftq_policy


# optimal trajectories

def get_trajectory(policy, env, max_ep_len=None, reset_pos=None, seed=None):
    rollout = envs.run_ep(
        policy,
        env,
        max_ep_len=max_ep_len,
        render=False,
        task_idx=None,
        seed=seed,
        reset_pos=reset_pos,
    )
    states = [x['prev_obs'] for x in rollout]
    states.append(rollout[-1]['obs'])
    return states


def featurize_state(s, gw_size=envs.GW_SIZE):
    return np.array([s // gw_size, s % gw_size])


def state_dist(s, sp):
    """L1 distance."""
    return np.sum(np.abs(featurize_state(s) - featurize_state(sp)))


def action_recommendation(
        state,
        action,
        target_trajectory,
        real_dyn,
        step_idx,
        n_act_dim=envs.N_ACT_DIM,
):
    """Return action recommenation and measure of how much better than user action

    Args:
        state: Current state.
        action: User action.
        target_trajectory: List of states.
        real_dyn: Dictionary from (s, a) -> s.
        step_idx: Current time index.
        n_act_dim: Number of actions.

    Returns:
        int: Best action
        int: Goodness of best action
        int: Goodness of user action.
        [float]: Lookup for a -> Action badness.

    """
    # TODO: Reddy-RSS (used Q value threshold and action similarity rather than next state similarity)
    next_states = [real_dyn[state, a] for a in range(n_act_dim)]
    try:
        next_target_state = target_trajectory[step_idx + 1]
    except IndexError:
        next_target_state = target_trajectory[-1]
    # Each state corresponds to diff action
    dists = np.array([state_dist(next_target_state, sp) for sp in next_states])
    best_action = np.argmax(-dists)

    return best_action, -dists[best_action], -dists[action], dists


def make_env(
        env_name,
        support_name,
        threshold=None,
        q_bumper_boltzmann=None,
        q_bumper_version=None,
        q_bumper_target_r=None,
        q_bumper_length_normalized=False,
        q_bumper_logistic_upper_prob=None,
        q_bumper_alpha=None,
        q_threshold=None,
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
    # TODO: Pass dirname to environments that do logging.
    env = gym.make(env_name)

    if support_name == 'unassisted':
        from .wrappers import Unassisted
        return Unassisted(env)
    elif support_name == 'random_pushes':
        from .wrappers import RandomPushes
        return RandomPushes(
            env,
            optimal_agent_training_timesteps=optimal_agent_training_timesteps,
            optimal_agent_smoothing_timesteps=optimal_agent_smoothing_timesteps,
        )
    elif support_name == 'bumpers':
        from .wrappers import Bumpers
        return Bumpers(
            env,
            threshold=threshold,
            trajectory_distance=trajectory_distance,
            p_override=p_override,
            undoing=undoing,
            p_suboptimal_override=p_suboptimal_override,
            override_next_best=override_next_best,
            optimal_agent_training_timesteps=optimal_agent_training_timesteps,
            optimal_agent_smoothing_timesteps=optimal_agent_smoothing_timesteps,
        )
    elif support_name == 'q_bumpers':
        from .wrappers import QBumpers
        return QBumpers(
            env,
            boltzmann_parameter=q_bumper_boltzmann,
            version=q_bumper_version,
            p_override=p_override,
            undoing=undoing,
            p_suboptimal_override=p_suboptimal_override,
            override_next_best=override_next_best,
            target_r=q_bumper_target_r,
            optimal_agent_training_timesteps=optimal_agent_training_timesteps,
            optimal_agent_smoothing_timesteps=optimal_agent_smoothing_timesteps,
            length_normalized=q_bumper_length_normalized,
            logistic_upper_prob=q_bumper_logistic_upper_prob,
            gamma=gamma,
            alpha=q_bumper_alpha,
        )
    elif support_name == 'reddy_rss':
        from .wrappers import QThreshold
        return QThreshold(
            env,
            q_threshold=q_threshold,
            p_override=p_override,
            undoing=undoing,
            p_suboptimal_override=p_suboptimal_override,
            override_next_best=override_next_best,
            optimal_agent_training_timesteps=optimal_agent_training_timesteps,
            optimal_agent_smoothing_timesteps=optimal_agent_smoothing_timesteps,
        )
    else:
        raise NotImplementedError


def make_val_assisted_env(
        act_labels,
        n_act_dim,
        obs_tp1=None,
        goal=None,
        seed=None,
        threshold=None,
        threshold_diff=None,
        dyn_transfer=False,
        optimal_Q=None,
        q_threshold=None,  # alpha in Reddy NIPS
        verbose=False,
        start_pos=None,
):
    """Env that takes corrective action.

    Always takes user action unless one of the following threshold
    conditions are met.

    Args:
        threshold: Maximum value allowed for user action, based
            on KL(s_user, s*).
        threshold_diff: Maximum value allowed for user action, based
            on ABS(KL(s_user, s*) - KL(s_best, s*)).

    """
    is_bumpers = threshold is not None or threshold_diff is not None
    if seed is not None:
        np.random.seed(seed)
    if goal is None:
        raise NotImplementedError
        goal = np.random.choice(gw_size, 2)
    test_reward_func = envs.make_reward_func(goal)
    env = envs.GridWorldNav(
        reward_func=test_reward_func, goal=goal, act_labels=act_labels
    )
    if verbose:
        print(goal)

    real_dyn = np.argmax(env.unwrapped.T, axis=2)

    env.unwrapped._reset_orig = env.unwrapped.reset
    def reset(self, pos=None, seed=None):
        if start_pos is not None and pos is not None:
            raise Exception('Environment has fixed start state. Cannot reset.')
        if start_pos is not None:
            pos = start_pos
        prev_obs = self._reset_orig(pos=pos, seed=seed)
        if is_bumpers:
            pos = prev_obs
            _env = envs.GridWorldNav(
                reward_func=test_reward_func, goal=goal, act_labels=act_labels
            )
            self.target_trajectory = get_trajectory(
                make_perfect_pilot_policy(goal=goal, act_labels=act_labels, seed=0),
                _env,
                reset_pos=pos,
            )
            if verbose:
                print('Target:')
                print([list(featurize_state(s)) for s in self.target_trajectory])
        return prev_obs
    env.unwrapped.reset = types.MethodType(reset, env.unwrapped)


    env.unwrapped._step_orig = env.unwrapped.step

    def step(self, action):
        new_info = {
            'user_action': action,
        }
        should_override = None
        recommended_act = None
        v_action = None
        v_best = None
        dists = None
        next_act = action
        target_trajectory = None
        if dyn_transfer:
            def f_dyn_transfer(state, action, obs_tp1=obs_tp1):
                desired_next_state = obs_tp1[state, action]
                next_states = [real_dyn[state, a] for a in range(n_act_dim)]
                # Each state corresponds to diff action
                dists = np.array([state_dist(desired_next_state, sp)
                                  for sp in next_states])
                return np.argmax(-dists)  # Return action for closest state

            next_act = f_dyn_transfer(self.curr_obs, action, obs_tp1=obs_tp1)
            recommended_act = next_act
        elif optimal_Q is not None and q_threshold is not None:
            q_normalized = optimal_Q[self.curr_obs, :] - np.min(optimal_Q[self.curr_obs,:])
            if q_normalized[action] >= (1 - q_threshold) * np.max(q_normalized):
                next_act = action
                recommended_act = None
                should_override = False
            else:
                next_act = np.argmax(q_normalized)
                recommmended_act = next_act
                should_override = True
            # TODO: Doesn't select action most similar to user action.
        elif is_bumpers:
            target_trajectory = self.target_trajectory
            recommended_act, v_best, v_action, dists = action_recommendation(
                self.curr_obs,
                action,
                target_trajectory=self.target_trajectory,
                real_dyn=real_dyn,
                #step_idx=env.unwrapped.curr_step
                step_idx=self.curr_step
            )
            next_act = action
            amount_better = v_best - v_action
            should_override = (
                (
                    threshold_diff is not None
                    and amount_better > threshold_diff
                )
                or (
                    threshold is not None
                    and dists[action] > threshold
                )
            )
            if should_override:
                next_act = recommended_act
                # print('OVERRIDING! {} with {} ({:.2f} better)'.format(action, recommended_act, amount_better))
        new_info.update({
            'final_action': next_act,
            'override_action': next_act != action,
            'recommended_act': recommended_act,
            'should_override': should_override,
            'v_best': v_best,
            'v_action': v_action,
            'dists': dists,
            'target_trajectory': target_trajectory,
        })
        obs, r, done, info = self._step_orig(next_act)
        info.update(new_info)
        return obs, r, done, info
    env.unwrapped.step = types.MethodType(step, env.unwrapped)
    return env
