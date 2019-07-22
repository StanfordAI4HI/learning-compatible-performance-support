from __future__ import division
import random

import gym
from gym import spaces
import numpy as np



GW_SIZE = 7
#N_TRAIN_TASKS = 49
N_TRAIN_TASKS = 1
N_ACT_DIM = 4
N_OBS_DIM = GW_SIZE**2 + 1
SUCC_REW_BONUS = 1
CRASH_REW_PENALTY = -1
GAMMA = 0.99
MAX_EP_LEN = 100
UNIV_SCRAMBLE = True


DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3



def make_reward_func(
        goal,
        gw_size=GW_SIZE,
        crash_rew_penalty=CRASH_REW_PENALTY,
        succ_rew_bonus=SUCC_REW_BONUS,
        gamma=GAMMA,
):
    def pos_from_obs(obs):
        x = obs // gw_size
        y = obs % gw_size
        return np.array([x, y])

    def reward_shaping(obs):
        """Reward for moving closer to goal"""
        return -np.linalg.norm((pos_from_obs(obs) - goal) / gw_size)

    def reward_func(prev_obs, action, obs):
        pos = pos_from_obs(obs)
        if (pos < 0).any() or (pos >= gw_size).any():
            r = crash_rew_penalty
        elif (pos == goal).all():
            r = succ_rew_bonus
        else:
            r = 0
        r += gamma * reward_shaping(obs) - reward_shaping(prev_obs)
        return r

    return reward_func


def make_act_labels(
        n_act_dim=N_ACT_DIM,
        n_obs_dim=N_OBS_DIM,
        univ_scramble=UNIV_SCRAMBLE,
):
    """Return newton and aristotle act labels."""
    newton_act_labels = [list(range(n_act_dim)) for _ in range(n_obs_dim)]
    newton_act_labels = np.array(newton_act_labels).astype(int)

    # Aristotle has scrambled actions
    aristotle_act_labels = [list(range(n_act_dim)) for _ in range(n_obs_dim)]
    for i in range(n_obs_dim):
        if univ_scramble:
            aristotle_act_labels[i] = list(reversed(aristotle_act_labels[i]))
        else:
            random.shuffle(aristotle_act_labels[i])
    aristotle_act_labels = np.array(aristotle_act_labels).astype(int)
    return newton_act_labels, aristotle_act_labels


def make_goals(gw_size=GW_SIZE, n_train_tasks=N_TRAIN_TASKS, seed=None):
    """Return all goals and training goals."""
    all_goals = list(
        zip(*[x.ravel() for x in np.meshgrid(
            np.arange(0, gw_size, 1), np.arange(0, gw_size, 1))])
    )

    # Select a random (without replacement) subset of n_train_tasks
    if seed is not None:
        np.random.seed(seed)
    train_goals = [
        all_goals[i] for i in np.random.choice(
            list(range(len(all_goals))), n_train_tasks, replace=False
        )
    ]
    train_goals = np.array(train_goals)
    return all_goals, train_goals


class GridWorldNav(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(
            self,
            act_labels=None,
            max_ep_len=MAX_EP_LEN,
            reward_func=None,
            goal=None,
            succ_rew_bonus=SUCC_REW_BONUS,
            crash_rew_penalty=CRASH_REW_PENALTY,
            n_act_dim=N_ACT_DIM,
            n_obs_dim=N_OBS_DIM,
            gw_size=GW_SIZE,
            gamma=GAMMA,
    ):
        self.observation_space = spaces.Discrete(n_obs_dim)
        self.action_space = spaces.Discrete(n_act_dim)

        self.pos = None
        self.curr_step = None
        self.viewer = None
        self.curr_obs = None
        self.next_obs = None

        self.succ_rew_bonus = succ_rew_bonus
        self.max_ep_len = max_ep_len
        self.reward_func = reward_func
        self.act_labels = act_labels
        self.goal = goal
        self.gw_size = gw_size

        self.is_succ = lambda r: r[-1][2] > succ_rew_bonus / 2
        self.is_crash = lambda r: r[-1][2] < crash_rew_penalty / 2

        if reward_func is None:
            self.reward_func = make_reward_func(
                goal=goal,
                gw_size=gw_size,
                succ_rew_bonus=succ_rew_bonus,
                crash_rew_penalty=crash_rew_penalty,
            )

        self.R = np.zeros((n_obs_dim, n_act_dim, n_obs_dim))
        for s in range(n_obs_dim):
            for sp in range(n_obs_dim):
                self.R[s, :, sp] = self.reward_func(s, None, sp)

        self.T = np.zeros((n_obs_dim, n_act_dim, n_obs_dim))
        for s in range(n_obs_dim-1):
            x = s // gw_size
            y = s % gw_size
            self.T[s, self.act_labels[s, DOWN], x *
                   gw_size+(y-1) if y > 0 else -1] = 1
            self.T[s, self.act_labels[s, UP], x*gw_size +
                   (y+1) if y < gw_size-1 else -1] = 1
            self.T[s, self.act_labels[s, LEFT],
                   (x-1)*gw_size+y if x > 0 else -1] = 1
            self.T[s, self.act_labels[s, RIGHT],
                   (x+1)*gw_size+y if x < gw_size-1 else -1] = 1
        self.T[-1, :, -1] = 1
        super(GridWorldNav, self).__init__()

    def obs(self):
        # Observe position
        self.curr_obs = int(self.pos[0]*self.gw_size + self.pos[1])
        # Map out-of-bounds to single observation
        if self.curr_obs < 0 or self.curr_obs >= self.gw_size**2:
            self.curr_obs = self.gw_size**2
        return self.curr_obs

    def step(self, action):
        if self.next_obs is None:
            if action == self.act_labels[self.curr_obs, DOWN]:  # left
                self.pos[1] -= 1
            elif action == self.act_labels[self.curr_obs, UP]:  # right
                self.pos[1] += 1
            elif action == self.act_labels[self.curr_obs, LEFT]:  # up
                self.pos[0] -= 1
            elif action == self.act_labels[self.curr_obs, RIGHT]:  # down
                self.pos[0] += 1
            else:
                raise ValueError('invalid action')
        else:
            self.pos = np.array(
                [self.next_obs // self.gw_size, self.next_obs % self.gw_size])

        self.curr_step += 1
        succ = (self.pos == self.goal).all()  # Success
        oob = (self.pos < 0).any() or (
            self.pos >= self.gw_size).any()  # Out of bounds
        oot = self.curr_step >= self.max_ep_len  # Out of time

        obs = self.obs()
        r = self.reward_func(self.prev_obs, action, obs)
        done = oot or succ or oob
        info = {}
        self.prev_obs = obs

        return obs, r, done, info

    def reset(self, pos=None, seed=None):
        # Choose a random distance from the goal
        if seed is not None:
            np.random.seed(seed)
        if pos is None:
            pos = (np.random.choice(self.gw_size**2-1) +
                   self.goal[0]*self.gw_size + self.goal[1]) % (self.gw_size**2)
        self.pos = np.array([pos // self.gw_size, pos % self.gw_size])

        self.curr_step = 0
        self.prev_obs = self.obs()
        self.next_obs = None
        return self.prev_obs

    def render(self, mode='rgb_array'):  # , mode='human', close=False):
        """
        if close:
        if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
        return



        if self.viewer is None:
        self.viewer = rendering.SimpleImageViewer()
        """
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        fig = plt.figure()
        canvas = FigureCanvas(fig)

        plt.scatter([self.goal[0]], [self.goal[1]], color='gray',
                    linewidth=0, alpha=0.75, marker='*')
        plt.scatter([self.pos[0]], [self.pos[1]],
                    color='orange', linewidth=0, alpha=0.75)
        plt.xlim([-1, self.gw_size+1])
        plt.ylim([-1, self.gw_size+1])
        plt.axis('off')

        agg = canvas.switch_backends(FigureCanvas)
        agg.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        #self.viewer.imshow(np.fromstring(agg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3))
        plt.close()
        return np.fromstring(agg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)



def run_ep(
        policy,
        env,
        max_ep_len=None,
        render=False,
        task_idx=None,
        seed=None,
        display=None,
        verbose=False,
        reset_pos=None,
        policy_explore=False,
        policy_update=False,
        user_action=None,
        override_penalty=0,
        override_penalty_only=False,
):
    """

    Args:
        seed: Seed for selecting same starting state only.

    """
    if seed is not None or reset_pos is not None:
        obs = env.reset(pos=reset_pos, seed=seed)
        raise NotImplementedError
    else:
        obs = env.reset()

    done = False
    totalr = 0.
    prev_obs = obs
    rollout = []
    #if seed is not None:
    #    np.random.seed(seed)
    for step_idx in range(max_ep_len+1 if max_ep_len is not None else 99999999):
        if done:
            break
        action = policy.act(obs, explore=policy_explore)
        obs, r, done, info = env.step(action)
        if policy_update:
            action_final = info['final_action']
            a_agent_sees = action_final
            if user_action == 'all_own':
                a_agent_sees = action

            r_agent_sees = r
            if override_penalty and override_penalty_only:
                r_agent_sees = override_penalty
            elif override_penalty:
                r_agent_sees += override_penalty

            if (
                    not user_action == 'own_only'
                    or action_final == action
            ):
                policy.update(
                    s=prev_obs,
                    a=a_agent_sees,
                    s1=obs,
                    r=r_agent_sees,
                    done=done,
                )
        rollout.append({
            'prev_obs': prev_obs,
            'action': action,
            'r': r,
            'obs': obs,
            'done': done,
            'info': info,
        })
        prev_obs = obs
        if render:
            from matplotlib import pyplot as plt
            #plt.figure()
            #plt.clf()
            plt.imshow(env.render(mode='rgb_array'))
            if display is not None:
                display.clear_output(wait=True)
                display.display(plt.gcf())

        totalr += r
    # env.render(close=True)
    env.close()
    return rollout
