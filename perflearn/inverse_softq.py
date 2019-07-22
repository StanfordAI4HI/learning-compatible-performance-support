import uuid
import pprint
import collections
import random
import numpy as np
import tensorflow as tf
from . import envs
from . import policies

FAKE_N = -999999

gamma = 0.99
iterations = 1000
learning_rate = 1e-3
batch_size = 512
sq_td_err_penalty = 1e-3

q_n_layers = 0
q_layer_size = None
q_activation = None
q_output_activation = None

n_layers = 0
layer_size = None
activation = None
output_activation = tf.nn.softmax

val_update_freq = 100
n_val_eval_rollouts = 10

im_scope = str(uuid.uuid4())
q_scope = str(uuid.uuid4())


def vectorize_rollouts(rollouts, user_action=None):
    """Split a list of rollouts into components.

    Args:
        user_action (bool): Assume that all actions taken were user actions.

    """
    if user_action is not None and user_action != 'all_own':
        raise NotImplementedError
    obs = []
    actions = []
    rewards = []
    next_obs = []
    dones = []
    task_idxes = []
    for rollout in rollouts:
        more_obs, more_actions, more_rewards, more_next_obs, more_dones, more_task_idxes, infos = list(
            zip(*rollout))
        if not user_action:
            more_actions = [d['final_action'] for d in infos]
        obs.extend(more_obs)
        actions.extend(more_actions)
        rewards.extend(more_rewards)
        next_obs.extend(more_next_obs)
        dones.extend(more_dones)
        task_idxes.extend(more_task_idxes)
    return (
        np.array(obs),
        np.array(actions),
        np.array(rewards),
        np.array(next_obs),
        np.array(dones),
        np.array(task_idxes),
    )





def process_demo_rollouts(
        demo_rollouts,
        train_frac=0.9,
        seed=None,
        user_action=False,
):
    """Process rollouts.

    Returns:
        {
            'demo_obs': Vector of observations,
            'demo_actions': Vector of observations,
            'demo_next_obs': Vector of next observations,
            'demo_task_idxes': Vector of optional task indexes,
            'train_demo_example_idxes': Rollout indices for training.
            'val_demo_batch': val_demo_batch,
        }

        Where val_demo_batch is a tuple containing the following for the
        test rollout indices:
            (
                demo_obs
                demo_actions,
                demo_next_obs,
                demo_task_idxes,
             )


    """
    demo_obs = None
    demo_actions = None
    demo_next_obs = None
    demo_task_idxes = None
    train_demo_example_idxes = None
    val_demo_batch = None

    vectorized_demo_rollouts = vectorize_rollouts(
        demo_rollouts,
        user_action=user_action,
    )

    demo_obs, demo_actions, demo_rewards, demo_next_obs, demo_done_masks, demo_task_idxes = vectorized_demo_rollouts
    demo_example_idxes = list(range(len(demo_obs)))

    if seed is not None:
        random.seed(seed)
    random.shuffle(demo_example_idxes)
    n_train_demo_examples = int(train_frac * len(demo_example_idxes))
    train_demo_example_idxes = demo_example_idxes[:n_train_demo_examples]
    val_demo_example_idxes = demo_example_idxes[n_train_demo_examples:]
    val_demo_batch = (
        demo_obs[val_demo_example_idxes],
        demo_actions[val_demo_example_idxes],
        demo_next_obs[val_demo_example_idxes],
        demo_task_idxes[val_demo_example_idxes]
    )

    return {
        'demo_obs': demo_obs,
        'demo_actions': demo_actions,
        'demo_next_obs': demo_next_obs,
        'demo_task_idxes': demo_task_idxes,
        'train_demo_example_idxes': train_demo_example_idxes,
        'val_demo_batch': val_demo_batch,
    }







class InverseSoftQModel:

    def __init__(
            self,
            train_envs,
            n_act_dim=envs.N_ACT_DIM,
            n_obs_dim=envs.N_OBS_DIM,
            true_dynamics=None,
    ):
        """

        Args:
            train_envs: List of environments. Uses known rewards.
            true_dynamics: If None, use dynamics from first training task.

        """
        n_train_tasks = len(train_envs)
        demo_obs_t_ph = tf.placeholder(tf.int32, [None])
        demo_act_t_ph = tf.placeholder(tf.int32, [None])
        demo_task_t_ph = tf.placeholder(tf.int32, [None])
        demo_batch_size_ph = tf.placeholder(tf.int32)

        demo_batch_idxes = tf.reshape(
            tf.range(0, demo_batch_size_ph, 1), [demo_batch_size_ph, 1])

        demo_q_t = tf.stack([
            self._build_mlp(
                self._featurize_obs(demo_obs_t_ph, n_obs_dim),
                n_act_dim,
                q_scope+'-'+str(train_task_idx),
                n_layers=q_n_layers,
                size=q_layer_size,
                activation=q_activation,
                output_activation=q_output_activation,
            ) for train_task_idx in range(n_train_tasks)
        ], axis=0)
        demo_q_t = tf.gather_nd(demo_q_t, tf.concat(
            [tf.expand_dims(demo_task_t_ph, 1), demo_batch_idxes], axis=1))

        demo_act_idxes = tf.concat([demo_batch_idxes, tf.reshape(
            demo_act_t_ph, [demo_batch_size_ph, 1])], axis=1)
        demo_act_val_t = tf.gather_nd(demo_q_t, demo_act_idxes)
        state_val_t = tf.reduce_logsumexp(demo_q_t, axis=1)
        act_log_likelihoods = demo_act_val_t - state_val_t

        neg_avg_log_likelihood = -tf.reduce_mean(act_log_likelihoods)

        obs_for_obs_tp1_probs = tf.cast(tf.floor(
            tf.range(0, n_obs_dim*n_act_dim, 1) / n_act_dim), dtype=tf.int32)

        act_for_obs_tp1_probs = tf.floormod(tf.range(
            0, n_obs_dim*n_act_dim, 1), n_act_dim)

        obs_tp1_probs_in = tf.one_hot(
            obs_for_obs_tp1_probs*n_act_dim+act_for_obs_tp1_probs, n_obs_dim*n_act_dim)

        obs_tp1_probs = self._build_mlp(
            obs_tp1_probs_in,
            n_obs_dim, im_scope, n_layers=n_layers, size=layer_size,
            activation=activation, output_activation=output_activation
        )
        obs_tp1_probs = tf.reshape(
            obs_tp1_probs, [n_obs_dim, n_act_dim, n_obs_dim])

        q_tp1 = tf.stack([
            self._build_mlp(
                self._featurize_obs(tf.range(0, n_obs_dim, 1), n_obs_dim),
                n_act_dim,
                q_scope+'-'+str(train_task_idx),
                n_layers=q_n_layers,
                size=q_layer_size,
                activation=q_activation,
                output_activation=q_output_activation,
                reuse=True,
            ) for train_task_idx in range(n_train_tasks)
        ], axis=0)

        v_tp1 = tf.reduce_logsumexp(q_tp1, axis=2)

        all_rew = tf.convert_to_tensor(np.stack(
            [env.unwrapped.R for env in train_envs], axis=0), dtype=tf.float32)

        v_tp1_broad = tf.reshape(v_tp1, [n_train_tasks, 1, 1, n_obs_dim])
        obs_tp1_probs_broad = tf.expand_dims(obs_tp1_probs, 0)

        exp_v_tp1 = tf.reduce_sum(obs_tp1_probs_broad * v_tp1_broad, axis=3)
        exp_rew_t = tf.reduce_sum(obs_tp1_probs_broad * all_rew, axis=3)
        target_t = exp_rew_t + gamma * exp_v_tp1

        q_t = tf.stack([
            self._build_mlp(
                self._featurize_obs(tf.range(0, n_obs_dim, 1), n_obs_dim),
                n_act_dim,
                q_scope+'-'+str(train_task_idx),
                n_layers=q_n_layers,
                size=q_layer_size,
                activation=q_activation,
                output_activation=q_output_activation,
                reuse=True,
            )
            for train_task_idx in range(n_train_tasks)
        ], axis=0)

        td_err = q_t - target_t
        sq_td_err = tf.reduce_mean(td_err**2)
        loss = neg_avg_log_likelihood + sq_td_err_penalty * sq_td_err

        update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        self.n_act_dim = n_act_dim
        self.n_obs_dim = n_obs_dim

        self.demo_obs_t_ph = demo_obs_t_ph
        self.demo_act_t_ph = demo_act_t_ph
        self.demo_task_t_ph = demo_task_t_ph
        self.demo_batch_size_ph = demo_batch_size_ph

        self.q_t = q_t
        self.loss = loss
        self.neg_avg_log_likelihood = neg_avg_log_likelihood
        self.sq_td_err = sq_td_err
        self.update_op = update_op
        self.obs_tp1_probs = obs_tp1_probs
        if true_dynamics is None:
            self.true_dynamics = np.argmax(train_envs[0].unwrapped.T, axis=2)
        else:
            self.true_dynamics = true_dynamics


    @staticmethod
    def _featurize_obs(obs, n_obs_dim):
        return tf.one_hot(obs, n_obs_dim)

    @staticmethod
    def _build_mlp(
            input_placeholder,
            output_size,
            scope,
            n_layers=2,
            size=500,
            activation=tf.nn.relu,
            output_activation=None,
            reuse=False
    ):
        out = input_placeholder
        with tf.variable_scope(scope, reuse=reuse):
            for _ in range(n_layers):
                out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
        return out

    def compute_int_dyn_acc(self, sess):
        states = np.repeat(np.arange(0, self.n_obs_dim, 1), self.n_act_dim)
        actions = np.tile(np.arange(0, self.n_act_dim, 1), self.n_obs_dim)
        probs = sess.run(self.obs_tp1_probs)
        obs_tp1_true = self.true_dynamics[states, actions]
        accuracy = np.mean(
            (np.argmax(probs[states, actions, :], axis=1) == obs_tp1_true).astype(int))
        kld = -np.mean(np.log(1e-9+probs[states, actions, obs_tp1_true]))
        return {'int_dyn_acc': accuracy, 'int_dyn_kld': kld}


    def get_internal_model(self, sess):
        _T = sess.run(self.obs_tp1_probs)
        return _T


    def compute_batch_loss(
            self,
            sess,
            test_act_labels,
            train_act_labels,
            demo_batch,
            step=False,
            t=None,
            goal=None,
    ):
        demo_batch_obs_t, demo_batch_act_t, demo_batch_obs_tp1, demo_batch_task_t = demo_batch

        feed_dict = {
            self.demo_obs_t_ph: demo_batch_obs_t,
            self.demo_act_t_ph: demo_batch_act_t,
            self.demo_task_t_ph: demo_batch_task_t,
            self.demo_batch_size_ph: demo_batch_obs_t.shape[0],
        }

        [loss_eval, neg_avg_log_likelihood_eval, sq_td_err_eval] = sess.run(
            [self.loss, self.neg_avg_log_likelihood, self.sq_td_err],
            feed_dict=feed_dict,
        )

        if step:
            sess.run(self.update_op, feed_dict=feed_dict)

        d = {
            'loss': loss_eval,
            'nll': neg_avg_log_likelihood_eval,
            'ste': sq_td_err_eval,
        }
        if not step:
            d.update(self.compute_int_dyn_acc(sess=sess))
            #d.update(self.compute_assisted_perf(
            #    sess=sess,
            #    test_act_labels=test_act_labels,
            #    train_act_labels=train_act_labels,
            #    goal=goal,
            #))
        return d


def run_learning(
        model, sess, rollouts, test_goal, test_act_labels,
        train_act_labels, n_iters=20000, user_action=False,
        train_frac=0.9,
):
    """Run learning.

    Args:
        rollouts: List of list of rollouts, one per task (environment).
            IMPORTANT: rollouts must have indexes corresponding to train_task indexes.

    """
    # TODO: n_iters = iterations * len(demo_obs) // batch_size
    master_train_logs = []
    processed_rollouts = process_demo_rollouts(
        sum(rollouts, []),
        user_action=user_action,
        train_frac=train_frac,
    )
    val_demo_batch = processed_rollouts['val_demo_batch']
    demo_obs = processed_rollouts['demo_obs']
    demo_actions = processed_rollouts['demo_actions']
    demo_next_obs = processed_rollouts['demo_next_obs']
    demo_task_idxes = processed_rollouts['demo_task_idxes']

    tf.global_variables_initializer().run(session=sess)

    train_logs = {
        'loss_evals': [],
        'nll_evals': [],
        'ste_evals': [],
        'val_loss_evals': [],
        'val_nll_evals': [],
        'val_ste_evals': [],
        'assisted_rew_evals': [],
        'assisted_succ_evals': [],
        'assisted_crash_evals': [],
        'dyn_assisted_rew_evals': [],
        'dyn_assisted_succ_evals': [],
        'dyn_assisted_crash_evals': [],
        'unassisted_rew_evals': [],
        'unassisted_succ_evals': [],
        'unassisted_crash_evals': [],
        'aristotle_unassisted_rew_evals': [],
        'aristotle_unassisted_succ_evals': [],
        'aristotle_unassisted_crash_evals': [],
        'aristotle_dyn_assisted_rew_evals': [],
        'aristotle_dyn_assisted_succ_evals': [],
        'aristotle_dyn_assisted_crash_evals': [],
        'softq_unassisted_rew_evals': [],
        'softq_unassisted_succ_evals': [],
        'softq_unassisted_crash_evals': [],
        'softq_dyn_assisted_rew_evals': [],
        'softq_dyn_assisted_succ_evals': [],
        'softq_dyn_assisted_crash_evals': [],
        'int_dyn_acc_evals': [],
        'int_dyn_kld_evals': [],
    }

    val_log = None
    while len(train_logs['loss_evals']) < n_iters:
        def sample_batch(size):
            idxes = random.sample(
                processed_rollouts['train_demo_example_idxes'],
                size,
            )
            demo_batch = (
                demo_obs[idxes],
                demo_actions[idxes],
                demo_next_obs[idxes],
                demo_task_idxes[idxes],
            )
            return demo_batch

        demo_batch = sample_batch(batch_size)

        t = len(train_logs['loss_evals'])
        train_log = model.compute_batch_loss(
            sess,
            test_act_labels,
            train_act_labels,
            demo_batch,
            step=True,
            t=t,
            goal=test_goal,
        )
        if val_demo_batch and (val_log is None or len(train_logs['loss_evals']) % val_update_freq == 0):
            val_log = model.compute_batch_loss(
                sess,
                test_act_labels,
                train_act_labels,
                val_demo_batch,
                step=False,
                t=t,
                goal=test_goal,
            )

        if t % 1000 == 0:
            print('%d %d %f %f %f' % (
                t, n_iters, train_log['loss'],
                train_log['nll'], train_log['ste'],
            ))
            pprint.pprint(val_log)
            """
                  val_log['loss'],
                val_log['nll'], val_log['ste'], val_log['assisted_rew'],
                val_log['unassisted_rew'], val_log['aristotle_unassisted_rew'],
                val_log['dyn_assisted_rew'], val_log['aristotle_dyn_assisted_rew'],
                val_log.get('softq_unassisted_rew', FAKE_N),
                val_log.get('softq_dyn_assisted_rew', FAKE_N),
            )
            )
            """

        for k, v in train_log.items():
            train_logs['%s_evals' % k].append(v)
        for k, v in val_log.items():
            train_logs['%s%s_evals' %
                       ('val_' if k in ['loss', 'nll', 'ste'] else '', k)].append(v)

    master_train_logs.append(train_logs)
    return master_train_logs
