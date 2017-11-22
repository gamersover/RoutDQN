import numpy as np
import tensorflow as tf
import random
from routgame.routing_game import Environment
from collections import deque

# np.random.seed(1)
# tf.set_random_seed(1)

class DeepQNetwork:

	def __init__(
			self, 
			input_features, 
			n_actions, 
			learning_rate=0.01,
			reward_decay=0.9,
			e_greedy=0.9,
			replace_target_iter=300,
			saved_network_step=300,
			memory_size=2000,
			observe_step=200,
			batch_size=32,
			e_greedy_increment=None,
				):
		self.input_features = input_features
		self.n_actions = n_actions
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon_max = e_greedy
		self.replace_target_iter = replace_target_iter
		self.saved_network_step = saved_network_step
		self.memory_size = memory_size
		self.observe_step = observe_step
		self.batch_size = batch_size
		self.epsilon_increment = e_greedy_increment
		self.epsilon = 0.1 if self.epsilon_increment is not None else self.epsilon_max

		self.learn_step_counter = 0
		self.replayMemory = deque()
		
		self._build_net()
		t_params = tf.get_collection('target_net_parmas')
		e_params = tf.get_collection('eval_net_params')
		self.replace_target_op = [tf.assign(t,e) for t,e in zip(t_params, e_params)]
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state('checkpoint')
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print('Successfully loaded: ',checkpoint.model_checkpoint_path)
		else:
			print('Could not find checkpoint')


	def _build_net(self):
		self.s = tf.placeholder(tf.float32, [None, self.input_features], name='s')
		self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
		with tf.variable_scope('eval_net'):
			
			c_names, n_l1, w_initializer, b_initializer = \
            ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
            tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
			
			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1', [self.input_features, n_l1], initializer=w_initializer, collections=c_names)
				b1 = tf.get_variable('b1', [n_l1], initializer=b_initializer, collections=c_names)
				h1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
				b2 = tf.get_variable('b2', [self.n_actions], initializer=b_initializer, collections=c_names)
				self.q_eval = tf.matmul(h1, w2) + b2

			with tf.variable_scope('loss'):
				self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

			with tf.variable_scope('train_op'):
				self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

		self.s_ = tf.placeholder(tf.float32, [None, self.input_features], name='s_')
		with tf.variable_scope('target_net'):
			c_names = ['target_net_parmas', tf.GraphKeys.GLOBAL_VARIABLES]
			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1', [self.input_features, n_l1], initializer=w_initializer, collections=c_names)
				b1 = tf.get_variable('b1', [n_l1], initializer=b_initializer, collections=c_names)
				h1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
				b2 = tf.get_variable('b2', [self.n_actions], initializer=b_initializer, collections=c_names)
				self.q_next = tf.matmul(h1, w2) + b2

	def store_transition(self, s, a, r, s_, terminal):
		self.replayMemory.append((s,a,r,s_,terminal))
		if len(self.replayMemory) > self.memory_size:
			self.replayMemory.popleft()

	def choose_action(self, observation):
		
		action = np.zeros(self.n_actions)
		if np.random.uniform() < self.epsilon:
			action_value = self.sess.run(self.q_eval, feed_dict={self.s:observation})
			action_idx = action_value.argmax()
		else:
			action_idx = np.random.randint(0, self.n_actions)
		
		action[action_idx] = 1
		return action

	def learn(self):
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.sess.run(self.replace_target_op)
			print('replace target network!')

		if self.learn_step_counter > self.observe_step:
			mini_batch = random.sample(self.replayMemory, self.batch_size)
			s_batch = np.array([data[0] for data in mini_batch]).reshape([-1,2])
			a_batch = [data[1] for data in mini_batch]
			r_batch = [data[2] for data in mini_batch]
			s_next_batch = np.array([data[3] for data in mini_batch]).reshape([-1,2])
			t_batch = [data[4] for data in mini_batch]
			q_next, q_eval = self.sess.run([self.q_next, self.q_eval], 
											feed_dict={self.s_:s_next_batch, self.s:s_batch})
			q_target = q_next.copy()

			for i in range(self.batch_size):
				terminal = mini_batch[i][4]
				if terminal:
					q_target[i, a_batch[i].argmax()] = r_batch[i]
				else:
					q_target[i, a_batch[i].argmax()] = r_batch[i] + self.gamma*np.max(q_next[i])
			_, loss = self.sess.run([self.train_op, self.loss],
									feed_dict={self.s:s_batch, self.q_target:q_target})

			self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		
		if self.learn_step_counter%self.saved_network_step == 0:
			self.saver.save(self.sess, 'checkpoint/model.ckpt')
			print('network saved')
		
		self.learn_step_counter += 1
		

def run():
	
	step = 0

	for episode in range(300):
		observation, _, _ = env.step([0,0,0,0])
		while True:
			action = RL.choose_action(observation)
			observation_, reward, terminal = env.step(action)
			RL.store_transition(observation, action, reward, observation_, terminal)
			if (step > 200) and (step % 5 == 0):
				RL.learn()

			observation = observation_
			if terminal:
				break
			step += 1

if __name__ == '__main__':

	env = Environment()
	RL = DeepQNetwork(2, 4)
	run()
