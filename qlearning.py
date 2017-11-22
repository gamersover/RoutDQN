import numpy as np
import pandas as pd
from routing_game import Environment

class Qlearn:

	def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):

		self.actions = actions
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.q_tables = pd.DataFrame(columns=self.actions, dtype=np.float64)

	def choose_action(self, observation):
		self.check_state_exist(observation)
		action_li = np.zeros(len(self.actions))
		state_action = self.q_tables.ix[observation,:]
		state_action = state_action.reindex(np.random.permutation(state_action.index))
		if np.random.uniform() < self.epsilon:
			action_idx = state_action.argmax()

		else:
			action_idx = np.random.randint(0,len(self.actions))

		action_li[action_idx] = 1	
		return action_li

	def learn(self, s, a, r, s_, done):
		self.check_state_exist(s_)
		a_idx = a.argmax()
		q_predict = self.q_tables.ix[s,a]
		if not done:
			q_target = r + self.gamma*self.q_tables.ix[s_, :].max()
		else:
			q_target = r

		self.q_tables.ix[s,a] = q_predict + self.lr*(q_target-q_predict)
	
	def check_state_exist(self, state):
	        if state not in self.q_tables.index:
	            # append new state to q table
	            self.q_tables = self.q_tables.append(
	                pd.Series(
	                    [0]*len(self.actions),
	                    index=self.q_tables.columns,
	                    name=state,
	                )
	            )

def update():

	for episode in range(100):
		
		observation,_,_ = env.step([0,0,0,0])
		
		while True:
			action = RL.choose_action(observation)
			observation_, reward, terminal = env.step(action)
			a = action.argmax()
			RL.learn(observation, a, reward, observation_, terminal)
			observation = observation_
			if terminal:
				break
		# print(RL.q_tables)


if __name__ == '__main__':

	env = Environment()
	RL = Qlearn(range(4))
	update()
	# print(RL.q_tables)