import tensorflow as tf
import numpy as np
from routgame.routing_game import Environment

epsilon = 0.9
env = Environment()
model_path = 'checkpoint/model.ckpt'
meta_path = model_path + '.meta'

saver = tf.train.import_meta_graph(meta_path)

with tf.Session() as sess:
	
	saver.restore(sess, model_path)
	graph = tf.get_default_graph()
	Q_eval = graph.get_tensor_by_name('eval_net/l2/add:0')
	s = graph.get_tensor_by_name('s:0')
	
	for i in range(10):
		observation,_,_ = env.step([0,0,0,0])
		terminal = False
		while not terminal:
			q_eval = sess.run(Q_eval, feed_dict={s:observation})
			action = np.zeros(4)
			if np.random.uniform() < epsilon:
				action[q_eval.argmax()] = 1
			else:
				action[np.random.randint(0,4)] = 1
			observation,_,terminal = env.step(action)

	

