import gym
import numpy as np
import random
import math
NUM_EPISODES = 100000
MAX_T = 250
ALPHA = 0.2
GAMMA = 1

EXPLORATION_RATE = 0.5
EXPLORATION_RATE_DECAY = 0.9

env = gym.make('CartPole-v0')

NUM_ACTIONS = env.action_space.n


CART_POS = np.linspace(-2.4, 2.4, 10)
POLE_ANGLE = np.linspace(-2, 2, 10)
CART_VEL = np.linspace(-1, 1, 10)
ANG_RATE = np.linspace(-3, 3, 10)

DEBUG_MODE = True
streaks = 0
Q = {}

NUM_STATES = 10**4

def to_bins(value, bins):
	return np.digitize(x=[value], bins=bins)[0]


def to_state(obs):
	x, theta, v, omega = obs
	state = (to_bins(x, CART_POS),
					to_bins(theta, POLE_ANGLE),
					to_bins(v, CART_VEL),
					to_bins(omega, ANG_RATE))
	return state

def get_action(state):
	p = np.random.uniform(0,1)
	#print p
	if p < EXPLORATION_RATE:
		return random.choice([0,1])
	x = []
	for action in [0,1]:
		if (state, action) not in Q:
			Q[(state, action)] = 0
		x.append(Q[(state, action)])

	return np.argmax(x)

avg = 0


for episode in range(NUM_EPISODES):
	obs = env.reset()
	state = to_state(obs)

	for t in range(MAX_T):
		#print t
		#env.render()
		action = get_action(state)

		obs, reward, done, _ = env.step(action)
		
		state_prime = to_state(obs)
		action_prime = get_action(state_prime)

		if (state_prime, action_prime) not in Q:
			Q[(state_prime, action_prime)] = np.random.uniform(1,-1)
		if (state, action) not in Q:
			Q[(state, action)] = np.random.uniform(1,-1)



		#print "olderstate: ", Q[(state, action)], "done", done,  "Reward:", reward
		Q[(state, action)] = (1-ALPHA)*Q[(state, action)] + ALPHA*(reward + GAMMA*Q[(state_prime, action_prime)])
		#print "newstate: ", Q[(state, action)]
		state = state_prime

		#print "done", done

		if done:
			#print("Episode %d completed in %d" % (episode, t))
			avg += t
			if t > 199:
				streaks += 1
			else:
				streaks = 0
			break

	EXPLORATION_RATE *= EXPLORATION_RATE_DECAY
	if episode%100 == 0:
		print "Average of 100:", avg/100.0
		avg = 0
	if streaks >= 120:
		print("Completed in %d episodes" % (episodes))
		break



	#print Q
	