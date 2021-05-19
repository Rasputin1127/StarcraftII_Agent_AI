from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import matplotlib.pylab as plt
import matplotlib as mpl

plt.style.use("ggplot") # always make it pretty
mpl.rcParams['font.size'] = 16 # Frank was here
mpl.rcParams['figure.figsize'] = 10, 8

from pysc2.lib import actions


class BaseAgent(object):
  """A base agent to write custom scripted agents.
  It can also act as a passive agent that does nothing but no-ops.
  """

  def __init__(self):
    self.reward = 0
    self.episodes = 0
    self.steps = 0
    self.obs_spec = None
    self.action_spec = None

  def setup(self, obs_spec, action_spec):
    self.obs_spec = obs_spec
    self.action_spec = action_spec

  def reset(self):
    self.episodes += 1

  def step(self, obs):
    self.steps += 1
    self.reward += obs.reward
    return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

    def step(self, obs):
        num_actions = self.action_spec
        num_states = self.obs_spec
        
        # create matrix of model-interpretable states
        obs_states = np.identity(num_states)
        
        # create the model
        model = Sequential()
        model.add(InputLayer(batch_input_shape=(1, num_states)))
        model.add(Dense(10, activation='sigmoid'))
        model.add(Dense(num_actions, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        # now execute the q learning
        gamma = 0.95
        eps = 0.2
        decay_factor = 0.999
        r_avg_list = []
        for i in range(num_episodes):
            obs = env.reset()
            eps *= decay_factor
    #         if i % 25 == 0:
            print("Episode {} of {}".format(i, num_episodes))
            done = False
            reward_sum = 0
            iteration = 0
            while not done: 
    #             print(f'iteration: {iteration}')
                iteration += 1
                if np.random.random() < eps:
                    action = np.random.randint(0, num_actions)
                else:
                    action = np.argmax(model.predict(obs_states[obs:obs + 1]))
                    
                new_obs, reward, done, _ = env.step(action)
                target_q = reward + gamma * np.max(model.predict(obs_states[new_obs:new_obs + 1]))
                target_vec = model.predict(obs_states[obs:obs + 1])[0]
                target_vec[action] = target_q
                model.fit(obs_states[obs:obs + 1], target_vec.reshape(-1, num_actions), epochs=1, verbose=0)
                obs = new_obs
                reward_sum += reward
                
                if iteration > 50: break
            r_avg_list.append(reward_sum/iteration)