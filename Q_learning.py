
import gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human")
env.action_space.seed(42)

c_learning_rate = 0.8
c_discount_value = 0.9
c_no_of_eps = 1
c_show_each = 1000

n = 20
length_of_list =  env.observation_space.shape[0]
q_table_size = [n] * length_of_list
q_table_segment_size = (env.observation_space.high - env.observation_space.low) / q_table_size

def convert_state(real_state):
    Q_state = (real_state - env.observation_space.low) // q_table_segment_size
    return tuple(Q_state.astype(np.int_))

q_table = np.random.uniform(low=-2, high=0, size=(q_table_size + [env.action_space.n]))


max_ep_reward = -999
max_ep_action_list = []
max_start_state = None


for ep in range(c_no_of_eps):
    print("Eps = ", ep)
    terminated = False
    ep_reward = 0
    action_list = []
    observation, info = env.reset(seed=42)
    current_overvation = convert_state(observation)
    ep_start_state = current_overvation

    while not terminated:
        action = np.argmax(q_table[current_overvation])
        action_list.append(action)

        new_real_observation, reward, terminated, truncated, info = env.step(action=action)

        ep_reward += reward
        if terminated:
            print("Đã đến cờ tại ep = {}, reward = {}".format(ep, ep_reward))
            if ep_reward > max_ep_reward:
                max_ep_reward = ep_reward
                max_ep_action_list = action_list
                max_start_state = ep_start_state
        else:
            new_observation = convert_state(new_real_observation)

            current_q_value = q_table[current_overvation + (action,)]

            new_q_value = ((1 - c_learning_rate) * current_q_value + c_learning_rate *
                           (reward + c_discount_value * np.max(q_table[new_observation])))

            q_table[current_overvation + (action,)] = new_q_value

            current_overvation = new_observation


print("Max reward = ", max_ep_reward)
print("Max action list = ", max_ep_action_list)


env.reset(seed=42)
for action in max_ep_action_list:
    env.step(action=action)





