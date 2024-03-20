import rooms
import agent as a
import matplotlib.pyplot as plot
import sys
# global temp

# def episode(env, agent, nr_episode=0):
#     state = env.reset()
#     discounted_return = 0
#     discount_factor = 0.997
#     done = False
#     time_step = 0
#     while not done:
#         # 1. Select action according to policy
#         action = agent.policy(state)
#         # 2. Execute selected action
#         next_state, reward, terminated, truncated, _ = env.step(action)
#         # 3. Integrate new experience into agent
#         global temp
#         temp = agent.update(state, action, reward, next_state, terminated, truncated)
#         state = next_state
#         done = terminated or truncated
#         discounted_return += (discount_factor**time_step)*reward
#         time_step += 1
#     print(nr_episode, ":", discounted_return)
#     return discounted_return


def episode(env, agent, nr_episode=0, q_value=None):
    state = env.reset()
    discounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    temp = {}
    while not done:
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, terminated, truncated, _ = env.step(action)
        # 3. Integrate new experience into agent
        # global temp
        if nr_episode == 0:
            temp = agent.update(state, action, reward, next_state, terminated, truncated, q_value)
        else:
            temp = agent.update(state, action, reward, next_state, terminated, truncated)
        state = next_state
        done = terminated or truncated
        discounted_return += (discount_factor**time_step)*reward
        time_step += 1
    print(nr_episode, ":", discounted_return)
    return (temp,discounted_return)
    
params = {}
rooms_instance = sys.argv[1]
env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.99
params["epsilon_decay"] = 0.0001
params["alpha"] = 0.1
params["env"] = env

# agent = a.RandomAgent(params)
# agent = a.SARSALearner(params)
agent = a.QLearner(params)
def train_test_func(no_episodes, agent, test=False, temp=None):
    temp_val = {}
    if test == False:
        returns = list()
        for i in range(no_episodes):
            temp_val, return_temp = episode(env, agent, i)
            returns.append(return_temp)
        return (temp_val, returns) 
    else:
        returns = list()
        for i in range(no_episodes):
            if i == 0:
                temp_val, return_temp = episode(env, agent, i,  temp)
                returns.append(return_temp)
            else:
                temp_val, return_temp = episode(env, agent, i)
                returns.append(return_temp)
        return returns
    

def plot_func(no_episodes, returns):
    x = range(no_episodes)
    y = returns
    plot.plot(x,y)
    plot.title("Progress")
    plot.xlabel("Episode")
    plot.ylabel("Discounted Return")
    plot.show()
    plot.close()

training_episodes = 1000
testing_episodes = 10
temp, returns = train_test_func(training_episodes, agent)
plot_func(training_episodes, returns)
returns_test = train_test_func(testing_episodes, agent, True, temp)

plot_func(testing_episodes, returns_test)



env.save_video()

