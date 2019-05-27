from dqn_torch import train_dqn
from ddqn_torch import train_ddqn
from policy_gradients_torch import train_pg
import matplotlib.pyplot as plt
import numpy as np

dqn_rewards = train_dqn()
ddqn_rewards = train_ddqn()
pg_rewards = train_pg()
print(dqn_rewards, ddqn_rewards, pg_rewards)
#dqn_rewards = [22.14, 26.84, 42.42, 47.76, 50.32, 51.49, 52.0, 52.02, 54.3, 57.73] 
#ddqn_rewards = [16.9, 13.88, 46.43, 49.33, 60.81, 70.14, 75.24, 82.95, 89.01, 91.46] 
#pg_rewards = [24.72, 61.28, 139.54, 168.64, 192.63, 197.34, 195.94, 192.67, 200.0, 199.47]
x = range(dqn_rewards.__len__())
x_label = [100*y for y in x]
plt.plot(x, dqn_rewards, '-r', label = 'DQN')
plt.plot(x, ddqn_rewards, '-g', label = 'DDQN')
plt.plot(x, pg_rewards, '-b', label = 'PolicyGradients')
plt.title('Average Reward over Episodes')
plt.xlabel('Episode Number')
plt.ylabel('Average Reward')
plt.legend(loc = 'upper left')
plt.show()
