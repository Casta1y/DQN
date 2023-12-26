import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy

# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100

env = gym.make("CartPole-v0")
env = env.unwrapped
NUM_ACTIONS = env.action_space.n # 2
NUM_STATES = env.observation_space.shape[0] # 4
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape # 0，用来检查动作空间是否是离散的，离散则为0，连续则设置为样本的形状

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob # 输出动作概率

class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy() # max返回(最大值，下标)
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action


    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict()) # 一定步数更新目标网络
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE) # 2000范围内随机取样128个
        # memory里存了很多种state下采取某个action获得reward以及下一个state
        batch_memory = self.memory[sample_index, :]
        # np.hstack((state, [action, reward], next_state))
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2]) # 每一步采取某个action后的reward是固定的
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action) # 收集采取每一次的动作概率，即每个动作对应的q值
        q_next = self.target_net(batch_next_state).detach() # 取得下一个状态下采取每个动作的qvalue
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1) # 目标q值：每个样本的即时奖励与对未来最大 Q 值的折扣预期得结合。（目标 Q 值是根据贝尔曼方程（Bellman Equation）计算的，表示在当前状态下采取某个动作后，未来能够获得的累计折扣奖励的期望值。）
        loss = self.loss_func(q_eval, q_target) # 目标是去学习一个完善的qvalue函数，以更准确地估计状态-动作对的价值，从而实现策略的优化。

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def reward_func(env, x, x_dot, theta, theta_dot): # 小车位置，小车速度，杆子角度，杆子角速度
    r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5 # thres默认2.4，x在正负2.4范围内，这里是为了让小车位置尽量在正负1.2之间？
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5 # theta_threshold_radians默认是12度，弧度下约0.209
    reward = r1 + r2 # 小车越中心，r1越大，可接受的偏离角度越大即r2范围可更大，越边上杆子要求越竖直
    return reward # 小车越中心，杆子越竖直，奖励越大，[0, 1]

def main():
    dqn = DQN()
    episodes = 400
    print("Collecting Experience....")
    reward_list = []
    plt.ion()
    fig, ax = plt.subplots()
    for i in range(episodes):
        state = env.reset()
        ep_reward = 0
        while True:
            env.render()
            action = dqn.choose_action(state)
            next_state, _ , done, info = env.step(action)
            x, x_dot, theta, theta_dot = next_state # 小车位置，小车速度，杆子角度，杆子角速度
            reward = reward_func(env, x, x_dot, theta, theta_dot)

            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                break
            state = next_state
        r = copy.copy(reward)
        reward_list.append(r)
        ax.set_xlim(0,300)
        #ax.cla()
        ax.plot(reward_list, 'g-', label='total_loss')
        plt.pause(0.001)
    env.close()
        

if __name__ == '__main__':
    main()
