import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from collections import namedtuple
import random


# ------------------------------------- #
# 策略网络
# ------------------------------------- #

class PolicyNet(nn.Module):
    def __init__(self, input_size, output_size, action_bound, hidden_size=[128, 128]):
        super(PolicyNet, self).__init__()
        # 环境可以接受的动作最大值
        self.action_bound = action_bound
        self.input_size = input_size
        # 只包含一个隐含层
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], output_size)

    # 前向传播
    def forward(self, x):
        x = x.view(x.size(0), 1, self.input_size)  # 改变输入形状，第一个维度是batch，第二个维度是1，第三个维度是35
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 输出
        x = torch.tanh(x)  # 将数值调整到 [-1,1]
        x = x * self.action_bound  # 缩放到 [-action_bound, action_bound]
        return x


# ------------------------------------- #
# 价值网络
# ------------------------------------- #

class QValueNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=[128, 128]):
        super(QValueNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size + output_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], 1)

    # 前向传播
    def forward(self, x, a):
        x1 = x.view(x.size(0), 1, self.input_size)
        x2= a.view(a.size(0), 1, self.output_size)
        # 拼接状态和动作
        cat = torch.cat([x1, x2], dim=2)  # [b, n_states + n_actions]
        x = self.fc1(cat)  # -->[b, n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # -->[b, n_hiddens]
        x = F.relu(x)
        x = self.fc3(x)  # -->[b, 1]
        return x


# ------------------------------------- #
# 算法主体
# ------------------------------------- #
MODELS_DIR="models"
class DDPG:
    def __init__(self,opt,agent_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent_name = agent_name
        self.opt = opt

        # 策略网络--训练
        self.actor = PolicyNet(self.opt.input_size, self.opt.output_size, self.opt.action_bound,self.opt.p_hidden_size).to(self.device)
        # 价值网络--训练
        self.critic = QValueNet(self.opt.input_size, self.opt.output_size,self.opt.q_hidden_size).to(self.device)
        # 策略网络--目标
        self.target_actor = PolicyNet(self.opt.input_size,self.opt.output_size, self.opt.action_bound,self.opt.p_hidden_size).to(self.device)
        # 价值网络--目标
        self.target_critic = QValueNet(self.opt.input_size, self.opt.output_size,self.opt.q_hidden_size).to(self.device)
        # 初始化价值网络的参数，两个价值网络的参数相同
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化策略网络的参数，两个策略网络的参数相同
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.opt.actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.opt.critic_lr)

        # 属性分配
        self.gamma = self.opt.gamma  # 折扣因子

        self.tau = self.opt.tau  # 目标网络的软更新参数

        self.memory = []  # 经验回放缓冲区
        self.position = 0  # 当前经验存储位置
        self.capacity = self.opt.memory_capacity  # 经验回放缓冲区容量

        # 定义经验存储结构
        self.transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward',"done"))

        if self.opt.train:
            # 保存模型路径
            self.model_save_path = os.path.join(self.opt.save_dir, MODELS_DIR, self.agent_name)
            os.makedirs(self.model_save_path, exist_ok=True)

    # 动作选择
    def choose_action(self, state,e):
        # 维度变换 list[n_states]-->tensor[1,n_states]-->gpu
        x =np.expand_dims(state, axis=0)  # 增加维度以匹配网络输入形状
        # 策略网络计算出当前状态下的动作价值 [1,n_states]-->[1,1]-->int
        action = self.actor(torch.FloatTensor(x).to(self.device)).squeeze(1).cpu().data.numpy()[0]
        # 给动作添加噪声，增加搜索
        action=np.clip(np.random.normal(action, e), -self.opt.action_bound, self.opt.action_bound)
        return action

    def push_memory(self, s, a, r, s_,done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # 如果经验回放缓冲区未满，添加新条目
        self.memory[self.position] = self.transition(torch.unsqueeze(torch.FloatTensor(s), 0).to(self.device),
                                                     torch.unsqueeze(torch.FloatTensor(s_), 0).to(self.device), \
                                                     torch.from_numpy(np.array([a], dtype='float32')).to(self.device),
                                                     torch.from_numpy(np.array([r], dtype='float32')).to(self.device),
                                                     torch.from_numpy(np.array([done],dtype='float32')).to(self.device))  #
        self.position = (self.position + 1) % self.capacity  # 更新经验存储位置

    # 从经验中获取样本的方法
    def get_sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)  # 随机采样一批经验样本
        return sample

    # 软更新, 意思是每次learn的时候更新部分参数
    def soft_update(self, net, target_net):
        # 获取训练网络和目标网络需要更新的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 训练网络的参数更新要综合考虑目标网络和训练网络
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)

    # 训练
    def learn(self):
        transitions = self.get_sample(self.opt.batch_size)  # 从经验回放缓冲区中获取批次样本
        batch = self.transition(*zip(*transitions))  # 解包样本元组

        b_s = Variable(torch.cat(batch.state))  # 转为Tensor并封装为Variable
        b_s_ = Variable(torch.cat(batch.next_state))
        b_a = Variable(torch.cat(batch.action))
        b_r = Variable(torch.cat(batch.reward))
        b_d= Variable(torch.cat(batch.done))

        # 价值目标网络获取下一时刻的动作[b,n_states]-->[b,n_actors]
        next_q_values = self.target_actor(b_s_)
        # 策略目标网络获取下一时刻状态选出的动作价值 [b,n_states+n_actions]-->[b,1]
        next_q_values = self.target_critic(b_s_, next_q_values)
        # 当前时刻的动作价值的目标值 [b,1]
        q_targets = b_r.unsqueeze(1) + self.gamma * next_q_values.squeeze(1) * (1 - b_d).unsqueeze(1)

        # 当前时刻动作价值的预测值 [b,n_states+n_actions]-->[b,1]
        q_values = self.critic(b_s, b_a).squeeze(1)

        # 预测值和目标值之间的均方差损失
        critic_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # 价值网络梯度
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 当前状态的每个动作的价值 [b, n_actions]
        actor_q_values = self.actor(b_s)
        # 当前状态选出的动作价值 [b,1]
        score = self.critic(b_s, actor_q_values)
        # 计算损失
        actor_loss = -torch.mean(score)
        # 策略网络梯度
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新策略网络的参数
        self.soft_update(self.actor, self.target_actor)
        # 软更新价值网络的参数
        self.soft_update(self.critic, self.target_critic)
    def save_model(self):
        torch.save(self.actor.state_dict(), os.path.join(self.model_save_path, "actor.pt"))
        torch.save(self.target_actor.state_dict(), os.path.join(self.model_save_path, "target_actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(self.model_save_path, "critic.pt"))
        torch.save(self.target_critic.state_dict(), os.path.join(self.model_save_path, "target_critic.pt"))


    def load_model(self,model_dir):
        self.actor.load_state_dict(torch.load(os.path.join(model_dir,self.agent_name,"actor.pt")))
        self.target_actor.load_state_dict(torch.load(os.path.join(model_dir,self.agent_name,"target_actor.pt")))
        self.critic.load_state_dict(torch.load(os.path.join(model_dir,self.agent_name,"critic.pt")))
        self.target_critic.load_state_dict(torch.load(os.path.join(model_dir,self.agent_name,"target_critic.pt")))


