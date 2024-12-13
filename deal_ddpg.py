import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from IPython import display
from matplotlib import pyplot as plt
from tqdm import tqdm

from ddpg import DDPG,MODELS_DIR
from multi_highway_env2 import Multi_Highway_Env2
from tools import get_log_path, logger_setting, print_args_ddpg, save_config, load_config, PlotUtil

def compute_avg_head_distace(head_distance_list):
    sum_data=0
    num=0
    for d in head_distance_list:
        if d is not None:
            sum_data+=d
            num+=1
    if num ==0:
        return None
    return sum_data/num


def train_ddpg(opt, multi_highway_config_ddpg):
    # opt增加input_size,output_size参数
    opt.input_size = len(multi_highway_config_ddpg["observation"]["observation_config"]["features"]) * \
                     multi_highway_config_ddpg["observation"]["observation_config"]["vehicles_count"]
    opt.output_size = opt.action_space
    # 生成不同的文件夹 run/exp1 run/exp2 ...
    save_dir = get_log_path(os.path.join(opt.save_path, "train"))
    # opt增加保存文件夹参数
    opt.save_dir = save_dir
    # opt 根据环境配置增加控制车辆数参数
    opt.num_agent = len(multi_highway_config_ddpg['controlled_vehicles'])
    # 日志记录初始化
    logger = logger_setting(save_dir)
    # 打印参数
    print_args_ddpg(logger, opt, multi_highway_config_ddpg)
    # 保存配置至 models目录
    save_config(os.path.join(save_dir, MODELS_DIR), opt, multi_highway_config_ddpg)
    # 游戏配置初始化游戏环境
    env = Multi_Highway_Env2(multi_highway_config_ddpg)
    count = 0  # 记录训练次数
    each_avg_reward = {}  # 多辆车每轮游戏的平均奖励
    each_total_reward = {}  # 多辆车每轮游戏的奖励总和
    each_avg_head_distance={} # 多辆车每轮游戏的平均车头距离
    total_total_rewards = []  # 每轮多辆车奖励总和的总和
    avg_total_rewards = []  # 每轮多辆车奖励总和的平均值
    total_avg_rewards = []  # 每轮多辆车平均奖励的总和
    avg_avg_rewards = []  # 每轮多辆车平均奖励的平均值
    avg_speeds = []  # 每轮中每个车辆的平均速度
    crasheds = []  # 每轮车辆的碰撞
    crashed_num = 0
    steps = []  # 每轮游戏的总轮数
    ep_list = []  # 游戏的轮数
    plot_process = PlotUtil()  # 画图进程
    # 初始化每辆车的奖励
    for i in range(opt.num_agent):
        each_avg_reward[i] = []
        each_total_reward[i] = []
        each_avg_head_distance[i] = []
        avg_speeds.append([])
    agent_list = [DDPG(opt, str(i)) for i in range(opt.num_agent)]  # 创建DDPG智能体
    if opt.render_mode == 1:
        img = plt.imshow(env.render(mode='rgb_array'))  # 仅仅用于ipynp展示游戏场面
    # 游戏开始前先填充经验池
    logger.info("填充经验池")
    while len(agent_list[0].memory) < opt.train_start_memory_capacity:  # 经验池未满
        print(len(agent_list[0].memory), "/", opt.train_start_memory_capacity)
        s = env.reset()[0]
        truncated = False  # 是否截断
        done = False  # 是否结束
        while not truncated and not done:
            a_list = []  # 每辆车的动作
            for agent_id, agent in enumerate(agent_list):  # 每辆车选择动作
                a = agent.choose_action(s[agent_id], 0.5)  # 选择动作
                a_list.append(a)  # 记录动作 is_random:是否随机选择
                # 与环境交互并观察奖励、新状态
            s_, r, done, truncated, info = env.step(tuple(a_list))  # 与环境交互
            # 展示游戏画面
            if opt.render_mode == 0:
                env.render()
            elif opt.render_mode == 1:  # ipynp展示游戏场面
                img.set_data(env.render(mode='rgb_array'))  # just update the data
                display.display(plt.gcf())
                display.clear_output(wait=True)
            for agent_id, agent in enumerate(agent_list):
                agent.push_memory(s[agent_id], a_list[agent_id], r[agent_id], s_[agent_id],
                                  float(done or truncated))  # 存储经验
            s = s_  # 更新状态
    logger.info("经验池填充完成，开始训练！")
    setbar = tqdm(range(opt.episodes))  # 进度条
    for ep_num in setbar:  # 循环训练 显示训练进度条
        setbar.set_description(f"Episode: {ep_num + 1}", refresh=True)  # 进度条描述
        s = env.reset()[0]  # 重置环境
        total_reward = {}  # 每轮游戏的每辆车的每步奖励
        _speeds = []
        head_distance = {} # 每轮游戏的每辆车的每步车头间距
        # 初始化每辆车的奖励
        for i in range(opt.num_agent):
            total_reward[i] = []
            head_distance[i] = []
            _speeds.append([])
        done = False  # 是否结束
        truncated = False  # 是否截断
        step_num = 0  # 游戏步数
        while not done and not truncated:
            e = min(opt.max_exploration, np.exp(-count / opt.exploration_decay) + opt.min_exploration)  # 减小探索率
            # 每个智能体选择动作
            a_list = []
            for agent_id, agent in enumerate(agent_list):
                a = agent.choose_action(s[agent_id], e)  # 选择动作
                a_list.append(a)  # 记录动作
            # 与环境交互并观察奖励、新状态
            s_, r, done, truncated, info = env.step(tuple(a_list))  # 与环境交互
            # 展示游戏画面
            if opt.render_mode == 0:
                env.render()
            elif opt.render_mode == 1:
                # 测试的时候试试展示游戏场面
                img.set_data(env.render(mode='rgb_array'))  # just update the data
                display.display(plt.gcf())
                display.clear_output(wait=True)
            postfix = ""  # 进度条描述
            step_num += 1  # 游戏步数加1
            # 存储经验
            for agent_id, agent in enumerate(agent_list):
                agent.push_memory(s[agent_id], a_list[agent_id], r[agent_id], s_[agent_id], float(done or truncated))
                total_reward[agent_id].append(r[agent_id])  # 记录每辆车每步的奖励
                _speeds[agent_id].append(info["speed"][agent_id])  # 记录每辆车的速度
                head_distance[agent_id].append(info["head_distance"][agent_id])
                count += 1  # 记录训练次数
                agent.learn()  # 学习更新
                postfix += f" {agent_id}:" + "{ " + f"a={a_list[agent_id]}, r={r[agent_id]}" + " }"
            setbar.set_postfix(step_num=step_num, e=e, info=postfix)  # 进度条描述
            s = s_  # 更新状态
        if done:
            crashed_num += 1

        # 一轮游戏结束后多辆车各自保存平均奖励和奖励总和
        for agent_id, agent in enumerate(agent_list):
            each_avg_reward[agent_id].append(np.mean(total_reward[agent_id]))  # 记录每辆车每轮游戏的平均奖励
            each_total_reward[agent_id].append(np.sum(total_reward[agent_id]))  # 记录每每辆车轮游戏的奖励总和
            avg_speeds[agent_id].append(np.mean(_speeds[agent_id]))  # 记录每辆车每轮的平均速度
            each_avg_head_distance[agent_id].append(compute_avg_head_distace(head_distance[agent_id]))
            agent.save_model()  # 保存模型

        crasheds.append(crashed_num)  # done表示碰撞结束的游戏
        # 一把游戏总轮数
        steps.append(step_num)
        ep_list.append(ep_num)
        # 一轮游戏结束后多辆车平均奖励的平均值和总和
        avg_avg_rewards.append(np.mean([each_avg_reward[agent_id][-1] for agent_id in range(opt.num_agent)]))
        total_avg_rewards.append(np.sum([each_avg_reward[agent_id][-1] for agent_id in range(opt.num_agent)]))
        # 一轮游戏结束后多辆车奖励总和的平均值和总和
        avg_total_rewards.append(np.mean([each_total_reward[agent_id][-1] for agent_id in range(opt.num_agent)]))
        total_total_rewards.append(np.sum([each_total_reward[agent_id][-1] for agent_id in range(opt.num_agent)]))
        # 每隔一定轮次保存模型并更新记录
        if ep_num % opt.save_step == 0:
            # 画图并保存cvs
            plot_eval(plot_process, save_dir, agent_list, ep_list.copy(), deepcopy(each_avg_reward),
                      deepcopy(each_total_reward), deepcopy(avg_speeds),
                      deepcopy(crasheds), deepcopy(steps), deepcopy(avg_avg_rewards), deepcopy(total_avg_rewards),
                      deepcopy(avg_total_rewards), deepcopy(total_total_rewards),deepcopy(each_avg_head_distance))




    # 最终画图
    plot_eval(plot_process, save_dir, agent_list, ep_list.copy(), deepcopy(each_avg_reward),
              deepcopy(each_total_reward), deepcopy(avg_speeds),
              deepcopy(crasheds), deepcopy(steps), deepcopy(avg_avg_rewards), deepcopy(total_avg_rewards),
              deepcopy(avg_total_rewards), deepcopy(total_total_rewards),deepcopy(each_avg_head_distance))
    logger.info("训练结束！")
    # 堆积过多则等待画图进程处理
    n = plot_process.get_queue_task_number()
    while n > 0:
        logger.info(f"正在完成剩余绘图任务, 数量：{n}")
        time.sleep(3)
        n = plot_process.get_queue_task_number()
    logger.info("绘制完成！")
    # 关闭画图进程
    plot_process.close()


def plot_eval(plot_process, save_dir, agent_list, ep_list, each_avg_reward, each_total_reward, avg_speeds, crasheds,
              steps, avg_avg_rewards, total_avg_rewards, avg_total_rewards, total_total_rewards, each_avg_head_distance):
    # 画图并保存csv
    for agent_id, agent in enumerate(agent_list):
        plot_process.push_task("each_avg_reward", "episodes", "avg_reward", ep_list, each_avg_reward[agent_id],
                               os.path.join(save_dir, f"each_avg_reward_{agent_id}.png"))
        plot_process.push_task("each_avg_head_distance", "episodes", "avg_head_distance", ep_list, each_avg_head_distance[agent_id],
                               os.path.join(save_dir, f"each_avg_head_distance_{agent_id}.png"))
        plot_process.push_task("each_total_reward", "episodes", "total_reward", ep_list,
                               each_total_reward[agent_id],
                               os.path.join(save_dir, f"each_total_reward_{agent_id}.png"))
        plot_process.push_task("each_avg_speed", "episodes", "avg_speed", ep_list,
                               avg_speeds[agent_id],
                               os.path.join(save_dir, f"each_avg_speed_{agent_id}.png"))
        data = {'avg_reward': each_avg_reward[agent_id],
                'total_reward': each_total_reward[agent_id],
                'avg_speed': avg_speeds[agent_id],
                'avg_head_distance':each_avg_head_distance[agent_id],
                }
        df = pd.DataFrame(data)
        df.to_excel(os.path.join(save_dir, f"agent_{agent_id}_train_data.xlsx"))
    # 保存游戏碰撞
    plot_process.push_task("crashed", "episodes", "crashed", ep_list, crasheds,
                           os.path.join(save_dir, f"crashed.png"))
    # 保存游戏轮数
    plot_process.push_task("steps", "episodes", "steps", ep_list, steps,
                           os.path.join(save_dir, f"steps.png"))
    # 保存多辆车平均奖励的平均值和总和
    plot_process.push_task("avg_avg_rewards", "episodes", "avg_avg_reward", ep_list, avg_avg_rewards,
                           os.path.join(save_dir, f"avg_avg_rewards.png"))
    plot_process.push_task("total_avg_rewards", "episodes", "total_avg_reward", ep_list, total_avg_rewards,
                           os.path.join(save_dir, f"total_avg_rewards.png"))
    # 保存多辆车奖励总和的平均值和奖励总和的总和
    plot_process.push_task("avg_total_rewards", "episodes", "avg_total_reward", ep_list, avg_total_rewards,
                           os.path.join(save_dir, f"avg_total_rewards.png"))
    plot_process.push_task("total_total_rewards", "episodes", "total_total_reward", ep_list, total_total_rewards,
                           os.path.join(save_dir, f"total_total_rewards.png"))

def test_ddpg(opt):
    save_dir = get_log_path(os.path.join(opt.save_path, "test"))  # 生成不同的文件夹 run/exp1 run/exp2 ...
    model_dir = os.path.join(opt.test_model_dir, MODELS_DIR)  # 训练好的模型文件夹
    opt_dict, multi_highway_config_ddpg = load_config(model_dir)  # 加载配置
    # opt增加input_size,output_size参数
    opt.input_size = opt_dict["input_size"]
    opt.output_size = opt_dict["output_size"]
    opt.num_agent = opt_dict["num_agent"]  # 智能体个数
    opt.save_dir = save_dir  # opt增加保存文件夹参数
    logger = logger_setting(save_dir)  # 日志记录初始化
    print_args_ddpg(logger, opt, multi_highway_config_ddpg)  # 打印参数
    env = Multi_Highway_Env2(multi_highway_config_ddpg)  # 游戏配置初始化游戏环境
    agent_list = [DDPG(opt, str(i)) for i in range(opt.num_agent)]  # 创建DDPG智能体
    # 加载模型
    for agent in agent_list:
        agent.load_model(model_dir)
    e = 0
    setbar = tqdm(range(opt.episodes))
    for ep_num in setbar:  # 循环训练 显示训练进度条
        setbar.set_description(f"Episode: {ep_num + 1}", refresh=True)
        s = env.reset()[0]  # 重置环境
        done = False
        truncated = False
        step_num = 0
        while not done and not truncated:
            # 每个智能体选择动作
            a_list = []
            for agent_id, agent in enumerate(agent_list):
                a= agent.choose_action(s[agent_id],e)
                a_list.append(a)
            # 与环境交互并观察奖励、新状态
            s_, r, done, truncated, info = env.step(tuple(a_list))
            # 展示游戏画面
            if opt.render_mode == 0:
                env.render()
            elif opt.render_mode == 1:
                # 测试的时候试试展示游戏场面
                plt.figure(3)
                plt.clf()
                plt.imshow(env.render(mode='rgb_array'))
                plt.title("Episode: %d" % (ep_num))
                plt.axis('off')
                display.clear_output(wait=True)
                display.display(plt.gcf())
            postfix = ""
            step_num += 1
            for agent_id, agent in enumerate(agent_list):
                postfix += f" {agent_id}:" + "{ " + f"a={a_list[agent_id]}, r={r[agent_id]}" + " }"
            setbar.set_postfix(step_num=step_num, info=postfix)
