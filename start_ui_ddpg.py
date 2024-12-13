# -*- coding:utf-8 -*-
import argparse
from copy import deepcopy
from tkinter import Tk, Button, StringVar, Label, Canvas, W, E, RAISED, Widget, Frame, Entry, IntVar, DoubleVar,BooleanVar
from tkinter.messagebox import showerror

import numpy as np

from deal_ddpg import train_ddpg, test_ddpg


def parse_opt(_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=True, help='True: 训练模式 False: 测试模式 ')
    parser.add_argument('--test_model_dir', type=str, default='runs/ddpg/train/exp1',
                        help='若当前为测试模式，则会导入该路径文件夹中的模型文件')
    parser.add_argument('--episodes', type=int, default=100000, help='游戏循环的场次')
    parser.add_argument('--p_hidden_size', type=list, default=[128, 64], help='策略网络的隐藏层大小')
    parser.add_argument('--q_hidden_size', type=list, default=[128, 64], help='价值网络的隐藏层大小')
    parser.add_argument('--batch_size', type=int, default=128, help='每次从经验池中随机抽取batch_size组数据用来训练')
    parser.add_argument('--save_step', type=int, default=100, help='每隔多少轮保存一次模型')
    parser.add_argument('--actor_lr', type=float, default=0.0001, help='策略网络学习率')
    parser.add_argument('--critic_lr', type=float, default=0.0001, help='价值网络学习率')
    parser.add_argument('--gamma', type=float, default=0.98, help='对未来的看重')
    parser.add_argument('--tau', type=float, default=0.05, help='目标网络的软更新参数')
    parser.add_argument('--memory_capacity', type=int, default=10000, help='经验回放池的大小')
    parser.add_argument('--train_start_memory_capacity', type=int, default=1000, help='经验池多大开始训练')
    parser.add_argument('--action_bound', type=float, default=1.0,
                        help='动作空间最大值，默认[-action_space,action_space]')
    parser.add_argument('--action_space', type=int, default=2, help='动作空间维度')
    parser.add_argument('--save_path', type=str, default='./runs/ddpg', help='运行结果文件夹保存路径')
    parser.add_argument('--desc', type=str, default='', help='进度条描述')
    parser.add_argument('--render_mode', type=int, default=0,
                        help='用于展示画面  0:用于在本地展示 1:用于在ipynp中展示 3:不展示')  # 训练时不推荐展示 会增加时间开销
    parser.add_argument('--exploration_decay', type=float, default=20000, help='随机探索衰减，值越大衰减的越慢')
    parser.add_argument('--min_exploration', type=float, default=0.01, help='最小随机探索概率')
    parser.add_argument('--max_exploration', type=float, default=0.9, help='最大随机探索概率')
    return parser.parse_args(_list)


controlled_vehicle_config_default = {
    "speed": 28,
    "initial_lane_id": 1,
    "spacing": 0.6,
    # =========奖励相关参数============
    "collision_reward": -1,  # 碰撞奖励，与碰撞时收到的奖励相关
    "high_speed_reward": 0.25,  # 全速行驶奖励，根据config["reward_speed_range"]，低速时线性映射到0
    "on_road_reward": 0.01,  # 在路上的奖励，与在路上时收到的奖励相关
    "right_lane_reward": -0.01,  # 右侧车道奖励，其他车道线性映射到0
    "lane_change_reward": -0.01,  # 每次变道奖励
    "head_distance_reward": 0.25,  # 车头间距奖励
    "head_distance_range": [10, 20],  # 车头间距奖励范围 在这个范围内才有车头间距奖励
    "acceleration_reward": 0.1,  # 默认的加速奖励
    "deceleration_reward": -0.1,  # 默认的减速奖励
}
multi_highway_config_ddpg_default = {
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",  # 每个控制车辆的状态信息类型
            "vehicles_count": 10,  # 每辆控制车观测到的最近的车辆数，包括自己
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],  # 每辆车的特征
            "absolute": False,
        },
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "ContinuousAction",
            "acceleration_range": [-5, 5],  # 加速度范围
            "steering_range": [-np.pi / 4, np.pi / 4],  # 转向值范围
            "speed_range": [25, 35],  # 速度范围
            "longitudinal": True,  # 启用油门控制
            "lateral": True  # 启用转弯控制
        },
    },
    "lanes_count": 3,  # 高速公路的车道
    "speed_limit": 35,  # 车道速度限制
    "vehicles_count": 10,  # 非控制车辆数目
    "duration": 300,  # 仿真时长 [s]  不是真实时长
    "vehicles_controlled": 3,  # 控制车辆数
    "reward_speed_range": [30, 35],  # 该速度范围才有速度奖励 超过最大值奖励达到最大
    "distance_between_controlled_vehicles": 15,  # 控制车辆之间的距离阈值m，阈值之内车辆奖励替换为平均奖励
    # 受控制的车辆数目以及配置
    "controlled_vehicles": {
        0: controlled_vehicle_config_default,
        1: controlled_vehicle_config_default,
        2: controlled_vehicle_config_default
    },
    "normalize_reward": False,  # 是否对奖励进行归一化
    "offroad_terminal": True,  # 是否在离开路面时终止仿真
    "simulation_frequency": 10,  # 仿真频率，每秒进行15次仿真步骤 [Hz]
    "policy_frequency": 2,  # 策略更新频率，每秒进行2次策略更新 [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",  # 其他车辆的类型，使用IDM模型
    "screen_width": 600,  # 屏幕宽度，用于图形渲染 [px]
    "screen_height": 150,  # 屏幕高度，用于图形渲染 [px]
    "centering_position": [0.3, 0.5],  # 屏幕中心位置的归一化坐标，x坐标为0.3，y坐标为0.5
    "scaling": 4,  # 屏幕缩放因子，用于图形渲染
    "show_trajectories": False,  # 是否显示车辆轨迹
    "render_agent": True,  # 是否渲染代理车辆
    "real_time_rendering": False  # 是否实时渲染
}

class Main_Windows:
    def __init__(self, vehicles_number, vehicles_count):
        self.root = Tk()
        self.root.title('start-up')  # 程序的标题名称
        self.win_width = 350
        self.win_height = 100
        self.k_width = 20
        self.root.geometry(f"{self.win_width}x{self.win_height}+512+288")  # 窗口的大小及页面的显示位置
        self.root.resizable(False, False)  # 固定页面不可放大缩小
        self.vehicles_number = IntVar()
        self.vehicles_number.set(vehicles_number)
        self.vehicles_count = IntVar()
        self.vehicles_count.set(vehicles_count)
        self.init_weight()

    def init_weight(self):
        try:
            lab1 = Label(self.root, text="Number of HDVs", width=self.k_width).grid(row=1)
            lab2 = Label(self.root, text="Number of ICVs", width=self.k_width).grid(row=2)
            e1 = Entry(self.root, width=self.k_width, textvariable=self.vehicles_number)  # 文本框1
            e2 = Entry(self.root, width=self.k_width, textvariable=self.vehicles_count)  # 文本框2
            e1.grid(row=1, column=1)  # 定位文本框1
            e2.grid(row=2, column=1)  # 定位文本框2
            b1 = Button(self.root, text="Submit", width=self.k_width // 2, command=self.close).place(x=130,y=60)
        except Exception as e:
            print(e)

    def get_var(self):
        return self.vehicles_number.get(), self.vehicles_count.get()

    def close(self):
        try:
            # 防止空值
            self.vehicles_number.get()
            self.vehicles_count.get()
            self.root.destroy()
        except Exception as e:
            showerror(title="错误",
                      message="提交失败,请检查参数!")

    def run(self):
        self.root.mainloop()  # 运行


class ControlledVehicleConfigWindows:
    def __init__(self,controlled_vehicle_config_default,title):
        self.root = Tk()
        self.root.title(title)  # 程序的标题名称
        self.win_width = 750
        self.win_height = 350
        self.k_width = 20
        self.root.geometry(f"{self.win_width}x{self.win_height}+512+288")  # 窗口的大小及页面的显示位置
        self.root.resizable(False, False)  # 固定页面不可放大缩小
        self.controlled_vehicle_config=deepcopy(controlled_vehicle_config_default)
        self.init_var()
        self.init_weight()

    def init_var(self):
        self.label_name_dict = {
            "Initial Speed": ["speed", IntVar(), "Initial speed of the vehicle"],  # 初始速度
            "Initial Lane": ["initial_lane_id", IntVar(), "Initial lane of the vehicle"],
            "Position Spacing": ["spacing", DoubleVar(), "Initial position of the vehicle"],  # 位置
            # =========Reward-related Parameters============
            "Collision Reward": ["collision_reward", DoubleVar(), "Reward for collision"],  # 碰撞奖励，与碰撞时收到的奖励相关
            "Speed Reward": ["high_speed_reward", DoubleVar(), "Maximum reward within specified speed range"],
            # 全速行驶奖励，根据config["reward_speed_range"]，低速时线性映射到0
            "Acceleration Reward": ["acceleration_reward", DoubleVar(), "Positive value, reward for acceleration"],
            "Deceleration Penalty": ["deceleration_reward", DoubleVar(), "Negative value, penalty for deceleration"],
            "On-road Reward": ["on_road_reward", DoubleVar(), "Reward for staying on the road"],  # 在路上的奖励，与在路上时收到的奖励相关
            "Lane Change Reward": ["lane_change_reward", DoubleVar(), "Reward for each lane change"],  # 每次变道奖励
            "Headway Reward": ["head_distance_reward", DoubleVar(), "Reward for headway"],  # 车头间距奖励
            "Minimum Headway": ["head_distance_min", DoubleVar(),
                                      "Headway must be greater than this value to get a reward"],
            # 车头间距奖励范围 在这个范围内才有车头间距奖励
            "Maximum Headway": ["head_distance_max", DoubleVar(),
                                      "Headway must be less than this value to get a reward"],
            # 车头间距奖励范围 在这个范围内才有车头间距奖励
        }

        self.label_names=sorted(self.label_name_dict.keys())

        for k,v in self.label_name_dict.items():
            if v[0]=="head_distance_min":
                v[1].set(self.controlled_vehicle_config["head_distance_range"][0])
            elif v[0]=="head_distance_max":
                v[1].set(self.controlled_vehicle_config["head_distance_range"][1])
            else:
                v[1].set(self.controlled_vehicle_config[v[0]])

    def set_controlled_vehicle_config(self):
        for k,v in self.label_name_dict.items():
            if v[0]=="head_distance_min":
                self.controlled_vehicle_config["head_distance_range"][0]=v[1].get()
            elif v[0]=="head_distance_max":
                self.controlled_vehicle_config["head_distance_range"][1]=v[1].get()
            else:
                self.controlled_vehicle_config[v[0]]=v[1].get()

    def init_weight(self):
        try:
            Label(self.root, text="reward", width=self.k_width).grid(row=0,column=0)
            Label(self.root, text="value", width=self.k_width).grid(row=0,column=1)
            Label(self.root, text="remarks", width=self.k_width).grid(row=0,column=2)
            for idx,label_name in enumerate(self.label_names):
                Label(self.root, text=label_name, width=self.k_width).grid(row=idx+1)
                e= Entry(self.root, width=self.k_width, textvariable=self.label_name_dict[label_name][1])  # 文本框1
                e.grid(row=idx+1, column=1)  # 定位文本框2
                Label(self.root, text=f"({self.label_name_dict[label_name][2]})", width=self.k_width * 3).grid(row=idx+1,column=2)
            b1 = Button(self.root, text="submit", width=self.k_width//2, command=self.close)
            b1.grid(row=idx+2, column=1)
        except Exception as e:
            print(e)

    def close(self):
        try:
            self.set_controlled_vehicle_config()
            self.root.destroy()
        except Exception as e:
            showerror(title="错误",
                      message="提交失败,请检查参数!")

    def run(self):
        self.root.mainloop()  # 运行

class TrainConfigWindows:
    def __init__(self,opt):
        self.root = Tk()
        self.root.title("Training parameter settings")  # 程序的标题名称
        self.win_width = 1000
        self.win_height = 550
        self.k_width = 30
        self.root.geometry(f"{self.win_width}x{self.win_height}+512+288")  # 窗口的大小及页面的显示位置
        self.root.resizable(False, False)  # 固定页面不可放大缩小
        self.opt=opt

        self.init_var()
        self.init_weight()

    def init_var(self):

        self.label_name_dict = {
            "train": [IntVar(), "Non-zero: training mode; 0: testing mode"],
            "test_model_dir": [StringVar(), "If in testing mode, load models from the specified directory"],
            "episodes": [IntVar(), "Number of game episodes"],
            "p_hidden_size": [StringVar(), "Hidden layer size of the policy network (two layers only)"],
            "q_hidden_size": [StringVar(), "Hidden layer size of the value network (two layers only)"],
            "batch_size": [IntVar(), "Number of samples drawn from the replay buffer for each training"],
            "save_step": [IntVar(), "Save the model every specified number of episodes"],
            "actor_lr": [DoubleVar(), "Learning rate of the policy network"],
            "critic_lr": [DoubleVar(), "Learning rate of the value network"],
            "gamma": [DoubleVar(), "Discount factor for future rewards"],
            "tau": [DoubleVar(), "Soft update parameter for the target network"],
            "memory_capacity": [IntVar(), "Size of the replay buffer"],
            "train_start_memory_capacity": [IntVar(), "Minimum replay buffer size before training starts"],
            "action_bound": [DoubleVar(), "Maximum action magnitude forming [-action_space, action_space]"],
            "action_space": [IntVar(), "Dimension of the action space"],
            "save_path": [StringVar(), "Directory to save training results"],
            "desc": [StringVar(), "Description for the progress bar"],
            "render_mode": [IntVar(), "Mode for rendering: 0 for local, 1 for IPYNB, 3 for no display"],
            "exploration_decay": [IntVar(), "Exploration decay factor (larger value means slower decay)"],
            "min_exploration": [DoubleVar(), "Minimum exploration probability"],
            "max_exploration": [DoubleVar(), "Maximum exploration probability"],
        }

        # 初始化值
        for k,v in self.label_name_dict.items():
            if k=="p_hidden_size":
                v[0].set(str(getattr(self.opt,"p_hidden_size")))
            elif k=="q_hidden_size":
                v[0].set(str(getattr(self.opt,"q_hidden_size")))
            else:
                v[0].set(getattr(self.opt,k))

    def get_list_from_string(self,string):
        return [int(i) for i in string[1:-1].split(",")]

    def set_config(self):
        for k,v in self.label_name_dict.items():
            if k=="p_hidden_size":
                setattr(self.opt,k,self.get_list_from_string(v[0].get()))
            elif k=="q_hidden_size":
                setattr(self.opt,k,self.get_list_from_string(v[0].get()))
            else:
                if isinstance(v,StringVar):
                    arg=v.get()
                    if arg.split()=="":
                        if k in ["q_hidden_size","p_hidden_size"]:
                            raise ValueError(f"{k}参数不能为空")
                setattr(self.opt, k, v[0].get())
        if self.opt.train==0 and self.opt.test_model_dir=="":
            raise ValueError("测试模式下必须输入模型路径")

    def init_weight(self):
        try:
            Label(self.root, text="parameter name", width=self.k_width).grid(row=0,column=0)
            Label(self.root, text="value", width=self.k_width).grid(row=0,column=1)
            Label(self.root, text="remarks", width=self.k_width).grid(row=0,column=2)
            for idx,label_name in enumerate(self.label_name_dict.keys()):
                Label(self.root, text=label_name, width=self.k_width).grid(row=idx+1)
                e= Entry(self.root, width=int(self.k_width*0.8), textvariable=self.label_name_dict[label_name][0])  # 文本框1
                e.grid(row=idx+1, column=1)  # 定位文本框2
                Label(self.root, text=f"({self.label_name_dict[label_name][1]})", width=int(self.k_width*2)).grid(row=idx+1,column=2)
            b1 = Button(self.root, text="submit", width=self.k_width//2, command=self.close)
            b1.grid(row=idx+2, column=1)
        except Exception as e:
            print(e)

    def close(self):
        try:
            self.set_config()
            self.root.destroy()
        except Exception as e:
            showerror(title="错误",
                      message=f"提交失败,请检查参数!{e}")

    def run(self):
        self.root.mainloop()  # 运行


def get_controlled_vehicle_config(controlled_vehicle_config_default,title):
    win=ControlledVehicleConfigWindows(controlled_vehicle_config_default,title)
    win.run()
    return win.controlled_vehicle_config



def main():
    multi_highway_config_ddpg = deepcopy(multi_highway_config_ddpg_default)
    win = Main_Windows(multi_highway_config_ddpg["vehicles_count"],
                       len(multi_highway_config_ddpg["controlled_vehicles"]))
    win.run()
    # 获取车辆密度和控制车辆数量
    vehicles_count, controlled_vehicles_number = win.get_var()
    multi_highway_config_ddpg["vehicles_count"]=vehicles_count

    controlled_vehicles={}
    controlled_vehicle_config_d=deepcopy(controlled_vehicle_config_default)
    for idx in range(controlled_vehicles_number):
        controlled_vehicle_config=get_controlled_vehicle_config(controlled_vehicle_config_d,f"车辆{idx+1}奖励设置-共{controlled_vehicles_number}辆")
        controlled_vehicles[idx]=controlled_vehicle_config

    # 控制车辆奖励设置
    multi_highway_config_ddpg["controlled_vehicles"]=controlled_vehicles

    opt = parse_opt()
    # 训练参数奖励设置
    train_win=TrainConfigWindows(opt)
    train_win.run()

    opt=train_win.opt


    print(opt)
    print(multi_highway_config_ddpg)
    #
    if opt.train:
        train_ddpg(opt, multi_highway_config_ddpg)
    else:
        test_ddpg(opt)



if __name__ == '__main__':
    main()