# 命令行获取参数
import argparse

import numpy as np

from deal_ddpg import train_ddpg, test_ddpg


def parse_opt(_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False, help='True: 训练模式 False: 测试模式 ')
    parser.add_argument('--test_model_dir', type=str, default='runs/ddpg/train/exp1',
                        help='若当前为测试模式，则会导入该路径文件夹中的模型文件')
    parser.add_argument('--episodes', type=int, default=100, help='游戏循环的场次')
    parser.add_argument('--p_hidden_size', type=list, default=[128, 64], help='策略网络的隐藏层大小')
    parser.add_argument('--q_hidden_size', type=list, default=[128, 64], help='价值网络的隐藏层大小')
    parser.add_argument('--batch_size', type=int, default=4, help='每次从经验池中随机抽取batch_size组数据用来训练')
    parser.add_argument('--save_step', type=int, default=20, help='每隔多少轮保存一次模型')
    parser.add_argument('--actor_lr', type=float, default=0.0001, help='策略网络学习率')
    parser.add_argument('--critic_lr', type=float, default=0.0001, help='价值网络学习率')
    parser.add_argument('--gamma', type=float, default=0.98, help='对未来的看重')
    parser.add_argument('--tau', type=float, default=0.05, help='目标网络的软更新参数')
    parser.add_argument('--memory_capacity', type=int, default=5000, help='经验回放池的大小')
    parser.add_argument('--train_start_memory_capacity', type=int, default=10, help='经验池多大开始训练')
    parser.add_argument('--action_bound', type=float, default=1.0, help='动作空间最大值，默认[-action_space,action_space]')
    parser.add_argument('--action_space', type=int, default=2, help='动作空间维度')
    parser.add_argument('--save_path', type=str, default='./runs/ddpg', help='运行结果文件夹保存路径')
    parser.add_argument('--desc', type=str, default='', help='进度条描述')
    parser.add_argument('--render_mode', type=int, default=0,
                        help='用于展示画面  0:用于在本地展示 1:用于在ipynp中展示 3:不展示')  # 训练时不推荐展示 会增加时间开销
    parser.add_argument('--exploration_decay', type=float, default=20000, help='随机探索衰减，值越大衰减的越慢')
    parser.add_argument('--min_exploration', type=float, default=0.01, help='最小随机探索概率')
    parser.add_argument('--max_exploration', type=float, default=0.9, help='最大随机探索概率')
    return parser.parse_args(_list)

multi_highway_config_ddpg = {
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",  # 每个控制车辆的状态信息类型
            "vehicles_count": 10,  # 每辆控制车观测到的最近的车辆数，包括自己
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"], #每辆车的特征
            "absolute": False,
        },
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "ContinuousAction",
            "acceleration_range": [-5, 5],  # 加速度范围
            "steering_range": [-np.pi / 4, np.pi / 4],  # 转向值范围
            "speed_range": [20, 60],  # 速度范围
            "longitudinal": True,  # 启用油门控制
            "lateral": True  # 启用转弯控制
        },
    },
    "lanes_count": 3,  # 高速公路的车道
    "speed_limit": 60,  # 车道速度限制
    "vehicles_count": 30,  # 非控制车辆数目
    "duration": 300,  # 仿真时长 [s]  不是真实时长
    "vehicles_density": 2,  # 车辆密度
    "reward_speed_range": [40, 60],  # 该速度范围才有速度奖励 超过最大值奖励达到最大
    "distance_between_controlled_vehicles": 10,  # 控制车辆之间的距离阈值m，阈值之内车辆奖励替换为平均奖励
    # 受控制的车辆数目以及配置
    "controlled_vehicles": {
        0: {
            "speed": 20,  # 初始速度
            "initial_lane_id": 0,
            "spacing": 0.36,  # 位置
            # =========奖励相关参数============
            "collision_reward": -1,  # 碰撞奖励，与碰撞时收到的奖励相关
            "high_speed_reward": 0.4,  # 全速行驶奖励，根据config["reward_speed_range"]，低速时线性映射到0
            "on_road_reward": 0.1,  # 在路上的奖励，与在路上时收到的奖励相关
            "right_lane_reward": 0,  # 右侧车道奖励，其他车道线性映射到0
            "lane_change_reward": 0,  # 每次变道奖励
            "head_distance_reward": 0.5,  # 车头间距奖励
            "head_distance_range": [7, 9],  # 车头间距奖励范围 在这个范围内才有车头间距奖励
        },
        1: {
            "speed": 20,  # 初始速度
            "initial_lane_id": 1,
            "spacing": 0.30,  # 位置
            # =========奖励相关参数============
            "collision_reward": -1,  # 碰撞奖励，与碰撞时收到的奖励相关
            "high_speed_reward": 0.4,  # 全速行驶奖励，根据config["reward_speed_range"]，低速时线性映射到0
            "on_road_reward": 0.1,  # 在路上的奖励，与在路上时收到的奖励相关
            "right_lane_reward": 0,  # 右侧车道奖励，其他车道线性映射到0
            "lane_change_reward": 0,  # 每次变道奖励
            "head_distance_reward": 0.5,  # 车头间距奖励
            "head_distance_range": [7, 9],  # 车头间距奖励范围 在这个范围内才有车头间距奖励
        }
    },
    "normalize_reward": True,  # 是否对奖励进行归一化
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

if __name__ == '__main__':
    opt = parse_opt()
    print(type(opt))
    print(opt.__dict__)
    # if opt.train:
    #     train_ddpg(opt, multi_highway_config_ddpg)
    # else:
    #     test_ddpg(opt)
