import random
from typing import Optional, TypeVar

import numpy as np
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env import utils

Observation = TypeVar("Observation")
default_config = {
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",  # 每个控制车辆的状态信息类型
            "vehicles_count": 12,  # 观测到的车辆最大数目
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
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
    "speed_limit": 40,  # 车道速度限制
    "vehicles_count": 10,  # 非控制车辆数目
    "duration": 30,  # 仿真时长 [s]  不是真实时长
    "reward_speed_range": [20, 40],  # 该速度范围才有速度奖励 超过最大值奖励达到最大
    "vehicles_controlled": 3,  # 控制车辆数
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
            "acceleration_reward": 0.1,  # 加速奖励
            "deceleration_reward": -0.1,  # 减速奖励

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
            "acceleration_reward": 0.1,  # 加速奖励
            "deceleration_reward": -0.1,  # 减速奖励

        }
    },
    "normalize_reward": True,  # 是否对奖励进行归一化
    "offroad_terminal": True,  # 是否在离开路面时终止仿真
    "simulation_frequency": 15,  # 仿真频率，每秒进行15次仿真步骤 [Hz]
    "policy_frequency": 1,  # 策略更新频率，每秒进行1次策略更新 [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",  # 其他车辆的类型，使用IDM模型
    "screen_width": 600,  # 屏幕宽度，用于图形渲染 [px]
    "screen_height": 150,  # 屏幕高度，用于图形渲染 [px]
    "centering_position": [0.3, 0.5],  # 屏幕中心位置的归一化坐标，x坐标为0.3，y坐标为0.5
    "scaling": 4,  # 屏幕缩放因子，用于图形渲染
    "show_trajectories": False,  # 是否显示车辆轨迹
    "render_agent": True,  # 是否渲染代理车辆
    "real_time_rendering": False  # 是否实时渲染
}


def get_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


class Multi_Highway_Env2(AbstractEnv):
    def __init__(self, config: dict = None, render_mode: Optional[str] = None):
        super().__init__(config, render_mode)
        self.time_step_length = 1 / self.config["simulation_frequency"]  # 假设时间步长是仿真频率的倒数
        self.initial_speeds = {}  # 新增：存储每辆控制车辆的初始速度
        self.previous_speeds = None  # 添加属性以存储受控车辆的上一步速度

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        return config

    def _reset(self):
        self._create_road()
        self._create_vehicles()
        self.previous_speeds = [v.speed for v in self.controlled_vehicles]
        # 初始化每辆车的初始速度，现在使用正确的键，假设键是根据车辆的顺序来设定的
        self.initial_speeds = {str(i): cfg["speed"] for i, cfg in enumerate(self.config["controlled_vehicles"].values())}
        self.time = 0

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"],
                                                                   speed_limit=self.config.get("speed_limit", 60)),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    # 创建车辆
    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        others = self.config["vehicles_count"]
        forward_num = int(others / 3)
        self.controlled_vehicles = []
        for _ in range(forward_num):
            vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_controlled"])
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)
        for id, cfg in self.config["controlled_vehicles"].items():
            vehicle = Vehicle.create_random(
                self.road,
                speed=cfg["speed"],
                lane_id=cfg["initial_lane_id"],
                spacing=cfg["spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)
        for _ in range(others - forward_num):
            vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_controlled"])
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

    def get_multi_avg_speed(self, vehicle_id, _rewards):
        '''
        获得受控制车辆和附近受控制车辆的vehicle_id的平均速度
        :param vehicle_id: 当前受控车辆id
        :param _rewards: 所有受控车辆的奖励信息
        :return: 受控车辆和附近受控车辆的vehicle_id的平均速度
        '''
        sum_speed = [_rewards[vehicle_id].get("high_speed_reward", 0)]
        for idx, reward in _rewards.items():
            if idx == vehicle_id:
                continue
            # 如果距离小于阈值，将速度加入到sum_speed中
            if self.config.get("distance_between_controlled_vehicles", 1000) > get_distance(
                    self.controlled_vehicles[vehicle_id].position, self.controlled_vehicles[idx].position):
                sum_speed.append(reward.get("high_speed_reward", 0))
        return sum(sum_speed) / len(sum_speed)

    def _reward(self, action: Action):
        multi_rewards = {}
        _rewards = self._rewards(action)
        current_speeds = [vehicle.speed for vehicle in self.controlled_vehicles]  # 获取当前速度

        for vehicle_id, rewards in _rewards.items():
            vehicle_config = self.config["controlled_vehicles"][vehicle_id]
            initial_speed = vehicle_config["speed"]  # 从配置中获取初始速度
            _reward = []

            # 计算加速度
            if self.time == 0:  # 对于第一个时间步，使用初始速度进行计算
                acceleration = (current_speeds[vehicle_id] - initial_speed) / self.time_step_length
            else:  # 对于后续时间步，使用前一个时间步的速度进行计算
                acceleration = (current_speeds[vehicle_id] - self.previous_speeds[vehicle_id]) / self.time_step_length

            # 根据加速度调整奖励值
            acceleration_reward = 0
            if acceleration > 0:
                acceleration_reward = vehicle_config["acceleration_reward"]
            elif acceleration < 0:
                acceleration_reward = vehicle_config["deceleration_reward"]

            # 计算其他奖励
            for name, reward in rewards.items():
                if name == "high_speed_reward":
                    # 如果根据阈值和速度获得高速奖励 若附近阈值距离内有车辆则取平均速度
                    reward = self.get_multi_avg_speed(vehicle_id, _rewards)
                _reward.append(vehicle_config.get(name, 0) * reward)

            # 加入加速奖励
            _reward.append(acceleration_reward)

            # 计算总奖励
            reward = sum(_reward)
            if self.config["normalize_reward"]:
                reward = utils.lmap(reward,
                                    [vehicle_config.get("collision_reward", 0),
                                     vehicle_config.get("high_speed_reward", 0) + vehicle_config.get(
                                         "right_lane_reward", 0) + vehicle_config.get(
                                         "head_distance_reward", 0) + vehicle_config.get(
                                         "on_road_reward", 0) + max(acceleration_reward, abs(acceleration_reward))],
                                    [0, 1])
            reward *= rewards.get('on_road_reward', 1)
            multi_rewards[vehicle_id] = reward

        # 更新上一步的速度为当前速度，准备下一时间步的计算
        self.previous_speeds = current_speeds

        return multi_rewards

    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "speed": [vehicle.speed for vehicle in self.controlled_vehicles],
            "crashed": [vehicle.crashed for vehicle in self.controlled_vehicles],
            "action": action,
            "head_distance":[self.get_head_distance(vehicle_id) for vehicle_id in range(len(self.controlled_vehicles))]
        }
        try:
            info["rewards"] = self._rewards(action)
        except NotImplementedError:
            pass
        return info

    def get_head_distance(self, control_vehicle_id):
        '''
        控制车辆同车道前方是否有车 有的话获得间距
        :return:
        '''
        control_vehicle=self.controlled_vehicles[control_vehicle_id]
        head_distance = float("inf")  # 前方车距
        control_lane_id = control_vehicle.lane_index[2]
        for idx,vehicle in enumerate(self.controlled_vehicles):
            if idx == control_vehicle_id:
                continue
            lane_id = vehicle.lane_index[2]
            if lane_id == control_lane_id:
                if vehicle.position[0] > control_vehicle.position[0]:
                    distance = vehicle.position[0] - control_vehicle.position[0]
                    if head_distance > distance:
                        head_distance = distance
        return head_distance if head_distance != float("inf") else None

    def multi_rewards_func(self, vehicle_id, control_vehicle: ControlledVehicle):
        '''
        获得受控制车辆的奖励信息
        :param control_vehicle: 受控制车辆
        :return: 受控制车辆的奖励信息
        '''
        neighbours = self.road.network.all_side_lanes(control_vehicle.lane_index)
        lane = control_vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = control_vehicle.speed * np.cos(control_vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        head_distance = self.get_head_distance(vehicle_id)
        head_distance_ok = False
        head_distance_range= self.config["controlled_vehicles"][vehicle_id]["head_distance_range"]
        if head_distance is not None and head_distance >=head_distance_range[0] and head_distance<=head_distance_range[1]:
            head_distance_ok=True

        return {
            "collision_reward": float(control_vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(control_vehicle.on_road),
            "head_distance_reward":float(head_distance_ok)
        }

    def _rewards(self, action: Action):
        rewards = {}
        for vehicle_id, controlled_vehicle in enumerate(self.controlled_vehicles):
            rewards[vehicle_id] = self.multi_rewards_func(vehicle_id, controlled_vehicle)
        return rewards

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _is_terminated(self) -> bool:
        for controlled_vehicle in self.controlled_vehicles:
            if controlled_vehicle.crashed:
                return True
            if self.config["offroad_terminal"] and not controlled_vehicle.on_road:
                return True
        return False


if __name__ == '__main__':
    env = Multi_Highway_Env2(default_config)
    obs = env.reset()
    control_vehicles = len(env.controlled_vehicles)
    eposides = 10
    rewards = [0 for _ in range(control_vehicles)]
    # 0: 'LANE_LEFT',
    # 1: 'IDLE',
    # 2: 'LANE_RIGHT',
    # 3: 'FASTER',
    # 4: 'SLOWER'
    print(env.action_space)
    for eq in range(eposides):
        obs = env.reset()
        # print(obs)
        env.render()
        done = False
        truncated = False
        while not done and not truncated:
            # action = env.action_space.sample()
            # action1 = random.sample([0,1,2,3,4], 1)[0]
            # action2 = random.sample([0,1,2,3,4], 1)[0]
            # action = tuple([(random.random()*2-1, random.random()*2-1) for _ in range(control_vehicles)])
            action = ((-1, 0.01), (0, 0))
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            for i in range(control_vehicles):
                rewards[i] += reward[i]
            control_vehicle = env.controlled_vehicles[0]  # type:ControlledVehicle
            print(info)
            # print(control_vehicle.speed)
            # print(env.controlled_vehicles[1].speed)
            # print("1\n")
            # for b in obs:
            #     print(b)
            print(reward)
            print(done)

            # print(obs)
        # print(rewards)