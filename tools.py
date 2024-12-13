import glob
import json
import logging
import os
from multiprocessing import Process,Queue
from matplotlib import pyplot as plt


def plot(q):
    while True:
        title=None
        try:
            data = q.get()  # 等待并从队列中取出任务
            close,title, xlabel, ylabel, x, y, save_path = data
            if close:
                break
            plt.figure(figsize=(20, 10), dpi=60)
            plt.plot(x, y)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(e)
            print(title)

class PlotUtil:
    def __init__(self,):
        self.queue=Queue() # 任务队列
        self.init_process()

    def init_process(self):
        self.process=Process(target=plot,args=(self.queue,)) # 画图进程
        self.process.start() # 启动进程

    def push_task(self,title,xlabel,ylabel,x,y,save_path):
        self.queue.put((False,title,xlabel,ylabel,x,y,save_path)) # 将任务放入队列

    def close(self):
        self.queue.put((True,1,1,1,1,1,1))
    def get_queue_task_number(self):
        return self.queue.qsize()

def print_args_ddpg(logger,opt,config):
    logger.info(
        f"训练参数配置：\n"
        f"    save_dir:{opt.save_dir}\n"
        f"    train:{opt.train}\n"
        f"    test_model_dir:{opt.test_model_dir}\n"
        f"    episodes:{opt.episodes}\n"
        f"    num_agent:{opt.num_agent}\n"
        f"    p_hidden_size:{opt.p_hidden_size}\n"
        f"    q_hidden_size:{opt.q_hidden_size}\n"
        f"    batch_size:{opt.batch_size}\n"
        f"    actor_lr:{opt.actor_lr}\n"
        f"    critic_lr:{opt.critic_lr}\n"
        f"    gamma:{opt.gamma}\n"
        f"    memory_capacity:{opt.memory_capacity}\n"
        f"    tau:{opt.tau}\n"
        f"    desc:{opt.desc}\n"
        f"    render_mode:{opt.render_mode}\n"
        f"    input_size:{opt.input_size}\n"
        f"    output_size:{opt.output_size}\n"
        f"    action_bound:{opt.action_bound}\n"
        f"    action_space:{opt.action_space}\n"
        f"    exploration_decay:{opt.exploration_decay}\n"
        f"    train_start_memory_capacity:{opt.train_start_memory_capacity}\n"
        f"    save_step:{opt.save_step}\n"
        f"    min_exploration:{opt.min_exploration}\n"
        f"    max_exploration:{opt.max_exploration}\n")

    logger.info("游戏环境配置：\n  "+config.__str__())

def print_args_dqn(logger,opt,config):
    logger.info(
        f"训练参数配置：\n"
        f"    save_dir:{opt.save_dir}\n"
        f"    train:{opt.train}\n"
        f"    test_model_dir:{opt.test_model_dir}\n"
        f"    episodes:{opt.episodes}\n"
        f"    num_agent:{opt.num_agent}\n"
        f"    hidden_size:{opt.hidden_size}\n"
        f"    batch_size:{opt.batch_size}\n"
        f"    lr:{opt.lr}\n"
        f"    gamma:{opt.gamma}\n"
        f"    memory_capacity:{opt.memory_capacity}\n"
        f"    copy_step:{opt.copy_step}\n"
        f"    action_space:{opt.action_space}\n"
        f"    desc:{opt.desc}\n"
        f"    render_mode:{opt.render_mode}\n"
        f"    input_size:{opt.input_size}\n"
        f"    output_size:{opt.output_size}\n"
        f"    exploration_decay:{opt.exploration_decay}\n"
        f"    train_memory_capacity:{opt.train_start_memory_capacity}\n"
        f"    save_step:{opt.save_step}\n"
        f"    min_exploration:{opt.min_exploration}\n"
        f"    max_exploration:{opt.max_exploration}\n")

    logger.info("游戏环境配置：\n  "+config.__str__())


def save_config(config_save_dir,opt,multi_highway_config):
    os.makedirs(config_save_dir,exist_ok=True)
    with open(os.path.join(config_save_dir,"multi_highway_config.json"),"w",encoding="utf-8") as f:
        json.dump(multi_highway_config,f)
    opt_dict=opt.__dict__
    with open(os.path.join(config_save_dir,"opt.json"),"w",encoding="utf-8") as f:
        json.dump(opt_dict,f)

def load_config(config_save_dir):
    with open(os.path.join(config_save_dir,"multi_highway_config.json"),"r",encoding="utf-8") as f:
        multi_highway_config=json.load(f, object_hook=int_keys_hook)
    with open(os.path.join(config_save_dir,"opt.json"),"r",encoding="utf-8") as f:
        opt_dict=json.load(f)
    return opt_dict,multi_highway_config

def int_keys_hook(obj):
    # Convert string keys to integers
    return {int(key) if key.isdigit() else key: value for key, value in obj.items()}


# 每次生成不同的文件夹 run/exp1 run/exp2 ...
def get_log_path(log_root="run", sub_dir="exp"):
    os.makedirs(log_root, exist_ok=True)
    files = glob.glob(os.path.join(log_root,f"{sub_dir}")+"*", recursive=False)
    log_dir = os.path.join(log_root, f"{sub_dir}{len(files)+1}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


#配置日志 输出到控制台和文件
def logger_setting(logs_dir,file_handler_level=logging.DEBUG,stream_handler_level=logging.DEBUG):
    # 配置日志记录器
    logger = logging.getLogger(__name__)
    # 配置日志文件路径
    log_file = os.path.join(logs_dir, "run.log")
    logger.setLevel(logging.DEBUG)
    # 配置文件处理器
    file_handler = logging.FileHandler(log_file,encoding="utf-8")
    file_handler.setLevel(file_handler_level)

    # 配置控制台处理器
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_handler_level)

    # 配置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',"%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

if __name__ == '__main__':
    print(load_config("runs/train/exp9/models"))