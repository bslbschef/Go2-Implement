import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    # torch.rand()生成的随机数服从 [0, 1) 的均匀分布。
    # 默认数据类型为 torch.float32。如果需要特定的数据类型，可以通过 dtype 参数指定。
    # size (或 shape)：指定输出张量的形状。例如 (2, 3) 表示 2 行 3 列的张量。
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

    # torch.randn()返回：标准正态分布 ~ N(0, 1)

class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True, device='cuda'):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)  # 返回一个设备对象，可以传递给 PyTorch 中的张量或模型，用于指定设备

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True
        self.dt = 0.02  # dt表示delta time（δ time）

        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)  # 向上取整，进一法

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scalses"]

        self.scene = gs.Scene(
            # 在这里将substeps-子步骤的数量设置为 2，在模拟过程中，每一个主要的时间步长 dt 会被细分为 2 个子步骤进行更精细的计算，以提高模拟的精度或稳定性。
            sim_options = gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options = gs.options.ViewerOptions(
                max_FPS = int(0.5 / self.dt),
                camera_pos = (2.0, 0.0, 2.5),
                camera_lookat = (0.0, 0.0, 0.5),
                camera_fov = 40,   # fov表示field of view！
            ),
            vis_options = gs.options.VisOptions(n_rendered_envs=1),
            rigid_options = gs.options.RigidOptions(
                dt = self.dt,
                constraint_solver = gs.constraint_solver.Newton,
                enable_collision = True,
                enable_joint_limit = True,
            ),
            show_viewer = show_viewer,
        )
        # fixed=True 表示添加到场景中的实体是固定的，即它不会受到物理引擎中重力或其他力的作用，也不会移动或旋转。这通常用于定义地面或其他静态物体。
        # 固定实体：不会参与动态计算，不会因为碰撞、重力等改变其位置或姿态。
        # 如果 fixed=False，实体会变成动态物体，可以受力移动、旋转或与其他动态物体交互。
        # 添加地面实体
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # 添加机器人实体
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )
        
        self.scene.build(n_envs=num_envs)
        # [robot是一个RigidEntity,见gs.RigidEntity] [get_joint()返回的是一个gs.RigidJoint] 
        # [dof_indx_local返回：Returns the local dof index of the joint in the entity.]
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        self.reward_functions, self.episode_sums = dict(), dict()
        # 方法一：使用 items() 方法遍历键值对： for key, value in my_dict.items():
        # 方法二：使用 keys() 方法遍历键:      for key in my_dict.keys():
        # 方法三：使用 values() 方法遍历值     for value in my_dict.values():
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            # 函数也是类的属性！所以也可以用getattr函数 
            # 调用getattr后，reward_functions[name]是一个对函数_reward_name的一个引用！
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor([self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
                                           device=self.device, dtype=gs.tc_float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict() 
    
    def _resample_commands(self, envs_idx):
        # * 运算符在函数调用时可用于解包可迭代对象。
        # (len(envs_idx),) 是一个元组。这里使用逗号 , 是为了将 len(envs_idx) 作为一个元组的元素，而不是简单的一个值。
        # 它作为 gs_rand_float 函数的一个参数，可能表示生成随机数的形状或大小。
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device) # 线速度x轴范围
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device) # 线速度y轴范围
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)   # 角速度范围

    def step(self, actions):
        # 对action数值进行裁剪，保持在正负clip_actions范围内，here is [-100, 100]
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        # 这是为了模拟真实机器人在执行动作时可能存在的延迟现象，增加模拟的真实性。 
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions 
        # 将动作值进行缩放并与默认关节位置相加，得到最终的关节位置目标值，为接下来的位置控制做准备。 
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos 
        # Set the PD controller’s target position for the entity’s dofs. This is used for position control.
        # 调用 self.robot 的 control_dofs_position 方法，将 target_dof_pos 作为目标位置，通过 self.motor_dofs 指定的关节自由度索引，
        # 使用 PD 控制器对机器人的关节位置进行控制。这是为了让机器人的关节移动到目标位置，实现机器人的动作执行。
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        # Runs a simulation step forward in time. 
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        # 虽然base_euler没有在init中初始化，但仍然是类的属性，所以可以这样使用！
        # base_euler 是类的属性，但在 __init__ 中对其进行显式初始化通常是更好的编程实践。
        self.base_euler = quat_to_xyz(transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat))
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang_vel(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)  # motor_dofs 是一个列表，包含了机器人关节的自由度索引。
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # resample commands
        # 该代码找出哪些环境的模拟周期长度正好是 resampling_time_s 的整数倍，这些环境将是需要重新采样命令的环境。
        # 通过 self._resample_commands(envs_idx) 对这些环境重新采样命令，以确保在特定的时间间隔内更新命令，从而保证模拟环境的动态性和多样性。
        # int(self.env_cfg["resampling_time_s"] / self.dt)表示重采样所需要的步数！
        envs_idx = (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(envs_idx)

        # 检查模拟是否要终止或重置！
        # 这行代码将 self.reset_buf 更新为一个布尔型张量，其中元素为 True 表示对应环境的模拟周期长度超过了 self.max_episode_length，需要进行重置，元素为 False 表示不需要。
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        # nonzero() 是 PyTorch 中的一个张量方法，用于找出张量中非零元素的索引
        # 当 as_tuple=True（默认）时，返回的结果是一个元组，元组中的每个元素是一个一维张量，代表了不同维度上的索引。
        # 先使用 .nonzero(as_tuple=False) 找出张量中所有非零元素的索引，并将结果存储在一个二维张量中，其中每一行是一个非零元素的索引。
        # 然后使用 .flatten() 将这个二维张量 “压平” 为一维张量，得到一个存储所有非零元素索引的一维张量。
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_out"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_out"][time_out_idx] = 1.0
        # reset_idx是一个函数：重置那些1）时间步超时；2）pitch 或 roll 超过阈值；3）位置超出阈值的索引。
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # 计算奖励
        self.rew_buf[:] = 0.0
        # .items()：是 Python 字典的一个方法，它返回一个包含字典中所有键值对的可迭代对象。每个键值对以元组的形式表示，元组的第一个元素是键，第二个元素是值。
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        
        # 计算观测值
        # 通过 axis=-1 将它们在最后一个维度上拼接
        self.obs_buf = torch.cat(
            [
                # 速度值与缩放因子相乘！
                self.base_ang_vel * self.obs_scales["ang_vel"],
                self.projected_gravity, 
                # 指令与指令缩放因子相乘！
                self.commands * self.commands_scale,
                # 关节位置与关节位置缩放因子相乘！
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                # 关节速度与关节速度缩放因子相乘！
                self.dof_vel * self.obs_scales["dof_vel"],
                self.actions,
            ],
            axis = -1,
        )     

        # self.last_actions[:] = self.actions[:]
        # 原地操作，表示直接修改 self.last_actions 存储的数据，而不改变 self.last_actions 的引用。
        # 这种操作在内存使用上比较高效，因为它避免了创建新的张量，而是直接在原 self.last_actions 张量的存储位置上更新数据。
        # self.last_actions = self.actions[:]
        # 这意味着 self.last_actions 现在指向了一个新的张量对象，即 self.actions 的切片副本，而不是修改原 self.last_actions 张量的数据。
        # 如果 self.last_actions 之前指向的张量不再有其他引用，它可能会被 Python 的垃圾回收机制回收。
        # 注意：二者有本质区别！！！
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # None是一个占位符，这与合作开发的沟通有关！可能对方接口没用到，所以这里为None
        # obs, _, rew, reset, extra = example.step()
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None
    
    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            # 当使用 return 语句时，如果后面没有跟任何表达式（如 return），函数将立即终止执行，并且不返回任何明确的值。
            return 

        # 重置自由度
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            #  The indices of the dofs to set (motor_dofs表示关节自由度的索引)
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # 重置基本条件
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # 重置buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # 填充字典日志
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["raw_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0
        
        self._resample_commands(envs_idx)
    
    def reset(self):
        # 当 self.reset_buf 是一个 PyTorch 张量或 NumPy 数组时，使用 self.reset_buf = True 会将 self.reset_buf 从一个向量变为一个标量。
        # 当使用 self.reset_buf[:] = True 时，self.reset_buf[:] 是对 self.reset_buf 的切片操作，这里的 [:] 表示选取整个张量或数组。
        # 该操作会将 self.reset_buf 中的所有元素都更新为 True
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None
    

    # 奖励函数的定义
    def _reward_tracking_lin_vel(self):
        # tracking of linear velocity (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # 求指数e的xxx次方！
        # 注意：这里使用指数函数将误差转换为奖励，使得智能体在最小化角速度误差（加了个负号！）时得到更高的奖励
        # tracking_sigma 在 _reward_tracking_ang_vel 函数中起到一个缩放因子的作用。
        # 当 tracking_sigma 较大时，-ang_vel_error / self.reward_cfg["tracking_sigma"] 的绝对值会较小，
        # 即使 ang_vel_error 较大，得到的指数结果也不会太小，意味着对误差的惩罚（即奖励的减少）会相对较小。
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])
    
    def _reward_tracking_ang_vel(self):
        # tracking of angular velocity (z axis)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])
    
    def _reward_lin_vel_z(self):
        # reward for moving up/down
        return torch.square(self.commands[:, 2] - self.base_lin_vel[:, 2])
    
    def _reward_action_rate(self):
        # penalize high action rates
        return torch.square(self.actions / self.action_scale).mean()




        

