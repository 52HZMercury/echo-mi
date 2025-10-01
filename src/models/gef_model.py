import torch
import torch.nn as nn
from .base_model import BaseModel
from .components.encoders import VideoEncoder, TextEncoder
# from .components.attention import HierarchicalFusionModule
from open_clip import tokenize
from contextlib import nullcontext  # 导入一个有用的上下文管理器


class GEFModel(BaseModel):
    """
    生成式证据流 (Generative Evidence Flow) 模型.
    它将诊断建模为一个由多模态证据驱动的信念演化过程。
    """

    def __init__(self, video_encoder_path, embed_dim=512, context_dim=1024,
                 velocity_net_hidden_dim=512, velocity_net_depth=2,
                 frozen_video_encoder=True, frozen_text_encoder=True):
        super().__init__()
        self.video_encoder = VideoEncoder(video_encoder_path, frozen=frozen_video_encoder)
        self.text_encoder = TextEncoder(frozen=frozen_text_encoder)

        self.context_fuser = nn.Sequential(
            nn.Linear(embed_dim * 3, context_dim),
            nn.LayerNorm(context_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(context_dim, context_dim)
        )
        # self.context_fuser = HierarchicalFusionModule(
        #     embed_dim=embed_dim,
        #     context_dim=context_dim,
        #     n_head=8,  # embed_dim // 64
        #     fusion_layers=2
        # )

        velocity_layers = []
        input_dim = context_dim + 1 + 1
        for _ in range(velocity_net_depth):
            velocity_layers.append(nn.Linear(input_dim, velocity_net_hidden_dim))
            velocity_layers.append(nn.ReLU())
            input_dim = velocity_net_hidden_dim
        velocity_layers.append(nn.Linear(velocity_net_hidden_dim, 1))
        self.velocity_net = nn.Sequential(*velocity_layers)

    @property
    def device(self):
        return next(self.velocity_net.parameters()).device

    def get_context(self, a2c_video, a4c_video, text_prompts):
        a2c_feat = self.video_encoder(a2c_video)
        a4c_feat = self.video_encoder(a4c_video)
        tokenized_prompts = tokenize(text_prompts).to(self.device)
        text_features_all = self.text_encoder(tokenized_prompts)
        text_features = text_features_all.mean(dim=0).unsqueeze(0).expand(a2c_feat.shape[0], -1)
        combined_features = torch.cat([a2c_feat, a4c_feat, text_features], dim=1)
        context_vector = self.context_fuser(combined_features)  # MLP
        return context_vector

    def forward(self, context, belief_t, time_t):
        belief_t = belief_t.unsqueeze(1) if belief_t.dim() == 1 else belief_t
        time_t = time_t.unsqueeze(1) if time_t.dim() == 1 else time_t
        net_input = torch.cat([context, belief_t, time_t], dim=1)
        return self.velocity_net(net_input)

    # --- **核心修正点 1**: 为ODE求解器添加梯度开关 ---
    def solve_ode(self, context, num_steps, enable_grad=False):
        """
        使用欧拉法求解ODE。
        Args:
            context (Tensor): 上下文向量.
            num_steps (int): 求解步数.
            enable_grad (bool): 是否为此过程开启梯度追踪.
        """
        batch_size = context.shape[0]
        belief_states = torch.zeros(batch_size, device=self.device)

        trace = [belief_states.clone().cpu()]
        velocities = []

        dt = 1.0 / num_steps

        # 根据`enable_grad`的值，选择合适的上下文管理器
        # 如果为True, nullcontext()什么都不做，梯度正常流动
        # 如果为False, torch.no_grad()会关闭梯度，节省内存
        grad_context = nullcontext() if enable_grad else torch.no_grad()

        with grad_context:
            for i in range(num_steps):
                t = torch.full((batch_size,), i * dt, device=self.device)
                # forward的计算现在处于可控的梯度上下文中
                v = self.forward(context, belief_states, t).squeeze(-1)
                belief_states = belief_states + v * dt

                # 我们只记录最终结果用于反向传播，中间过程的张量可以detach
                if enable_grad:
                    # 在梯度模式下，为下一次迭代分离状态，避免计算图中出现过长的链条
                    belief_states_detached = belief_states.detach().clone()
                    trace.append(belief_states_detached.cpu())
                    velocities.append(v.detach().clone().cpu())
                else:
                    trace.append(belief_states.clone().cpu())
                    velocities.append(v.clone().cpu())

        # 对于GEF模型，我们将上下文向量作为其特征表示
        return belief_states, torch.stack(trace), torch.stack(velocities), context