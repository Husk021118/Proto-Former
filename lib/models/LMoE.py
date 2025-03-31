import torch
import torch.nn as nn

from lib.models.Gating import GatedExpertModule


class LowRankConvExpert(nn.Module):
    def __init__(self, in_channels, out_channels, rank, kernel_size=3, stride=1, padding=1):
        """
        低秩卷积专家模块
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param rank: 分解后的秩
        :param kernel_size: 卷积核大小
        :param stride: 步长
        :param padding: 填充
        """
        super(LowRankConvExpert, self).__init__()

        # 低秩分解矩阵A (模拟卷积权重的低秩分解)
        self.conv_A = nn.Conv2d(in_channels, rank, kernel_size=kernel_size, stride=stride, padding=padding)

        # 低秩分解矩阵B
        self.conv_B = nn.Conv2d(rank, out_channels, kernel_size=1)  # 1x1卷积层，用于降维

    def forward(self, x):
        # 通过矩阵A的卷积操作
        x = self.conv_A(x)  # 第一层卷积，降维到秩
        # 通过矩阵B的卷积操作
        x = self.conv_B(x)  # 第二层卷积，将秩恢复到原始输出通道数
        return x


class GatingNetwork(nn.Module):
    def __init__(self, num_experts, input_channels,top_k):
        super(GatingNetwork, self).__init__()
        num_experts=num_experts
        top_k=top_k
        # self.gating = nn.Linear(input_dim, top_k)
        self.gating2= GatedExpertModule(input_channels=input_channels,num_experts=num_experts,top_k=top_k)


    def forward_with_linear(self,x):
        gating_weights = self.gating(x)  # (bs, c*h*w)
        gating_weights = torch.softmax(gating_weights, dim=-1)
        return gating_weights  # [32,8]

    def forward_with_conv(self,x):
        gating_weights=self.gating2(x)
        return gating_weights


    def forward(self, x, type='conv'):
        if type=='conv':
            gating_weights=GatingNetwork.forward_with_conv(self,x)
        else:
            x = x.view(bs, -1)  # 将输入flatten为(bs, c*h*w)
            gating_weights=GatingNetwork.forward_with_linear(self,x)
        return gating_weights#[32,8]


class LMoEWithConv(nn.Module):
    def __init__(self, num_experts, in_channels, out_channels, rank, height, width,top_k):#num_experts=16
        """
        使用低秩分解卷积的LMoE模块
        :param num_experts: 专家网络的数量
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param rank: 分解秩
        :param height: 输入图像高度
        :param width: 输入图像宽度
        """
        super(LMoEWithConv, self).__init__()
        self.num_experts = num_experts

        # 初始化专家网络，使用低秩分解卷积
        self.experts = nn.ModuleList([LowRankConvExpert(in_channels, out_channels, rank) for _ in range(num_experts)])

        # Gating网络，输入为flatten后的特征维度
        self.gating_network = GatingNetwork(num_experts, in_channels,top_k)

    def forward(self, x, type):
        """
        :param x: 输入特征 (bs, c, h, w)[8,512,64,64]
        :return: 加权的专家输出 (bs, out_channels, h, w)
        """
        # 1. 计算每个专家的输出
        expert_outputs = [expert(x) for expert in self.experts]  # 列表 (bs, out_channels, h, w)8,512,64,64
        expert_outputs = torch.stack(expert_outputs, dim=0)  # (num_experts, bs, out_channels, h, w)8,8,512,64,64

        # 2. Gating网络的权重计算
        gating_weights,top_k_indices = self.gating_network(x, type)  # (bs, num_experts)
        # top_k_indices = top_k_indices.cpu()
        # expert_outputs = expert_outputs.cpu()

        expert_outputs = expert_outputs.permute(1, 0 ,2, 3, 4)
        selected_expert_outputs = []
        for i in range(expert_outputs.shape[0]):
            selected_expert_outputs.append(expert_outputs[i][top_k_indices[i].tolist()])
        selected_expert_outputs = torch.stack(selected_expert_outputs, dim=0)#[bs,8,] 8,8,1024,32,32
        selected_expert_outputs=selected_expert_outputs.permute(1,0,2,3,4)

        gating_weights = gating_weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # (bs, num_experts, 1, 1, 1)

        # expert_outputs=expert_outputs.permute(1,0,2,3,4)
        # top_k_indices=top_k_indices.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # selected_expert_outputs=expert_outputs[top_k_indices]

        # 3. 将专家的输出和对应的 gating 权重结合，得到加权特征图
        # weighted_expert_outputs = gating_weights * expert_outputs.permute(1, 0, 2, 3,4)  # (bs, num_experts, out_channels, h, w)
        weighted_expert_outputs = gating_weights * selected_expert_outputs.permute(1, 0, 2, 3,
                                                                          4)  # (bs, num_experts, out_channels, h, w)
        # 4. 逐元素相乘原始特征图与加权特征图
        # 在此之前需要调整原始特征图的形状以便进行逐元素相乘
        x_expanded = x.unsqueeze(1)  # (bs, 1, c, h, w)
        # # 逐元素相乘
        combined_outputs = weighted_expert_outputs * x_expanded  # (bs, num_experts, out_channels, h, w)

        # 5. 对所有专家的加权特征图进行累加
        final_output = torch.sum(combined_outputs, dim=1)  # (bs, out_channels, h, w)
        # final_output = torch.sum(weighted_expert_outputs, dim=1)  # (bs, out_channels, h, w)

        return final_output, gating_weights  # 返回加权后的输出和 gating 权重

if __name__ == '__main__':

    # DEMO: 如何使用 LMoEWithConv
    bs, c, h, w = 32, 64, 32, 32  # 假设输入特征大小
    x = torch.randn(bs, c, h, w)  # 假设的输入

    num_experts = 8  # 专家数量
    in_channels = c  # 输入通道数
    out_channels = c  # 假设输出通道数
    rank = 16  # 低秩分解的秩

    # 初始化LMoE模块
    lmoe_conv = LMoEWithConv(num_experts=num_experts, in_channels=in_channels, out_channels=out_channels, rank=rank, height=h, width=w)

    # 通过LMoE模块得到加权的专家输出
    output_conv,gating_weights_conv = lmoe_conv(x,type='conv')
    output_linear,gating_weights_linear = lmoe_conv(x,type='linear')#output_conv和output_linear就是具有具体特征的特征图
    # print("conv_gating")
    # print(gating_weights_conv)
    # print("linear_gating")
    # print(gating_weights_linear)
    # output_conv可以作为后续模块的输入
