import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedExpertModule(nn.Module):
    def __init__(self, input_channels, num_experts, top_k=8):
        super(GatedExpertModule, self).__init__()
        self.top_k = top_k
        self.num_experts=num_experts

        # 3x3 convolution to reduce channel dimensions
        self.conv_reduce = nn.Conv2d(input_channels, input_channels // 4, kernel_size=3, padding=1)

        # Multi-head self-attention (MHSA) layer
        self.mhsa = nn.MultiheadAttention(embed_dim=input_channels // 4, num_heads=8)

        # MLPs for position-aware branch
        self.mlp_pos1 = nn.Linear(input_channels // 4, input_channels // 4)  # R^HW x C -> R^1 x C
        self.mlp_pos2 = nn.Linear(input_channels // 4, num_experts)  # R^1 x C -> R^1 x N

        # Final convolution to get task-specific routing features
        self.conv_final = nn.Conv1d(input_channels // 4+num_experts, num_experts, kernel_size=1)#【20，10】

    def forward(self, x):#[8,512,64,64]
        # Step 1: Channel reduction
        x_reduced = self.conv_reduce(x)  # Shape: [B, C/4, H, W]【8，128，64，64】

        # Step 2: Reshape for MHSA
        B, C, H, W = x_reduced.size()#
        x_seq = x_reduced.view(B, C, -1).permute(0, 2, 1)  # Shape: [B, HW, C/4][8,64*64(4096),16]

        # Step 3: Multi-head self-attention
        attn_out, _ = self.mhsa(x_seq, x_seq, x_seq)  # Shape: [B, 4096, C/4(128)]

        # Step 4: Position-aware branch
        pos_out = self.mlp_pos1(attn_out.mean(dim=1))  # Shape: [B, C/4]【8，128】
        pos_out = self.mlp_pos2(pos_out)  # Shape: [B, N]【8，16】

        # Step 5: Stack outputs and convolution for final routing features，从c维度拼接
        combined_out = torch.cat((attn_out.mean(dim=1), pos_out), dim=1)  # Shape: [B, C/4 + N][8,144]
        combined_out = combined_out.unsqueeze(1)  # Shape: [B, 1, C/4 + N][8,1,144]

        combined_out = combined_out.permute(0, 2, 1)#【8，144，1】
        # Step 6: Final convolution
        final_out = self.conv_final(combined_out)  # Shape: [B, N, 1] [8,16,1]

        # Step 7: Calculate task gating scores
        gating_scores = F.softmax(final_out, dim=1)  # Shape: [B, N, 1] [8,16,1]

        # Step 8: Select top K gating scores
        top_k_scores, top_k_indices = torch.topk(gating_scores, self.top_k, dim=1)  # Shape: [B, K, 1][8,8,1]

        top_k_scores=top_k_scores.squeeze(2)#[8,8]
        top_k_indices = top_k_indices.squeeze(2)  # [8, 8]
        return top_k_scores,top_k_indices#[32,8]


# # 示例用法
# input_tensor = torch.randn(32, 64, 32, 32)  # 假设输入为 [B, C, H, W] 形状
# gated_expert = GatedExpertModule(input_channels=64,  num_experts=16, top_k=8)#k是用来选择前k个最好的专家
# output = gated_expert(input_tensor)
#
# print(output.shape)  # 应输出 [B, K, 1]