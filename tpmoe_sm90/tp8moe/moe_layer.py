"""One Kimi K3 MoE layer on TP8 (intra-node), v0 composition: the routed experts run in tp8moe (fused FC1 / SiTU / FC2 + 8-way one-shot
LL reduce of the latent sum); the rest is torch + NCCL:
  router (replicated, [896, 7168])            -> ids / weights          (torch sigmoid gate, K3 KimiMoEGate semantics)
  W_down sharded by output rows (448 / rank)  -> z_r, all-gather -> z   (NCCL)
  kernel: z -> u = sum over ranks and slots   (one-shot LL reduce)
  RMSNorm(u) (replicated, 3584)
  W_up sharded by K (448 latent rows / rank)  -> partial y_r [M, 7168]
  shared expert sharded by intermediate (768 / rank): situ(x Wg_r^T, x Wu_r^T) Wd_r^T -> partial s_r
  all-reduce(y_r + s_r) (NCCL, fp32)          -> y
"""
import torch
import torch.distributed as dist

import tp8moe_host as H
import moe_ref as R

HID, LAT, SH_INTER, TOPK, NDEV = 7168, 3584, 6144, 16, 8


class MoELayerTP8:
    def __init__(self, rank, state, experts, gate_w, gate_bias, down_w, up_w, norm_w, sh_gate, sh_up, sh_down, group=None):
        """experts: kernel-layout weights of this rank's intermediate shard (dict w1 s1 f1 w2 s2 f2).  Full BF16 tensors for the
        dense parts are sharded here (down_w [3584, 7168] -> rows, up_w [7168, 3584] -> columns, shared [6144, 7168] / [7168, 6144])."""
        self.rank, self.st, self.w, self.group = rank, state, experts, group
        self.gate_w32 = gate_w.float().contiguous()   # the K3 gate computes its logits in fp32 (KimiMoEGate); fp32 copy made once
        self.gate_bias, self.norm_w = gate_bias.float().contiguous(), norm_w.contiguous()
        r0, r1 = rank * (LAT // NDEV), (rank + 1) * (LAT // NDEV)
        self.down_r = down_w[r0:r1].contiguous()            # [448, 7168]
        self.up_r = up_w[:, r0:r1].contiguous()             # [7168, 448]
        s0, s1 = rank * (SH_INTER // NDEV), (rank + 1) * (SH_INTER // NDEV)
        self.shg_r, self.shu_r, self.shd_r = sh_gate[s0:s1].contiguous(), sh_up[s0:s1].contiguous(), sh_down[:, s0:s1].contiguous()
        self.z_full = None

    def forward(self, x):
        """x BF16 [M, 7168] (identical on every rank) -> BF16 [M, 7168] (identical on every rank)."""
        M = x.shape[0]
        logits = x.float() @ self.gate_w32.t()                           # fp32 router logits (as the model code)
        ids, rw = R.gate_ref(logits, self.gate_bias, topk=TOPK)
        z_r = (x @ self.down_r.t())                                      # BF16 [M, 448] (bf16 GEMM, fp32 accumulate)
        if self.z_full is None or self.z_full.shape[1] != M:
            self.z_full = torch.empty(NDEV, M, LAT // NDEV, dtype=torch.bfloat16, device=x.device)
        dist.all_gather_into_tensor(self.z_full, z_r.contiguous(), group=self.group)
        z = self.z_full.permute(1, 0, 2).reshape(M, LAT).contiguous()   # [M, 3584]
        u = torch.empty(M, LAT, dtype=torch.bfloat16, device=x.device)
        H.run(self.st, self.w, z, ids.contiguous(), rw.contiguous(), u, self.rank, NDEV)
        un = R.rmsnorm(u, self.norm_w).to(torch.bfloat16)
        c0, c1 = self.rank * (LAT // NDEV), (self.rank + 1) * (LAT // NDEV)
        y_r = (un[:, c0:c1] @ self.up_r.t()).float()                    # partial [M, 7168] (bf16 GEMM output, fp32 sum)
        g = (x @ self.shg_r.t()).float(); p = (x @ self.shu_r.t()).float()
        a = R.situ_glu(g, p).to(torch.bfloat16)
        s_r = (a @ self.shd_r.t()).float()
        part = y_r + s_r
        dist.all_reduce(part, group=self.group)
        return part.to(torch.bfloat16)
