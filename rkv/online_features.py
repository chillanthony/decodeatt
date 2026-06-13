"""Phase 2：解码时**增量维护**每页四维签名，并用导出的线性打分器在线打分。

与离线 page_features 的特征定义对齐（保证用 Step 0 训出的权重）：
- entropy       : 页内生成 token 的平均预测熵（解码时每步 logits 免费拿）。
- cum_attn      : 页被各因果有效 query 步访问的**平均**注意力（运行均值）。
- position      : 页下标 / 当前页数（打分时即时算）。
- concentration : 页被访问注意力的峰均比 max/(mean+eps)。

每步喂入：当前 query 对各 page 的注意力行（[num_pages]，层/头平均，行和=1）+ 该步熵。
注意力行的页 p 项 = 该步分配给页 p 的注意力；据此更新 cum_attn 的运行和/计数与 max。
熵按 token 落入的页累加。
"""
from __future__ import annotations

import json

import torch

_EPS = 1e-12


class OnlineSignature:
    """逐页签名的增量状态 + 线性打分。页数随解码增长，按需扩容。"""

    def __init__(self, scorer_path: str, page_size: int = 16, device="cpu"):
        with open(scorer_path) as f:
            s = json.load(f)
        self.names = s["feature_names"]
        self.w = torch.tensor(s["w_raw"], dtype=torch.float32, device=device)
        self.b = float(s["b_raw"])
        self.page_size = page_size
        self.device = device
        imp = s.get("impute_mean")
        self.ent_impute = float(imp[s["feature_names"].index("entropy")]) if imp else 0.0
        # 各列在 FEATURE_NAMES 里的位置
        self.i_ent = self.names.index("entropy")
        self.i_cum = self.names.index("cum_attn")
        self.i_pos = self.names.index("position")
        self.i_con = self.names.index("concentration")

        z = lambda: torch.zeros(0, dtype=torch.float32, device=device)
        self.ent_sum, self.ent_cnt = z(), z()      # 熵累加 / token 计数
        self.cum_sum, self.cum_cnt = z(), z()      # 被访问注意力累加 / 步计数
        self.cum_max = z()                         # 峰值被访问注意力

    def _grow(self, num_pages: int):
        cur = self.ent_sum.numel()
        if num_pages <= cur:
            return
        pad = num_pages - cur
        z = torch.zeros(pad, dtype=torch.float32, device=self.device)
        for attr in ("ent_sum", "ent_cnt", "cum_sum", "cum_cnt", "cum_max"):
            setattr(self, attr, torch.cat([getattr(self, attr), z.clone()]))

    @torch.no_grad()
    def update(self, attn_row: torch.Tensor, step_entropy: float, token_pos: int):
        """吸收一个解码步：attn_row[num_pages] = 该 query 对各页注意力；
        step_entropy = 该步预测熵；token_pos = 新 token 的全序列位置。"""
        num_pages = attn_row.numel()
        self._grow(num_pages)
        a = attn_row.to(self.device, torch.float32)
        # cum_attn：只对因果有效页（页首 <= 当前 query 位置）计步——与离线一致
        valid = torch.arange(num_pages, device=self.device) * self.page_size <= token_pos
        self.cum_sum[:num_pages] += torch.where(valid, a, torch.zeros_like(a))
        self.cum_cnt[:num_pages] += valid.float()
        self.cum_max[:num_pages] = torch.maximum(self.cum_max[:num_pages],
                                                 torch.where(valid, a, self.cum_max[:num_pages]))
        # entropy：新 token 落入的页
        p = token_pos // self.page_size
        self.ent_sum[p] += step_entropy
        self.ent_cnt[p] += 1.0

    @torch.no_grad()
    def scores(self, num_pages: int) -> torch.Tensor:
        """返回各页签名分 [num_pages]（分越高越像纠错锚点 R_t）。"""
        self._grow(num_pages)
        n = num_pages
        ent = self.ent_sum[:n] / self.ent_cnt[:n].clamp(min=1)
        cum = self.cum_sum[:n] / self.cum_cnt[:n].clamp(min=1)
        con = self.cum_max[:n] / (cum + _EPS)
        pos = torch.arange(n, device=self.device).float() / max(n, 1)
        feat = torch.zeros(n, len(self.names), device=self.device)
        feat[:, self.i_ent] = ent
        feat[:, self.i_cum] = cum
        feat[:, self.i_pos] = pos
        feat[:, self.i_con] = con
        # entropy 为 0 的纯 prompt 页：用插补均值（与离线一致，避免误判）
        no_ent = self.ent_cnt[:n] == 0
        if no_ent.any():
            feat[no_ent, self.i_ent] = self.ent_impute
        return feat @ self.w + self.b


def _selftest():
    """合成数据：手算单页签名与运行状态一致（不需 GPU/模型）。"""
    import tempfile
    import os

    scorer = {
        "feature_names": ["entropy", "cum_attn", "position", "concentration"],
        "w_raw": [1.0, 0.0, 0.0, 0.0], "b_raw": 0.0, "impute_mean": [2.0, 0, 0, 0],
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    os.write(fd, json.dumps(scorer).encode()); os.close(fd)
    sig = OnlineSignature(path, page_size=2)

    # 步0：token_pos=0（页0），熵 1.0，注意力全给页0
    sig.update(torch.tensor([1.0]), 1.0, token_pos=0)
    # 步1：token_pos=1（页0），熵 3.0，注意力 [1.0]
    sig.update(torch.tensor([1.0]), 3.0, token_pos=1)
    # 步2：token_pos=2（页1），熵 5.0，注意力 [0.3, 0.7]
    sig.update(torch.tensor([0.3, 0.7]), 5.0, token_pos=2)

    # 页0 熵 = mean(1,3)=2；页1 熵 = 5
    sc = sig.scores(2)
    assert abs(sc[0].item() - 2.0) < 1e-5, sc
    assert abs(sc[1].item() - 5.0) < 1e-5, sc
    # cum_attn 页0：步0,1,2 都因果有效 -> mean(1,1,0.3)=0.7667
    cum0 = (1.0 + 1.0 + 0.3) / 3
    assert abs((sig.cum_sum[0] / sig.cum_cnt[0]).item() - cum0) < 1e-5
    # cum_attn 页1：仅步2有效 -> 0.7
    assert abs((sig.cum_sum[1] / sig.cum_cnt[1]).item() - 0.7) < 1e-5
    print("OK: OnlineSignature 增量签名与手算一致")
    os.unlink(path)


if __name__ == "__main__":
    _selftest()
