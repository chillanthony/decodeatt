"""信号 1：复发周期 MRI（Maximum Recurrence Interval）。

离线版：遍历所有步，记录每 page 进入该步 Top-mri_topk 注意的步序列，
MRI = 最大相邻间隔。在线版留到 Step 1。
"""
# TODO(step4): def compute_mri(per_step_topk_pages) -> dict[int, int]
