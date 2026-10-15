## 0907 关于ICLR KV Eviction的新调研

### 五个技术路线

- **启发式驱逐(H2O/SnapKV/PyramidKV/Quest)**
  - 设计固定的策略
  - baseline 堆,已经到头了。
  - attention-score 在聚合 benchmark 上常"及格",但**恰好漏掉那种"出现过一次、得分很低、却最关键"的针**。
  
- **学习型驱逐**:
  - 刚起步，有空间，预测token的未来效用
  - [KVpop](https://www.semanticscholar.org/paper/KVpop-Key-Value-Cache-Compression-with-Predictive-Hauzenberger-Schmidinger/ac7ed322fd7d9bd3c63d65808e2df18100b2f173)(延迟记忆打分器)和
  - [Sigmoid Attention 作为驱逐基座](https://arxiv.org/abs/2608.23296)(2×2×2 分解,发现 sigmoid+学习型硬删除才有 softmax 给不出的工作点)都在**优化带预算的困惑度/任务分**,不是检索鲁棒性。

- **可恢复驱逐**
  - 做多分层：驱逐-量化-全量
  - [QEvict](https://www.zhongzhuzhou.org/blog/2026-08-08-qevict-technical-review-en/)(INT2/全精度分层 + 升降级)
  - [AnchorKV](https://www.zhongzhuzhou.org/blog/2026-08-05-anchorkv-technical-review-en/)(softmax Jacobian 一阶近似用anchor token表示其他token)。

- **分层/摘要记忆**
  - 保存粗细粒度层dump到CPU 按需要恢复细节层
  - [SeKV](https://arxiv.org/abs/2606.31145)(surprisal 切分 + 双表示 + zoom-in)
  - [ZoomR](https://aclanthology.org/2026.acl-long.76/)(多粒度，用摘要选取细节)
  - [KSA](https://github.com/Kuaishou-OneRec/KSA)(摘要 token)。

- **换信号的角度**
  -  [FASA](https://mlanthology.org/iclr/2026/wang2026iclr-fasa/)(ICLR 2026, RoPE 频段代理预测重要token)
  -  [HitKV](https://ojs.aaai.org/index.php/AAAI/article/view/40105)(激活频次换累积分数)
  -  [VaSE](https://huggingface.co/papers/2606.03928)(值幅值+随机驱逐)
  -  [REAL](https://aclanthology.org/2026.acl-long.1811/)(ACL 2026,首次分成功/失败样本刻画注意力行为)。

**三个方向**

**方向一:查询感知的"软驱逐 + 可恢复救援"**
核心论点:**token 的重要性是相对于查询的,不是 token 自身的属性;而在 prefill 时刻查询还不可知。** 现在所有驱逐(包括学习型的)都在 prefill 做一次不可逆的 keep/drop 决定。做法:双层预算——大部分进一个便宜的压缩层,少量全精度;查询到来时用一个**训练出来的查询条件化"救援门"**把被误删的针从压缩层拉回全精度。差异化定位:QEvict 有升降级但规则是静态/随机的、不感知查询;SeKV 有 zoom-in 但需要训练一个摘要模型、改注意力。你的贡献点是"学习一个对未来任意查询分布鲁棒的救援门",而不是"驱逐多了一层救援"。冻结预训练模型即可跑,**不用重训 LLM**。

**方向二:现有 benchmark 抓不住的"针"——检索压力测试 + 定点修复**
这是偏方法/评估的贡献,但正好戳中所有论文都承认的盲区。构造"出现过一次、多跳绑定、反事实、带干扰物"的 hard needle,先系统性展示当前驱逐方法在它上面崩掉,再配一个定点的修复。风险:审稿人会当"只是 benchmark"或"修复是增量"怼。

**方向三:跨层/跨头"重要性一致性"信号**
单层 attention score 是弱点(REAL 那篇就是在骂只看头的平均行为)。把重要性定义成跨层、跨头的**一致性**,或用注意力分布的**熵**去标出"重要性不确定"的 token 并保护它们;再用 attention-matrix-free(表征变化)的办法躲开 FA2 的矩阵物化,让它在融合核上能跑。这个最不拥挤、故事最"新",但"so what"和对语义的深度分析要求也最高。

**我最推荐:方向一**,并把方向二的 stress-test 作为整篇论文的动机和评估骨干。