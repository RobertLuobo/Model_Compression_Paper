# Model_Compression_Paper

### Type of Pruning

| Type        |      `F`       |      `W`       |   `Other`   |
| :---------- | :------------: | :------------: | :---------: |
| Explanation | Filter pruning | Weight pruning | other types |



| `Conf`    | `2015`  | `2016`  | `2017` | `2018`  |  `2019`   |   `2020`   | `2021`  |
| :-------- | :-----: | :-----: | :----: | :-----: | :-------: | :--------: | :-----: |
| `AAAI`    |   539   |   548   |  649   |   938   |   1147    |    1591    |  1692   |
| `CVPR`    | 602(71) | 643(83) | 783-71 | 979(70) | 1300(288) | 1470(335)  |         |
| `NeurIPS` |   479   |   645   |  954   |  1011   |   1428    | 1900 (105) |         |
| `ICLR`    |         | oral-15 |  198   | 336(23) |  502(24)  |    687     | 860(53) |
| `ICML`    |         |         | `433`  |  `621`  |   `774`   |   `1088`   |         |
| `IJCAI`   |   572   |   551   |  660   |   710   |    850    |    592     |         |
| `ICCV`    |         |   `-`   | `621`  |   `-`   |   1077    |    `-`     |         |
| `ECCV`    |         |  `415`  |  `-`   |  `778`  |    `-`    |   `1360`   |         |
| `MLsys`   |         |         |        |         |           |            |         |



`MLsys`:https://proceedings.mlsys.org/paper/2019

`ICCV`https://dblp.org/db/conf/iccv/iccv2019.html

`ICCV` https://dblp.org/db/conf/iccv/iccv2017.html

`ECCV` https://link.springer.com/conference/eccv

`ECCV` https://zhuanlan.zhihu.com/p/157569669

`CVPR` https://dblp.org/db/conf/cvpr/cvpr2020.html



`ICDE` `ECAI` `ACCV` `WACV` `BMVC`

`WACV`:(Applications of Computer Vision)

---

### 量化2015 & 2016 & 2017

| Title                                                        |   Venue   | Type |          Notion           |
| :----------------------------------------------------------- | :-------: | :--: | :-----------------------: |
| HWGQ-Deep Learning With Low Precision by Half-wave Gaussian Quantization |  `CVPR`   |      |           孙剑            |
| Weighted-Entropy-based Quantization for Deep Neural Networks |  `CVPR`   |      |        `not code`         |
| WRPN Wide Reduced-Precision Networks                         |  `ICLR`   |      | `intel`+distiller框架集成 |
| DoReFa-Net: training low bitwidth convolutional neural networks with low bitwidth gradients |  `ICLR`   |      |          超低bit          |
| XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks |  `ECCV`   |      |          超低bit          |
| Binaryconnect Training deep neural networks with binary weights during propagations | `NeurIPS` |      |          超低bit          |
| INQ-Incremental network quantization Towards lossless cnns with low-precision weight |  `ICLR`   |      |          `intel`          |



---

### 剪枝 2017

| Title                                                        |  Venue  | Type |        Notion         |
| :----------------------------------------------------------- | :-----: | :--: | :-------------------: |
| [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) |  ICLR   | `F`  |      abs(filter)      |
| [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440) |  ICLR   | `F`  | 基于一阶泰勒展开近似  |
| [ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](https://arxiv.org/abs/1707.06342) |  ICCV   | `F`  | 找一组channel近似全集 |
| [Channel pruning for accelerating very deep neural networks](https://arxiv.org/abs/1707.06168) |  ICCV   | `F`  |    LASSO回归、孙剑    |
| [Learning Efficient Convolutional Networks Through Network Slimming](https://arxiv.org/abs/1708.06519) |  ICCV   | `F`  |       基于BN层        |
| [Net-Trim: Convex Pruning of Deep Neural Networks with Performance Guarantee](https://arxiv.org/abs/1611.05162) | NeurIPS | `W`  |      **还没看**       |
| [Learning to Prune Deep Neural Networks via Layer-wise Optimal Brain Surgeon](https://arxiv.org/abs/1705.07565) | NeurIPS | `W`  |      **还没看**       |
| [Runtime Neural Pruning](https://papers.NeurIPS.cc/paper/6813-runtime-neural-pruning) | NeurIPS | `F`  |      **还没看**       |
| [Designing Energy-Efficient Convolutional Neural Networks using Energy-Aware Pruning](https://arxiv.org/abs/1611.05128) |  CVPR   | `F`  |      **还没看**       |






---

### 量化 2018

| Title                                                        |   Venue   | Type |      Notion      |
| :----------------------------------------------------------- | :-------: | :--: | :--------------: |
| PACT: Parameterized Clipping Activation for Quantized Neural Networks |  `ICLR`   |      |                  |
| Scalable methods for 8-bit training of neural networks       | `NeurIPS` |      |                  |
| Two-step quantization for low-bit neural networks            |  `CVPR`   |      |                  |
| Quantization and Training of Neural Networks for Efﬁcient Integer-Arithmetic-Only Inference |  `CVPR`   |      | **QAT和fold Bn** |
| Joint training of low-precision neural network with quantization interval Parameters | `NeurIPS` |      |     samsung      |
| Lq-nets Learned quantization for highly accurate and compact deep neural networks |  `ECCV`   |      |                  |
|                                                              |           |      |                  |
|                                                              |           |      |                  |
|                                                              |           |      |                  |



---

### 剪枝 2018

| Title                                                        |  Venue  | Type |   Notion   |
| :----------------------------------------------------------- | :-----: | :--: | :--------: |
| [Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://arxiv.org/abs/1802.00124) |  ICLR   | `F`  |            |
| [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878) |  ICLR   | `w`  | **还没看** |
| [Discrimination-aware Channel Pruning for Deep Neural Networks](https://arxiv.org/abs/1810.11809) | NeurIPS | `F`  | **还没看** |
| [Amc: Automl for model compression and acceleration on mobile devices](https://arxiv.org/abs/1802.03494) |  ECCV   | `F`  | **还没看** |
| [Coreset-Based Neural Network Compression](https://arxiv.org/abs/1807.09810) |  ECCV   | `F`  | **还没看** |
| [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://arxiv.org/abs/1711.05769) |  CVPR   | `F`  | **还没看** |
| [NISP: Pruning Networks using Neuron Importance Score Propagation](https://arxiv.org/abs/1711.05908) |  CVPR   | `F`  | **还没看** |
| [Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1808.06866) |  IJCAI  | `F`  | **还没看** |
| [Accelerating Convolutional Networks via Global & Dynamic Filter Pruning](https://www.ijcai.org/proceedings/2018/0336.pdf) |  IJCAI  | `F`  | **还没看** |



---

### 量化 2019

| Title                                                        |      Venue       | Type | Notion |
| :----------------------------------------------------------- | :--------------: | :--: | :----: |
| ACIQ-Analytical Clipping for Integer Quantization of Neural Networks |      `ICLR`      |      |        |
| OCS-Improving Neural Network  Quantization without Retraining using Outlier Channel Splitting. |      `ICML`      |      |        |
| Data-Free Quantization Through Weight Equalization and Bias Correction | `ICCV`**(Oral)** |      |        |
|                                                              |                  |      |        |
|                                                              |                  |      |        |
|                                                              |                  |      |        |
|                                                              |                  |      |        |
|                                                              |                  |      |        |
|                                                              |                  |      |        |

---

### 2019

| Title                                                        |      Venue      | Type |     Notion     |
| :----------------------------------------------------------- | :-------------: | :--: | :------------: |
| [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) | ICLR **(Best)** | `W`  |                |
| [Rethinking the Value of Network Pruning](https://arxiv.org/abs/1810.05270) |      ICLR       | `F`  |                |
| [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://arxiv.org/abs/1811.00250) | CVPR **(Oral)** | `F`  | 基于几何平均数 |
| [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](https://arxiv.org/abs/1904.03837) |      CVPR       | `F`  |                |
| [Network Pruning via Transformable Architecture Search](https://arxiv.org/abs/1905.09717) |     NeurIPS     | `F`  |                |
| [Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1909.08174) |     NeurIPS     | `F`  |                |
| [Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask](https://arxiv.org/abs/1905.01067) |     NeurIPS     | `W`  |                |
| [One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers](https://arxiv.org/abs/1906.02773) |     NeurIPS     | `W`  |                |
| [AutoPrune: Automatic Network Pruning by Regularizing Auxiliary Parameters](https://papers.nips.cc/paper/9521-autoprune-automatic-network-pruning-by-regularizing-auxiliary-parameters) |     NeurIPS     | `W`  |                |
| [Accelerate CNN via Recursive Bayesian Pruning](https://arxiv.org/abs/1812.00353) |      ICCV       | `F`  |                |
| [Structured Pruning of Neural Networks with Budget-Aware Regularization](https://arxiv.org/abs/1811.09332) |      CVPR       | `F`  |                |
| [Importance Estimation for Neural Network Pruning](http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf) |      CVPR       | `F`  |                |
| [Dynamic Channel Pruning: Feature Boosting and Suppression](https://arxiv.org/abs/1810.05331) |      ICLR       | `F`  |                |
| [Collaborative Channel Pruning for Deep Networks](http://proceedings.mlr.press/v97/peng19c.html) |      ICML       | `F`  |                |
| [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization github](https://arxiv.org/abs/1905.04748) |      ICML       | `F`  |                |



---

### 量化 2020

| Title                                                        |      Venue       | Type |      Notion      |
| :----------------------------------------------------------- | :--------------: | :--: | :--------------: |
| Precision Gating Improving Neural Network Efficiency with Dynamic Dual-Precision Activations |      `ICLR`      |      |                  |
| Post-training Quantization with Multiple Points  Mixed Precision without Mixed Precision |      `ICML`      |      |                  |
| Towards Unified INT8 Training for Convolutional Neural Network |      `CVPR`      |      |    商汤bp+qat    |
| APoT-addive powers-of-two quantization an efficient non-uniform discretization for neural networks |      `ICLR`      |      | 非线性量化scheme |
| Post-Training Piecewise Linear Quantization for Deep Neural Networks | `ECCV`**(oral)** |      |                  |
| Training Quantized Neural Networks With a Full-Precision Auxiliary Module. | `CVPR`**(oral)** |      |                  |
|                                                              |                  |      |                  |
|                                                              |                  |      |                  |
|                                                              |                  |      |                  |



---

### 剪枝 2020

| Title                                                        |        Venue         |  Type   |                             Code                             |
| :----------------------------------------------------------- | :------------------: | :-----: | :----------------------------------------------------------: |
| [EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning](https://arxiv.org/abs/2007.02491) |   ECCV **(Oral)**    |   `F`   | [PyTorch(Author)](https://github.com/anonymous47823493/EagleEye) |
| [DSA: More Efficient Budgeted Pruning via Differentiable Sparsity Allocation](https://arxiv.org/abs/2004.02164) |         ECCV         |   `F`   |                              -                               |
| [DHP: Differentiable Meta Pruning via HyperNetworks](https://arxiv.org/abs/2003.13683) |         ECCV         |   `F`   |     [PyTorch(Author)](https://github.com/ofsoundof/dhp)      |
| [Towards Efficient Model Compression via Learned Global Ranking](https://arxiv.org/abs/1904.12368) |   CVPR  **(Oral)**   |   `F`   |     [Pytorch(Author)](https://github.com/cmu-enyac/LeGR)     |
| [HRank: Filter Pruning using High-Rank Feature Map](https://arxiv.org/abs/2002.10179) |   CVPR **(Oral)**    |   `F`   |      [Pytorch(Author)](https://github.com/lmbxmu/HRank)      |
| [Differentiable Joint Pruning and Quantization for Hardware Efficiency](https://arxiv.org/abs/2007.10463) |         ECCV         | `Other` |                              -                               |
| [Channel Pruning via Automatic Structure Search](https://arxiv.org/abs/2001.08565) |        IJCAI         |   `F`   |    [PyTorch(Author)](https://github.com/lmbxmu/ABCPruner)    |
| [Proving the Lottery Ticket Hypothesis: Pruning is All You Need](https://arxiv.org/abs/2002.00585) |         ICML         |   `W`   |                              -                               |
| [Soft Threshold Weight Reparameterization for Learnable Sparsity](https://arxiv.org/abs/2002.03231) |         ICML         |  `WF`   |      [Pytorch(Author)](https://github.com/RAIVNLab/STR)      |
| [Network Pruning by Greedy Subnetwork Selection](https://arxiv.org/abs/2003.01794) |         ICML         |   `F`   |                              -                               |
| [Operation-Aware Soft Channel Pruning using Differentiable Masks](https://arxiv.org/abs/2007.03938) |         ICML         |   `F`   |                              -                               |
| [DropNet: Reducing Neural Network Complexity via Iterative Pruning](https://proceedings.icml.cc/static/paper_files/icml/2020/2026-Paper.pdf) |         ICML         |   `F`   |                              -                               |
| [Neural Network Pruning with Residual-Connections and Limited-Data](https://arxiv.org/abs/1911.08114) |   CVPR **(Oral)**    |   `F`   |                              -                               |
| [Multi-Dimensional Pruning: A Unified Framework for Model Compression](http://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Multi-Dimensional_Pruning_A_Unified_Framework_for_Model_Compression_CVPR_2020_paper.html) |   CVPR **(Oral)**    |  `WF`   |                              -                               |
| [DMCP: Differentiable Markov Channel Pruning for Neural Networks](https://arxiv.org/abs/2005.03354) |   CVPR **(Oral)**    |   `F`   |      [TensorFlow(Author)](https://github.com/zx55/dmcp)      |
| [Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression](https://arxiv.org/abs/2003.08935) |         CVPR         |   `F`   | [PyTorch(Author)](https://github.com/ofsoundof/group_sparsity) |
| [Few Sample Knowledge Distillation for Efficient Network Compression](https://arxiv.org/abs/1812.01839) |         CVPR         |   `F`   |                              -                               |
| [Discrete Model Compression With Resource Constraint for Deep Neural Networks](http://openaccess.thecvf.com/content_CVPR_2020/html/Gao_Discrete_Model_Compression_With_Resource_Constraint_for_Deep_Neural_Networks_CVPR_2020_paper.html) |         CVPR         |   `F`   |                              -                               |
| [Structured Compression by Weight Encryption for Unstructured Pruning and Quantization](https://arxiv.org/abs/1905.10138) |         CVPR         |   `W`   |                              -                               |
| [Learning Filter Pruning Criteria for Deep Convolutional Neural Networks Acceleration](http://openaccess.thecvf.com/content_CVPR_2020/html/He_Learning_Filter_Pruning_Criteria_for_Deep_Convolutional_Neural_Networks_Acceleration_CVPR_2020_paper.html) |         CVPR         |   `F`   |                              -                               |
| [APQ: Joint Search for Network Architecture, Pruning and Quantization Policy](https://arxiv.org/abs/2006.08509l) |         CVPR         |   `F`   |                              -                               |
| [Comparing Rewinding and Fine-tuning in Neural Network Pruning](https://arxiv.org/abs/2003.02389) |   ICLR **(Oral)**    |  `WF`   | [TensorFlow(Author)](https://github.com/lottery-ticket/rewinding-iclr20-public) |
| [A Signal Propagation Perspective for Pruning Neural Networks at Initialization](https://arxiv.org/abs/1906.06307) | ICLR **(Spotlight)** |   `W`   |                              -                               |
| [ProxSGD: Training Structured Neural Networks under Regularization and Constraints](https://openreview.net/forum?id=HygpthEtvr) |         ICLR         |   `W`   |     [TF+PT(Author)](https://github.com/optyang/proxsgd)      |
| [One-Shot Pruning of Recurrent Neural Networks by Jacobian Spectrum Evaluation](https://arxiv.org/abs/1912.00120) |         ICLR         |   `W`   |                              -                               |
| [Lookahead: A Far-sighted Alternative of Magnitude-based Pruning](https://arxiv.org/abs/2002.04809) |         ICLR         |   `W`   | [PyTorch(Author)](https://github.com/alinlab/lookahead_pruning) |
| [Dynamic Model Pruning with Feedback](https://openreview.net/forum?id=SJem8lSFwB) |         ICLR         |  `WF`   |                              -                               |
| [Provable Filter Pruning for Efficient Neural Networks](https://arxiv.org/abs/1911.07412) |         ICLR         |   `F`   |                              -                               |
| [Data-Independent Neural Pruning via Coresets](https://arxiv.org/abs/1907.04018) |         ICLR         |   `W`   |                              -                               |
| [AutoCompress: An Automatic DNN Structured Pruning Framework for Ultra-High Compression Rates](https://arxiv.org/abs/1907.03141) |         AAAI         |   `F`   |                              -                               |
| [DARB: A Density-Aware Regular-Block Pruning for Deep Neural Networks](http://arxiv.org/abs/1911.08020) |         AAAI         | `Other` |                              -                               |
| [Pruning from Scratch](http://arxiv.org/abs/1909.12579)      |         AAAI         | `Other` |                              -                               |



---

### 蒸馏 2020

| Title                                                        |      Venue       | Type | Notion |
| :----------------------------------------------------------- | :--------------: | :--: | :----: |
| Neural Networks Are More Productive Teachers Than Human Raters: Active Mixup for Data-Efficient Knowledge Distillation From a Blackbox Model. | `CVPR`**(oral)** |      |        |
|                                                              |                  |      |        |
|                                                              |                  |      |        |
|                                                              |                  |      |        |
|                                                              |                  |      |        |
|                                                              |                  |      |        |
|                                                              |                  |      |        |
|                                                              |                  |      |        |
|                                                              |                  |      |        |



### Pruning by Heyang

https://github.com/he-y/Awesome-Pruning#2018



### Papers-Lottery Ticket Hypothesis (LTH)

- 2019-ICLR-[The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://openreview.net/forum?id=rJl-b3RcF7) (best paper!)
- 2019-NIPS-[Deconstructing lottery tickets: Zeros, signs, and the supermask](https://papers.nips.cc/paper/2019/hash/1113d7a76ffceca1bb350bfe145467c6-Abstract.html)
- 2019-NIPS-[One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers](https://papers.nips.cc/paper/2019/hash/a4613e8d72a61b3b69b32d040f89ad81-Abstract.html)
- 2020-ICLR-[GraSP: Picking Winning Tickets Before Training By Preserving Gradient Flow](https://openreview.net/pdf?id=SkgsACVKPH) [[Code](https://github.com/alecwangcq/GraSP)]
- 2020-ICLR-[Playing the lottery with rewards and multiple languages: Lottery tickets in rl and nlp](https://openreview.net/forum?id=S1xnXRVFwH)
- 2020-ICLR-[The Early Phase of Neural Network Training](https://openreview.net/forum?id=Hkl1iRNFwS)
- 2020-[The Sooner The Better: Investigating Structure of Early Winning Lottery Tickets](https://openreview.net/forum?id=BJlNs0VYPB)
- 2020-ICML-[Proving the Lottery Ticket Hypothesis: Pruning is All You Need](https://arxiv.org/abs/2002.00585)
- 2020-ICML-[Rigging the Lottery: Making All Tickets Winners](https://arxiv.org/abs/1911.11134) [[Code](https://github.com/google-research/rigl)]
- 2020-ICML-[Linear Mode Connectivity and the Lottery Ticket Hypothesis](https://arxiv.org/abs/1912.05671)
- 2020-ICML-[Finding trainable sparse networks through neural tangent transfer](https://arxiv.org/abs/2006.08228)
- 2020-NIPS-[Sanity-Checking Pruning Methods: Random Tickets can Win the Jackpot](https://proceedings.neurips.cc//paper/2020/hash/eae27d77ca20db309e056e3d2dcd7d69-Abstract.html)
- 2020-ICLRo-[Comparing Rewinding and Fine-tuning in Neural Network Pruning](https://openreview.net/forum?id=S1gSj0NKvB) [[Code](https://github.com/lottery-ticket/rewinding-iclr20-public)]
- 2020-NIPS-[Logarithmic Pruning is All You Need](https://papers.nips.cc/paper/2020/hash/1e9491470749d5b0e361ce4f0b24d037-Abstract.html)
- 2020-NIPS-[Winning the Lottery with Continuous Sparsification](https://papers.nips.cc/paper/2020/hash/83004190b1793d7aa15f8d0d49a13eba-Abstract.html)
- 2020.2-[Calibrate and Prune: Improving Reliability of Lottery Tickets Through Prediction Calibration](https://arxiv.org/abs/2002.03875)

### Papers-Bayesian Compression

- 1995-Neural Computation-[Bayesian Regularisation and Pruning using a Laplace Prior](https://www.researchgate.net/profile/Peter_Williams19/publication/2719575_Bayesian_Regularisation_and_Pruning_using_a_Laplace_Prior/links/58fde123aca2728fa70f6aab/Bayesian-Regularisation-and-Pruning-using-a-Laplace-Prior.pdf)
- 1997-Neural Networks-[Regularization with a Pruning Prior](https://www.sciencedirect.com/science/article/pii/S0893608097000270?casa_token=sLb4dFBnyH8AAAAA:a9WwAAoYl5CgLepZGXjZ5DKQ4YBEjINgGd7Jl2bPHqrbhIWZHso-uC_gpL-85JmdxG7g8x71)
- 2015-NIPS-[Bayesian dark knowledge](http://papers.nips.cc/paper/5965-bayesian-dark-knowledge.pdf)
- 2017-NIPS-[Bayesian Compression for Deep Learning](http://papers.nips.cc/paper/6921-bayesian-compression-for-deep-learning) [[Code](https://github.com/KarenUllrich/Tutorial_BayesianCompressionForDL)]
- 2017-ICML-[Variational dropout sparsifies deep neural networks](https://arxiv.org/pdf/1701.05369.pdf)
- 2017-NIPSo-[Structured Bayesian Pruning via Log-Normal Multiplicative Noise](http://papers.nips.cc/paper/7254-structured-bayesian-pruning-via-log-normal-multiplicative-noise)
- 2017-ICMLw-[Bayesian Sparsification of Recurrent Neural Networks](https://arxiv.org/abs/1708.00077)
- 2020-NIPS-[Bayesian Bits: Unifying Quantization and Pruning](https://papers.nips.cc/paper/2020/hash/3f13cf4ddf6fc50c0d39a1d5aeb57dd8-Abstract.html)
