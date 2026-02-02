# arXiv AI Daily Paper Report
Date: 2026-02-02

Total Papers: 116

---

### 1. cs.CV - VideoGPA: Distilling Geometry Priors for 3D-Consistent Video Generation

**Authors:** Hongyang Du, Junjie Ye, Xiaoyan Cong, Runhao Li, Jingcheng Ni, Aman Agarwal, Zeqi Zhou, Zekun Li, Randall Balestriero, Yue Wang

**Date:** 2026-01-30

**Summary:** While recent video diffusion models (VDMs) produce visually impressive results, they fundamentally struggle to maintain 3D structural consistency, often resulting in object deformation or spatial drift. We hypothesize that these failures arise because standard denoising objectives lack explicit incentives for geometric coherence. To address this, we introduce VideoGPA (Video Geometric Preference A...

**Plain Summary:** While recent video diffusion models (VDMs) produce visually impressive results, they fundamentally struggle to maintain 3D structural consistency, often resulting in object deformation or spatial drift. We hypothesize that these failures arise because standard denoising objectives lack explicit incentives for geometric coherence. To address this, we introduce VideoGPA (Video Geometric Preference Alignment), a data-速度快、资源消耗少（undefined） self-supervised 提供结构的基础代码库（undefined） that leverages a geometry foundation model to automatically derive dense preference signals that guide VDMs via Direct Preference 寻找最佳参数或解决方案的过程（undefined） (DPO). This approach effectively steers the generative distribution toward inherent 3D consistency without requiring human annotations. VideoGPA significantly enhances temporal stability, physical plausibility, and motion coherence using minimal preference pairs, consistently outperforming 当前最好的、领先的方法（undefined） baselines in 大量实验.

**One Sentence:** 【cs.CV】Hongyang Du等VideoGPA，使用To address this, we introduce ...，在cs.CV取得新进展。

**Contributions:**
- To address this, we introduce VideoGPA (Video Geometric Preference Alignment), a data-速度快、资源消耗少（undefined） self-supervised 提供结构的基础代码库（undefined） that leverages a geometry foundation model to automatically derive dense preference signals that guide VDMs via Direct Preference 寻找最佳参数或解决方案的过程（undefined） (DPO)

**Links:** [Abstract](https://arxiv.org/abs/2601.23286v1) | [PDF](https://arxiv.org/pdf/2601.23286v1)

---

### 2. cs.RO - End-to-end 寻找最佳参数或解决方案的过程（undefined） of Belief and Policy Learning in Shared Autonomy Paradigms

**Authors:** MH Farhadi, Ali Rabiee, Sima Ghafoori, Anna Cetera, Andrew Fisher, Reza Abiri

**Date:** 2026-01-30

**Summary:** Shared autonomy systems require principled methods for inferring user intent and determining appropriate assistance levels. This is a central challenge in human-robot interaction, where systems must be successful while being mindful of user agency. Previous approaches relied on static blending ratios or separated goal inference from assistance arbitration, leading to suboptimal performance in unst...

**Plain Summary:** Shared autonomy systems require principled methods for inferring user intent and determining appropriate assistance levels. This is a central challenge in human-robot interaction, where systems must be successful while being mindful of user agency. Previous approaches relied on static blending ratios or separated goal inference from assistance arbitration, leading to suboptimal performance in unstructured environments. We introduce BRACE (Bayesian Reinforcement Assistance with Context Encoding), a 创新的、前人未做过的（undefined） 提供结构的基础代码库（undefined） that fine-tunes Bayesian intent inference and context-adaptive assistance through an architecture enabling end-to-end gradient flow between intent inference and assistance arbitration. Our 数据处理或模型训练的完整流程（undefined） conditions collaborative control policies on environmental context and complete goal probability distributions. We provide analysis showing (1) optimal assistance levels should decrease with goal uncertainty and increase with environmental constraint severity, and (2) integrating belief information into policy learning yields a quadratic expected regret advantage over sequential approaches. We validated our algorithm against SOTA methods (IDA, DQN) using a three-part evaluation progressively isolating distinct challenges of end-effector control: (1) core human-interaction dynamics in a 2D human-in-the-loop cursor task, (2) non-linear dynamics of a robotic arm, and (3) integrated manipulation under goal ambiguity and environmental constraints. We demonstrate improvements over SOTA, achieving 6.3% higher success rates and 41% increased path efficiency, and 36.3% success rate and 87% path efficiency improvement over unassisted control. Our results confirmed that integrated 寻找最佳参数或解决方案的过程（undefined） is most beneficial in complex, goal-ambiguous scenarios, and is 能够适用于新场景（undefined） across robotic domains requiring goal-directed assistance, advancing the SOTA for adaptive shared autonomy.

**One Sentence:** 【cs.RO】MH Farhadi等End-to-end Optimization of Belief and Policy Learning in Shared Autonomy Paradigms，使用We validated our algorithm aga...，在cs.RO取得新进展。

**Contributions:**
- We introduce BRACE (Bayesian Reinforcement Assistance with Context Encoding), a 创新的、前人未做过的（undefined） 提供结构的基础代码库（undefined） that fine-tunes Bayesian intent inference and context-adaptive assistance through an architecture enabling end-to-end gradient flow between intent inference and assistance arbitration

**Links:** [Abstract](https://arxiv.org/abs/2601.23285v1) | [PDF](https://arxiv.org/pdf/2601.23285v1)

---

### 3. cs.CV - User Prompting Strategies and Prompt Enhancement Methods for Open-Set 在图像中识别和定位特定物体（undefined） in XR Environments

**Authors:** Junfeng Lin, Yanming Xiu, Maria Gorlatova

**Date:** 2026-01-30

**Summary:** Open-set 在图像中识别和定位特定物体（undefined） (OSOD) localizes objects while identifying and rejecting unknown classes at inference. While recent OSOD models perform well on benchmarks, their behavior under realistic user prompting remains underexplored. In interactive XR settings, user-generated prompts are often ambiguous, underspecified, or overly detailed. To study prompt-conditioned robustness, we evalua...

**Plain Summary:** Open-set 在图像中识别和定位特定物体（undefined） (OSOD) localizes objects while identifying and rejecting unknown classes at inference. While recent OSOD models perform well on benchmarks, their behavior under realistic user prompting remains underexplored. In interactive XR settings, user-generated prompts are often ambiguous, underspecified, or overly detailed. To study prompt-conditioned robustness, we evaluate two OSOD models, GroundingDINO and YOLO-E, on real-world XR images and simulate diverse user prompting behaviors using vision-language models. We consider four prompt types: standard, underdetailed, overdetailed, and pragmatically ambiguous, and examine the impact of two enhancement strategies on these prompts. Results show that both models exhibit stable performance under underdetailed and standard prompts, while they suffer degradation under ambiguous prompts. Overdetailed prompts primarily affect GroundingDINO. Prompt enhancement substantially improves robustness under ambiguity, yielding gains exceeding 55% mIoU and 41% average confidence. Based on the findings, 我们提出 several prompting strategies and prompt enhancement methods for OSOD models in XR environments.

**One Sentence:** 【cs.CV】Junfeng Lin等User Prompting Strategies and Prompt Enhancement Methods for Open-Set Object Detection in XR Environments，使用To study prompt-conditioned ro...，在cs.CV取得新进展。

**Contributions:**
- Based on the findings, we propose several prompting strategies and prompt enhancement methods for OSOD models in XR environments

**Links:** [Abstract](https://arxiv.org/abs/2601.23281v1) | [PDF](https://arxiv.org/pdf/2601.23281v1)

---

### 4. cs.LG - Decoupled Diffusion Sampling for Inverse Problems on Function Spaces

**Authors:** Thomas Y. L. Lin, Jiachen Yao, Lufang Chiang, Julius Berner, Anima Anandkumar

**Date:** 2026-01-30

**Summary:** We propose a data-速度快、资源消耗少（undefined）, physics-aware generative 提供结构的基础代码库（undefined） in function space for inverse PDE problems. Existing plug-and-play diffusion 观察到数据后的概率（undefined） samplers represent physics implicitly through joint coefficient-solution modeling, requiring substantial paired supervision. In contrast, our Decoupled Diffusion Inverse Solver (DDIS) employs a decoupled design: an ...

**Plain Summary:** 我们提出 a data-速度快、资源消耗少（undefined）, physics-aware generative 提供结构的基础代码库（undefined） in function space for inverse PDE problems. Existing plug-and-play diffusion 观察到数据后的概率（undefined） samplers represent physics implicitly through joint coefficient-solution modeling, requiring substantial paired supervision. In contrast, our Decoupled Diffusion Inverse Solver (DDIS) employs a decoupled design: an unconditional diffusion learns the coefficient 观察到数据前的概率（undefined）, while a neural operator explicitly models the forward PDE for guidance. This decoupling enables superior data efficiency and effective physics-informed learning, while naturally supporting Decoupled Annealing 观察到数据后的概率（undefined） Sampling (DAPS) to avoid over-smoothing in Diffusion 观察到数据后的概率（undefined） Sampling (DPS). Theoretically, we prove that DDIS avoids the guidance attenuation failure of joint models when training data is scarce. Empirically, DDIS achieves 当前最好的、领先的方法（undefined） performance under sparse observation, improving error by 11% and spectral error by 54% on average; when data is limited to 1%, DDIS maintains 正确预测占总预测的比例（undefined） with 40% advantage in error compared to joint models.

**One Sentence:** 【cs.LG】Thomas Y. L. Lin等Decoupled Diffusion Sampling for Inverse Problems on Function Spaces，使用In contrast, our Decoupled Dif...，在cs.LG取得新进展。

**Contributions:**
- We propose a data-速度快、资源消耗少（undefined）, physics-aware generative 提供结构的基础代码库（undefined） in function space for inverse PDE problems
- Existing plug-and-play diffusion 观察到数据后的概率（undefined） samplers represent physics implicitly through joint coefficient-solution modeling, requiring substantial paired supervision

**Links:** [Abstract](https://arxiv.org/abs/2601.23280v1) | [PDF](https://arxiv.org/pdf/2601.23280v1)

---

### 5. cs.LG - FOCUS: DLLMs Know How to Tame Their Compute Bound

**Authors:** Kaihua Liang, Xin Tan, An Zhong, Hong Xu, Marco Canini

**Date:** 2026-01-30

**Summary:** Diffusion Large Language Models (DLLMs) offer a compelling alternative to Auto-Regressive models, but their deployment is constrained by high decoding cost. In this work, we identify a key inefficiency in DLLM decoding: while computation is parallelized over token blocks, only a small subset of tokens is decodable at each diffusion step, causing most compute to be wasted on non-decodable tokens. W...

**Plain Summary:** Diffusion Large Language Models (DLLMs) offer a compelling alternative to Auto-Regressive models, but their deployment is constrained by high decoding cost. In this work, we identify a key inefficiency in DLLM decoding: while computation is parallelized over token blocks, only a small subset of tokens is decodable at each diffusion step, causing most compute to be wasted on non-decodable tokens. We further observe a strong correlation between attention-derived token importance and token-wise decoding probability. Based on this insight, 我们提出 FOCUS -- an inference system designed for DLLMs. By dynamically focusing computation on decodable tokens and evicting non-decodable ones on-the-fly, FOCUS increases the effective batch size, alleviating compute limitations and enabling 能够处理更大规模数据（undefined） throughput. 基于实验和观察的（undefined） evaluations demonstrate that FOCUS achieves up to 3.52 throughput improvement over the production-grade engine LMDeploy, while preserving or improving generation quality across multiple benchmarks. The FOCUS system is publicly available on GitHub: https://github.com/sands-lab/FOCUS.

**One Sentence:** 【cs.LG】Kaihua Liang等FOCUS，使用In this work, we identify a ke...，在cs.LG取得新进展。

**Contributions:**
- Based on this insight, we propose FOCUS -- an inference system designed for DLLMs

**Links:** [Abstract](https://arxiv.org/abs/2601.23278v1) | [PDF](https://arxiv.org/pdf/2601.23278v1)

---

### 6. astro-ph.IM - Denoising the Deep Sky: Physics-Based CCD Noise Formation for Astronomical Imaging

**Authors:** Shuhong Liu, Xining Ge, Ziying Gu, Lin Gu, Ziteng Cui, Xuangeng Chu, Jun Liu, Dong Li, Tatsuya Harada

**Date:** 2026-01-30

**Summary:** Astronomical imaging remains noise-limited under practical observing constraints, while standard calibration pipelines mainly remove structured artifacts and leave stochastic noise largely unresolved. Learning-based denoising is promising, yet progress is hindered by scarce paired training data and the need for physically 能够解释其决策过程（undefined） and reproducible models in scientific workflows. We pro...

**Plain Summary:** Astronomical imaging remains noise-limited under practical observing constraints, while standard calibration pipelines mainly remove structured artifacts and leave stochastic noise largely unresolved. Learning-based denoising is promising, yet progress is hindered by scarce paired training data and the need for physically 能够解释其决策过程（undefined） and reproducible models in scientific workflows. 我们提出 a physics-based noise synthesis 提供结构的基础代码库（undefined） tailored to CCD noise formation. The 数据处理或模型训练的完整流程（undefined） models photon shot noise, photo-response non-uniformity, dark-current noise, readout effects, and localized outliers arising from cosmic-ray hits and hot pixels. To obtain low-noise inputs for synthesis, we average multiple unregistered exposures to produce high-SNR bases. Realistic noisy counterparts synthesized from these bases using our noise model enable the construction of abundant paired datasets for 使用标注数据训练模型（undefined）. We further introduce a real-world dataset across multi-bands acquired with two twin ground-based telescopes, providing paired raw frames and instrument-数据处理或模型训练的完整流程（undefined） calibrated frames, together with calibration data and stacked high-SNR bases for real-world evaluation.

**One Sentence:** 【astro-ph.IM】Shuhong Liu等Denoising the Deep Sky，使用Realistic noisy counterparts s...，在astro-ph.IM取得新进展。

**Contributions:**
- We propose a physics-based noise synthesis 提供结构的基础代码库（undefined） tailored to CCD noise formation
- We further introduce a real-world dataset across multi-bands acquired with two twin ground-based telescopes, providing paired raw frames and instrument-数据处理或模型训练的完整流程（undefined） calibrated frames, together with calibration data and stacked high-SNR bases for real-world evaluation

**Links:** [Abstract](https://arxiv.org/abs/2601.23276v1) | [PDF](https://arxiv.org/pdf/2601.23276v1)

---

### 7. cs.CL - UPA: Unsupervised Prompt Agent via Tree-Based Search and Selection

**Authors:** Siran Peng, Weisong Zhao, Tianyu Fu, Chenxu Zhao, Tianshuo Zhang, Haoyuan Zhang, Xiangyu Zhu, Minghui Wu, Zhen Lei

**Date:** 2026-01-30

**Summary:** Prompt agents have recently emerged as a promising paradigm for automated prompt 寻找最佳参数或解决方案的过程（undefined）, framing refinement as a sequential decision-making problem over a structured prompt space. While this formulation enables the use of advanced planning algorithms, these methods typically assume access to supervised reward signals, which are often unavailable in practical scenarios. In this w...

**Plain Summary:** Prompt agents have recently emerged as a promising paradigm for automated prompt 寻找最佳参数或解决方案的过程（undefined）, framing refinement as a sequential decision-making problem over a structured prompt space. While this formulation enables the use of advanced planning algorithms, these methods typically assume access to supervised reward signals, which are often unavailable in practical scenarios. In this work, 我们提出 UPA, an Unsupervised Prompt Agent that realizes structured search and selection without relying on supervised feedback. Specifically, during search, UPA iteratively constructs an evolving tree structure to navigate the prompt space, guided by fine-grained and order-invariant pairwise comparisons from Large Language Models (LLMs). Crucially, as these local comparisons do not inherently yield a consistent global scale, we decouple systematic prompt exploration from final selection, introducing a two-stage 提供结构的基础代码库（undefined） grounded in the Bradley-Terry-Luce (BTL) model. This 提供结构的基础代码库（undefined） first performs path-wise Bayesian aggregation of local comparisons to filter candidates under uncertainty, followed by global tournament-style comparisons to infer latent prompt quality and identify the optimal prompt. Experiments across multiple tasks demonstrate that UPA consistently outperforms existing prompt 寻找最佳参数或解决方案的过程（undefined） methods, showing that agent-style 寻找最佳参数或解决方案的过程（undefined） remains highly effective even in fully unsupervised settings.

**One Sentence:** 【cs.CL】Siran Peng等UPA，在cs.CL取得新进展。

**Contributions:**
- In this work, we propose UPA, an Unsupervised Prompt Agent that realizes structured search and selection without relying on supervised feedback

**Links:** [Abstract](https://arxiv.org/abs/2601.23273v1) | [PDF](https://arxiv.org/pdf/2601.23273v1)

---

### 8. cs.RO - IRL-DAL: Safe and Adaptive Trajectory Planning for Autonomous Driving via Energy-Guided Diffusion Models

**Authors:** Seyed Ahmad Hosseini Miangoleh, Amin Jalal Aghdasian, Farzaneh Abdollahi

**Date:** 2026-01-30

**Summary:** This paper proposes a 创新的、前人未做过的（undefined） inverse 通过试错学习最佳策略的机器学习方法（undefined） 提供结构的基础代码库（undefined） using a diffusion-based adaptive lookahead planner (IRL-DAL) for autonomous vehicles. Training begins with imitation from an expert finite state machine (FSM) controller to provide a stable initialization. Environment terms are combined with an IRL discriminator signal to align with expert goals....

**Plain Summary:** This paper proposes a 创新的、前人未做过的（undefined） inverse 通过试错学习最佳策略的机器学习方法（undefined） 提供结构的基础代码库（undefined） using a diffusion-based adaptive lookahead planner (IRL-DAL) for autonomous vehicles. Training begins with imitation from an expert finite state machine (FSM) controller to provide a stable initialization. Environment terms are combined with an IRL discriminator signal to align with expert goals. 通过试错学习最佳策略的机器学习方法（undefined） (RL) is then performed with a hybrid reward that combines diffuse environmental feedback and targeted IRL rewards. A conditional diffusion model, which acts as a safety supervisor, plans safe paths. It stays in its lane, avoids obstacles, and moves smoothly. Then, a learnable adaptive mask (LAM) improves perception. It shifts visual attention based on vehicle speed and nearby hazards. After FSM-based imitation, the policy is fine-tuned with Proximal Policy 寻找最佳参数或解决方案的过程（undefined） (PPO). Training is run in the Webots simulator with a two-stage curriculum. A 96\% success rate is reached, and collisions are reduced to 0.05 per 1k steps, marking a new 用于比较性能的标准数据集或方法（undefined） for safe navigation. By applying the proposed approach, the agent not only drives in lane but also handles unsafe conditions at an expert level, increasing robustness.We make our code publicly available.

**One Sentence:** 【cs.RO】Seyed Ahmad Hosseini Miangoleh等IRL-DAL，使用This paper proposes a 创新的、前人未做...，在cs.RO取得新进展。

**Contributions:**
- This paper proposes a 创新的、前人未做过的（undefined） inverse 通过试错学习最佳策略的机器学习方法（undefined） 提供结构的基础代码库（undefined） using a diffusion-based adaptive lookahead planner (IRL-DAL) for autonomous vehicles
- By applying the proposed approach, the agent not only drives in lane but also handles unsafe conditions at an expert level, increasing robustness

**Links:** [Abstract](https://arxiv.org/abs/2601.23266v1) | [PDF](https://arxiv.org/pdf/2601.23266v1)

---

### 9. cs.CL - PaperBanana: Automating Academic Illustration for AI Scientists

**Authors:** Dawei Zhu, Rui Meng, Yale Song, Xiyu Wei, Sujian Li, Tomas Pfister, Jinsung Yoon

**Date:** 2026-01-30

**Summary:** Despite rapid advances in autonomous AI scientists powered by language models, generating publication-ready illustrations remains a labor-intensive bottleneck in the research workflow. To lift this burden, we introduce PaperBanana, an agentic 提供结构的基础代码库（undefined） for automated generation of publication-ready academic illustrations. Powered by 当前最好的、领先的方法（undefined） VLMs and image generation model...

**Plain Summary:** Despite rapid advances in autonomous AI scientists powered by language models, generating publication-ready illustrations remains a labor-intensive bottleneck in the research workflow. To lift this burden, we introduce PaperBanana, an agentic 提供结构的基础代码库（undefined） for automated generation of publication-ready academic illustrations. Powered by 当前最好的、领先的方法（undefined） VLMs and image generation models, PaperBanana orchestrates specialized agents to retrieve references, plan content and style, render images, and iteratively refine via self-critique. To rigorously evaluate our 提供结构的基础代码库（undefined）, we introduce PaperBananaBench, comprising 292 test cases for methodology diagrams curated from NeurIPS 2025 publications, covering diverse research domains and illustration styles. 覆盖广泛的、详细的（undefined） experiments demonstrate that PaperBanana consistently outperforms leading baselines in faithfulness, conciseness, readability, and aesthetics. We further show that 我们的方法 effectively extends to the generation of high-quality statistical plots. Collectively, PaperBanana paves the way for the automated generation of publication-ready illustrations.

**One Sentence:** 【cs.CL】Dawei Zhu等PaperBanana，在cs.CL取得新进展。

**Contributions:**
- To lift this burden, we introduce PaperBanana, an agentic 提供结构的基础代码库（undefined） for automated generation of publication-ready academic illustrations
- To rigorously evaluate our 提供结构的基础代码库（undefined）, we introduce PaperBananaBench, comprising 292 test cases for methodology diagrams curated from NeurIPS 2025 publications, covering diverse research domains and illustration styles

**Links:** [Abstract](https://arxiv.org/abs/2601.23265v1) | [PDF](https://arxiv.org/pdf/2601.23265v1)

---

### 10. cs.LG - Particle-Guided Diffusion Models for Partial Differential Equations

**Authors:** Andrew Millard, Fredrik Lindsten, Zheng Zhao

**Date:** 2026-01-30

**Summary:** We introduce a guided stochastic sampling method that augments sampling from diffusion models with physics-based guidance derived from partial differential equation (PDE) residuals and observational constraints, ensuring generated samples remain physically admissible. We embed this sampling procedure within a new Sequential Monte Carlo (SMC) 提供结构的基础代码库（undefined）, yielding a 能够处理更大规模数据（undefined） ...

**Plain Summary:** We introduce a guided stochastic sampling method that augments sampling from diffusion models with physics-based guidance derived from partial differential equation (PDE) residuals and observational constraints, ensuring generated samples remain physically admissible. We embed this sampling procedure within a new Sequential Monte Carlo (SMC) 提供结构的基础代码库（undefined）, yielding a 能够处理更大规模数据（undefined） generative PDE solver. Across multiple 用于比较性能的标准数据集或方法（undefined） PDE systems as well as multiphysics and interacting PDE systems, 我们的方法 produces solution fields with lower numerical error than existing 当前最好的、领先的方法（undefined） generative methods.

**One Sentence:** 【cs.LG】Andrew Millard等Particle-Guided Diffusion Models for Partial Differential Equations，在cs.LG取得新进展。

**Contributions:**
- We introduce a guided stochastic sampling method that augments sampling from diffusion models with physics-based guidance derived from partial differential equation (PDE) residuals and observational constraints, ensuring generated samples remain physically admissible

**Links:** [Abstract](https://arxiv.org/abs/2601.23262v1) | [PDF](https://arxiv.org/pdf/2601.23262v1)

---

### 11. cs.LG - TEON: Tensorized Orthonormalization Beyond Layer-Wise Muon for 基于海量文本训练的语言模型，如GPT（undefined） Pre-Training

**Authors:** Ruijie Zhang, Yequan Zhao, Ziyue Liu, Zhengyang Wang, Dongyang Li, Yupeng Su, Sijia Liu, Zheng Zhang

**Date:** 2026-01-30

**Summary:** The Muon optimizer has demonstrated strong 基于实验和观察的（undefined） performance in pre-training large language models by performing matrix-level gradient (or momentum) orthogonalization in each layer independently. In this work, we propose TEON, a principled generalization of Muon that extends orthogonalization beyond individual layers by modeling the gradients of a 一种受人脑启发的计算模型，由许多互相连接的节点组成（undefined）...

**Plain Summary:** The Muon optimizer has demonstrated strong 基于实验和观察的（undefined） performance in pre-training large language models by performing matrix-level gradient (or momentum) orthogonalization in each layer independently. In this work, 我们提出 TEON, a principled generalization of Muon that extends orthogonalization beyond individual layers by modeling the gradients of a 一种受人脑启发的计算模型，由许多互相连接的节点组成（undefined） as a structured higher-order tensor. We present TEON's improved convergence guarantee over layer-wise Muon, and further develop a practical instantiation of TEON based on the 基于数学推导的（undefined） analysis with corresponding ablation. We evaluate our approach on two widely adopted architectures: GPT-style models, ranging from 130M to 774M parameters, and LLaMA-style models, ranging from 60M to 1B parameters. 实验结果表明 that TEON consistently improves training and validation 语言模型预测能力的度量，越低越好（undefined） across model scales and exhibits strong robustness under various approximate SVD schemes.

**One Sentence:** 【cs.LG】Ruijie Zhang等TEON，使用We present TEON's improved con...，在cs.LG取得新进展。

**Contributions:**
- In this work, we propose TEON, a principled generalization of Muon that extends orthogonalization beyond individual layers by modeling the gradients of a 一种受人脑启发的计算模型，由许多互相连接的节点组成（undefined） as a structured higher-order tensor
- We present TEON's improved convergence guarantee over layer-wise Muon, and further develop a practical instantiation of TEON based on the 基于数学推导的（undefined） analysis with corresponding ablation

**Links:** [Abstract](https://arxiv.org/abs/2601.23261v1) | [PDF](https://arxiv.org/pdf/2601.23261v1)

---

### 12. cs.LG - Agnostic Language Identification and Generation

**Authors:** Mikael Møller Høgsgaard, Chirag Pabbaraju

**Date:** 2026-01-30

**Summary:** Recent works on language identification and generation have established tight statistical rates at which these tasks can be achieved. These works typically operate under a strong realizability assumption: that the input data is drawn from an unknown distribution necessarily supported on some language in a given collection. In this work, we relax this assumption of realizability entirely, and impos...

**Plain Summary:** Recent works on language identification and generation have established tight statistical rates at which these tasks can be achieved. These works typically operate under a strong realizability assumption: that the input data is drawn from an unknown distribution necessarily supported on some language in a given collection. In this work, we relax this assumption of realizability entirely, and impose no restrictions on the distribution of the input data. 我们提出 objectives to study both language identification and generation in this more general "agnostic" setup. Across both problems, we obtain 创新的、前人未做过的（undefined） interesting characterizations and nearly tight rates.

**One Sentence:** 【cs.LG】Mikael Møller Høgsgaard等Agnostic Language Identification and Generation，在cs.LG取得新进展。

**Contributions:**
- We propose objectives to study both language identification and generation in this more general "agnostic" setup

**Links:** [Abstract](https://arxiv.org/abs/2601.23258v1) | [PDF](https://arxiv.org/pdf/2601.23258v1)

---

### 13. cs.CL - Now You Hear Me: Audio Narrative Attacks Against Large Audio-Language Models

**Authors:** Ye Yu, Haibo Jin, Yaoning Yu, Jun Zhuang, Haohan Wang

**Date:** 2026-01-30

**Summary:** Large audio-language models increasingly operate on raw speech inputs, enabling more seamless integration across domains such as voice assistants, education, and clinical triage. This transition, however, introduces a distinct class of vulnerabilities that remain largely uncharacterized. We examine the security implications of this modality shift by designing a text-to-audio jailbreak that embeds ...

**Plain Summary:** Large audio-language models increasingly operate on raw speech inputs, enabling more seamless integration across domains such as voice assistants, education, and clinical triage. This transition, however, introduces a distinct class of vulnerabilities that remain largely uncharacterized. We examine the security implications of this modality shift by designing a text-to-audio jailbreak that embeds disallowed directives within a narrative-style audio stream. The attack leverages an advanced instruction-following text-to-speech (TTS) model to exploit structural and acoustic properties, thereby circumventing safety mechanisms primarily calibrated for text. When delivered through synthetic speech, the narrative format elicits restricted outputs from 当前最好的、领先的方法（undefined） models, including Gemini 2.0 Flash, achieving a 98.26% success rate that substantially exceeds text-only baselines. These results highlight the need for safety frameworks that jointly reason over linguistic and paralinguistic representations, particularly as speech-based interfaces become more prevalent.

**One Sentence:** 【cs.CL】Ye Yu等Now You Hear Me，使用The attack leverages an advanc...，在cs.CL取得新进展。

**Contributions:**
- This transition, however, introduces a distinct class of vulnerabilities that remain largely uncharacterized
- We examine the security implications of this modality shift by designing a text-to-audio jailbreak that embeds disallowed directives within a narrative-style audio stream

**Links:** [Abstract](https://arxiv.org/abs/2601.23255v1) | [PDF](https://arxiv.org/pdf/2601.23255v1)

---

### 14. cs.CV - Training-Free Test-Time Adaptation with Brownian Distance Covariance in Vision-Language Models

**Authors:** Yi Zhang, Chun-Wun Cheng, Angelica I. Aviles-Rivero, Zhihai He, Liang-Jie Zhang

**Date:** 2026-01-30

**Summary:** Vision-language models suffer performance degradation under domain shift, limiting real-world applicability. Existing test-time adaptation methods are computationally intensive, rely on back-propagation, and often focus on single modalities. To address these issues, we propose Training-free Test-Time Adaptation with Brownian Distance Covariance (TaTa). TaTa leverages Brownian Distance Covariance-a...

**Plain Summary:** Vision-language models suffer performance degradation under domain shift, limiting real-world applicability. Existing test-time adaptation methods are computationally intensive, rely on back-propagation, and often focus on single modalities. To address these issues, 我们提出 Training-free Test-Time Adaptation with Brownian Distance Covariance (TaTa). TaTa leverages Brownian Distance Covariance-a powerful statistical measure that captures both linear and nonlinear dependencies via pairwise distances-to dynamically adapt VLMs to new domains without training or back-propagation. This not only improves efficiency but also enhances stability by avoiding disruptive weight updates. TaTa further integrates attribute-enhanced prompting to improve vision-language inference with descriptive visual cues. Combined with dynamic clustering and pseudo-label refinement, it effectively recalibrates the model for 创新的、前人未做过的（undefined） visual contexts. Experiments across diverse datasets show that TaTa significantly reduces computational cost while achieving 当前最好的、领先的方法（undefined） performance in domain and cross-dataset generalization.

**One Sentence:** 【cs.CV】Yi Zhang等Training-Free Test-Time Adaptation with Brownian Distance Covariance in Vision-Language Models，使用TaTa leverages Brownian Distan...，在cs.CV取得新进展。

**Contributions:**
- To address these issues, we propose Training-free Test-Time Adaptation with Brownian Distance Covariance (TaTa)

**Links:** [Abstract](https://arxiv.org/abs/2601.23253v1) | [PDF](https://arxiv.org/pdf/2601.23253v1)

---

### 15. stat.CO - Nested Slice Sampling: Vectorized Nested Sampling for GPU-Accelerated Inference

**Authors:** David Yallup, Namu Kroupa, Will Handley

**Date:** 2026-01-30

**Summary:** Model comparison and calibrated uncertainty quantification often require integrating over parameters, but 能够处理更大规模数据（undefined） inference can be challenging for complex, multimodal targets. Nested Sampling is a 对噪声和扰动不敏感（undefined） alternative to standard MCMC, yet its typically sequential structure and hard constraints make 速度快、资源消耗少（undefined） accelerator implementations difficult. This paper in...

**Plain Summary:** Model comparison and calibrated uncertainty quantification often require integrating over parameters, but 能够处理更大规模数据（undefined） inference can be challenging for complex, multimodal targets. Nested Sampling is a 对噪声和扰动不敏感（undefined） alternative to standard MCMC, yet its typically sequential structure and hard constraints make 速度快、资源消耗少（undefined） accelerator implementations difficult. This paper introduces Nested Slice Sampling (NSS), a GPU-friendly, vectorized formulation of Nested Sampling that uses Hit-and-Run Slice Sampling for constrained updates. A tuning analysis yields a simple near-optimal rule for setting the slice width, improving high-dimensional behavior and making per-step compute more predictable for parallel execution. Experiments on challenging synthetic targets, high dimensional Bayesian inference, and Gaussian process hyperparameter marginalization show that NSS maintains accurate evidence estimates and high-quality 观察到数据后的概率（undefined） samples, and is particularly 对噪声和扰动不敏感（undefined） on difficult multimodal problems where current 当前最好的、领先的方法（undefined） methods such as tempered SMC baselines can struggle. An open-source implementation is released to facilitate adoption and reproducibility.

**One Sentence:** 【stat.CO】David Yallup等Nested Slice Sampling，在stat.CO取得新进展。

**Contributions:**
- This paper introduces Nested Slice Sampling (NSS), a GPU-friendly, vectorized formulation of Nested Sampling that uses Hit-and-Run Slice Sampling for constrained updates

**Links:** [Abstract](https://arxiv.org/abs/2601.23252v1) | [PDF](https://arxiv.org/pdf/2601.23252v1)

---

### 16. cs.CV - Structured Over Scale: Learning Spatial Reasoning from Educational Video

**Authors:** Bishoy Galoaa, Xiangyu Bai, Sarah Ostadabbas

**Date:** 2026-01-30

**Summary:** Vision-language models (VLMs) demonstrate impressive performance on standard video understanding benchmarks yet fail systematically on simple reasoning tasks that preschool children can solve, including counting, spatial reasoning, and compositional understanding. We hypothesize that the pedagogically-structured content of educational videos provides an ideal training signal for improving these ca...

**Plain Summary:** Vision-language models (VLMs) demonstrate impressive performance on standard video understanding benchmarks yet fail systematically on simple reasoning tasks that preschool children can solve, including counting, spatial reasoning, and compositional understanding. We hypothesize that the pedagogically-structured content of educational videos provides an ideal training signal for improving these capabilities. We introduce DoraVQA, a dataset of 5,344 question-answer pairs automatically extracted from 8 seasons of Dora the Explorer with precise timestamp alignment. Each episode follows a consistent \textit{context-question-pause-answer} structure that creates a self-contained learning environment analogous to interactive tutoring. We fine-tune both Qwen2 and Qwen3 using Group Relative Policy 寻找最佳参数或解决方案的过程（undefined） (GRPO), leveraging the clear correctness signals and structured reasoning traces inherent in educational content. Despite training exclusively on 38 hours of children's educational videos, our approach achieves improvements of 8-14 points on DoraVQA and 当前最好的、领先的方法（undefined） 86.16\% on CVBench, with strong transfer to Video-MME and NExT-QA, demonstrating effective generalization from narrow pedagogical content to broad multimodal understanding. Through cross-domain benchmarks, we show that VLMs can perform tasks that require 对噪声和扰动不敏感（undefined） reasoning learned from structured educational content, suggesting that content structure matters as much as content scale.

**One Sentence:** 【cs.CV】Bishoy Galoaa等Structured Over Scale，使用We fine-tune both Qwen2 and Qw...，在cs.CV取得新进展。

**Contributions:**
- We introduce DoraVQA, a dataset of 5,344 question-answer pairs automatically extracted from 8 seasons of Dora the Explorer with precise timestamp alignment
- Each episode follows a consistent \textit{context-question-pause-answer} structure that creates a self-contained learning environment analogous to interactive tutoring

**Links:** [Abstract](https://arxiv.org/abs/2601.23251v1) | [PDF](https://arxiv.org/pdf/2601.23251v1)

---

### 17. stat.ML - Graph Attention Network for Node Regression on Random Geometric Graphs with Erdős--Rényi contamination

**Authors:** Somak Laha, Suqi Liu, Morgane Austern

**Date:** 2026-01-30

**Summary:** Graph attention networks (GATs) are widely used and often appear 对噪声和扰动不敏感（undefined） to noise in node covariates and edges, yet rigorous statistical guarantees demonstrating a provable advantage of GATs over non-attention graph neural networks~(GNNs) are scarce. We partially address this gap for node regression with graph-based errors-in-variables models under simultaneous covariate and edge corr...

**Plain Summary:** Graph attention networks (GATs) are widely used and often appear 对噪声和扰动不敏感（undefined） to noise in node covariates and edges, yet rigorous statistical guarantees demonstrating a provable advantage of GATs over non-attention graph neural networks~(GNNs) are scarce. We partially address this gap for node regression with graph-based errors-in-variables models under simultaneous covariate and edge corruption: responses are generated from latent node-level covariates, but only noise-perturbed versions of the latent covariates are observed; and the sample graph is a random geometric graph created from the node covariates but contaminated by independent Erdős--Rényi edges. 我们提出 and analyze a carefully designed, task-specific GAT that constructs denoised proxy features for regression. We prove that regressing the response variables on the proxies achieves lower error asymptotically in (a) estimating the regression coefficient compared to the ordinary least squares (OLS) estimator on the noisy node covariates, and (b) predicting the response for an unlabelled node compared to a vanilla graph convolutional network~(GCN) -- under mild growth conditions. Our analysis leverages high-dimensional geometric tail bounds and concentration for neighbourhood counts and sample covariances. We verify our 基于数学推导的（undefined） findings through experiments on synthetically generated data. We also perform experiments on real-world graphs and demonstrate the effectiveness of the 让模型关注输入中最相关部分的技术（undefined） in several node regression tasks.

**One Sentence:** 【stat.ML】Somak Laha等Graph Attention Network for Node Regression on Random Geometric Graphs with Erdős--Rényi contamination，使用Our analysis leverages high-di...，在stat.ML取得新进展。

**Contributions:**
- We partially address this gap for node regression with graph-based errors-in-variables models under simultaneous covariate and edge corruption: responses are generated from latent node-level covariates, but only noise-perturbed versions of the latent covariates are observed; and the sample graph is a random geometric graph created from the node covariates but contaminated by independent Erdős--Rényi edges
- We propose and analyze a carefully designed, task-specific GAT that constructs denoised proxy features for regression

**Links:** [Abstract](https://arxiv.org/abs/2601.23239v1) | [PDF](https://arxiv.org/pdf/2601.23239v1)

---

### 18. cs.LG - How well do generative models solve inverse problems? A 用于比较性能的标准数据集或方法（undefined） study

**Authors:** Patrick Krüger, Patrick Materne, Werner Krebs, Hanno Gottschalk

**Date:** 2026-01-30

**Summary:** Generative learning generates high dimensional data based on low dimensional conditions, also called prompts. Therefore, generative learning algorithms are eligible for solving (Bayesian) inverse problems. In this article we compare a traditional Bayesian inverse approach based on a forward regression model and a 观察到数据前的概率（undefined） sampled with the Markov Chain Monte Carlo method with three stat...

**Plain Summary:** Generative learning generates high dimensional data based on low dimensional conditions, also called prompts. Therefore, generative learning algorithms are eligible for solving (Bayesian) inverse problems. In this article we compare a traditional Bayesian inverse approach based on a forward regression model and a 观察到数据前的概率（undefined） sampled with the Markov Chain Monte Carlo method with three 最先进 generative learning models, namely conditional Generative Adversarial Networks, Invertible Neural Networks and Conditional Flow Matching. We apply them to a problem of gas turbine combustor design where we map six independent design parameters to three performance labels. 我们提出 several metrics for the evaluation of this inverse design approaches and measure the 正确预测占总预测的比例（undefined） of the labels of the generated designs along with the diversity. We also study the performance as a function of the training dataset size. Our 用于比较性能的标准数据集或方法（undefined） has a clear winner, as Conditional Flow Matching consistently outperforms all competing approaches.

**One Sentence:** 【cs.LG】Patrick Krüger等How well do generative models solve inverse problems? A benchmark study，使用Generative learning generates ...，在cs.LG取得新进展。

**Contributions:**
- We apply them to a problem of gas turbine combustor design where we map six independent design parameters to three performance labels
- We propose several metrics for the evaluation of this inverse design approaches and measure the 正确预测占总预测的比例（undefined） of the labels of the generated designs along with the diversity

**Links:** [Abstract](https://arxiv.org/abs/2601.23238v1) | [PDF](https://arxiv.org/pdf/2601.23238v1)

---

### 19. cs.LG - YuriiFormer: A Suite of Nesterov-Accelerated Transformers

**Authors:** Aleksandr Zimin, Yury Polyanskiy, Philippe Rigollet

**Date:** 2026-01-30

**Summary:** We propose a variational 提供结构的基础代码库（undefined） that interprets 一种处理序列数据的神经网络架构，特别擅长处理语言（undefined） layers as iterations of an 寻找最佳参数或解决方案的过程（undefined） algorithm acting on token embeddings. In this view, self-attention implements a gradient step of an interaction energy, while MLP layers correspond to gradient updates of a potential energy. Standard GPT-style transformers emerge as vanilla 通过计算梯度来...

**Plain Summary:** 我们提出 a variational 提供结构的基础代码库（undefined） that interprets 一种处理序列数据的神经网络架构，特别擅长处理语言（undefined） layers as iterations of an 寻找最佳参数或解决方案的过程（undefined） algorithm acting on token embeddings. In this view, self-attention implements a gradient step of an interaction energy, while MLP layers correspond to gradient updates of a potential energy. Standard GPT-style transformers emerge as vanilla 通过计算梯度来最小化损失函数的优化方法（undefined） on the resulting composite objective, implemented via Lie--Trotter splitting between these two energy functionals. This perspective enables principled architectural design using classical 寻找最佳参数或解决方案的过程（undefined） ideas. As a proof of concept, we introduce a Nesterov-style accelerated 一种处理序列数据的神经网络架构，特别擅长处理语言（undefined） that preserves the same attention and MLP oracles. The resulting architecture consistently outperforms a nanoGPT 用于对比的基准方法（undefined） on TinyStories and OpenWebText, demonstrating that 寻找最佳参数或解决方案的过程（undefined）-theoretic insights can translate into practical gains.

**One Sentence:** 【cs.LG】Aleksandr Zimin等YuriiFormer，使用This perspective enables princ...，在cs.LG取得新进展。

**Contributions:**
- We propose a variational 提供结构的基础代码库（undefined） that interprets 一种处理序列数据的神经网络架构，特别擅长处理语言（undefined） layers as iterations of an 寻找最佳参数或解决方案的过程（undefined） algorithm acting on token embeddings
- This perspective enables principled architectural design using classical 寻找最佳参数或解决方案的过程（undefined） ideas

**Links:** [Abstract](https://arxiv.org/abs/2601.23236v1) | [PDF](https://arxiv.org/pdf/2601.23236v1)

---

### 20. cs.LG - Sequence Diffusion Model for Temporal Link Prediction in Continuous-Time Dynamic Graph

**Authors:** Nguyen Minh Duc, Viet Cuong Ta

**Date:** 2026-01-30

**Summary:** Temporal link prediction in dynamic graphs is a fundamental problem in many real-world systems. Existing temporal graph neural networks mainly focus on learning representations of historical interactions. Despite their strong performance, these models are still purely discriminative, producing point estimates for future links and lacking an explicit mechanism to capture the uncertainty and sequent...

**Plain Summary:** Temporal link prediction in dynamic graphs is a fundamental problem in many real-world systems. Existing temporal graph neural networks mainly focus on learning representations of historical interactions. Despite their strong performance, these models are still purely discriminative, producing point estimates for future links and lacking an explicit mechanism to capture the uncertainty and sequential structure of future temporal interactions. In this paper, 我们提出 SDG, a 创新的、前人未做过的（undefined） sequence-level diffusion 提供结构的基础代码库（undefined） that unifies dynamic graph learning with generative denoising. Specifically, SDG injects noise into the entire historical interaction sequence and jointly reconstructs all interaction embeddings through a conditional denoising process, thereby enabling the model to capture more 覆盖广泛的、详细的（undefined） interaction distributions. To align the generative process with temporal link prediction, we employ a cross-attention denoising decoder to guide the reconstruction of the destination sequence and optimize the model in an end-to-end manner. 大量实验 on various temporal graph benchmarks show that SDG consistently achieves 当前最好的、领先的方法（undefined） performance in the temporal link prediction task.

**One Sentence:** 【cs.LG】Nguyen Minh Duc等Sequence Diffusion Model for Temporal Link Prediction in Continuous-Time Dynamic Graph，在cs.LG取得新进展。

**Contributions:**
- Existing temporal graph neural networks mainly focus on learning representations of historical interactions
- In this paper, we propose SDG, a 创新的、前人未做过的（undefined） sequence-level diffusion 提供结构的基础代码库（undefined） that unifies dynamic graph learning with generative denoising

**Links:** [Abstract](https://arxiv.org/abs/2601.23233v1) | [PDF](https://arxiv.org/pdf/2601.23233v1)

---

