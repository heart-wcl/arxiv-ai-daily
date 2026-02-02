# 📚 arXiv AI 每日论文报告
**日期:** 2026年2月2日星期一
**论文数:** 116 篇

---

## 1. cs.CV 👁️

**VideoGPA: Distilling Geometry Priors for 3D-Consistent Video Generation**

👥 **作者:** Hongyang Du, Junjie Ye, Xiaoyan Cong 等10人

📝 **一句话总结:** 【cs.CV】Hongyang Du等VideoGPA，使用To address this, we introduce ...，在cs.CV取得新进展。

📖 **通俗解读:**
While recent video diffusion models (VDMs) produce visually impressive results, they fundamentally struggle to maintain 3D structural consistency, often resulting in object deformation or spatial drift. We hypothesize that these failures arise because standard denoising objectives lack explicit incentives for geometric coherence. To address this, we introduce VideoGPA (Video Geometric Preference Alignment), a data-高效的（速度快、资源消耗少） self-supervised 框架（提供结构的基础代码库） that leverages a geometry foundation model to automatically derive dense preference signals that guide VDMs via Direct Preference 优化（寻找最佳参数或解决方案的过程） (DPO). This approach effectively steers the generative distribution toward inherent 3D consistency without requiring human annotations. VideoGPA significantly enhances temporal stability, physical plausibility, and motion coherence using minimal preference pairs, consistently outperforming 最先进（当前最好的、领先的方法） baselines in 大量实验.

💡 **核心贡献:**
- To address this, we introduce VideoGPA (Video Geometric Preference Alignment), a data-高效的（速度快、资源消耗少） self-supervised 框架（提供结构的基础代码库） that leverages a g...

🔗 **链接:** [论文](https://arxiv.org/abs/2601.23286v1) | [PDF](https://arxiv.org/pdf/2601.23286v1)

---

## 2. cs.RO 🦾

**End-to-end 优化（寻找最佳参数或解决方案的过程） of Belief and Policy Learning in Shared Autonomy Paradigms**

👥 **作者:** MH Farhadi, Ali Rabiee, Sima Ghafoori 等6人

📝 **一句话总结:** 【cs.RO】MH Farhadi等End-to-end Optimization of Belief and Policy Learning in Shared Autonomy Paradigms，使用We validated our algorithm aga...，在cs.RO取得新进展。

📖 **通俗解读:**
Shared autonomy systems require principled methods for inferring user intent and determining appropriate assistance levels. This is a central challenge in human-robot interaction, where systems must be successful while being mindful of user agency. Previous approaches relied on static blending ratios or separated goal inference from assistance arbitration, leading to suboptimal performance in unstructured environments. We introduce BRACE (Bayesian Reinforcement Assistance with Context Encoding), a 新颖的（创新的、前人未做过的） 框架（提供结构的基础代码库） that fine-tunes Bayesian intent inference and context-adaptive assistance through an architecture enabling end-to-end gradient flow between intent inference and assistance arbitration. Our 流程（数据处理或模型训练的完整流程） conditions collaborative control policies on environmental context and complete goal probability distributions. We provide analysis showing (1) optimal assistance levels should decrease with goal uncertainty and increase with environmental constraint severity, and (2) integrating belief information into policy learning yields a quadratic expected regret advantage over sequential approaches. We validated our algorithm against SOTA methods (IDA, DQN) using a three-part evaluation progressively isolating distinct challenges of end-effector control: (1) core human-interaction dynamics in a 2D human-in-the-loop cursor task, (2) non-linear dynamics of a robotic arm, and (3) integrated manipulation under goal ambiguity and environmental constraints. We demonstrate improvements over SOTA, achieving 6.3% higher success rates and 41% increased path efficiency, and 36.3% success rate and 87% path efficiency improvement over unassisted control. Our results confirmed that integrated 优化（寻找最佳参数或解决方案的过程） is most beneficial in complex, goal-ambiguous scenarios, and is 可泛化的（能够适用于新场景） across robotic domains requiring goal-directed assistance, advancing the SOTA for adaptive shared autonomy.

💡 **核心贡献:**
- We introduce BRACE (Bayesian Reinforcement Assistance with Context Encoding), a 新颖的（创新的、前人未做过的） 框架（提供结构的基础代码库） that fine-tunes Bayesian intent inferen...

🔗 **链接:** [论文](https://arxiv.org/abs/2601.23285v1) | [PDF](https://arxiv.org/pdf/2601.23285v1)

---

## 3. cs.CV 👁️

**User Prompting Strategies and Prompt Enhancement Methods for Open-Set 目标检测（在图像中识别和定位特定物体） in XR Environments**

👥 **作者:** Junfeng Lin, Yanming Xiu, Maria Gorlatova

📝 **一句话总结:** 【cs.CV】Junfeng Lin等User Prompting Strategies and Prompt Enhancement Methods for Open-Set Object Detection in XR Environments，使用To study prompt-conditioned ro...，在cs.CV取得新进展。

📖 **通俗解读:**
Open-set 目标检测（在图像中识别和定位特定物体） (OSOD) localizes objects while identifying and rejecting unknown classes at inference. While recent OSOD models perform well on benchmarks, their behavior under realistic user prompting remains underexplored. In interactive XR settings, user-generated prompts are often ambiguous, underspecified, or overly detailed. To study prompt-conditioned robustness, we evaluate two OSOD models, GroundingDINO and YOLO-E, on real-world XR images and simulate diverse user prompting behaviors using vision-language models. We consider four prompt types: standard, underdetailed, overdetailed, and pragmatically ambiguous, and examine the impact of two enhancement strategies on these prompts. Results show that both models exhibit stable performance under underdetailed and standard prompts, while they suffer degradation under ambiguous prompts. Overdetailed prompts primarily affect GroundingDINO. Prompt enhancement substantially improves robustness under ambiguity, yielding gains exceeding 55% mIoU and 41% average confidence. Based on the findings, 我们提出 several prompting strategies and prompt enhancement methods for OSOD models in XR environments.

💡 **核心贡献:**
- Based on the findings, we propose several prompting strategies and prompt enhancement methods for OSOD models in XR environments

🔗 **链接:** [论文](https://arxiv.org/abs/2601.23281v1) | [PDF](https://arxiv.org/pdf/2601.23281v1)

---

## 4. cs.LG 🧠

**Decoupled Diffusion Sampling for Inverse Problems on Function Spaces**

👥 **作者:** Thomas Y. L. Lin, Jiachen Yao, Lufang Chiang 等5人

📝 **一句话总结:** 【cs.LG】Thomas Y. L. Lin等Decoupled Diffusion Sampling for Inverse Problems on Function Spaces，使用In contrast, our Decoupled Dif...，在cs.LG取得新进展。

📖 **通俗解读:**
我们提出 a data-高效的（速度快、资源消耗少）, physics-aware generative 框架（提供结构的基础代码库） in function space for inverse PDE problems. Existing plug-and-play diffusion 后验概率（观察到数据后的概率） samplers represent physics implicitly through joint coefficient-solution modeling, requiring substantial paired supervision. In contrast, our Decoupled Diffusion Inverse Solver (DDIS) employs a decoupled design: an unconditional diffusion learns the coefficient 先验概率（观察到数据前的概率）, while a neural operator explicitly models the forward PDE for guidance. This decoupling enables superior data efficiency and effective physics-informed learning, while naturally supporting Decoupled Annealing 后验概率（观察到数据后的概率） Sampling (DAPS) to avoid over-smoothing in Diffusion 后验概率（观察到数据后的概率） Sampling (DPS). Theoretically, we prove that DDIS avoids the guidance attenuation failure of joint models when training data is scarce. Empirically, DDIS achieves 最先进（当前最好的、领先的方法） performance under sparse observation, improving error by 11% and spectral error by 54% on average; when data is limited to 1%, DDIS maintains 准确率（正确预测占总预测的比例） with 40% advantage in error compared to joint models.

💡 **核心贡献:**
- We propose a data-高效的（速度快、资源消耗少）, physics-aware generative 框架（提供结构的基础代码库） in function space for inverse PDE problems
- Existing plug-and-play diffusion 后验概率（观察到数据后的概率） samplers represent physics implicitly through joint coefficient-solution modeling, requiring substant...

🔗 **链接:** [论文](https://arxiv.org/abs/2601.23280v1) | [PDF](https://arxiv.org/pdf/2601.23280v1)

---

## 5. cs.LG 🧠

**FOCUS: DLLMs Know How to Tame Their Compute Bound**

👥 **作者:** Kaihua Liang, Xin Tan, An Zhong 等5人

📝 **一句话总结:** 【cs.LG】Kaihua Liang等FOCUS，使用In this work, we identify a ke...，在cs.LG取得新进展。

📖 **通俗解读:**
Diffusion Large Language Models (DLLMs) offer a compelling alternative to Auto-Regressive models, but their deployment is constrained by high decoding cost. In this work, we identify a key inefficiency in DLLM decoding: while computation is parallelized over token blocks, only a small subset of tokens is decodable at each diffusion step, causing most compute to be wasted on non-decodable tokens. We further observe a strong correlation between attention-derived token importance and token-wise decoding probability. Based on this insight, 我们提出 FOCUS -- an inference system designed for DLLMs. By dynamically focusing computation on decodable tokens and evicting non-decodable ones on-the-fly, FOCUS increases the effective batch size, alleviating compute limitations and enabling 可扩展的（能够处理更大规模数据） throughput. 经验性的（基于实验和观察的） evaluations demonstrate that FOCUS achieves up to 3.52 throughput improvement over the production-grade engine LMDeploy, while preserving or improving generation quality across multiple benchmarks. The FOCUS system is publicly available on GitHub: https://github.com/sands-lab/FOCUS.

💡 **核心贡献:**
- Based on this insight, we propose FOCUS -- an inference system designed for DLLMs

🔗 **链接:** [论文](https://arxiv.org/abs/2601.23278v1) | [PDF](https://arxiv.org/pdf/2601.23278v1)

---

## 6. astro-ph.IM 📄

**Denoising the Deep Sky: Physics-Based CCD Noise Formation for Astronomical Imaging**

👥 **作者:** Shuhong Liu, Xining Ge, Ziying Gu 等9人

📝 **一句话总结:** 【astro-ph.IM】Shuhong Liu等Denoising the Deep Sky，使用Realistic noisy counterparts s...，在astro-ph.IM取得新进展。

📖 **通俗解读:**
Astronomical imaging remains noise-limited under practical observing constraints, while standard calibration pipelines mainly remove structured artifacts and leave stochastic noise largely unresolved. Learning-based denoising is promising, yet progress is hindered by scarce paired training data and the need for physically 可解释的（能够解释其决策过程） and reproducible models in scientific workflows. 我们提出 a physics-based noise synthesis 框架（提供结构的基础代码库） tailored to CCD noise formation. The 流程（数据处理或模型训练的完整流程） models photon shot noise, photo-response non-uniformity, dark-current noise, readout effects, and localized outliers arising from cosmic-ray hits and hot pixels. To obtain low-noise inputs for synthesis, we average multiple unregistered exposures to produce high-SNR bases. Realistic noisy counterparts synthesized from these bases using our noise model enable the construction of abundant paired datasets for 监督学习（使用标注数据训练模型）. We further introduce a real-world dataset across multi-bands acquired with two twin ground-based telescopes, providing paired raw frames and instrument-流程（数据处理或模型训练的完整流程） calibrated frames, together with calibration data and stacked high-SNR bases for real-world evaluation.

💡 **核心贡献:**
- We propose a physics-based noise synthesis 框架（提供结构的基础代码库） tailored to CCD noise formation
- We further introduce a real-world dataset across multi-bands acquired with two twin ground-based telescopes, providing paired raw frames and instrumen...

🔗 **链接:** [论文](https://arxiv.org/abs/2601.23276v1) | [PDF](https://arxiv.org/pdf/2601.23276v1)

---

## 7. cs.CL 💬

**UPA: Unsupervised Prompt Agent via Tree-Based Search and Selection**

👥 **作者:** Siran Peng, Weisong Zhao, Tianyu Fu 等9人

📝 **一句话总结:** 【cs.CL】Siran Peng等UPA，在cs.CL取得新进展。

📖 **通俗解读:**
Prompt agents have recently emerged as a promising paradigm for automated prompt 优化（寻找最佳参数或解决方案的过程）, framing refinement as a sequential decision-making problem over a structured prompt space. While this formulation enables the use of advanced planning algorithms, these methods typically assume access to supervised reward signals, which are often unavailable in practical scenarios. In this work, 我们提出 UPA, an Unsupervised Prompt Agent that realizes structured search and selection without relying on supervised feedback. Specifically, during search, UPA iteratively constructs an evolving tree structure to navigate the prompt space, guided by fine-grained and order-invariant pairwise comparisons from Large Language Models (LLMs). Crucially, as these local comparisons do not inherently yield a consistent global scale, we decouple systematic prompt exploration from final selection, introducing a two-stage 框架（提供结构的基础代码库） grounded in the Bradley-Terry-Luce (BTL) model. This 框架（提供结构的基础代码库） first performs path-wise Bayesian aggregation of local comparisons to filter candidates under uncertainty, followed by global tournament-style comparisons to infer latent prompt quality and identify the optimal prompt. Experiments across multiple tasks demonstrate that UPA consistently outperforms existing prompt 优化（寻找最佳参数或解决方案的过程） methods, showing that agent-style 优化（寻找最佳参数或解决方案的过程） remains highly effective even in fully unsupervised settings.

💡 **核心贡献:**
- In this work, we propose UPA, an Unsupervised Prompt Agent that realizes structured search and selection without relying on supervised feedback

🔗 **链接:** [论文](https://arxiv.org/abs/2601.23273v1) | [PDF](https://arxiv.org/pdf/2601.23273v1)

---

## 8. cs.RO 🦾

**IRL-DAL: Safe and Adaptive Trajectory Planning for Autonomous Driving via Energy-Guided Diffusion Models**

👥 **作者:** Seyed Ahmad Hosseini Miangoleh, Amin Jalal Aghdasian, Farzaneh Abdollahi

📝 **一句话总结:** 【cs.RO】Seyed Ahmad Hosseini Miangoleh等IRL-DAL，使用This paper proposes a 新颖的（创新的、...，在cs.RO取得新进展。

📖 **通俗解读:**
This paper proposes a 新颖的（创新的、前人未做过的） inverse 强化学习（通过试错学习最佳策略的机器学习方法） 框架（提供结构的基础代码库） using a diffusion-based adaptive lookahead planner (IRL-DAL) for autonomous vehicles. Training begins with imitation from an expert finite state machine (FSM) controller to provide a stable initialization. Environment terms are combined with an IRL discriminator signal to align with expert goals. 强化学习（通过试错学习最佳策略的机器学习方法） (RL) is then performed with a hybrid reward that combines diffuse environmental feedback and targeted IRL rewards. A conditional diffusion model, which acts as a safety supervisor, plans safe paths. It stays in its lane, avoids obstacles, and moves smoothly. Then, a learnable adaptive mask (LAM) improves perception. It shifts visual attention based on vehicle speed and nearby hazards. After FSM-based imitation, the policy is fine-tuned with Proximal Policy 优化（寻找最佳参数或解决方案的过程） (PPO). Training is run in the Webots simulator with a two-stage curriculum. A 96\% success rate is reached, and collisions are reduced to 0.05 per 1k steps, marking a new 基准（用于比较性能的标准数据集或方法） for safe navigation. By applying the proposed approach, the agent not only drives in lane but also handles unsafe conditions at an expert level, increasing robustness.We make our code publicly available.

💡 **核心贡献:**
- This paper proposes a 新颖的（创新的、前人未做过的） inverse 强化学习（通过试错学习最佳策略的机器学习方法） 框架（提供结构的基础代码库） using a diffusion-based adaptive lookahead planner (IRL-DAL) for ...
- By applying the proposed approach, the agent not only drives in lane but also handles unsafe conditions at an expert level, increasing robustness

🔗 **链接:** [论文](https://arxiv.org/abs/2601.23266v1) | [PDF](https://arxiv.org/pdf/2601.23266v1)

---

## 9. cs.CL 💬

**PaperBanana: Automating Academic Illustration for AI Scientists**

👥 **作者:** Dawei Zhu, Rui Meng, Yale Song 等7人

📝 **一句话总结:** 【cs.CL】Dawei Zhu等PaperBanana，在cs.CL取得新进展。

📖 **通俗解读:**
Despite rapid advances in autonomous AI scientists powered by language models, generating publication-ready illustrations remains a labor-intensive bottleneck in the research workflow. To lift this burden, we introduce PaperBanana, an agentic 框架（提供结构的基础代码库） for automated generation of publication-ready academic illustrations. Powered by 最先进（当前最好的、领先的方法） VLMs and image generation models, PaperBanana orchestrates specialized agents to retrieve references, plan content and style, render images, and iteratively refine via self-critique. To rigorously evaluate our 框架（提供结构的基础代码库）, we introduce PaperBananaBench, comprising 292 test cases for methodology diagrams curated from NeurIPS 2025 publications, covering diverse research domains and illustration styles. 全面的（覆盖广泛的、详细的） experiments demonstrate that PaperBanana consistently outperforms leading baselines in faithfulness, conciseness, readability, and aesthetics. We further show that 我们的方法 effectively extends to the generation of high-quality statistical plots. Collectively, PaperBanana paves the way for the automated generation of publication-ready illustrations.

💡 **核心贡献:**
- To lift this burden, we introduce PaperBanana, an agentic 框架（提供结构的基础代码库） for automated generation of publication-ready academic illustrations
- To rigorously evaluate our 框架（提供结构的基础代码库）, we introduce PaperBananaBench, comprising 292 test cases for methodology diagrams curated from NeurIPS 2025...

🔗 **链接:** [论文](https://arxiv.org/abs/2601.23265v1) | [PDF](https://arxiv.org/pdf/2601.23265v1)

---

## 10. cs.LG 🧠

**Particle-Guided Diffusion Models for Partial Differential Equations**

👥 **作者:** Andrew Millard, Fredrik Lindsten, Zheng Zhao

📝 **一句话总结:** 【cs.LG】Andrew Millard等Particle-Guided Diffusion Models for Partial Differential Equations，在cs.LG取得新进展。

📖 **通俗解读:**
We introduce a guided stochastic sampling method that augments sampling from diffusion models with physics-based guidance derived from partial differential equation (PDE) residuals and observational constraints, ensuring generated samples remain physically admissible. We embed this sampling procedure within a new Sequential Monte Carlo (SMC) 框架（提供结构的基础代码库）, yielding a 可扩展的（能够处理更大规模数据） generative PDE solver. Across multiple 基准（用于比较性能的标准数据集或方法） PDE systems as well as multiphysics and interacting PDE systems, 我们的方法 produces solution fields with lower numerical error than existing 最先进（当前最好的、领先的方法） generative methods.

💡 **核心贡献:**
- We introduce a guided stochastic sampling method that augments sampling from diffusion models with physics-based guidance derived from partial differe...

🔗 **链接:** [论文](https://arxiv.org/abs/2601.23262v1) | [PDF](https://arxiv.org/pdf/2601.23262v1)

---

## 📊 统计信息

**分类分布:**
- 🧠 cs.LG: 24 篇
- 👁️ cs.CV: 16 篇
- 💬 cs.CL: 15 篇
- 🧬 cs.NE: 15 篇
- 🦾 cs.RO: 14 篇
- 📈 stat.ML: 12 篇
- 🤖 cs.AI: 8 篇
- 📄 eess.IV: 4 篇
- 📄 astro-ph.IM: 1 篇
- 📄 stat.CO: 1 篇
- 📄 cs.MA: 1 篇
- 📄 q-bio.BM: 1 篇
- 📄 cs.SD: 1 篇
- 📄 cs.GR: 1 篇
- ♿ cs.HC: 1 篇
- 📄 stat.ME: 1 篇

