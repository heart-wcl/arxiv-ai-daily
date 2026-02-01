# 📚 arXiv AI 每日论文报告
**日期:** 2026年2月1日星期日
**论文数:** 116 篇

---

## 1. cs.CR 📄

**RedSage: A Cybersecurity Generalist LLM**

👥 **作者:** Naufal Suryanto, Muzammal Naseer, Pengfei Li 等8人

📝 **一句话总结:** 【cs.CR】Naufal Suryanto等RedSage，在cs.CR取得新进展。

📖 **通俗解读:**
Cybersecurity operations demand assistant LLMs that support diverse workflows without exposing sensitive data. Existing solutions either rely on proprietary APIs with privacy risks or on open models lacking domain adaptation. To bridge this gap, we curate 11.8B tokens of cybersecurity-focused continual pretraining data via large-scale web filtering and manual collection of high-quality resources, spanning 28.6K documents across frameworks, offensive techniques, and security tools. Building on this, we design an agentic augmentation 数据处理或模型训练的完整流程（undefined） that simulates expert workflows to generate 266K multi-turn cybersecurity samples for supervised 在预训练模型基础上进行小幅调整（undefined）. Combined with general open-source LLM data, these resources enable the training of RedSage, an open-source, locally deployable cybersecurity assistant with domain-aware pretraining and post-training. To rigorously evaluate the models, we introduce RedSage-Bench, a 用于比较性能的标准数据集或方法（undefined） with 30K multiple-choice and 240 open-ended Q&A items covering cybersecurity knowledge, skills, and tool expertise. RedSage is further evaluated on established cybersecurity benchmarks (e.g., CTI-Bench, CyberMetric, SECURE) and general LLM benchmarks to assess broader generalization. At the 8B scale, RedSage achieves consistently better results, surpassing the 用于对比的基准方法（undefined） models by up to +5.59 points on cybersecurity benchmarks and +5.05 points on Open LLM Leaderboard tasks. These findings demonstrate that domain-aware agentic augmentation and pre/post-training can not only enhance cybersecurity-specific expertise but also help to improve general reasoning and instruction-following. All models, datasets, and code are publicly available.

💡 **核心贡献:**
- Building on this, we design an agentic augmentation 数据处理或模型训练的完整流程（undefined） that simulates expert workflows to generate 266K multi-turn cybersecurit...
- To rigorously evaluate the models, we introduce RedSage-Bench, a 用于比较性能的标准数据集或方法（undefined） with 30K multiple-choice and 240 open-ended Q&A items cove...

🔗 **链接:** [论文](https://arxiv.org/abs/2601.22159v1) | [PDF](https://arxiv.org/pdf/2601.22159v1)

---

## 2. cs.CV 👁️

**One-step Latent-free Image Generation with Pixel Mean Flows**

👥 **作者:** Yiyang Lu, Susie Lu, Qiao Sun 等9人

📝 **一句话总结:** 【cs.CV】Yiyang Lu等One-step Latent-free Image Generation with Pixel Mean Flows，使用Modern diffusion/flow-based mo...，在cs.CV取得新进展。

📖 **通俗解读:**
Modern diffusion/flow-based models for image generation typically exhibit two core characteristics: (i) using multi-step sampling, and (ii) operating in a 数据的压缩表示空间（undefined）. Recent advances have made encouraging progress on each aspect individually, paving the way toward one-step diffusion/flow without latents. In this work, we take a further step towards this goal and propose "pixel MeanFlow" (pMF). Our core guideline is to formulate the network output space and the loss space separately. The network target is designed to be on a presumed low-dimensional image manifold (i.e., x-prediction), while the loss is defined via MeanFlow in the velocity space. We introduce a simple transformation between the image manifold and the average velocity field. In experiments, pMF achieves strong results for one-step latent-free generation on ImageNet at 256x256 resolution (2.22 FID) and 512x512 resolution (2.48 FID), filling a key missing piece in this regime. We hope that our study will further advance the boundaries of diffusion/flow-based generative models.

💡 **核心贡献:**
- In this work, we take a further step towards this goal and propose "pixel MeanFlow" (pMF)
- The network target is designed to be on a presumed low-dimensional image manifold (i

🔗 **链接:** [论文](https://arxiv.org/abs/2601.22158v1) | [PDF](https://arxiv.org/pdf/2601.22158v1)

---

## 3. cs.LG 🧠

**Discovering Hidden Gems in Model Repositories**

👥 **作者:** Jonathan Kahana, Eliahu Horwitz, Yedid Hoshen

📝 **一句话总结:** 【cs.LG】Jonathan Kahana等Discovering Hidden Gems in Model Repositories，使用We therefore formulate model d...，在cs.LG取得新进展。

📖 **通俗解读:**
Public repositories host millions of fine-tuned models, yet community usage remains disproportionately concentrated on a small number of foundation checkpoints. We investigate whether this concentration reflects 速度快、资源消耗少（undefined） market selection or if superior models are systematically overlooked. Through an extensive evaluation of over 2,000 models, we show the prevalence of "hidden gems", unpopular fine-tunes that significantly outperform their popular counterparts. Notably, within the Llama-3.1-8B family, we find rarely downloaded checkpoints that improve math performance from 83.2% to 96.0% without increasing inference costs. However, discovering these models through exhaustive evaluation of every uploaded model is computationally infeasible. We therefore formulate model discovery as a Multi-Armed Bandit problem and accelerate the Sequential Halving search algorithm by using shared query sets and aggressive elimination schedules. 我们的方法 retrieves top models with as few as 50 queries per candidate, accelerating discovery by over 50x.

🔗 **链接:** [论文](https://arxiv.org/abs/2601.22157v1) | [PDF](https://arxiv.org/pdf/2601.22157v1)

---

## 4. cs.CL 💬

**Hybrid Linear Attention Done Right: 速度快、资源消耗少（undefined） Distillation and Effective Architectures for Extremely Long Contexts**

👥 **作者:** Yingfa Chen, Zhen Leng Thai, Zihan Zhou 等9人

📝 **一句话总结:** 【cs.CL】Yingfa Chen等Hybrid Linear Attention Done Right，使用We convert the Qwen3 series in...，在cs.CL取得新进展。

📖 **通俗解读:**
Hybrid 一种处理序列数据的神经网络架构，特别擅长处理语言（undefined） architectures, which combine softmax attention blocks and recurrent neural networks (RNNs), have shown a desirable performance-throughput tradeoff for long-context modeling, but their adoption and studies are hindered by the prohibitive cost of large-scale pre-training from scratch. Some recent studies have shown that pre-trained softmax attention blocks can be converted into RNN blocks through parameter transfer and knowledge distillation. However, these transfer methods require substantial amounts of training data (more than 10B tokens), and the resulting hybrid models also exhibit poor long-context performance, which is the scenario where hybrid models enjoy significant inference speedups over 一种处理序列数据的神经网络架构，特别擅长处理语言（undefined）-based models. In this paper, we present HALO (Hybrid Attention via Layer 寻找最佳参数或解决方案的过程（undefined）), a 数据处理或模型训练的完整流程（undefined） for distilling 一种处理序列数据的神经网络架构，特别擅长处理语言（undefined） models into RNN-attention hybrid models. We then present HypeNet, a hybrid architecture with superior length generalization enabled by a 创新的、前人未做过的（undefined） position encoding scheme (named HyPE) and various architectural modifications. We convert the Qwen3 series into HypeNet using HALO, achieving performance comparable to the original 一种处理序列数据的神经网络架构，特别擅长处理语言（undefined） models while enjoying superior long-context performance and efficiency. The conversion requires just 2.3B tokens, less than 0.01% of their pre-training data

💡 **核心贡献:**
- In this paper, we present HALO (Hybrid Attention via Layer 寻找最佳参数或解决方案的过程（undefined）), a 数据处理或模型训练的完整流程（undefined） for distilling 一种处理序列数据的神经网络架构，特别擅长...
- We then present HypeNet, a hybrid architecture with superior length generalization enabled by a 创新的、前人未做过的（undefined） position encoding scheme (named ...

🔗 **链接:** [论文](https://arxiv.org/abs/2601.22156v1) | [PDF](https://arxiv.org/pdf/2601.22156v1)

---

## 5. cs.AI 🤖

**Exploring Reasoning Reward Model for Agents**

👥 **作者:** Kaixuan Fan, Kaituo Feng, Manyuan Zhang 等10人

📝 **一句话总结:** 【cs.AI】Kaixuan Fan等Exploring Reasoning Reward Model for Agents，在cs.AI取得新进展。

📖 **通俗解读:**
Agentic 通过试错学习最佳策略的机器学习方法（undefined） (Agentic RL) has achieved notable success in enabling agents to perform complex reasoning and tool use. However, most methods still relies on sparse outcome-based reward for training. Such feedback fails to differentiate intermediate reasoning quality, leading to suboptimal training results. In this paper, we introduce Agent Reasoning Reward Model (Agent-RRM), a multi-faceted reward model that produces structured feedback for agentic trajectories, including (1) an explicit reasoning trace , (2) a focused critique that provides refinement guidance by highlighting reasoning flaws, and (3) an overall score that evaluates process performance. Leveraging these signals, we systematically investigate three integration strategies: Reagent-C (text-augmented refinement), Reagent-R (reward-augmented guidance), and Reagent-U (unified feedback integration). Extensive evaluations across 12 diverse benchmarks demonstrate that Reagent-U yields substantial performance leaps, achieving 43.7% on GAIA and 46.2% on WebWalkerQA, validating the effectiveness of our reasoning reward model and training schemes. Code, models, and datasets are all released to facilitate future research.

💡 **核心贡献:**
- In this paper, we introduce Agent Reasoning Reward Model (Agent-RRM), a multi-faceted reward model that produces structured feedback for agentic traje...

🔗 **链接:** [论文](https://arxiv.org/abs/2601.22154v1) | [PDF](https://arxiv.org/pdf/2601.22154v1)

---

## 6. cs.CV 👁️

**UEval: A 用于比较性能的标准数据集或方法（undefined） for Unified Multimodal Generation**

👥 **作者:** Bo Li, Yida Yin, Wenhao Chai 等5人

📝 **一句话总结:** 【cs.CV】Bo Li等UEval，在cs.CV取得新进展。

📖 **通俗解读:**
We introduce UEval, a 用于比较性能的标准数据集或方法（undefined） to evaluate unified models, i.e., models capable of generating both images and text. UEval comprises 1,000 expert-curated questions that require both images and text in the model output, sourced from 8 real-world tasks. Our curated questions cover a wide range of reasoning types, from step-by-step guides to textbook explanations. Evaluating open-ended multimodal generation is non-trivial, as simple LLM-as-a-judge methods can miss the subtleties. Different from previous works that rely on multimodal Large Language Models (MLLMs) to rate image quality or text 正确预测占总预测的比例（undefined）, we design a rubric-based scoring system in UEval. For each question, reference images and text answers are provided to a MLLM to generate an initial rubric, consisting of multiple evaluation criteria, and human experts then refine and validate these rubrics. In total, UEval contains 10,417 validated rubric criteria, enabling 能够处理更大规模数据（undefined） and fine-grained automatic scoring. UEval is challenging for current unified models: GPT-5-Thinking scores only 66.4 out of 100, while the best open-source model reaches merely 49.1. We observe that reasoning models often outperform non-reasoning ones, and transferring reasoning traces from a reasoning model to a non-reasoning model significantly narrows the gap. This suggests that reasoning may be important for tasks requiring complex multimodal understanding and generation.

💡 **核心贡献:**
- We introduce UEval, a 用于比较性能的标准数据集或方法（undefined） to evaluate unified models, i
- Different from previous works that rely on multimodal Large Language Models (MLLMs) to rate image quality or text 正确预测占总预测的比例（undefined）, we design a ...

🔗 **链接:** [论文](https://arxiv.org/abs/2601.22155v1) | [PDF](https://arxiv.org/pdf/2601.22155v1)

---

## 7. cs.RO 🦾

**DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation**

👥 **作者:** Haozhe Xie, Beichen Wen, Jiarui Zheng 等7人

📝 **一句话总结:** 【cs.RO】Haozhe Xie等DynamicVLA，使用4B VLA using a convolutional v...，在cs.RO取得新进展。

📖 **通俗解读:**
Manipulating dynamic objects remains an open challenge for Vision-Language-Action (VLA) models, which, despite strong generalization in static manipulation, struggle in dynamic scenarios requiring rapid perception, temporal anticipation, and continuous control. We present DynamicVLA, a 提供结构的基础代码库（undefined） for dynamic object manipulation that integrates temporal reasoning and closed-loop adaptation through three key designs: 1) a compact 0.4B VLA using a convolutional vision encoder for spatially 速度快、资源消耗少（undefined）, structurally faithful encoding, enabling fast multimodal inference; 2) Continuous Inference, enabling overlapping reasoning and execution for lower latency and timely adaptation to object motion; and 3) Latent-aware Action Streaming, which bridges the perception-execution gap by enforcing temporally aligned action execution. To fill the missing foundation of dynamic manipulation data, we introduce the Dynamic Object Manipulation (DOM) 用于比较性能的标准数据集或方法（undefined）, built from scratch with an auto data collection 数据处理或模型训练的完整流程（undefined） that efficiently gathers 200K synthetic episodes across 2.8K scenes and 206 objects, and enables fast collection of 2K real-world episodes without teleoperation. Extensive evaluations demonstrate remarkable improvements in response speed, perception, and generalization, positioning DynamicVLA as a unified 提供结构的基础代码库（undefined） for general dynamic object manipulation across embodiments.

💡 **核心贡献:**
- We present DynamicVLA, a 提供结构的基础代码库（undefined） for dynamic object manipulation that integrates temporal reasoning and closed-loop adaptation through t...
- To fill the missing foundation of dynamic manipulation data, we introduce the Dynamic Object Manipulation (DOM) 用于比较性能的标准数据集或方法（undefined）, built from...

🔗 **链接:** [论文](https://arxiv.org/abs/2601.22153v1) | [PDF](https://arxiv.org/pdf/2601.22153v1)

---

## 8. cs.LG 🧠

**Late Breaking Results: Conversion of Neural Networks into Logic Flows for Edge Computing**

👥 **作者:** Daniel Stein, Shaoyi Huang, Rolf Drechsler 等5人

📝 **一句话总结:** 【cs.LG】Daniel Stein等Late Breaking Results，在cs.LG取得新进展。

📖 **通俗解读:**
Neural networks have been successfully applied in various resource-constrained edge devices, where usually central processing units (CPUs) instead of graphics processing units exist due to limited power availability. 当前最好的、领先的方法（undefined） research still focuses on efficiently executing enormous numbers of multiply-accumulate (MAC) operations. However, CPUs themselves are not good at executing such mathematical operations on a large scale, since they are more suited to execute control flow logic, i.e., computer algorithms. To enhance the computation efficiency of neural networks on CPUs, in this paper, 我们提出 to convert them into logic flows for execution. Specifically, neural networks are first converted into equivalent decision trees, from which decision paths with constant leaves are then selected and compressed into logic flows. Such logic flows consist of if and else structures and a reduced number of MAC operations. Experimental results demonstrate that the latency can be reduced by up to 14.9 % on a simulated RISC-V CPU without any 正确预测占总预测的比例（undefined） degradation. The code is open source at https://github.com/TUDa-HWAI/NN2Logic

💡 **核心贡献:**
- To enhance the computation efficiency of neural networks on CPUs, in this paper, we propose to convert them into logic flows for execution

🔗 **链接:** [论文](https://arxiv.org/abs/2601.22151v1) | [PDF](https://arxiv.org/pdf/2601.22151v1)

---

## 9. cs.CV 👁️

**Do VLMs Perceive or 真正正例中被正确预测的比例（undefined）? Probing Visual Perception vs. Memory with Classic Visual Illusions**

👥 **作者:** Xiaoxiao Sun, Mingyang Li, Kun yuan 等10人

📝 **一句话总结:** 【cs.CV】Xiaoxiao Sun等Do VLMs Perceive or Recall? Probing Visual Perception vs. Memory with Classic Visual Illusions，使用Unlike 观察到数据前的概率（undefined） wo...，在cs.CV取得新进展。

📖 **通俗解读:**
Large Vision-Language Models (VLMs) often answer classic visual illusions "correctly" on original images, yet persist with the same responses when illusion factors are inverted, even though the visual change is obvious to humans. This raises a fundamental question: do VLMs perceive visual changes or merely 真正正例中被正确预测的比例（undefined） memorized patterns? While several studies have noted this phenomenon, the underlying causes remain unclear. To move from observations to systematic understanding, this paper introduces VI-Probe, a controllable visual-illusion 提供结构的基础代码库（undefined） with graded perturbations and matched visual controls (without illusion inducer) that disentangles visually grounded perception from language-driven 真正正例中被正确预测的比例（undefined）. Unlike 观察到数据前的概率（undefined） work that focuses on averaged 正确预测占总预测的比例（undefined）, we measure stability and sensitivity using Polarity-Flip Consistency, Template Fixation Index, and an illusion multiplier normalized against matched controls. Experiments across different families reveal that response persistence arises from heterogeneous causes rather than a single mechanism. For instance, GPT-5 exhibits memory override, Claude-Opus-4.1 shows perception-memory competition, while Qwen variants suggest visual-processing limits. Our findings challenge single-cause views and motivate probing-based evaluation that measures both knowledge and sensitivity to controlled visual change. Data and code are available at https://sites.google.com/view/vi-probe/.

💡 **核心贡献:**
- To move from observations to systematic understanding, this paper introduces VI-Probe, a controllable visual-illusion 提供结构的基础代码库（undefined） with grade...

🔗 **链接:** [论文](https://arxiv.org/abs/2601.22150v1) | [PDF](https://arxiv.org/pdf/2601.22150v1)

---

## 10. cs.CL 💬

**DynaWeb: Model-Based 通过试错学习最佳策略的机器学习方法（undefined） of Web Agents**

👥 **作者:** Hang Ding, Peidong Liu, Junqiao Wang 等10人

📝 **一句话总结:** 【cs.CL】Hang Ding等DynaWeb，在cs.CL取得新进展。

📖 **通俗解读:**
The development of autonomous web agents, powered by Large Language Models (LLMs) and 通过试错学习最佳策略的机器学习方法（undefined） (RL), represents a significant step towards general-purpose AI assistants. However, training these agents is severely hampered by the challenges of interacting with the live internet, which is inefficient, costly, and fraught with risks. Model-based 通过试错学习最佳策略的机器学习方法（undefined） (MBRL) offers a promising solution by learning a world model of the environment to enable simulated interaction. This paper introduces DynaWeb, a 创新的、前人未做过的（undefined） MBRL 提供结构的基础代码库（undefined） that trains web agents through interacting with a web world model trained to predict naturalistic web page representations given agent actions. This model serves as a synthetic web environment where an agent policy can dream by generating vast quantities of rollout action trajectories for 速度快、资源消耗少（undefined） online 通过试错学习最佳策略的机器学习方法（undefined）. Beyond free policy rollouts, DynaWeb incorporates real expert trajectories from training data, which are randomly interleaved with on-policy rollouts during training to improve stability and sample efficiency. Experiments conducted on the challenging WebArena and WebVoyager benchmarks demonstrate that DynaWeb consistently and significantly improves the performance of 当前最好的、领先的方法（undefined） open-source web agent models. Our findings establish the viability of training web agents through imagination, offering a 能够处理更大规模数据（undefined） and 速度快、资源消耗少（undefined） way to scale up online agentic RL.

💡 **核心贡献:**
- The development of autonomous web agents, powered by Large Language Models (LLMs) and 通过试错学习最佳策略的机器学习方法（undefined） (RL), represents a significant step...
- This paper introduces DynaWeb, a 创新的、前人未做过的（undefined） MBRL 提供结构的基础代码库（undefined） that trains web agents through interacting with a web world model tr...

🔗 **链接:** [论文](https://arxiv.org/abs/2601.22149v1) | [PDF](https://arxiv.org/pdf/2601.22149v1)

---

## 📊 统计信息

**分类分布:**
- 🧠 cs.LG: 21 篇
- 🦾 cs.RO: 17 篇
- 👁️ cs.CV: 16 篇
- 💬 cs.CL: 16 篇
- 🧬 cs.NE: 14 篇
- 🤖 cs.AI: 10 篇
- 📈 stat.ML: 8 篇
- 📄 math.OC: 2 篇
- 📄 cs.CR: 1 篇
- 📄 cs.GR: 1 篇
- 📄 cs.SE: 1 篇
- 📄 q-fin.CP: 1 篇
- 📄 q-fin.TR: 1 篇
- 📄 cs.CY: 1 篇
- 📄 physics.flu-dyn: 1 篇
- 📄 cs.MA: 1 篇
- 📄 quant-ph: 1 篇
- 📄 eess.SP: 1 篇
- 📄 eess.IV: 1 篇
- 📄 stat.CO: 1 篇

