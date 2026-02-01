# arXiv AI Daily Paper Report
Date: 2026-02-01

Total Papers: 116

---

### 1. cs.CR - RedSage: A Cybersecurity Generalist LLM

**Authors:** Naufal Suryanto, Muzammal Naseer, Pengfei Li, Syed Talal Wasim, Jinhui Yi, Juergen Gall, Paolo Ceravolo, Ernesto Damiani

**Date:** 2026-01-29

**Summary:** Cybersecurity operations demand assistant LLMs that support diverse workflows without exposing sensitive data. Existing solutions either rely on proprietary APIs with privacy risks or on open models lacking domain adaptation. To bridge this gap, we curate 11.8B tokens of cybersecurity-focused continual pretraining data via large-scale web filtering and manual collection of high-quality resources, ...

**Plain Summary:** Cybersecurity operations demand assistant LLMs that support diverse workflows without exposing sensitive data. Existing solutions either rely on proprietary APIs with privacy risks or on open models lacking domain adaptation. To bridge this gap, we curate 11.8B tokens of cybersecurity-focused continual pretraining data via large-scale web filtering and manual collection of high-quality resources, spanning 28.6K documents across frameworks, offensive techniques, and security tools. Building on this, we design an agentic augmentation 数据处理或模型训练的完整流程（undefined） that simulates expert workflows to generate 266K multi-turn cybersecurity samples for supervised 在预训练模型基础上进行小幅调整（undefined）. Combined with general open-source LLM data, these resources enable the training of RedSage, an open-source, locally deployable cybersecurity assistant with domain-aware pretraining and post-training. To rigorously evaluate the models, we introduce RedSage-Bench, a 用于比较性能的标准数据集或方法（undefined） with 30K multiple-choice and 240 open-ended Q&A items covering cybersecurity knowledge, skills, and tool expertise. RedSage is further evaluated on established cybersecurity benchmarks (e.g., CTI-Bench, CyberMetric, SECURE) and general LLM benchmarks to assess broader generalization. At the 8B scale, RedSage achieves consistently better results, surpassing the 用于对比的基准方法（undefined） models by up to +5.59 points on cybersecurity benchmarks and +5.05 points on Open LLM Leaderboard tasks. These findings demonstrate that domain-aware agentic augmentation and pre/post-training can not only enhance cybersecurity-specific expertise but also help to improve general reasoning and instruction-following. All models, datasets, and code are publicly available.

**One Sentence:** 【cs.CR】Naufal Suryanto等RedSage，在cs.CR取得新进展。

**Contributions:**
- Building on this, we design an agentic augmentation 数据处理或模型训练的完整流程（undefined） that simulates expert workflows to generate 266K multi-turn cybersecurity samples for supervised 在预训练模型基础上进行小幅调整（undefined）
- To rigorously evaluate the models, we introduce RedSage-Bench, a 用于比较性能的标准数据集或方法（undefined） with 30K multiple-choice and 240 open-ended Q&A items covering cybersecurity knowledge, skills, and tool expertise

**Links:** [Abstract](https://arxiv.org/abs/2601.22159v1) | [PDF](https://arxiv.org/pdf/2601.22159v1)

---

### 2. cs.CV - One-step Latent-free Image Generation with Pixel Mean Flows

**Authors:** Yiyang Lu, Susie Lu, Qiao Sun, Hanhong Zhao, Zhicheng Jiang, Xianbang Wang, Tianhong Li, Zhengyang Geng, Kaiming He

**Date:** 2026-01-29

**Summary:** Modern diffusion/flow-based models for image generation typically exhibit two core characteristics: (i) using multi-step sampling, and (ii) operating in a 数据的压缩表示空间（undefined）. Recent advances have made encouraging progress on each aspect individually, paving the way toward one-step diffusion/flow without latents. In this work, we take a further step towards this goal and propose "pixel MeanFlow" ...

**Plain Summary:** Modern diffusion/flow-based models for image generation typically exhibit two core characteristics: (i) using multi-step sampling, and (ii) operating in a 数据的压缩表示空间（undefined）. Recent advances have made encouraging progress on each aspect individually, paving the way toward one-step diffusion/flow without latents. In this work, we take a further step towards this goal and propose "pixel MeanFlow" (pMF). Our core guideline is to formulate the network output space and the loss space separately. The network target is designed to be on a presumed low-dimensional image manifold (i.e., x-prediction), while the loss is defined via MeanFlow in the velocity space. We introduce a simple transformation between the image manifold and the average velocity field. In experiments, pMF achieves strong results for one-step latent-free generation on ImageNet at 256x256 resolution (2.22 FID) and 512x512 resolution (2.48 FID), filling a key missing piece in this regime. We hope that our study will further advance the boundaries of diffusion/flow-based generative models.

**One Sentence:** 【cs.CV】Yiyang Lu等One-step Latent-free Image Generation with Pixel Mean Flows，使用Modern diffusion/flow-based mo...，在cs.CV取得新进展。

**Contributions:**
- In this work, we take a further step towards this goal and propose "pixel MeanFlow" (pMF)
- The network target is designed to be on a presumed low-dimensional image manifold (i

**Links:** [Abstract](https://arxiv.org/abs/2601.22158v1) | [PDF](https://arxiv.org/pdf/2601.22158v1)

---

### 3. cs.LG - Discovering Hidden Gems in Model Repositories

**Authors:** Jonathan Kahana, Eliahu Horwitz, Yedid Hoshen

**Date:** 2026-01-29

**Summary:** Public repositories host millions of fine-tuned models, yet community usage remains disproportionately concentrated on a small number of foundation checkpoints. We investigate whether this concentration reflects 速度快、资源消耗少（undefined） market selection or if superior models are systematically overlooked. Through an extensive evaluation of over 2,000 models, we show the prevalence of "hidden gems", un...

**Plain Summary:** Public repositories host millions of fine-tuned models, yet community usage remains disproportionately concentrated on a small number of foundation checkpoints. We investigate whether this concentration reflects 速度快、资源消耗少（undefined） market selection or if superior models are systematically overlooked. Through an extensive evaluation of over 2,000 models, we show the prevalence of "hidden gems", unpopular fine-tunes that significantly outperform their popular counterparts. Notably, within the Llama-3.1-8B family, we find rarely downloaded checkpoints that improve math performance from 83.2% to 96.0% without increasing inference costs. However, discovering these models through exhaustive evaluation of every uploaded model is computationally infeasible. We therefore formulate model discovery as a Multi-Armed Bandit problem and accelerate the Sequential Halving search algorithm by using shared query sets and aggressive elimination schedules. 我们的方法 retrieves top models with as few as 50 queries per candidate, accelerating discovery by over 50x.

**One Sentence:** 【cs.LG】Jonathan Kahana等Discovering Hidden Gems in Model Repositories，使用We therefore formulate model d...，在cs.LG取得新进展。

**Links:** [Abstract](https://arxiv.org/abs/2601.22157v1) | [PDF](https://arxiv.org/pdf/2601.22157v1)

---

### 4. cs.CL - Hybrid Linear Attention Done Right: 速度快、资源消耗少（undefined） Distillation and Effective Architectures for Extremely Long Contexts

**Authors:** Yingfa Chen, Zhen Leng Thai, Zihan Zhou, Zhu Zhang, Xingyu Shen, Shuo Wang, Chaojun Xiao, Xu Han, Zhiyuan Liu

**Date:** 2026-01-29

**Summary:** Hybrid 一种处理序列数据的神经网络架构，特别擅长处理语言（undefined） architectures, which combine softmax attention blocks and recurrent neural networks (RNNs), have shown a desirable performance-throughput tradeoff for long-context modeling, but their adoption and studies are hindered by the prohibitive cost of large-scale pre-training from scratch. Some recent studies have shown that pre-trained softmax attention blocks ...

**Plain Summary:** Hybrid 一种处理序列数据的神经网络架构，特别擅长处理语言（undefined） architectures, which combine softmax attention blocks and recurrent neural networks (RNNs), have shown a desirable performance-throughput tradeoff for long-context modeling, but their adoption and studies are hindered by the prohibitive cost of large-scale pre-training from scratch. Some recent studies have shown that pre-trained softmax attention blocks can be converted into RNN blocks through parameter transfer and knowledge distillation. However, these transfer methods require substantial amounts of training data (more than 10B tokens), and the resulting hybrid models also exhibit poor long-context performance, which is the scenario where hybrid models enjoy significant inference speedups over 一种处理序列数据的神经网络架构，特别擅长处理语言（undefined）-based models. In this paper, we present HALO (Hybrid Attention via Layer 寻找最佳参数或解决方案的过程（undefined）), a 数据处理或模型训练的完整流程（undefined） for distilling 一种处理序列数据的神经网络架构，特别擅长处理语言（undefined） models into RNN-attention hybrid models. We then present HypeNet, a hybrid architecture with superior length generalization enabled by a 创新的、前人未做过的（undefined） position encoding scheme (named HyPE) and various architectural modifications. We convert the Qwen3 series into HypeNet using HALO, achieving performance comparable to the original 一种处理序列数据的神经网络架构，特别擅长处理语言（undefined） models while enjoying superior long-context performance and efficiency. The conversion requires just 2.3B tokens, less than 0.01% of their pre-training data

**One Sentence:** 【cs.CL】Yingfa Chen等Hybrid Linear Attention Done Right，使用We convert the Qwen3 series in...，在cs.CL取得新进展。

**Contributions:**
- In this paper, we present HALO (Hybrid Attention via Layer 寻找最佳参数或解决方案的过程（undefined）), a 数据处理或模型训练的完整流程（undefined） for distilling 一种处理序列数据的神经网络架构，特别擅长处理语言（undefined） models into RNN-attention hybrid models
- We then present HypeNet, a hybrid architecture with superior length generalization enabled by a 创新的、前人未做过的（undefined） position encoding scheme (named HyPE) and various architectural modifications

**Links:** [Abstract](https://arxiv.org/abs/2601.22156v1) | [PDF](https://arxiv.org/pdf/2601.22156v1)

---

### 5. cs.AI - Exploring Reasoning Reward Model for Agents

**Authors:** Kaixuan Fan, Kaituo Feng, Manyuan Zhang, Tianshuo Peng, Zhixun Li, Yilei Jiang, Shuang Chen, Peng Pei, Xunliang Cai, Xiangyu Yue

**Date:** 2026-01-29

**Summary:** Agentic 通过试错学习最佳策略的机器学习方法（undefined） (Agentic RL) has achieved notable success in enabling agents to perform complex reasoning and tool use. However, most methods still relies on sparse outcome-based reward for training. Such feedback fails to differentiate intermediate reasoning quality, leading to suboptimal training results. In this paper, we introduce Agent Reasoning Reward Model (Agent-RRM), ...

**Plain Summary:** Agentic 通过试错学习最佳策略的机器学习方法（undefined） (Agentic RL) has achieved notable success in enabling agents to perform complex reasoning and tool use. However, most methods still relies on sparse outcome-based reward for training. Such feedback fails to differentiate intermediate reasoning quality, leading to suboptimal training results. In this paper, we introduce Agent Reasoning Reward Model (Agent-RRM), a multi-faceted reward model that produces structured feedback for agentic trajectories, including (1) an explicit reasoning trace , (2) a focused critique that provides refinement guidance by highlighting reasoning flaws, and (3) an overall score that evaluates process performance. Leveraging these signals, we systematically investigate three integration strategies: Reagent-C (text-augmented refinement), Reagent-R (reward-augmented guidance), and Reagent-U (unified feedback integration). Extensive evaluations across 12 diverse benchmarks demonstrate that Reagent-U yields substantial performance leaps, achieving 43.7% on GAIA and 46.2% on WebWalkerQA, validating the effectiveness of our reasoning reward model and training schemes. Code, models, and datasets are all released to facilitate future research.

**One Sentence:** 【cs.AI】Kaixuan Fan等Exploring Reasoning Reward Model for Agents，在cs.AI取得新进展。

**Contributions:**
- In this paper, we introduce Agent Reasoning Reward Model (Agent-RRM), a multi-faceted reward model that produces structured feedback for agentic trajectories, including (1) an explicit reasoning trace , (2) a focused critique that provides refinement guidance by highlighting reasoning flaws, and (3) an overall score that evaluates process performance

**Links:** [Abstract](https://arxiv.org/abs/2601.22154v1) | [PDF](https://arxiv.org/pdf/2601.22154v1)

---

### 6. cs.CV - UEval: A 用于比较性能的标准数据集或方法（undefined） for Unified Multimodal Generation

**Authors:** Bo Li, Yida Yin, Wenhao Chai, Xingyu Fu, Zhuang Liu

**Date:** 2026-01-29

**Summary:** We introduce UEval, a 用于比较性能的标准数据集或方法（undefined） to evaluate unified models, i.e., models capable of generating both images and text. UEval comprises 1,000 expert-curated questions that require both images and text in the model output, sourced from 8 real-world tasks. Our curated questions cover a wide range of reasoning types, from step-by-step guides to textbook explanations. Evaluating open-end...

**Plain Summary:** We introduce UEval, a 用于比较性能的标准数据集或方法（undefined） to evaluate unified models, i.e., models capable of generating both images and text. UEval comprises 1,000 expert-curated questions that require both images and text in the model output, sourced from 8 real-world tasks. Our curated questions cover a wide range of reasoning types, from step-by-step guides to textbook explanations. Evaluating open-ended multimodal generation is non-trivial, as simple LLM-as-a-judge methods can miss the subtleties. Different from previous works that rely on multimodal Large Language Models (MLLMs) to rate image quality or text 正确预测占总预测的比例（undefined）, we design a rubric-based scoring system in UEval. For each question, reference images and text answers are provided to a MLLM to generate an initial rubric, consisting of multiple evaluation criteria, and human experts then refine and validate these rubrics. In total, UEval contains 10,417 validated rubric criteria, enabling 能够处理更大规模数据（undefined） and fine-grained automatic scoring. UEval is challenging for current unified models: GPT-5-Thinking scores only 66.4 out of 100, while the best open-source model reaches merely 49.1. We observe that reasoning models often outperform non-reasoning ones, and transferring reasoning traces from a reasoning model to a non-reasoning model significantly narrows the gap. This suggests that reasoning may be important for tasks requiring complex multimodal understanding and generation.

**One Sentence:** 【cs.CV】Bo Li等UEval，在cs.CV取得新进展。

**Contributions:**
- We introduce UEval, a 用于比较性能的标准数据集或方法（undefined） to evaluate unified models, i
- Different from previous works that rely on multimodal Large Language Models (MLLMs) to rate image quality or text 正确预测占总预测的比例（undefined）, we design a rubric-based scoring system in UEval

**Links:** [Abstract](https://arxiv.org/abs/2601.22155v1) | [PDF](https://arxiv.org/pdf/2601.22155v1)

---

### 7. cs.RO - DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation

**Authors:** Haozhe Xie, Beichen Wen, Jiarui Zheng, Zhaoxi Chen, Fangzhou Hong, Haiwen Diao, Ziwei Liu

**Date:** 2026-01-29

**Summary:** Manipulating dynamic objects remains an open challenge for Vision-Language-Action (VLA) models, which, despite strong generalization in static manipulation, struggle in dynamic scenarios requiring rapid perception, temporal anticipation, and continuous control. We present DynamicVLA, a 提供结构的基础代码库（undefined） for dynamic object manipulation that integrates temporal reasoning and closed-loop adaptati...

**Plain Summary:** Manipulating dynamic objects remains an open challenge for Vision-Language-Action (VLA) models, which, despite strong generalization in static manipulation, struggle in dynamic scenarios requiring rapid perception, temporal anticipation, and continuous control. We present DynamicVLA, a 提供结构的基础代码库（undefined） for dynamic object manipulation that integrates temporal reasoning and closed-loop adaptation through three key designs: 1) a compact 0.4B VLA using a convolutional vision encoder for spatially 速度快、资源消耗少（undefined）, structurally faithful encoding, enabling fast multimodal inference; 2) Continuous Inference, enabling overlapping reasoning and execution for lower latency and timely adaptation to object motion; and 3) Latent-aware Action Streaming, which bridges the perception-execution gap by enforcing temporally aligned action execution. To fill the missing foundation of dynamic manipulation data, we introduce the Dynamic Object Manipulation (DOM) 用于比较性能的标准数据集或方法（undefined）, built from scratch with an auto data collection 数据处理或模型训练的完整流程（undefined） that efficiently gathers 200K synthetic episodes across 2.8K scenes and 206 objects, and enables fast collection of 2K real-world episodes without teleoperation. Extensive evaluations demonstrate remarkable improvements in response speed, perception, and generalization, positioning DynamicVLA as a unified 提供结构的基础代码库（undefined） for general dynamic object manipulation across embodiments.

**One Sentence:** 【cs.RO】Haozhe Xie等DynamicVLA，使用4B VLA using a convolutional v...，在cs.RO取得新进展。

**Contributions:**
- We present DynamicVLA, a 提供结构的基础代码库（undefined） for dynamic object manipulation that integrates temporal reasoning and closed-loop adaptation through three key designs: 1) a compact 0
- To fill the missing foundation of dynamic manipulation data, we introduce the Dynamic Object Manipulation (DOM) 用于比较性能的标准数据集或方法（undefined）, built from scratch with an auto data collection 数据处理或模型训练的完整流程（undefined） that efficiently gathers 200K synthetic episodes across 2

**Links:** [Abstract](https://arxiv.org/abs/2601.22153v1) | [PDF](https://arxiv.org/pdf/2601.22153v1)

---

### 8. cs.LG - Late Breaking Results: Conversion of Neural Networks into Logic Flows for Edge Computing

**Authors:** Daniel Stein, Shaoyi Huang, Rolf Drechsler, Bing Li, Grace Li Zhang

**Date:** 2026-01-29

**Summary:** Neural networks have been successfully applied in various resource-constrained edge devices, where usually central processing units (CPUs) instead of graphics processing units exist due to limited power availability. 当前最好的、领先的方法（undefined） research still focuses on efficiently executing enormous numbers of multiply-accumulate (MAC) operations. However, CPUs themselves are not good at executing suc...

**Plain Summary:** Neural networks have been successfully applied in various resource-constrained edge devices, where usually central processing units (CPUs) instead of graphics processing units exist due to limited power availability. 当前最好的、领先的方法（undefined） research still focuses on efficiently executing enormous numbers of multiply-accumulate (MAC) operations. However, CPUs themselves are not good at executing such mathematical operations on a large scale, since they are more suited to execute control flow logic, i.e., computer algorithms. To enhance the computation efficiency of neural networks on CPUs, in this paper, 我们提出 to convert them into logic flows for execution. Specifically, neural networks are first converted into equivalent decision trees, from which decision paths with constant leaves are then selected and compressed into logic flows. Such logic flows consist of if and else structures and a reduced number of MAC operations. Experimental results demonstrate that the latency can be reduced by up to 14.9 % on a simulated RISC-V CPU without any 正确预测占总预测的比例（undefined） degradation. The code is open source at https://github.com/TUDa-HWAI/NN2Logic

**One Sentence:** 【cs.LG】Daniel Stein等Late Breaking Results，在cs.LG取得新进展。

**Contributions:**
- To enhance the computation efficiency of neural networks on CPUs, in this paper, we propose to convert them into logic flows for execution

**Links:** [Abstract](https://arxiv.org/abs/2601.22151v1) | [PDF](https://arxiv.org/pdf/2601.22151v1)

---

### 9. cs.CV - Do VLMs Perceive or 真正正例中被正确预测的比例（undefined）? Probing Visual Perception vs. Memory with Classic Visual Illusions

**Authors:** Xiaoxiao Sun, Mingyang Li, Kun yuan, Min Woo Sun, Mark Endo, Shengguang Wu, Changlin Li, Yuhui Zhang, Zeyu Wang, Serena Yeung-Levy

**Date:** 2026-01-29

**Summary:** Large Vision-Language Models (VLMs) often answer classic visual illusions "correctly" on original images, yet persist with the same responses when illusion factors are inverted, even though the visual change is obvious to humans. This raises a fundamental question: do VLMs perceive visual changes or merely 真正正例中被正确预测的比例（undefined） memorized patterns? While several studies have noted this phenomeno...

**Plain Summary:** Large Vision-Language Models (VLMs) often answer classic visual illusions "correctly" on original images, yet persist with the same responses when illusion factors are inverted, even though the visual change is obvious to humans. This raises a fundamental question: do VLMs perceive visual changes or merely 真正正例中被正确预测的比例（undefined） memorized patterns? While several studies have noted this phenomenon, the underlying causes remain unclear. To move from observations to systematic understanding, this paper introduces VI-Probe, a controllable visual-illusion 提供结构的基础代码库（undefined） with graded perturbations and matched visual controls (without illusion inducer) that disentangles visually grounded perception from language-driven 真正正例中被正确预测的比例（undefined）. Unlike 观察到数据前的概率（undefined） work that focuses on averaged 正确预测占总预测的比例（undefined）, we measure stability and sensitivity using Polarity-Flip Consistency, Template Fixation Index, and an illusion multiplier normalized against matched controls. Experiments across different families reveal that response persistence arises from heterogeneous causes rather than a single mechanism. For instance, GPT-5 exhibits memory override, Claude-Opus-4.1 shows perception-memory competition, while Qwen variants suggest visual-processing limits. Our findings challenge single-cause views and motivate probing-based evaluation that measures both knowledge and sensitivity to controlled visual change. Data and code are available at https://sites.google.com/view/vi-probe/.

**One Sentence:** 【cs.CV】Xiaoxiao Sun等Do VLMs Perceive or Recall? Probing Visual Perception vs. Memory with Classic Visual Illusions，使用Unlike 观察到数据前的概率（undefined） wo...，在cs.CV取得新进展。

**Contributions:**
- To move from observations to systematic understanding, this paper introduces VI-Probe, a controllable visual-illusion 提供结构的基础代码库（undefined） with graded perturbations and matched visual controls (without illusion inducer) that disentangles visually grounded perception from language-driven 真正正例中被正确预测的比例（undefined）

**Links:** [Abstract](https://arxiv.org/abs/2601.22150v1) | [PDF](https://arxiv.org/pdf/2601.22150v1)

---

### 10. cs.CL - DynaWeb: Model-Based 通过试错学习最佳策略的机器学习方法（undefined） of Web Agents

**Authors:** Hang Ding, Peidong Liu, Junqiao Wang, Ziwei Ji, Meng Cao, Rongzhao Zhang, Lynn Ai, Eric Yang, Tianyu Shi, Lei Yu

**Date:** 2026-01-29

**Summary:** The development of autonomous web agents, powered by Large Language Models (LLMs) and 通过试错学习最佳策略的机器学习方法（undefined） (RL), represents a significant step towards general-purpose AI assistants. However, training these agents is severely hampered by the challenges of interacting with the live internet, which is inefficient, costly, and fraught with risks. Model-based 通过试错学习最佳策略的机器学习方法（undefined） (MBRL)...

**Plain Summary:** The development of autonomous web agents, powered by Large Language Models (LLMs) and 通过试错学习最佳策略的机器学习方法（undefined） (RL), represents a significant step towards general-purpose AI assistants. However, training these agents is severely hampered by the challenges of interacting with the live internet, which is inefficient, costly, and fraught with risks. Model-based 通过试错学习最佳策略的机器学习方法（undefined） (MBRL) offers a promising solution by learning a world model of the environment to enable simulated interaction. This paper introduces DynaWeb, a 创新的、前人未做过的（undefined） MBRL 提供结构的基础代码库（undefined） that trains web agents through interacting with a web world model trained to predict naturalistic web page representations given agent actions. This model serves as a synthetic web environment where an agent policy can dream by generating vast quantities of rollout action trajectories for 速度快、资源消耗少（undefined） online 通过试错学习最佳策略的机器学习方法（undefined）. Beyond free policy rollouts, DynaWeb incorporates real expert trajectories from training data, which are randomly interleaved with on-policy rollouts during training to improve stability and sample efficiency. Experiments conducted on the challenging WebArena and WebVoyager benchmarks demonstrate that DynaWeb consistently and significantly improves the performance of 当前最好的、领先的方法（undefined） open-source web agent models. Our findings establish the viability of training web agents through imagination, offering a 能够处理更大规模数据（undefined） and 速度快、资源消耗少（undefined） way to scale up online agentic RL.

**One Sentence:** 【cs.CL】Hang Ding等DynaWeb，在cs.CL取得新进展。

**Contributions:**
- The development of autonomous web agents, powered by Large Language Models (LLMs) and 通过试错学习最佳策略的机器学习方法（undefined） (RL), represents a significant step towards general-purpose AI assistants
- This paper introduces DynaWeb, a 创新的、前人未做过的（undefined） MBRL 提供结构的基础代码库（undefined） that trains web agents through interacting with a web world model trained to predict naturalistic web page representations given agent actions

**Links:** [Abstract](https://arxiv.org/abs/2601.22149v1) | [PDF](https://arxiv.org/pdf/2601.22149v1)

---

### 11. cs.CL - FineInstructions: Scaling Synthetic Instructions to Pre-Training Scale

**Authors:** Ajay Patel, Colin Raffel, Chris Callison-Burch

**Date:** 2026-01-29

**Summary:** Due to limited supervised training data, large language models (LLMs) are typically pre-trained via a self-supervised "predict the next word" objective on a vast amount of unstructured text data. To make the resulting model useful to users, it is further trained on a far smaller amount of "instruction-tuning" data comprised of supervised training examples of instructions and responses. To overcome...

**Plain Summary:** Due to limited supervised training data, large language models (LLMs) are typically pre-trained via a self-supervised "predict the next word" objective on a vast amount of unstructured text data. To make the resulting model useful to users, it is further trained on a far smaller amount of "instruction-tuning" data comprised of supervised training examples of instructions and responses. To overcome the limited amount of supervised data, 我们提出 a procedure that can transform the knowledge in internet-scale pre-training documents into billions of synthetic instruction and answer training pairs. The resulting dataset, called FineInstructions, uses ~18M instruction templates created from real user-written queries and prompts. These instruction templates are matched to and instantiated with human-written source documents from unstructured pre-training corpora. With "supervised" synthetic training data generated at this scale, an LLM can be pre-trained from scratch solely with the instruction-tuning objective, which is far more in-distribution with the expected downstream usage of LLMs (responding to user prompts). We conduct controlled token-for-token training experiments and find pre-training on FineInstructions outperforms standard pre-training and other proposed synthetic pre-training techniques on standard benchmarks measuring free-form response quality. Our resources can be found at https://huggingface.co/fineinstructions .

**One Sentence:** 【cs.CL】Ajay Patel等FineInstructions，在cs.CL取得新进展。

**Contributions:**
- To overcome the limited amount of supervised data, we propose a procedure that can transform the knowledge in internet-scale pre-training documents into billions of synthetic instruction and answer training pairs
- The resulting dataset, called FineInstructions, uses ~18M instruction templates created from real user-written queries and prompts

**Links:** [Abstract](https://arxiv.org/abs/2601.22146v1) | [PDF](https://arxiv.org/pdf/2601.22146v1)

---

### 12. cs.GR - JUST-DUB-IT: Video Dubbing via Joint Audio-Visual Diffusion

**Authors:** Anthony Chen, Naomi Ken Korem, Tavi Halperin, Matan Ben Yosef, Urska Jelercic, Ofir Bibi, Or Patashnik, Daniel Cohen-Or

**Date:** 2026-01-29

**Summary:** Audio-Visual Foundation Models, which are pretrained to jointly generate sound and visual content, have recently shown an unprecedented ability to model multi-modal generation and editing, opening new opportunities for downstream tasks. Among these tasks, video dubbing could greatly benefit from such priors, yet most existing solutions still rely on complex, task-specific pipelines that struggle i...

**Plain Summary:** Audio-Visual Foundation Models, which are pretrained to jointly generate sound and visual content, have recently shown an unprecedented ability to model multi-modal generation and editing, opening new opportunities for downstream tasks. Among these tasks, video dubbing could greatly benefit from such priors, yet most existing solutions still rely on complex, task-specific pipelines that struggle in real-world settings. In this work, we introduce a single-model approach that adapts a foundational audio-video diffusion model for video-to-video dubbing via a lightweight LoRA. The LoRA enables the model to condition on an input audio-video while jointly generating translated audio and synchronized facial motion. To train this LoRA, we leverage the 能够创建新数据的AI模型（undefined） itself to synthesize paired multilingual videos of the same speaker. Specifically, we generate multilingual videos with language switches within a single clip, and then inpaint the face and audio in each half to match the language of the other half. By leveraging the rich generative 观察到数据前的概率（undefined） of the audio-visual model, our approach preserves speaker identity and lip synchronization while remaining 对噪声和扰动不敏感（undefined） to complex motion and real-world dynamics. We demonstrate that our approach produces high-quality dubbed videos with improved visual fidelity, lip synchronization, and robustness compared to existing dubbing pipelines.

**One Sentence:** 【cs.GR】Anthony Chen等JUST-DUB-IT，在cs.GR取得新进展。

**Contributions:**
- In this work, we introduce a single-model approach that adapts a foundational audio-video diffusion model for video-to-video dubbing via a lightweight LoRA

**Links:** [Abstract](https://arxiv.org/abs/2601.22143v1) | [PDF](https://arxiv.org/pdf/2601.22143v1)

---

### 13. cs.AI - Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data

**Authors:** Grzegorz Stefanski, Alberto Presta, Michal Byra

**Date:** 2026-01-29

**Summary:** In pruning, the Lottery Ticket Hypothesis posits that large networks contain sparse subnetworks, or winning tickets, that can be trained in isolation to match the performance of their dense counterparts. However, most existing approaches assume a single universal winning ticket shared across all inputs, ignoring the inherent heterogeneity of real-world data. In this work, we propose Routing the Lo...

**Plain Summary:** In pruning, the Lottery Ticket Hypothesis posits that large networks contain sparse subnetworks, or winning tickets, that can be trained in isolation to match the performance of their dense counterparts. However, most existing approaches assume a single universal winning ticket shared across all inputs, ignoring the inherent heterogeneity of real-world data. In this work, 我们提出 Routing the Lottery (RTL), an adaptive pruning 提供结构的基础代码库（undefined） that discovers multiple specialized subnetworks, called adaptive tickets, each tailored to a class, semantic cluster, or environmental condition. Across diverse datasets and tasks, RTL consistently outperforms single- and multi-model baselines in balanced 正确预测占总预测的比例（undefined） and 真正正例中被正确预测的比例（undefined）, while using up to 10 times fewer parameters than independent models and exhibiting semantically aligned. Furthermore, we identify subnetwork collapse, a performance drop under aggressive pruning, and introduce a subnetwork similarity score that enables label-free diagnosis of oversparsification. Overall, our results recast pruning as a mechanism for aligning model structure with data heterogeneity, paving the way toward more modular and context-aware 使用多层神经网络来处理复杂模式的技术（undefined）.

**One Sentence:** 【cs.AI】Grzegorz Stefanski等Routing the Lottery，使用Across diverse datasets and ta...，在cs.AI取得新进展。

**Contributions:**
- In this work, we propose Routing the Lottery (RTL), an adaptive pruning 提供结构的基础代码库（undefined） that discovers multiple specialized subnetworks, called adaptive tickets, each tailored to a class, semantic cluster, or environmental condition
- Furthermore, we identify subnetwork collapse, a performance drop under aggressive pruning, and introduce a subnetwork similarity score that enables label-free diagnosis of oversparsification

**Links:** [Abstract](https://arxiv.org/abs/2601.22141v1) | [PDF](https://arxiv.org/pdf/2601.22141v1)

---

### 14. cs.CL - Reasoning While Asking: Transforming Reasoning Large Language Models from Passive Solvers to Proactive Inquirers

**Authors:** Xin Chen, Feng Jiang, Yiqian Zhang, Hardy Chen, Shuo Yan, Wenya Xie, Min Yang, Shujian Huang

**Date:** 2026-01-29

**Summary:** Reasoning-oriented Large Language Models (LLMs) have achieved remarkable progress with Chain-of-Thought (CoT) prompting, yet they remain fundamentally limited by a \emph{blind self-thinking} paradigm: performing extensive internal reasoning even when critical information is missing or ambiguous. We propose Proactive Interactive Reasoning (PIR), a new reasoning paradigm that transforms LLMs from pa...

**Plain Summary:** Reasoning-oriented Large Language Models (LLMs) have achieved remarkable progress with Chain-of-Thought (CoT) prompting, yet they remain fundamentally limited by a \emph{blind self-thinking} paradigm: performing extensive internal reasoning even when critical information is missing or ambiguous. 我们提出 Proactive Interactive Reasoning (PIR), a new reasoning paradigm that transforms LLMs from passive solvers into proactive inquirers that interleave reasoning with clarification. Unlike existing search- or tool-based frameworks that primarily address knowledge uncertainty by querying external environments, PIR targets premise- and intent-level uncertainty through direct interaction with the user. PIR is implemented via two core components: (1) an uncertainty-aware supervised 在预训练模型基础上进行小幅调整（undefined） procedure that equips models with interactive reasoning capability, and (2) a user-simulator-based policy 寻找最佳参数或解决方案的过程（undefined） 提供结构的基础代码库（undefined） driven by a composite reward that aligns model behavior with user intent. 大量实验 on mathematical reasoning, code generation, and document editing demonstrate that PIR consistently outperforms strong baselines, achieving up to 32.70\% higher 正确预测占总预测的比例（undefined）, 22.90\% higher pass rate, and 41.36 BLEU improvement, while reducing nearly half of the reasoning computation and unnecessary interaction turns. Further reliability evaluations on factual knowledge, 理解问题并给出答案的AI系统（undefined）, and missing-premise scenarios confirm the strong generalization and robustness of PIR. Model and code are publicly available at: \href{https://github.com/SUAT-AIRI/Proactive-Interactive-R1}

**One Sentence:** 【cs.CL】Xin Chen等Reasoning While Asking，在cs.CL取得新进展。

**Contributions:**
- We propose Proactive Interactive Reasoning (PIR), a new reasoning paradigm that transforms LLMs from passive solvers into proactive inquirers that interleave reasoning with clarification

**Links:** [Abstract](https://arxiv.org/abs/2601.22139v1) | [PDF](https://arxiv.org/pdf/2601.22139v1)

---

### 15. cs.LG - PRISM: Distribution-free Adaptive Computation of Matrix Functions for Accelerating 一种受人脑启发的计算模型，由许多互相连接的节点组成（undefined） Training

**Authors:** Shenghao Yang, Zhichao Wang, Oleg Balabanov, N. Benjamin Erichson, Michael W. Mahoney

**Date:** 2026-01-29

**Summary:** Matrix functions such as square root, inverse roots, and orthogonalization play a central role in preconditioned gradient methods for 一种受人脑启发的计算模型，由许多互相连接的节点组成（undefined） training. This has motivated the development of iterative algorithms that avoid explicit eigendecompositions and rely primarily on matrix multiplications, making them well suited for modern GPU accelerators. We present PRISM (Pol...

**Plain Summary:** Matrix functions such as square root, inverse roots, and orthogonalization play a central role in preconditioned gradient methods for 一种受人脑启发的计算模型，由许多互相连接的节点组成（undefined） training. This has motivated the development of iterative algorithms that avoid explicit eigendecompositions and rely primarily on matrix multiplications, making them well suited for modern GPU accelerators. We present PRISM (Polynomial-fitting and Randomized Iterative Sketching for Matrix functions computation), a general 提供结构的基础代码库（undefined） for accelerating iterative algorithms for computing matrix functions. PRISM combines adaptive polynomial approximation with randomized sketching: at each iteration, it fits a polynomial surrogate to the current spectrum via a sketched least-squares problem, adapting to the instance at hand with minimal overhead. We apply PRISM to accelerate Newton-Schulz-like iterations for matrix square roots and orthogonalization, which are core primitives in 让计算机通过数据自动学习和改进的技术（undefined）. Unlike 观察到数据前的概率（undefined） methods, PRISM requires no explicit spectral bounds or singular value estimates; and it adapts automatically to the evolving spectrum. Empirically, PRISM accelerates training when integrated into Shampoo and Muon optimizers.

**One Sentence:** 【cs.LG】Shenghao Yang等PRISM，在cs.LG取得新进展。

**Contributions:**
- This has motivated the development of iterative algorithms that avoid explicit eigendecompositions and rely primarily on matrix multiplications, making them well suited for modern GPU accelerators
- We present PRISM (Polynomial-fitting and Randomized Iterative Sketching for Matrix functions computation), a general 提供结构的基础代码库（undefined） for accelerating iterative algorithms for computing matrix functions

**Links:** [Abstract](https://arxiv.org/abs/2601.22137v1) | [PDF](https://arxiv.org/pdf/2601.22137v1)

---

### 16. cs.LG - StepShield: When, Not Whether to Intervene on Rogue Agents

**Authors:** Gloria Felicia, Michael Eniolade, Jinfeng He, Zitha Sasindran, Hemant Kumar, Milan Hussain Angati, Sandeep Bandarupalli

**Date:** 2026-01-29

**Summary:** Existing agent safety benchmarks report binary 正确预测占总预测的比例（undefined）, conflating early intervention with post-mortem analysis. A detector that flags a violation at step 8 enables intervention; one that reports it at step 48 provides only forensic value. This distinction is critical, yet current benchmarks cannot measure it. We introduce StepShield, the first 用于比较性能的标准数据集或方法（undefined） to evaluate...

**Plain Summary:** Existing agent safety benchmarks report binary 正确预测占总预测的比例（undefined）, conflating early intervention with post-mortem analysis. A detector that flags a violation at step 8 enables intervention; one that reports it at step 48 provides only forensic value. This distinction is critical, yet current benchmarks cannot measure it. We introduce StepShield, the first 用于比较性能的标准数据集或方法（undefined） to evaluate when violations are detected, not just whether. StepShield contains 9,213 code agent trajectories, including 1,278 meticulously annotated training pairs and a 7,935-trajectory test set with a realistic 8.1% rogue rate. Rogue behaviors are grounded in real-world security incidents across six categories. 我们提出 three 创新的、前人未做过的（undefined） temporal metrics: Early Intervention Rate (EIR), Intervention Gap, and Tokens Saved. Surprisingly, our evaluation reveals that an LLM-based judge achieves 59% EIR while a static analyzer achieves only 26%, a 2.3x performance gap that is entirely invisible to standard 正确预测占总预测的比例（undefined） metrics. We further show that early detection has direct economic benefits: our cascaded HybridGuard detector reduces monitoring costs by 75% and projects to $108M in cumulative savings over five years at enterprise scale. By shifting the focus of evaluation from whether to when, StepShield provides a new foundation for building safer and more economically viable AI agents. The code and data are released under an Apache 2.0 license.

**One Sentence:** 【cs.LG】Gloria Felicia等StepShield，在cs.LG取得新进展。

**Contributions:**
- We introduce StepShield, the first 用于比较性能的标准数据集或方法（undefined） to evaluate when violations are detected, not just whether
- We propose three 创新的、前人未做过的（undefined） temporal metrics: Early Intervention Rate (EIR), Intervention Gap, and Tokens Saved

**Links:** [Abstract](https://arxiv.org/abs/2601.22136v1) | [PDF](https://arxiv.org/pdf/2601.22136v1)

---

### 17. cs.CV - PI-Light: Physics-Inspired Diffusion for Full-Image Relighting

**Authors:** Zhexin Liang, Zhaoxi Chen, Yongwei Chen, Tianyi Wei, Tengfei Wang, Xingang Pan

**Date:** 2026-01-29

**Summary:** Full-image relighting remains a challenging problem due to the difficulty of collecting large-scale structured paired data, the difficulty of maintaining physical plausibility, and the limited generalizability imposed by data-driven priors. Existing attempts to bridge the synthetic-to-real gap for full-scene relighting remain suboptimal. To tackle these challenges, we introduce Physics-Inspired di...

**Plain Summary:** Full-image relighting remains a challenging problem due to the difficulty of collecting large-scale structured paired data, the difficulty of maintaining physical plausibility, and the limited generalizability imposed by data-driven priors. Existing attempts to bridge the synthetic-to-real gap for full-scene relighting remain suboptimal. To tackle these challenges, we introduce Physics-Inspired diffusion for full-image reLight (-Light, or PI-Light), a two-stage 提供结构的基础代码库（undefined） that leverages physics-inspired diffusion models. Our design incorporates (i) batch-aware attention, which improves the consistency of intrinsic predictions across a collection of images, (ii) a physics-guided neural rendering module that enforces physically plausible light transport, (iii) physics-inspired losses that regularize training dynamics toward a physically meaningful landscape, thereby enhancing generalizability to real-world image editing, and (iv) a carefully curated dataset of diverse objects and scenes captured under controlled lighting conditions. Together, these components enable 速度快、资源消耗少（undefined） finetuning of pretrained diffusion models while also providing a solid 用于比较性能的标准数据集或方法（undefined） for downstream evaluation. Experiments demonstrate that -Light synthesizes specular highlights and diffuse reflections across a wide variety of materials, achieving superior generalization to real-world scenes compared with 观察到数据前的概率（undefined） approaches.

**One Sentence:** 【cs.CV】Zhexin Liang等PI-Light，使用To tackle these challenges, we...，在cs.CV取得新进展。

**Contributions:**
- To tackle these challenges, we introduce Physics-Inspired diffusion for full-image reLight ($π$-Light, or PI-Light), a two-stage 提供结构的基础代码库（undefined） that leverages physics-inspired diffusion models
- Our design incorporates (i) batch-aware attention, which improves the consistency of intrinsic predictions across a collection of images, (ii) a physics-guided neural rendering module that enforces physically plausible light transport, (iii) physics-inspired losses that regularize training dynamics toward a physically meaningful landscape, thereby enhancing generalizability to real-world image editing, and (iv) a carefully curated dataset of diverse objects and scenes captured under controlled lighting conditions

**Links:** [Abstract](https://arxiv.org/abs/2601.22135v1) | [PDF](https://arxiv.org/pdf/2601.22135v1)

---

### 18. cs.CV - Early and Prediagnostic Detection of Pancreatic Cancer from Computed Tomography

**Authors:** Wenxuan Li, Pedro R. A. S. Bassi, Lizhou Wu, Xinze Zhou, Yuxuan Zhao, Qi Chen, Szymon Plotka, Tianyu Lin, Zheren Zhu, Marisa Martin, Justin Caskey, Shanshan Jiang, Xiaoxi Chen, Jaroslaw B. Ćwikla, Artur Sankowski, Yaping Wu, Sergio Decherchi, Andrea Cavalli, Chandana Lall, Cristian Tomasetti, Yaxing Guo, Xuan Yu, Yuqing Cai, Hualin Qiao, Jie Bao, Chenhan Hu, Ximing Wang, Arkadiusz Sitek, Kai Ding, Heng Li, Meiyun Wang, Dexin Yu, Guang Zhang, Yang Yang, Kang Wang, Alan L. Yuille, Zongwei Zhou

**Date:** 2026-01-29

**Summary:** Pancreatic ductal adenocarcinoma (PDAC), one of the deadliest solid malignancies, is often detected at a late and inoperable stage. Retrospective reviews of prediagnostic CT scans, when conducted by expert radiologists aware that the patient later developed PDAC, frequently reveal lesions that were previously overlooked. To help detecting these lesions earlier, we developed an automated system nam...

**Plain Summary:** Pancreatic ductal adenocarcinoma (PDAC), one of the deadliest solid malignancies, is often detected at a late and inoperable stage. Retrospective reviews of prediagnostic CT scans, when conducted by expert radiologists aware that the patient later developed PDAC, frequently reveal lesions that were previously overlooked. To help detecting these lesions earlier, we developed an automated system named ePAI (early Pancreatic cancer detection with 让机器模拟人类智能的技术（undefined）). It was trained on data from 1,598 patients from a single medical center. In the internal test involving 1,009 patients, ePAI achieved an area under the receiver operating characteristic curve (AUC) of 0.939-0.999, a sensitivity of 95.3%, and a specificity of 98.7% for detecting small PDAC less than 2 cm in diameter, precisely localizing PDAC as small as 2 mm. In an external test involving 7,158 patients across 6 centers, ePAI achieved an AUC of 0.918-0.945, a sensitivity of 91.5%, and a specificity of 88.0%, precisely localizing PDAC as small as 5 mm. Importantly, ePAI detected PDACs on prediagnostic CT scans obtained 3 to 36 months before clinical diagnosis that had originally been overlooked by radiologists. It successfully detected and localized PDACs in 75 of 159 patients, with a median lead time of 347 days before clinical diagnosis. Our multi-reader study showed that ePAI significantly outperformed 30 board-certified radiologists by 50.3% (P < 0.05) in sensitivity while maintaining a comparable specificity of 95.4% in detecting PDACs early and prediagnostic. These findings suggest its potential of ePAI as an assistive tool to improve early detection of pancreatic cancer.

**One Sentence:** 【cs.CV】Wenxuan Li等Early and Prediagnostic Detection of Pancreatic Cancer from Computed Tomography，在cs.CV取得新进展。

**Contributions:**
- Retrospective reviews of prediagnostic CT scans, when conducted by expert radiologists aware that the patient later developed PDAC, frequently reveal lesions that were previously overlooked
- To help detecting these lesions earlier, we developed an automated system named ePAI (early Pancreatic cancer detection with 让机器模拟人类智能的技术（undefined）)

**Links:** [Abstract](https://arxiv.org/abs/2601.22134v1) | [PDF](https://arxiv.org/pdf/2601.22134v1)

---

### 19. cs.LG - Pay for Hints, Not Answers: LLM Shepherding for Cost-速度快、资源消耗少（undefined） Inference

**Authors:** Ziming Dong, Hardik Sharma, Evan O'Toole, Jaya Prakash Champati, Kui Wu

**Date:** 2026-01-29

**Summary:** Large Language Models (LLMs) deliver 当前最好的、领先的方法（undefined） performance on complex reasoning tasks, but their inference costs limit deployment at scale. Small Language Models (SLMs) offer dramatic cost savings yet lag substantially in 正确预测占总预测的比例（undefined）. Existing approaches - routing and cascading - treat the LLM as an all-or-nothing resource: either the query bypasses the LLM entirely, or the...

**Plain Summary:** Large Language Models (LLMs) deliver 当前最好的、领先的方法（undefined） performance on complex reasoning tasks, but their inference costs limit deployment at scale. Small Language Models (SLMs) offer dramatic cost savings yet lag substantially in 正确预测占总预测的比例（undefined）. Existing approaches - routing and cascading - treat the LLM as an all-or-nothing resource: either the query bypasses the LLM entirely, or the LLM generates a complete response at full cost. We introduce LLM Shepherding, a 提供结构的基础代码库（undefined） that requests only a short prefix (a hint) from the LLM and provides it to SLM. This simple mechanism is surprisingly effective for math and coding tasks: even hints comprising 10-30% of the full LLM response improve SLM 正确预测占总预测的比例（undefined） significantly. Shepherding generalizes both routing and cascading, and it achieves lower cost under oracle decision-making. We develop a two-stage predictor that jointly determines whether a hint is needed and how many tokens to request. On the widely-used mathematical reasoning (GSM8K, CNK12) and code generation (HumanEval, MBPP) benchmarks, Shepherding reduces costs by 42-94% relative to LLM-only inference. Compared to 当前最好的、领先的方法（undefined） routing and cascading baselines, shepherding delivers up to 2.8x cost reduction while matching 正确预测占总预测的比例（undefined）. To our knowledge, this is the first work to exploit token-level budget control for SLM-LLM collaboration.

**One Sentence:** 【cs.LG】Ziming Dong等Pay for Hints, Not Answers，在cs.LG取得新进展。

**Contributions:**
- We introduce LLM Shepherding, a 提供结构的基础代码库（undefined） that requests only a short prefix (a hint) from the LLM and provides it to SLM
- We develop a two-stage predictor that jointly determines whether a hint is needed and how many tokens to request

**Links:** [Abstract](https://arxiv.org/abs/2601.22132v1) | [PDF](https://arxiv.org/pdf/2601.22132v1)

---

### 20. cs.LG - SMOG: 能够处理更大规模数据（undefined） Meta-Learning for Multi-Objective Bayesian 寻找最佳参数或解决方案的过程（undefined）

**Authors:** Leonard Papenmeier, Petru Tighineanu

**Date:** 2026-01-29

**Summary:** Multi-objective 寻找最佳参数或解决方案的过程（undefined） aims to solve problems with competing objectives, often with only black-box access to a problem and a limited budget of measurements. In many applications, historical data from related 寻找最佳参数或解决方案的过程（undefined） tasks is available, creating an opportunity for meta-learning to accelerate the 寻找最佳参数或解决方案的过程（undefined）. Bayesian 寻找最佳参数或解决方案的过程（undefined）, as a...

**Plain Summary:** Multi-objective 寻找最佳参数或解决方案的过程（undefined） aims to solve problems with competing objectives, often with only black-box access to a problem and a limited budget of measurements. In many applications, historical data from related 寻找最佳参数或解决方案的过程（undefined） tasks is available, creating an opportunity for meta-learning to accelerate the 寻找最佳参数或解决方案的过程（undefined）. Bayesian 寻找最佳参数或解决方案的过程（undefined）, as a promising technique for black-box 寻找最佳参数或解决方案的过程（undefined）, has been extended to meta-learning and multi-objective 寻找最佳参数或解决方案的过程（undefined） independently, but methods that simultaneously address both settings - meta-learned priors for multi-objective Bayesian 寻找最佳参数或解决方案的过程（undefined） - remain largely unexplored. 我们提出 SMOG, a 能够处理更大规模数据（undefined） and modular meta-learning model based on a multi-output Gaussian process that explicitly learns correlations between objectives. SMOG builds a structured joint Gaussian process 观察到数据前的概率（undefined） across meta- and target tasks and, after conditioning on metadata, yields a closed-form target-task 观察到数据前的概率（undefined） augmented by a flexible residual multi-output kernel. This construction propagates metadata uncertainty into the target surrogate in a principled way. SMOG supports hierarchical, parallel training: meta-task Gaussian processes are fit once and then cached, achieving linear scaling with the number of meta-tasks. The resulting surrogate integrates seamlessly with standard multi-objective Bayesian 寻找最佳参数或解决方案的过程（undefined） acquisition functions.

**One Sentence:** 【cs.LG】Leonard Papenmeier等SMOG，使用We propose SMOG, a 能够处理更大规模数据（...，在cs.LG取得新进展。

**Contributions:**
- We propose SMOG, a 能够处理更大规模数据（undefined） and modular meta-learning model based on a multi-output Gaussian process that explicitly learns correlations between objectives

**Links:** [Abstract](https://arxiv.org/abs/2601.22131v1) | [PDF](https://arxiv.org/pdf/2601.22131v1)

---

