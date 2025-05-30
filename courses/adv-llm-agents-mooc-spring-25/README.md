# Advanced LLM Agents, MOOC, Spring 2025

https://llmagents-learning.org/sp25

## Lecture 1: Inference-Time Techniques for LLM Reasoning

<details>

## Overview:

The lecture emphasized strategies for optimizing reasoning tasks using advanced prompting methods, multi-candidate exploration, and iterative self-improvement, all aimed at improving accuracy and adaptability during inference.

Here are the main takeaways:
- Standard prompting struggles with reasoning benchmarks, but Chain-of-Thought (CoT) prompting significantly improves performance by modeling intermediate steps.
- Zero-shot CoT prompting uses simple instructions like “Let’s think step by step” to elicit reasoning without exemplars.
- Analogical prompting enables LLMs to self-generate tailored exemplars and reasoning structures, improving task-specific performance.
- Self-consistency boosts accuracy by sampling multiple solutions and selecting the most consistent final answer.
- Tree of Thoughts (ToT) allows step-by-step evaluation and iterative exploration, excelling in complex tasks.
- Reflexion and Self-Refine techniques empower LLMs to iteratively self-correct and improve their outputs using internal and external feedback.
- Self-correction without external feedback (oracle) can worsen reasoning performance, highlighting the need for effective evaluation mechanisms.
- Balancing inference budgets and model size is crucial for optimizing multi-sample solutions and computational efficiency.
- General-purpose and scalable methods remain essential for designing effective reasoning strategies in LLMs.

Ref:
- https://www.youtube.com/live/g0Dwtf3BH-0
- https://llmagents-learning.org/slides/inference_time_techniques_lecture_sp25.pdf


## Briefing:

### Introduction

This document summarises a lecture on inference-time techniques for enhancing the reasoning capabilities of Large Language Models (LLMs). The lecture highlights the significant advancements in LLM reasoning, particularly with models like OpenAI's "o1" and "o3", which demonstrate impressive performance on complex tasks like math, coding, and STEM. However, these high levels of performance are often achieved by using substantial inference-time computation. The lecture explores various strategies to optimise this, categorising them into three main areas: using more tokens for a single solution, searching and selecting from multiple candidates, and iterative self-improvement.

### Part 1: Basic Prompting Techniques - Increasing Token Budget for Single Solution

#### Standard Prompting Limitations
- Prior to advanced post-training techniques, standard prompting struggled with reasoning benchmarks.
- Few-shot examples only provided the format of the final solution, not the reasoning behind it.

#### Chain-of-Thought (CoT) Prompting
- Prompts the model to generate reasoning steps before arriving at the final solution.
- Can be implemented via few-shot examples or instructions.
- **Scaling with Model Size:** CoT performance improves significantly with larger models.
- **Zero-Shot CoT:** Using instructions like "Let's think step by step" can elicit CoT without needing few-shot examples, though it is less effective.

#### Analogical Prompting
- Enhances CoT performance by instructing the model to recall relevant exemplars before solving the test problem.
- Outperforms both zero-shot CoT and manual few-shot CoT, particularly with stronger models.

#### LLM-Driven Prompt Optimisation
- Leverages LLMs to automatically design and optimise prompts.
- Uses past optimisation trajectories to generate improved instructions.
- A meta-prompt enables the LLM to propose new instructions based on previous ones.
- Optimised prompts can outperform standard zero-shot and few-shot prompts.

#### CoT & Reasoning Strategies
- CoT allows variable computation based on task complexity.
- **Least-to-Most Prompting:** Decomposes complex problems into simpler sub-tasks solved sequentially.
- **Dynamic Least-to-Most Prompting:** Customises prompts for each sub-problem.
- **Self-Discover:** Instructs the LLM to autonomously compose reasoning structures without manually created demonstrations.

### Part 2: Search and Selection from Multiple Candidates - Increasing Width of Solution Space

#### Rationale
- Exploring multiple branches allows the model to recover from single-generation errors.

#### Self-Consistency
- Generates multiple candidate solutions and selects the most consistent final answer.
- Effective across models and tasks, scaling well with the number of samples.
- **Diversity in Sampling:** Ensures response variety using high-temperature sampling instead of beam search.

#### Clustering by Execution (AlphaCode)
- In code generation, predicted code is clustered based on execution consistency.
- Improves performance by selecting a program from the largest semantically equivalent clusters.

#### Universal Self-Consistency (USC)
- Extends self-consistency to free-form generation tasks, where consistency is evaluated within the LLM itself.

#### LLM Rankers
- Training verifiers or reward models enhances solution selection.
- **Outcome-Supervised Reward Model (ORM):** Evaluates final solutions.
- **Process-Supervised Reward Model (PRM):** Evaluates step-by-step reasoning.

#### Tree-of-Thoughts (ToT)
- Combines LLMs with tree search.
- Generates possible next reasoning steps, evaluates each, and prioritises promising solutions.
- Scales well with increased token budget.

### Part 3: Iterative Self-Improvement - Increasing Depth of Solution Search

#### Rationale
- Mistakes occur even in strong LLMs.
- Sampling multiple solutions is insufficient without a feedback loop for error correction.

#### Reflexion and Self-Refine
- The LLM generates feedback on its own output and refines it.
- Effective when external evaluation is available.

#### Self-Debugging (Code)
- Uses execution feedback, such as unit tests, to improve generated code.
- More informative feedback yields better results.

#### Limitations of Self-Correction (QA)
- Self-correction without an oracle verifier can reduce accuracy.
- General-purpose feedback prompts and multi-agent debates are often ineffective.

#### Budget Optimisation
- Optimal inference budget depends on the task and model.
- Smaller models may generate more solutions within the same computational budget.

### Key Takeaways and General Principles

- **Adaptability:** Best practices for LLM interaction should evolve with model capabilities.
- **Chain-of-Thought:** Fundamental for reasoning enhancement.
- **Consistency-Based Selection:** A simple yet effective principle for better response selection.
- **Search:** Exploring multiple solution paths improves accuracy.
- **The "Bitter Lesson":** Emphasises general-purpose methods that scale well with computation.

### Conclusion

The lecture provides a comprehensive overview of inference-time techniques for improving LLM reasoning. These techniques focus on:
1. **Using more token budget for better single-solution generation.**
2. **Searching multiple branches in the solution space.**
3. **Iterative self-improvement of responses.**

They range from basic CoT prompting to advanced methods like tree-of-thought and self-debugging. The lecture underscores the importance of continuous adaptation, scalable general-purpose methods, and fostering models capable of independent discovery rather than pre-programmed intelligence.

</details>

## Lecture 2: Self-Improving and Reasoning with Large Language Models

<details>

## Briefing

This document outlines the research and development of self-improving and reasoning Large Language Models (LLMs), which aim to create AI that trains itself, evaluates its performance, and updates itself based on its understanding. The ultimate goal is to achieve superhuman performance through these methods.

### System 1 vs System 2
The document introduces two systems for how LLMs function, System 1 and System 2:
*   **System 1**: This is reactive, relies on associations, has fixed compute per token, directly outputs answers, and is prone to failures like hallucinations and spurious correlations. Standard LLMs are considered System 1.
*   **System 2**: This is more deliberate and effortful, involving multiple "calls" to the System 1 LLM. It uses planning, search, verification, and reasoning with dynamic computation. Techniques like Chain-of-Thought (CoT) and Tree-of-Thoughts (ToT) fall under System 2.

### Historical Context and Evolution of LLMs
The document provides a brief history of LLMs and related technologies:
*   **Early 2000s:** Support Vector Machines were prevalent.
*   **2014:** The LLM attention mechanism was developed.
*   **2019-2023:** A rapid evolution of LLMs occurred, from GPT-2 to GPT-4, including models like T5, Jurassic-1, Megatron-Turing NLG, Gopher, Chinchilla, PaLM, OPT, BLOOM, and LLaMA.
*   **Pre-2020:** Language models were trained by predicting the next token on "positive examples" of language.
*   **Post-2020:**  LLMs began using techniques like supervised fine-tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF).
*   **2022:** InstructGPT was developed using SFT and RLHF on GPT3.
*   **2023:** Models like Claude and GPT-4 began using extensive RLHF for safety and accuracy, and Direct Preference Optimization (DPO) was introduced.

### Improving Reasoning with System 2
*   **Prompting Approaches:** Early attempts to improve reasoning focused on prompting techniques.
*   **Chain-of-Verification (CoVe):** This method aims to reduce hallucinations by adding verification steps to the generation process. It includes variants like joint left-to-right generation, factored attention, and a factored-revise approach.
*   **System 2 Attention (S2A):** This method focuses on making attention more explicit and effortful by prompting the LLM to rewrite inputs, removing irrelevant or biased content, to improve the relevance of answers.
*   **Branch-Solve-Merge:** This approach breaks down complex tasks into subproblems, solves them individually, and merges the solutions to improve complex tasks where instructions are hard.

### Self-Improvement and Self-Rewarding LLMs
*   **Self-Training:** LLMs improve by assigning rewards to their own outputs and optimizing accordingly.
*   **Self-Rewarding LMs:** These models are trained to have both instruction-following and evaluation capabilities. They can generate responses to instructions and judge the quality of those responses, creating an iterative process of improvement.
*   **Iterative Training:** This involves two steps: self-instruction creation (generating prompts, responses, and self-rewards) and instruction training (using DPO on selected preference pairs).
*   **The "Superalignment challenge":** As LLMs improve they will become harder for humans to supervise.
*   **Initial Model:** Experiments start with a pre-trained LLaMA-2-70B model (M0) which is multitask trained using seed instruction following (IFT) and evaluation data (EFT). The model then goes through iterative training.
*    **Evaluation:** The self-rewarding models are evaluated on their ability to follow instructions and their ability to act as a reward model. The models are tested using internal instruction following test sets, AlpacaEval 2.0, and MT-Bench. They are also evaluated using the OpenAssistant validation set.
*    **Improvements:** The models show continuous improvement through iterative training.

### Iterative Reasoning and Meta-Rewarding
*   **Iterative Reasoning Preference Optimisation:** This technique uses self-rewarding techniques for reasoning tasks by generating multiple Chain-of-Thoughts (CoTs) and selecting preferences based on answer correctness.
*    **Thinking LLMs:**  This approach trains LLMs to think and respond for all instruction following tasks, not just math, using Thought Preference Optimization (TPO). It has achieved strong results on benchmarks like AlpacaEval and ArenaHard.
*   **Meta-Rewarding LLMs:** These models improve their judgments by meta-judging them. The LLM acts as an actor, judge, and meta-judge. Meta-rewards provide an additional training signal.
*   **LLM-as-a-Meta-Judge:** This is used to assess judgments. The method involves generating multiple judgments for pairs of responses and calculating pairwise meta-judgments.
*   **EvalPlanner:** This method trains LLMs to generate planning and reasoning CoTs for evaluation, converting evaluation tasks into verifiable tasks by generating similar prompts with high and low quality responses.

### Future Directions
*   **Latent System 2 Thoughts:** Explores the use of latent thoughts rather than tokens, with research into self-evaluation and learning from interaction.
*   **Improved System 1:** Research is needed to improve the fundamental architecture of System 1, such as better attention mechanisms and world models.

### Conclusion
The document highlights the significant progress in developing self-improving and reasoning LLMs. By using techniques like self-rewarding, iterative training, and meta-reasoning, LLMs are approaching and potentially surpassing human-level performance. Further research is needed to address limitations, improve reasoning, and explore the potential of more advanced approaches.

### Ref:
- https://www.youtube.com/live/_MNlLhU33H0
- https://llmagents-learning.org/slides/Jason-Weston-Reasoning-Alignment-Berkeley-Talk.pdf

</details>

## Lecture 3: On Memory, Reasoning, and Planning of Language Agents

<details>

**Briefing: On Memory, Reasoning, and Planning of Language Agents**

### **Overview**
This document provides a comprehensive analysis of **Language Agents**, AI systems that leverage language for reasoning, memory, and planning. It contrasts two main perspectives in developing these agents—**LLM-first** and **Agent-first**—highlighting their respective challenges and opportunities. The discussion is structured around three core competencies essential to advancing intelligent AI agents:
1. **Memory** – HippoRAG, a neurobiologically inspired long-term memory system.
2. **Reasoning** – The concept of implicit reasoning and "grokking" in Transformers.
3. **Planning** – Model-based planning techniques, particularly in web navigation, as demonstrated in WebDreamer.

While language agents are a significant step toward more intelligent AI, the field faces ongoing hurdles in areas like **continual learning, safety, world models, and adaptability**.

---

### **Key Themes and Takeaways**

#### **The Rise of Language Agents**
- Language agents are expected to revolutionize computing, as reflected in statements from key AI leaders:
  - Bill Gates: *"Agents are bringing about the biggest revolution in computing..."*
  - Andrew Ng: *"AI agentic workflows will drive massive AI progress this year."*
  - Sam Altman: *"2025 is when agents will work."*
- Current agents **rely heavily on LLMs** but still lack robust reasoning, memory, and planning capabilities.
- Following Russel & Norvig’s definition, an agent is *“anything that perceives its environment through sensors and acts upon it through actuators.”* Language agents stand out by using **language as the primary tool for reasoning and communication**.
- The document suggests we are entering a **new evolutionary stage** of AI, moving from **Logical Agents → Neural Agents → Language Agents**, characterized by increasing **expressiveness, reasoning, and adaptivity**.

#### **LLM-First vs. Agent-First Approaches**
- **LLM-First:** Builds agents around LLMs, **relying on prompting and engineering solutions** to scaffold agent behavior.
- **Agent-First:** Treats LLMs as a component of a broader AI system that incorporates **perception, memory, world modeling, and planning**.
- The **Agent-First** approach requires tackling **synthetic data generation, self-reflection, and internalized search**, bringing both traditional and novel AI challenges.

#### **Language as a Vehicle for Reasoning and Communication**
- **Language is the foundation** for instruction following, in-context learning, and customized outputs.
- Reasoning within an LLM-based agent functions as an **inner monologue**, where decisions are made via token generation.
- The integration of reasoning helps in:
  - **State inference** (understanding the environment’s current state).
  - **Self-reflection** (evaluating its own thought process).
  - **Replanning** (adjusting actions dynamically).

#### **Memory: HippoRAG – Neurobiologically Inspired Long-Term Memory**
- Human memory is crucial for learning, as reflected in Eric Kandel’s quote: *"Memory is everything. Without it, we are nothing."*
- **Current Retrieval-Augmented Generation (RAG) models have limitations** in retrieving relevant knowledge reliably.
- **HippoRAG** is introduced as a **solution inspired by the hippocampal indexing theory**, aiming to improve retrieval accuracy by:
  - **Indexing associations between stored knowledge**.
  - **Enabling pattern separation and pattern completion**, mimicking the way humans recall and differentiate information.
- HippoRAG is composed of three key components:
  - **Neocortex** – Handles perception, linguistic abilities, and reasoning.
  - **Parahippocampus** – Acts as a bridge between memory areas, akin to working memory.
  - **Hippocampus** – Provides indexing and auto-associative memory functions.

#### **Reasoning: Grokking in Transformers**
- The phenomenon of **"grokking"** describes the transition where **Transformers shift from memorization to generalization**.
- Key research questions include:
  - Can Transformers develop **implicit reasoning**, or are there fundamental limitations?
  - What factors influence the acquisition of reasoning skills, such as **data scale, distribution, and model architecture**?
- The document describes **grokking** as a phase transition, where generalization emerges as the dominant capability over rote memorization.

#### **Planning: Model-Based Planning for Web Agents (WebDreamer)**
- Planning remains a key challenge for AI agents, particularly in **open-ended digital environments** like the web.
- **Challenges in planning** include:
  - Expanding the **action space** while maintaining control.
  - Ensuring **goal verification**, as many tasks have **non-binary success criteria**.
  - Developing **world models** to predict the consequences of actions.
- **WebDreamer** is introduced as a **model-based planner for web agents**, addressing these challenges with:
  - **Stage 1: Simulation** – The LLM predicts state transitions before taking real-world actions.
  - **Stage 2: Execution** – The agent follows an optimal path based on the simulated outcomes.
- This approach ensures **safer and more efficient web navigation**, overcoming the drawbacks of purely reactive planning.

---

### **Future Directions and Challenges**
1. **Memory:** Enhancing **personalization and continual learning** for AI agents.
2. **Reasoning:** Developing models that integrate **external actions and environmental awareness**.
3. **Planning:** Building **better world models** while balancing reactive and model-based planning approaches.
4. **Safety:** Addressing both **endogenous (internal) and exogenous (external) risks**.
5. **Applications:** Expanding AI capabilities in **agentic search, workflow automation, and scientific reasoning**.

The author concludes that we are **at the beginning of a new AI era**, with key challenges in **multimodal perception, memory embodiment, reasoning, world models, grounding, planning, tool use, multi-agent dynamics, and continual learning**.

---

### **Quotes of Significance**
- *"Agents are bringing about the biggest revolution in computing..."* – Bill Gates
- *"2025 is when agents will work."* – Sam Altman
- *"An agent is anything that perceives and acts upon its environment."* – Russel & Norvig
- *"Memory is everything. Without it, we are nothing."* – Eric Kandel
- *"We find that LLMs can be highly receptive to external evidence even when that conflicts with their parametric memory, given that the external evidence is coherent and convincing."*

---

### **Conclusion**
The presentation offers a **detailed exploration of language agents**, illustrating their potential to reshape AI-driven interactions. While **significant progress has been made**, major obstacles remain in **memory, reasoning, planning, safety, and continual learning**. Research efforts such as **HippoRAG and WebDreamer** offer promising solutions, but further innovations are necessary to **fully realize the potential of AI-powered language agents**.


### Ref:
- https://www.youtube.com/live/zvI4UN2_i-w
- https://llmagents-learning.org/slides/language_agents_YuSu_Berkeley.pdf
- https://github.com/OSU-NLP-Group/WebDreamer
- https://github.com/OSU-NLP-Group/HippoRAG

</details>

## Lecture 4: Open Training Recipes for Reasoning in Language Models

<details>

**Unified Briefing on Open Language Models (LMs)**

This document summarizes key aspects of open language models (LMs), focusing on the OLMo family and the Tülu post-training recipe. It underscores the importance of **open science, transparency, and accessibility** in LM development, covering the stages of pre-training, post-training, test-time inference, and risk mitigation.

---

### **Key Themes and Insights**

#### **The Importance of Open Science in LM Research**
- Fully open LMs accelerate innovation by ensuring transparency, reproducibility, and accessibility.
- Analogy: Relying on proprietary models for AI research is like studying astronomy through newspaper pictures.
- OLMo is developed as a truly open AI, empowering the research community and enhancing public AI literacy.

#### **OLMo: A Fully Open Language Model**
- Designed for accessibility and reproducibility within an open ecosystem.
- Competes with **Llama3, Qwen2.5, DeepSeek, and GPT4-o** in performance.

#### **Tülu: Open Post-Training Recipe**
- Iterative development (*Tülu 1 → 2 → 2.5 → 3*), systematically refining LMs post-training.
- Toolkit includes **OpenInstruct and Safety Data & Toolkit - Instruct2**.
- Successful adaptation involves:
  1. **Targeted evaluations** for meaningful improvements.
  2. **Representative prompts** for testing and finetuning.
  3. **License verification** for compliance.
  4. **Data decontamination** to ensure integrity.

---

### **Core LM Development Stages**

#### **1. Pre-training**
- Predicting the next word across diverse contexts to create a foundation model.

#### **2. Post-training**
- Enhances model performance through:
  - **Supervised Fine-Tuning (SFT)**: Improves model outputs using structured prompt-completion training.
  - **Tool use and agents**: Equipping LMs with task-specific tools.
  - **Reasoning**: Enhancing logical and analytical capabilities.
  - **Safety alignment**: Ensuring ethical and responsible AI behavior.
  - **Hybrid data curation**: Mixing diverse datasets for skill refinement.

#### **3. Test-time Inference & Scaling**
- Techniques like **budget forcing** and computational scaling optimize real-time model performance.

---

### **Advanced LM Optimization Techniques**

#### **Supervised Fine-Tuning (SFT) / Instruction Tuning**
- **Purpose**: Finetunes pretrained LMs with structured prompts.
- **Challenges**: Data curation is expensive, time-consuming, and has high variance.
- **Solution**: **Hybrid data creation** (mixing curated data with persona-driven synthesis) ensures efficiency and diversity.

#### **Chain-of-Thought (CoT) Reasoning**
- **Improves** multi-step problem-solving and logical reasoning.
- **Challenges**: Manual annotation is expensive and difficult to scale.
- **Solution**: Hybrid data generation bridges the scalability gap.

#### **Preference Tuning**
- Aligns LMs with human preferences for better interaction quality.
- **DPO vs. PPO**:
  - **Direct Preference Optimization (DPO)**: Efficient, lower complexity, and high throughput.
  - **Proximal Policy Optimization (PPO)**: Slightly better performance (~1%) but more computationally intensive.
- **Key Factor**: **Data quality** is paramount for both methods.

#### **Reinforcement Learning with Verifiable Rewards (RLVR)**
- Addresses **over-optimization** by using rule-based rewards for tasks with ground-truth answers (e.g., math problems).
- RLVR involves **targeted datasets, verifiers, and PPO training** for enhanced accuracy.

#### **Mid-training Strategy**
- **99% of training budget** is allocated to **trillions of diverse text tokens**.
- **1% of budget** is reserved for upsampling high-quality SFT data, maximizing efficiency.

---

### **Overall Takeaways**
- **Open models (OLMo) and structured post-training (Tülu) are critical for AI progress**.
- **Data quality, hybrid data creation, and advanced tuning techniques** (RLVR, DPO, SFT) drive LMs' reasoning capabilities.
- **Open ecosystems foster reproducible research**, empowering the AI community with accessible knowledge and tools.

---

### Ref:
- https://www.youtube.com/live/cMiu3A7YBks
- https://llmagents-learning.org/slides/OLMo-Tulu-Reasoning-Hanna.pdf

</details>

## Lecture 5: Coding Agents and AI for Vulnerability Detection

<details>

### **Briefing Document: Coding Agents and AI for Vulnerability Detection**  
**Author**: Based on excerpts from *L5_Code_Agents_Vulnerability_Detection.pdf* by Charles Sutton  
**Date**: October 26, 2024  

---

## **Executive Summary**  
This document explores the application of **Large Language Model (LLM) agents** in **software engineering and security**, particularly for **vulnerability detection**. It covers the **evolution of evaluation metrics**, introduces **coding agents (SWE-Agent, Agentless, AutoCodeRover, Passerine)**, and discusses their use in **Capture the Flag (CTF) competitions** and **real-world security applications** through **Google’s Big Sleep project**.  

The key takeaway is that **LLM agents provide a promising yet underexplored approach to AI-driven security**, with **evaluation-driven improvements, trade-offs in agent design, and practical implementations in vulnerability detection**.  

---

## **Key Themes and Ideas**  

### **1. Rise of Coding Agents & AI for Software Engineering**  
**Definition of LLM Agents**:  
LLM agents are defined as **"multi-turn LLMs with tool use"**, characterized by:  
- **Dynamic computation time**  
- **Information retrieval from external tools**  
- **Hypothesis testing and validation**  
- **Action execution based on results**  

#### **Agent Designs & Trade-offs:**  
The document explores different **coding agent designs**:  
1. **SWE-Agent (Dynamic Approach)**  
   - Uses **planning, chain of thought reasoning, tool use, and execution feedback.**  
   - Implements the **ReACT loop** (*LLM generates output → Runs tools → Updates trajectory*).  
   - More **adaptive** for complex problem-solving.  

2. **Agentless (Procedural Approach)**  
   - **No persistent agent loop; control flow is in Python code.**  
   - Preferred when the **workflow is simple** and doesn’t require dynamic decision-making.  
   - Avoids **LLM trajectory errors**, making it more **robust but less flexible**.  

3. **Hybrid Models (AutoCodeRover, Passerine)**  
   - Combine **some procedural control** with **agent-driven exploration**.  
   - Useful for **structured software improvements** and **debugging workflows**.  

---

### **2. Evaluation Metrics: The Backbone of Model Design**  
Evaluation metrics **drive LLM model and agent design**.  
- **Early benchmarks (MBPP, HumanEval)**: Useful in 2021 but now **leaked, too easy, and limited in test cases.**  
- **SWE-Bench & SWE-Bench Verified**:  
  - A **realistic evaluation benchmark** that has **driven advances in agent development**.  
  - Verified variant removes **underspecified and less relevant test cases**, improving signal-to-noise ratio.  
- **Challenges in Evaluation**:  
  - "All evaluations have a shelf life"—new tests are needed as models improve.  
  - Data leakage, inexact verifiers, and overfitting to leaderboards are growing concerns.  

**Design Considerations in Evaluation:**  
- **Level of difficulty** must be balanced.  
- **Realism** should match real-world coding scenarios.  
- **Generalizability** ensures models **aren’t just learning shortcuts to pass benchmarks**.  

---

### **3. AI for Computer Security: Capture the Flag (CTF) & Vulnerability Detection**  
**CTF competitions as AI Benchmarks**  
LLM agents are adapted for **security tasks** using datasets like:  
- **NYU CtF Bench (2025):** 200 problems from real-world CTF challenges.  
- **InterCode-CtF (2023):** 100 tasks from **high school-level** PicoCTF.  

#### **Agent Adaptation for Security Tasks**  
To tackle **CTF challenges**, coding agents integrate:  
- **Command-line execution (sandboxed)**  
- **Decompilers & disassemblers**  
- **Python libraries (for cryptography, forensics, etc.)**  
- **Debuggers (GDB, Pwntools for memory exploits)**  

**"Arbitrary command line (use a sandbox!)" is critical for security-related LLM applications.**  

---

### **4. Big Sleep: AI for Real-World Vulnerability Detection**  
The **Big Sleep project at Google** represents a **major leap in AI-driven security research**.  

**Objective**: Find security vulnerabilities using **LLM agents with reasoning, execution, and verification**.  

#### **Big Sleep's Methodology**:  
- **Step 1: Code Navigation** – LLM browses code, jumps to definitions, and follows references.  
- **Step 2: Hypothesis Generation** – AI **predicts possible security flaws** (e.g., buffer overflow).  
- **Step 3: Dynamic Testing** – Executes **debugging tools, interpreters, and test inputs**.  
- **Step 4: Verification** – Confirms vulnerabilities using **sanitizer crashes, memory safety checks, and execution feedback**.  

#### **Results & Impact**:  
- **Achieved a 1.00 score in buffer overflow detection (from 0.05).**  
- **76% success rate in advanced memory corruption detection (up from 24%).**  
- **Discovered a real-world vulnerability in SQLite.**  

**Key Takeaways:**  
- **Dynamic AI agents outperform traditional fuzzing & static analysis** in some security domains.  
- **Execution-based verification makes AI more reliable in detecting real vulnerabilities.**  
- **Big Sleep proves that AI agents can autonomously detect security flaws at an expert level.**  

---

## **Future Directions & Implications**  
- **AI for security is still a "wide open area" with untapped potential.**  
- **More research needed in network security, malware detection, and adversarial attacks.**  
- **Real-world deployment of AI security agents will require stronger evaluation benchmarks.**  
- **Moving from CTFs to real-world tasks is the next major step.**  

---

## **Important Quotes**  
- *"LLM agents are multi-turn LLMs with tool use."*  
- *"Evaluations drive the design of the models. Organizational-level Bayesian optimization."*  
- *"All evaluations have a shelf life."*  
- *"If workflow really is simple, why make the LLM figure it out?"*  
- *"Agentic techniques seem particularly natural. Can require larger-scale understanding of software or system."*  
- *"Arbitrary command line (use a sandbox!)"*  

---

## **Final Thoughts**  
LLM-powered security agents are **emerging as a viable method for vulnerability detection**, surpassing traditional **fuzzing and static analysis** in some domains. However, the **design space for AI security remains vast**, with **open challenges in evaluation, real-world deployment, and system-level reasoning**.  

---

### **Best of Both Approaches**  
✅ **Quick, structured overview** (from the first summary)  
✅ **Deep, in-depth analysis with key details** (from the second briefing)  
✅ **Formatted for clarity, readability, and impact**  

### Ref:
- https://www.youtube.com/live/JCk6qJtaCSU
- https://llmagents-learning.org/slides/Code%20Agents%20and%20AI%20for%20Vulnerability%20Detection.pdf

</details>

## Lecture 6: Autonomous Multimodal AI Agents: Web and Beyond

<details>

  ### **Multimodal Autonomous AI Agents: Comprehensive Summary**

This summary combines the best aspects of brevity and technical depth, providing a **structured yet detailed** overview of the key concepts from **Russ Salakhutdinov's lecture on multimodal AI agents**.

---

## **1. Large Language Models (LLMs) as a Foundation for AI Agents**
The lecture begins by emphasizing **LLMs as the backbone** of autonomous AI agents, detailing their strengths and limitations.

### **Key Capabilities of LLMs:**
- **In-context learning**: Ability to generalize based on contextual cues in the input.
- **Zero-shot abilities**: Can perform tasks they haven't been explicitly trained on.
- **Long and coherent text generation**: Essential for complex reasoning and decision-making.
- **Strong text representations**: Helps in understanding and encoding textual information.
- **Sensitivity to word ordering**: Crucial for nuanced comprehension and response generation.
- **World knowledge**: Learned from vast training data, but with limitations in real-time updates.

---

## **2. The Emergence of Autonomous AI Agents**
AI agents are being developed to **automate digital tasks** and **enhance human productivity**, particularly for web-based activities.

### **Potential Use Cases:**
- **Automating repetitive digital tasks** (e.g., generating PowerPoint slides from a research paper).
- **Navigating the web** to retrieve structured data.
- **Performing human-like browsing and decision-making**.

These agents combine **LLMs, reinforcement learning, and multimodal capabilities** to operate in increasingly complex environments.

---

## **3. Challenges in Web-Based AI Agents**
The lecture highlights the **current limitations of web agents**, particularly when dealing with **HTML and text-based environments**.

### **WebArena: Early Benchmarks & Limitations**
- Designed as **the most realistic web-based evaluation** for AI agents.
- Uses real-world data from **Amazon, Reddit, GitHub**, etc.
- Tasks involve interacting with only **text and HTML**, leading to **low success rates** for AI.

### **Why HTML Alone is Insufficient**
- **Messy HTML & JavaScript**: Code is often **minified and compressed**, making parsing difficult.
- **Interactive elements**: JavaScript-driven UI elements **don't render well in raw HTML**.
- **Spatial layout issues**: Webpages rely on **visual structure**, which HTML alone cannot capture.
- **Context length constraints**: HTML pages often exceed **100k tokens**, making them difficult for LLMs to process.

---

## **4. VisualWebArena: A Benchmark for Multimodal Web Agents**
To overcome the limitations of **text-based web navigation**, the researchers introduced **VisualWebArena**, a **multimodal** testbed.

### **Key Features of VisualWebArena:**
- Includes **visual inputs** to complement **HTML parsing**.
- Uses **POMDP (Partially Observable Markov Decision Process)** framework:
  - **Observations**: Captures both **visual and textual** information.
  - **Actions**: Agents can **click, type, hover, and stop**.
  - **Reward function**: Evaluates success based on **task completion**.

### **Example Tasks in VisualWebArena:**
- **Shopping Task**: "Buy the cheapest color photo printer and send it to Emily."
- **Classifieds Task**: "Find a specific bike for $300-$500 and negotiate $10 less."
- **Reddit Task**: "Find the 2022 total GDP of the region producing the most sugarcane in 2021."

**Results:**
- **Human success rate: 78%**
- **AI agent success rate: 14%** (indicating significant room for improvement)

---

## **5. Architectures for Multimodal Web Agents**
The lecture introduces **several architectural advancements** aimed at improving AI interaction with web environments.

### **Key Approaches:**
#### **1. Web Agent Architecture**
- **Input**: HTML + Image.
- **Processing**: **Observation Parsing + High-Level Planning + Low-Level Action Generation**.
- **Actions**: Stop, Type, Click, Hover.

#### **2. LLM + Visual Encoder + Web Grounding**
- Uses **Set-of-Marks (SoM) prompting** to help agents **interact with UI elements**.
- Alternative to **cluttered HTML trees** and inefficient **accessibility parsers**.

---

## **6. Search & Planning for Long-Horizon Tasks**
One of the biggest **challenges in autonomous agents** is **long-horizon reasoning**.

### **Common AI Failure Modes:**
1. **Looping behavior**: Agents get stuck switching between pages.
2. **Undoing progress**: Performing the right action but reversing it later.
3. **Early stopping**: Ending tasks prematurely.
4. **Visual processing failures**: Misclicking or failing to identify elements.

To address these, the researchers propose **tree search methods** to improve decision-making.

### **Tree Search for Language Model Agents**
- **Baseline Approach:** AI models perform **repeated action sampling** until they reach a solution.
- **Proposed Method:** A **Best-First Search algorithm** with:
  - **Baseline agent** for action proposals.
  - **Backtracking mechanism** to correct errors.
  - **Value function scoring** (using GPT-4o to rank best states).

**Results:**
- **Search-based models significantly outperform baseline LLMs** on long-horizon tasks.
- **However, search methods are computationally expensive and require further optimization**.

---

## **7. AI Agents & Internet-Scale Training (InSTA)**
One of the **biggest challenges in AI agents** is the **lack of high-quality training data**. 

### **Proposed Solution: Synthetic Task Generation**
Researchers use **Llama models** to **generate realistic web tasks** for AI training.

### **Synthetic Data Generation Pipeline:**
1. **Stage 1 – Task Generation**: Llama proposes web-based tasks.
2. **Stage 2 – Task Evaluation**: AI performs tasks and Llama scores them.
3. **Stage 3 – Data Collection**: High-confidence tasks are stored for training.

### **Key Findings:**
- **LLM performance is 68.92% lower than humans** on **VisualWebArena**.
- **Synthetic data significantly improves generalization**:
  - **Mind2Web: +156.3% improvement**
  - **WebLINX: +149.0% improvement**

### **Scaling Up:**
- Researchers use **Common Crawl PageRank** to identify **150k+ useful websites**.
- **AI filtering achieves 97% accuracy** in detecting valid training data.

---

## **8. Robotics & Physical AI Agents**
The lecture briefly extends **multimodal agents** to **robotic manipulation**.

### **Plan-Sequence-Learn (PSL) Framework**
- Uses **structured language plans** for **long-horizon robotic tasks**.
- Combines **reinforcement learning (RL)** with **LLM-guided decision-making**.
- Successfully **generalizes to new object geometries**.

**Results:**
- PSL achieves **85%+ success rates** across **25+ long-horizon robotic tasks**.

---

## **9. Future Directions & AI Safety Considerations**
### **Key Areas for Future Work:**
1. **Improved long-term reasoning**: AI should maintain consistency over **multi-step tasks**.
2. **Multimodal vision-language models**: Stronger **visual grounding** is needed for **real-world applications**.
3. **Parallel execution & task coordination**: Agents should **search, execute, and verify multiple instances** in real-time.
4. **AI Safety**:
   - Ensuring **robustness against adversarial inputs**.
   - Addressing **biases in training data**.
   - Handling **destructive actions** (e.g., real-world purchasing).

---

## **10. Conclusion**
- **VisualWebArena** provides a **realistic benchmark** for evaluating multimodal agents.
- **Inference-time search & tree search** improve **long-horizon AI decision-making**.
- **Synthetic data generation** helps **scale AI training for web-based tasks**.
- **Plan-Sequence-Learn (PSL)** extends AI agents to **robotic manipulation**.

🚀 **The future of AI agents lies in improving multimodal understanding, data efficiency, and safety for real-world applications.**


  ### Ref:
  - https://www.youtube.com/live/RPINOYM12RU
  - https://llmagents-learning.org/slides/ruslan-multimodal.pdf

</details>


## Lecture 7: Multimodal Agents – From Perception to Action

<details>


## 📌 Overview

This document presents a comprehensive review of the latest advancements in **Multimodal Agents (MMAs)**—AI systems that integrate vision, language, and action to operate across complex digital interfaces. It explores new environments, benchmarks, models, and data generation methods that drive progress in AI agents capable of reasoning and taking actions in real or virtual computing environments.

---

## 🚀 Key Themes

### 1. Rise of Multimodal Agents
- Multimodal agents powered by **Vision-Language-Action Models (VLA-Ms)** can execute real-world digital tasks across web, OS, and mobile environments.
- These agents aim to boost digital productivity, accessibility, and autonomy.
- Supported by models like GPT-4V, Claude-3, Gemini-Pro, and open-source Mixtral, CogAgent.

---

## 🧪 Limitations of Existing Benchmarks

### 🔴 Problems:
- Existing platforms (e.g., **Mind2Web**, **WebArena**) are:
  - Non-executable.
  - App/domain specific.
  - Not scalable or realistic.

### ✅ Solution: OSWorld
- **OSWorld** is a VM-based benchmark with:
  - **369 annotated tasks** using real apps and workflows.
  - Full environment simulation (input/output, keyboard/mouse, screenshot, a11y-tree).
  - **Execution-based evaluation** using final environment states.

---

## 👀 Agent Interaction Mechanics

| Component         | Description                                               |
|-------------------|-----------------------------------------------------------|
| Inputs            | Natural language, screenshots, accessibility tree (a11y)  |
| Actions           | Executable keyboard and mouse operations (e.g., pyautogui)|
| Interaction Loop  | Agent runs iteratively until termination condition met    |
| Evaluation        | Compared against expected output ("gold") via config      |

### Observations:
- High-res screenshots improve accuracy.
- Text-based action history > screenshot-only (but less efficient).
- Models show strong OS-level performance correlation but poor layout robustness.

---

## 📉 Data Scarcity & Cost

### Challenge:
- Unlike LLMs trained on vast text corpora, agent models require **expensive, human-annotated trajectory data**.

---

## 🧪 Synthetic Data Solutions

### 1. AgentTrek (Trajectory Synthesis)
- Extracts and classifies **web tutorials** to guide agent replay.
- Parses task steps, environments, outcomes.
- Focused on imitation learning; ideal when combined with **reinforcement learning in OSWorld**.

### 2. TACO + CoTA (Action + Reasoning Data)
- Generates **Chains-of-Thought-and-Action (CoTA)** using programmatic templates.
- Boosts reasoning and action-calling in Multimodal LLMs.
- Actions include: OCR, GETOBJECTS, QUERYKNOWLEDGEBASE, etc.

#### Key Learnings:
- **Quality > Quantity** in training data.
- CoTA fine-tuning consistently outperforms instruction-only training or few-shot prompting.

---

## 🧠 Unified GUI Agent Framework

### AGUVIS (Pure Vision-Based GUI Agent)
- Solves:
  - Heterogeneous textual GUI representation (HTML, AXTree, XML).
  - Lack of visual grounding and inner reasoning.
- Features:
  - Unified vision-based perception and action space.
  - Two-stage training:
    - 1M+ GUI grounding examples.
    - 35K multi-step reasoning tasks with **inner monologue** augmentation.
- Demonstrates strong **cross-platform generalization** (web/mobile → desktop).

---

## 🎥 Long Video Understanding

### Problem:
- Long-form videos → huge token sequences → compute overload.

### Solutions:

#### 1. xGen-MM-Vid (BLIP-3-Video)
- Temporal encoder compresses videos into just **32–128 visual tokens**.
- More efficient than prior SOTA models (e.g., 4608 tokens).
- Scales to long-video question answering and description tasks.

#### 2. GenS (Generative Frame Sampler)
- Trained on **GenS-Video-150K** dataset.
- Predicts **salient frame spans** using instructions and confidence scores.
- Outperforms CLIP-based sampling in temporal QA and event localization.
- Supports **sliding-window inference** with JSON-formatted outputs.

---

## 🧩 Ecosystem Overview

| Module           | Functionality                                             |
|------------------|-----------------------------------------------------------|
| `OSWorld`        | Realistic, execution-based VM environment for agents      |
| `AgentTrek`      | Synthesizes trajectories from tutorial-based knowledge     |
| `TACO + CoTA`    | Reasoning and action-call fine-tuning for MLLMs           |
| `AGUVIS`         | GUI agent with visual-only perception + inner monologue   |
| `BLIP-3-Video`   | Token-efficient video understanding model                 |
| `GenS`           | Intelligent frame selector for long-video processing      |

---

## ✅ Final Takeaways

- **Realistic training environments** are essential (e.g., OSWorld).
- **Synthetic data** (AgentTrek, CoTA) solves scale and diversity issues in agent training.
- **Unified perception-action space + structured reasoning** unlocks general-purpose agents (e.g., AGUVIS).
- **Efficient video models + frame sampling** (xGen-MM-Vid, GenS) allow scalable long-video understanding.

---

## 📣 Quotes from the Source

> “Intelligence grows rapidly, even surpassing humans.”

> “No real, scalable interactive environments... Only demos without executable environment...”

> “OSWorld: The first scalable, real computer environment.”

> “LLMs and VLMs are still far from being digital agents on real computers.”

> “AgentTrek… leverages tutorial-like web content to create step-by-step synthetic trajectories.”

> “CoTA fine-tuning significantly boosts reasoning and action performance, outperforming few-shot prompting.”

> “AGUVIS enables autonomous GUI agents to operate across platforms using only visual observations.”

> “xGen-MM-Vid uses significantly fewer tokens (32 vs. 4608) than other state-of-the-art models.”

> “GenS predicts frame spans as a natural language generation task with confidence scores.”

---

## 📥 Additional Resources

- [OSWorld Benchmark](https://os-world.github.io)  
- [xGen-MM-Vid Paper](https://arxiv.org/abs/2410.16267)  
- [BLIP-3-Video Overview](https://www.salesforceairesearch.com/opensource/xGen-MM-Vid/index.html)

---

  ### Ref:
  - https://www.youtube.com/live/n__Tim8K2IY
  - https://llmagents-learning.org/slides/Multimodal_Agent_caiming.pdf


</details>

## Lecture 8: AlphaProof: RL Meets Formal Mathematics

<details>

**Author**: Thomas Hubert
**Date**: March 2025
**Affiliation**: Google DeepMind
**Event**: Performance at the IMO 2024

---

## 📌 Overview

AlphaProof is a pioneering research project at the intersection of **Reinforcement Learning (RL)** and **Formal Mathematics**, developed by **Google DeepMind**. Inspired by the successes of the **AlphaZero series** in mastering complex environments through scaled-up self-play, AlphaProof applies similar methods to **mathematical theorem proving** using the **Lean proof assistant** and its mathematical library, **Mathlib**.

The central thesis:

> *Formal mathematics provides a perfect environment and feedback signal for RL agents to learn mathematical reasoning, potentially achieving superhuman capabilities and uncovering new mathematical knowledge.*

The project culminated in AlphaProof’s participation in the **International Mathematical Olympiad (IMO) 2024**, where it achieved performance equivalent to a **Silver medallist**, marking a major milestone in AI-driven mathematical discovery.

---

## 🧠 Motivation: Why Formal Maths + RL?

### Mathematics: A Root Node to Intelligence

* Involves **reasoning, generalisation, planning, creativity**, and **open-ended complexity**.
* Even considered to require an *“eye for beauty”*.

### From Ancient Proofs to Code

* Formal proof has evolved from prose (e.g., Babylonian algebra) → symbolic notation → machine-verifiable proofs.
* Computer formalisation brings:

  * **Rigor & clarity**
  * **Efficient communication**
  * **Unification across fields**
  * **Discovery of new theorems**

---

## 🖥️ Key Technologies

### 🧮 Lean

* **Programming language**, **interactive theorem prover**, and **proof assistant**.
* Hosts a growing, **vibrant open-source community**.
* Has formalized even **Fields Medal-level** mathematics.

### 📚 Mathlib

* The core math library for Lean.
* Built entirely open-source by volunteers.
* Covers \~80% of undergraduate curriculum, but with **coverage gaps** (notably in **2D Euclidean Geometry**).

---

## 💡 Computer Formalisation: Opportunities and Limitations

### ✅ Synergies ("Instant Wins")

* **Perfect verification**: Eliminate correctness concerns.
* **Giant proofs**: Can be trusted via automated checking.
* **Education**: Turns mathematics into an interactive "video game".
* **Collaboration**: Allows large-scale coordination, like software engineering.

### ❌ Current Challenges

* <1% adoption among mathematicians.
* Steep learning curve and time investment.
* Tooling and library maturity are still evolving.
* Creative or intuitive steps in proofs remain elusive to current systems.

---

## 🧬 Reinforcement Learning (RL)

### Definition

> "RL is trial and error learning" — agents interact with environments to maximize rewards.

### Proven Track Record

* **AlphaGo**, **AlphaZero**, **MuZero**, **AlphaTensor**, **AlphaStar**, etc.
* Common attributes of success:

  * **Scaled trial and error**
  * **Grounded feedback**
  * **Search and Curriculum**
  * **Tabula rasa learning** ("Zero Philosophy")

---

## 🚀 AlphaProof: The Architecture

### Core Components

| Component              | Function                                                     |
| ---------------------- | ------------------------------------------------------------ |
| **Formaliser Model**   | Translates natural language problems → Lean formalisation    |
| **Prover Model**       | Suggests tactics (actions) based on current Lean proof state |
| **AlphaZero-style RL** | Trains the agent by simulating and verifying proof steps     |
| **Test-Time RL**       | Fine-tunes model on IMO-level problems via curated variants  |

---

## 🧪 The IMO 2024 Apollo Program

### Goal

> *Can AlphaProof solve real IMO problems with sufficient time and compute?*

### IMO Overview

* World’s most prestigious math competition for high school students.
* Only **6 elite students** represent a country out of \~17 million globally.
* Problems are **extremely hard**, test **reasoning** not knowledge, and often take **hours to solve**.

### Execution Timeline

| Date          | Highlights                                                                                |
| ------------- | ----------------------------------------------------------------------------------------- |
| **Jan 2024**  | Realized 2D geometry coverage in Mathlib was weak. Joined forces with **AlphaGeometry**.  |
| **Mar 2024**  | Decided to run in “hard mode”: AlphaProof must **generate** and **prove** answers itself. |
| **Jul 16-17** | AlphaProof and AlphaGeometry tackled all 6 IMO problems using formalised inputs.          |
| **Jul 18-19** | Progress reviewed; email logs show post-contest improvements and completed proofs.        |

---

## 🧾 Performance Summary

### Scores

| Problem | Domain        | Result           | Agent           | Points    |
| ------- | ------------- | ---------------- | --------------- | --------- |
| P1      | Algebra       | Partially Solved | AlphaProof      | 3/7       |
| P2      | Number Theory | Largely Solved   | AlphaProof      | 6/7       |
| P3      | Combinatorics | No Progress      | AlphaProof      | 0/7       |
| P4      | Geometry      | Fully Solved     | AlphaGeometry   | 7/7       |
| P5      | Combinatorics | No Progress      | AlphaProof      | 0/7       |
| P6      | Algebra       | Partially Solved | AlphaProof      | 2/7       |
|         |               |                  | **Total Score** | **18/42** |

> “We reached the score of a Silver medallist and missed the Gold threshold by one point (with more time and more compute!).”

---

## 🧠 Methodology Pipeline

### 1. **Auto-Formalisation**

* Train a formaliser to translate millions of natural language problems into Lean.

### 2. **Supervised Pretraining**

* Learn from Mathlib’s 300k+ lines of human-written proofs.

### 3. **AlphaZero RL**

* Use Lean to simulate proof steps.
* Update the prover with reward for each successful proof.

### 4. **Test-Time RL**

* Generate **problem variants**.
* Focus train the agent to tackle IMO-level “hard” problems using nearby easier examples.

---

## 🧗 Challenges

* **Data mismatch**: Most mathematical knowledge is in *natural language*, not formal systems.
* **Coverage gaps** in Mathlib (especially geometry and combinatorics).
* **Computational cost**: Orders of magnitude greater than human effort.
* **Creativity** and **mathematical elegance** remain difficult to encode.

---

## 🔮 What’s Next?

* Expand coverage to the **entire mathematical landscape**.
* Contribute to **frontiers of mathematical research**.
* Make AlphaProof a **tool for every mathematical thinker**, not just a benchmark system.

> “Individually, this was almost impossible. Together, it felt impossible to fail.”

---

## 🧵 Key Quotes

* *"Mathematics, a root node to intelligence?"*
* *"Computer Formalisation unlocks enormous synergies"*
* *"Transforms mathematics into a video game 🎉!"*
* *"Perfect verification will in the long run be the most important property for mathematics."*
* *"SuperScale RL: A Proven Recipe to Superintelligence"*
* *"Lean gives us a way to scale up trial and error..."*
* *"I find the fact that the program can come up with a somewhat complicated construction like this very impressive."* – Prof. Sir Timothy Gowers

---

### Ref:
- https://www.youtube.com/live/3gaEMscOMAU
- https://llmagents-learning.org/slides/alphaproof.pdf


</details>

## Lecture 9: Formal Reasoning Meets LLMs in Math and Verification

<details>


# 🧠 AI for Mathematics and Verification

**Source**: *“L9\_Language\_Models\_For\_Autoformalization\_And\_Theorem\_Proving.pdf” by Kaiyu Yang (Meta FAIR)*
**Date**: Late 2024 – Early 2025 (inferred from citations)
**Subject**: Application of Large Language Models (LLMs) to mathematical reasoning, formal proof generation, and automated verification.

---

## 🎯 Motivation: Why Math and Coding?

Mathematics and programming are ideal domains for testing LLM capabilities because they:

* Act as **proxies for complex reasoning and planning**.
* Offer **automatic evaluation**:

  * **Math**: answers can be directly verified.
  * **Code**: correctness is testable via unit tests.
* Represent a structured path toward general AI reasoning and planning.

---

## 🛠️ How LLMs Are Trained for Mathematical Reasoning

### 1. **Supervised Finetuning (SFT)**

* Uses curated datasets of math problems with solutions (step-by-step or tool-integrated).
* Leverages tools like `sympy` for symbolic reasoning.
* Datasets reach \~900K problems but often lack intermediate reasoning steps.
* **Slogan**: *“Good data is all you need!”*

### 2. **Reinforcement Learning (RL)**

* Optimizes LLMs using **verifiable final answers** as a reward signal.
* Especially useful for numeric problems (e.g., DeepSeekMath).
* Inadequate for proofs where final answers aren’t scalar.
* **Slogan**: *“Verifiability is all you need!”*

> **State-of-the-art LLMs ≈ Pretrained foundation model + Finetuning + RL + Engineering**

---

## 🚧 Current Gaps and Limitations

### Gap 1: **Pre-College → Advanced Math**

* LLMs excel in contests like AIME or USAMO.
* Struggle with **research-level math** involving abstract reasoning and deep theorems.

### Gap 2: **Answer Guessing → Proof Generation**

* LLMs often produce plausible but **invalid or incomplete proofs**.
* Unable to **bridge logical gaps** common in informal proofs.

### Root Causes:

* **Data Scarcity**: Advanced math proofs are rarely available in structured form.
* **Lack of Verifiability**: Current evaluation lacks fine-grained, formal correctness checking.

---

## 🔍 The Missing Ingredient: Formal Mathematical Reasoning

To overcome these limitations, the document advocates for integrating **formal logic-based systems**:

* **Foundations**:

  * First-order / higher-order logic
  * Dependent type theory
  * Formal specs in programming languages (e.g., Lean)

### Benefits of Formal Systems:

* **Verification**: Ensures logical soundness, eliminating hallucination.
* **Feedback**: Auto-checkable proofs enable iterative learning.
* **Scarcity Mitigation**: Feedback can substitute for labeled training data.

---

## 🧰 Proof Assistants and the Role of Lean

* **Lean** is a proof assistant used to write **machine-checkable formal mathematics**.
* Provides:

  * Formal syntax and semantics.
  * Tactic-based proof construction (e.g., `induction`, `simp`, `rfl`).
* Lean formalizations mirror structured programming, making them suitable for AI learning.

---

## 🤝 AI Meets Formal Mathematics

### Key Projects:

#### 🧠 **AlphaProof**

* Combines **LLMs + Reinforcement Learning + Lean**.
* Proves theorems using search-based strategies and feedback from formal verification.
* Avoids hallucination by deferring correctness checking to Lean.

#### 🧪 **LeanDojo** (Open Source)

* A comprehensive platform offering:

  * \~98K theorems
  * \~217K proof tactics
  * Data tools, trained checkpoints, and Lean interop.
* Benchmarks and trains LLMs for theorem proving in Lean.

#### 🔍 **ReProver (Retrieval-Augmented Proving)**

* Retrieves relevant premises to guide proof generation.
* Reduces search complexity by narrowing context.
* Embeds premise selection and tactic generation in a neural pipeline.

---

## ⚖️ Tackling Vast Action Spaces: LIPS

**LIPS** = *LLM-based Inequality Prover with Symbolic reasoning*

* A hybrid system that handles **Olympiad-level inequality proofs**.
* Combines:

  * **Symbolic engines** for scaling tactics (e.g., Cauchy-Schwarz)
  * **LLMs** for rewriting and ranking plausible proof states
* Outperforms IMO gold medalists and discovers new, human-inaccessible strategies.

---

## 🔄 Autoformalization: From Informal to Formal Math

### Tasks:

1. **Theorem Translation**: Informal → Formal statement
2. **Proof Translation**: Informal + formal theorem → Formal proof

### Major Challenges:

* **Evaluation Difficulties**: No automatic equivalence checker for formalized outputs.
* **Reasoning Gaps**: Informal math relies on assumptions, skipped steps, and diagrams.

---

## 🧭 Euclidean Geometry as a Case Study

### Why Geometry?

* Relies heavily on **diagrammatic reasoning** and implicit logic.
* Informal proofs are rich in **unstated assumptions**.

### Contributions:

* **LeanEuclid**: First faithful formalization of Euclid’s *Elements* and UniGeo dataset.
* **System E**:

  * Models **diagrammatic rules**.
  * Uses **SMT solvers** to bridge visual-logical gaps in proofs.

---

## 📌 Final Takeaways

### Core Ideas:

* Math and code are **perfect testbeds** for reasoning due to their evaluability.
* **LLMs are great at answers, not yet at rigorous reasoning.**
* Formal systems like **Lean are essential** for reliable mathematical AI.
* **Hybrid models (symbolic + neural)** show promise in narrow domains (e.g., LIPS).
* **Autoformalization** is a promising but nascent field, limited by evaluation and generalizability.

### Grand Challenge:

> **Can we generalize autoformalization across all domains of mathematics?**
> This remains an open frontier in AI research.

---

### Ref:

- https://llmagents-learning.org/slides/mathverification.pdf
- https://www.youtube.com/live/cLhWEyMQ4mQ


</details>

## Lecture 10: Bridging Informal and Formal Mathematical Reasoning

<details>

**Author**: Sean Welleck
**Institution**: Carnegie Mellon University
**Date**: April 14, 2025
**Source**: *L10\_Bridging\_Informal\_and\_Formal\_Mathematical\_Reasoning.pdf*

---

## 📌 Executive Summary

This presentation explores how Artificial Intelligence (AI)—specifically Large Language Models (LLMs)—is being used to bridge the *informal-formal gap* in mathematics. While informal mathematics is flexible and intuitive but unstructured, formal mathematics offers rigorous, checkable proof systems like Lean. The talk identifies three primary strategies to unify these domains:

1. **Training models to generate informal "thoughts"** that guide formal reasoning (Lean-STaR).
2. **Developing hybrid informal-formal provers** (Draft-Sketch-Prove, LeanHammer).
3. **Supporting research-level formalization projects** through contextual tools (MiniCTX).

These tools and methods show promising advances in both accuracy and usability, helping bring AI closer to assisting human mathematicians in authentic research workflows.

---

## 🔍 Key Themes and Ideas

### 🧠 AI in Expert Domains

* AI agents are increasingly used in **finance, medicine, and mathematics**.
* In math, AI can:

  * Engage in **open-ended dialogue**
  * Generate **counterexamples**
  * Assist in **writing formal proofs**

---

## ✍️ Informal vs. Formal Mathematics

| Aspect          | Informal Mathematics                     | Formal Mathematics                          |
| --------------- | ---------------------------------------- | ------------------------------------------- |
| Format          | Natural language, images, intuition      | Code-like proofs (e.g. Lean, Coq, Isabelle) |
| Characteristics | Flexible, expressive, but hard to verify | Rigid, verifiable, “math as source code”    |
| Tooling         | Language models (e.g., GPT-4)            | Proof assistants and theorem provers        |
| Challenge       | Difficult to check                       | Difficult to write                          |

Visual examples on pages 4–6 show how both types are expressed and used in practice.

---

## 📈 The Rise of Formal Methods

* **Lean Mathlib**: 1M+ lines of code, 300+ contributors
* **Terence Tao’s formalization project** (Oct 2023) serves as a case study
* Benefits of formal math:

  * **Collaboration** through modular problem solving
  * **Trustworthy output** via automatic verification
  * **Instant feedback** and **guaranteed correctness**

---

## 🤝 Why Formal Math Matters for AI

* Prevents incorrect code or math generation
* Offers verifiable reasoning benchmarks
* Serves as a **feedback mechanism for model training**
* Supports reasoning tasks from "1 + 1 = 2" to "Fermat’s Last Theorem"

---

## 🔧 Bridging the Informal–Formal Gap

### 1. **Informal Thoughts – Lean-STaR**

* **Goal**: Train LLMs to “think” before applying formal tactics.
* **Method**: Uses reinforcement learning to generate intermediate thoughts.
* **Impact**: Boosts performance on **miniF2F** benchmark tasks.
* **Result**: Inspired adoption in systems like DeepSeek Prover and OpenAI’s o1 model.

📌 *Key Insight*: More expressive thoughts lead to better proof search (page 46).

---

### 2. **Informal Provers – Draft-Sketch-Prove & LeanHammer**

#### 🔹 Draft-Sketch-Prove (DSP)

* **Workflow**:

  1. Draft an informal proof (LLM or human).
  2. Convert it into a formal sketch.
  3. Use a **low-level prover** (Sledgehammer) to fill in gaps.
* **Analogy**: Like a mathematician writing an outline, then formalizing it with rigorous steps.

#### 🔹 LeanHammer

* Builds a “hammer” for **Lean**, integrating:

  * Neural **premise selection**
  * **Tree search** (Aesop)
  * Automated Theorem Provers (ATP)
* **Premise selection** is framed as a **retrieval task** using transformer models.
* **Command**: Users can simply type `hammer` to fill proof steps.

📊 *Proof rate improves dramatically* with LeanHammer vs. no retriever (page 89).

---

### 3. **Research-Level Mathematics – MiniCTX**

* **Problem**: Benchmarks like IMO are too “clean.”
* **Solution**: MiniCTX evaluates on real Lean repositories:

  * Projects: **PFR**, **PrimeNumberTheorem**, etc.
  * Context-aware: evaluates both **in-file** and **cross-file** dependencies.
* **Methods Compared**:

  * “State-tactic tuning” vs. “File tuning”
  * File tuning consistently outperforms in real-world settings.

📎 Tool integration: **LLMLean** combines MiniCTX-trained models with Lean IDEs for real-time suggestions.

🛠 Open-Source:

* [GitHub: LLMLean](https://github.com/cmu-l3/llmlean)
* [Data/models: Hugging Face](https://huggingface.co/l3lab)

---

## 📌 Noteworthy Quotes

> **“Code compiles ≡ correct proof”**
> **“Can we train a model to ‘think’ before each step of formal reasoning?”**
> **“Draft an informal proof, translate it into a formal sketch, then use a low-level prover to fill in the gaps.”**
> **“Premise selection” is the key challenge for hammers.**
> **“Test models on real Lean projects” – MiniCTX**

---

## 🧠 Key Takeaways

* **AI can now meaningfully assist in mathematics**, moving beyond rote computation to logical reasoning.
* **Lean-STaR** demonstrates the value of training LLMs to generate intermediate “thoughts.”
* **Draft-Sketch-Prove and LeanHammer** provide practical mechanisms to automate theorem proving.
* **MiniCTX** benchmarks the ability of AI to handle real-world formalization scenarios.
* The field is moving toward **AI-supported collaborative formalization**, not just educational use cases.

---

### Ref:

 - https://llmagents-learning.org/slides/welleck2025_berkeley_bridging.pdf
 - https://www.youtube.com/live/Gy5Nm17l9oo

</details>

## Lecture 11: Abstraction and Discovery with Large Language Model Agents

<details>

**Title**: *Abstraction and Discovery with Large Language Model Agents*
**Author**: Swarat Chaudhuri (University of Texas at Austin)
**Context**: Exploring how LLM agents accelerate mathematical and scientific discovery via abstraction, reasoning, and concept learning.

---

## 🎯 Core Theme: Leveraging LLM Agents for Enhanced Discovery

LLM agents are presented as **powerful tools for automated discovery**, enabled by four key capabilities:

1. **Systematic exploration** of hypotheses, conjectures, and proofs
2. **Use of prior knowledge** to guide search
3. **Learning from experience** to improve search strategies
4. **Discovery of abstract tools and concepts** that boost both search and learning efficiency

> 📌 *"This talk: LLM agents with all four capabilities."* – Page 4

---

## 🧮 Section I: Mathematical Discovery with LLMs

### 🧠 Key Problems in the Neural-Only Approach

* **Data Scarcity**: Difficult to obtain structured reasoning traces beyond high-school level problems
* **Lack of Verifiability**: Natural-language reasoning is hard to validate in domains like formal verification

> *“Natural-language reasoning is hard to verify. In applications like system verification, edge cases are especially critical.”* – Page 8

---

### 🧩 Formal Reasoning Pipeline

A better alternative involves using **formal representations** and **proof assistants**:

#### ✅ Example Tools:

* **Neural Autoformalizer**: Converts informal math problems into formal statements
* **Neural Prover**: Suggests tactics using the context of formal math libraries (e.g., Lean)
* **Formal Proof Assistant**: Validates the generated proofs

🖼️ *Illustrated on Page 9–12 with a Lean proof of: “if a number is even, its square is even”*

---

### 🔁 Reinforcement Learning & Copra

#### 🔸 **AlphaProof** (DeepMind)

* Combines formalization + reinforcement learning to learn from **both successful and failed** proofs
* Uses test-time RL on problem variants

#### 🔸 **Copra**: In-Context Learning Agent for Theorem Proving

* Integrates formal proof reasoning with LLMs via:

  * Prompt synthesis
  * Tactic parsing
  * Execution in proof environments
  * Augmentation via lemma databases
* **Outperforms GPT-4** on theorem correctness (Page 18–19)

> 📊 *"Copra (+ Retrieval + Informal)" achieves \~30.74% on miniF2F test vs. \~6% from GPT-3.5 Few-Shot* – Page 19

---

### 🧠 Hierarchical Problem Solving

A unique pipeline integrates **natural-language and formal reasoning** by:

1. Asking an LLM for an informal outline
2. Splitting it into formal subgoals
3. Letting Copra solve each sequentially

🧪 **Example**: IMO 1959 Problem on irreducibility of a fraction – Pages 21–24

---

### 🧾 Application: **Compiler Verification**

A real-world example where:

* A source/target language is formally defined
* A compiler is implemented as a translator
* The LLM initially **fails to prove correctness**, but:

  * Invents an auxiliary lemma
  * Automatically proves both the lemma and the final theorem

🛠️ *Detailed walkthrough on Pages 26–39*

---

## 🔬 Section II: Scientific Discovery

### 🔁 Scientific Process

Follows classical lifecycle:
**Problem → Hypothesis → Data → Analysis → Interpretation**

🛰️ **Example**: Heliocentric model, Tycho Brahe’s Mars observations, Kepler's Law → Newton's Law
🖼️ Visuals of this process on Pages 47–50

---

### 📈 Symbolic Regression

#### ✅ PySR:

* Uses evolutionary algorithms to discover symbolic equations from data

#### 🚀 LaSR (Symbolic Regression + LLMs):

A new system combining symbolic search with LLM-guided evolution

#### LaSR Innovations:

* **Concept Library**: Abstractions like “Power Law Trend”, “Exponential Growth”
* **LLM Guidance**:

  * Crossover / Mutation / Initialization
  * Concept abstraction & evolution
* **Joint Concept & Program Learning**:

  * Evolving equations *and* abstract mathematical ideas simultaneously
    🖼️ Visual “islands” of search space: Pages 55, 69–72

> *"Concept guidance accelerates discovery."* – Page 73
> *"Smaller models synthesize simpler equations!"* – Pages 75–76 (Coulomb’s Law example)

---

## 📉 Finding LLM Scaling Laws with LaSR

### 🔍 Standard Approach (e.g., Hoffman et al.):

* Postulate scaling law
* Measure model loss vs. training hyperparameters
* Fit best function

### 🔬 LaSR Approach:

* Uses symbolic regression to fit *simpler, more generalizable scaling laws*
* Applied to BIG-Bench (204 tasks, 55 LLMs)
* Captures trends like:

  * *“More shots hurt low-capability models but help high-capability ones.”* – Page 80

---

## 👁️ Visual Discovery with LLMs & VLMs

### 🔎 Zero-Shot Transfer

* Uses **visual concept descriptors** like “white head”, “stacked containers” for identification (e.g., eagle, container ship)
  🖼️ Pages 83–85

### 🔁 Concept Evolution

* VLMs compute similarity scores between visual features and concept descriptors
* Refinement through **contrastive learning + evolutionary feedback**
  🖼️ *Beignet vs. Donut misclassification improved via concept refinement* – Page 87–88

---

## 🧠 Summary: The Power of LLM Agents

* **Mathematical Discovery**:

  * LLMs + proof assistants → non-trivial formal theorem solving
  * Hierarchical + lemma-driven proof construction

* **Scientific Discovery**:

  * Symbolic regression, concept abstraction, and neural evolution
  * Enhanced interpretability and accelerated knowledge generation

* **Visual Reasoning**:

  * Vision-language critics enable self-evolving visual concept libraries

---

## 🚧 Open Challenges & Future Directions

| Area              | Challenge                                   | Strategy                                         |
| ----------------- | ------------------------------------------- | ------------------------------------------------ |
| **Data**          | Lack of high-quality proof traces           | Synthetic data, crowdsourcing, multilingual data |
| **Formalization** | Converting informal → formal logic          | Process-driven autoformalization (Lu et al.)     |
| **Exploration**   | Open-ended conjecturing                     | Self-play between conjecturer & prover           |
| **Verification**  | Validating hypotheses & concepts            | More rigorous semantics & testing loops          |
| **Scalability**   | Expanding input space and design complexity | Hierarchical, modular agent designs              |

---

## Ref:

 - https://llmagents-learning.org/slides/swarat.pdf
 - https://www.youtube.com/live/IHc0TEMrEdY

</details> 

## Lecture 12: Towards building safe and secure agentic AI

<details>

**Authors**: Dawn Song, Xinyun Chen, Kaiyu Yang
**Institution**: UC Berkeley
**Source**: *L12\_Towards\_Building\_Safe\_And\_Secure\_Agentic\_AI.pdf*
**Date**: December 2024 (based on context)

---

## 🧭 Executive Summary

This document addresses the **urgent need to secure agentic AI systems**—complex systems powered by large language models (LLMs) capable of autonomous action, reasoning, and interaction. As these systems become more powerful and prevalent in 2025, so do the associated risks. The report provides a framework for understanding and defending against misuse, malfunction, and adversarial attacks across the agentic AI lifecycle.

---

## 🚀 1. The Rise of Agentic AI & Emerging Risks

> **“2025 is the year of Agents”** — Agents are transforming AI from passive tools into active systems that can act in the world.

### Key Risks:

* **Misuse/Malicious Use**: Scams, misinformation, cyber offense, child abuse material, weaponization (e.g., bioweapons).
* **Malfunction**: Systemic biases, inappropriate deployment, and unintended consequences.
* **Systemic Risks**: Privacy violations, copyright infringement, labor market disruption, and systemic failure.

### Attack Context:

> “History has shown attacker always follows footsteps of new technology.”

As AI agents gain real-world control, attacker incentives grow. Thus, **security must be an integral part of system design from the start**.

---

## 🧱 2. Distinguishing Safety vs. Security

| **AI Safety**                           | **AI Security**                                |
| --------------------------------------- | ---------------------------------------------- |
| Prevents harm caused *by* the AI system | Protects the AI system *from* malicious actors |

The document emphasizes the need for **safety mechanisms to be secure themselves**—e.g., **alignment techniques must withstand adversarial prompt manipulation**.

---

## 🧠 3. Understanding Agentic Systems & Their Vulnerabilities

### Agentic Hybrid Systems:

These are **compound architectures** integrating:

* LLMs
* Symbolic and neural components
* Memory, retrieval, tools, code execution
* Real-world actions and feedback loops

### CIA Goals Extended:

| Traditional Goal | Agentic Extension                         |
| ---------------- | ----------------------------------------- |
| Confidentiality  | Model inputs, prompts, memory, outputs    |
| Integrity        | Model behavior, poisoned data, tool use   |
| Availability     | DoS on agents, long-term task reliability |

### Failure Points:

At each stage—user input, system processing, model execution, external world interaction—**LLM output can be hijacked** for:

* **SQL injection**
* **Remote Code Execution (RCE)**
* **System misuse**

---

## 🧨 4. Attack Vectors in Agentic AI

### Attack Chain Roles of LLM Output:

1. **As UI output** → Info leakage
2. **As computation parameter** → Error propagation
3. **As conditional logic** → Control flow hijacking
4. **As API argument** → Injection attacks (SQL, SSRF)
5. **As executable code** → Arbitrary execution

### Attack Types:

* **Prompt Injection** (Direct/Indirect)
* **System Prompt Leakage** (e.g., Bing Chat)
* **Backdoors** (e.g., via RAG – *AgentPoison*)
* **Code Injection** (via tools like `llama_index`, `SuperAGI`)

> *Model Security Levels*:
> From **L0 (perfect and secure)** to **L4 (malicious by design)**.

---

## 🧪 5. Evaluation & Risk Assessment

### 🔍 Evaluation Frameworks:

| Tool              | Focus                                                  |
| ----------------- | ------------------------------------------------------ |
| **DecodingTrust** | LLM trustworthiness (performance, robustness, privacy) |
| **MMDT**          | Safety of Multimodal Foundation Models                 |
| **RedCode**       | Code agent risk (generation + execution)               |
| **AgentXploit**   | Fuzzing-based red-teaming of black-box agents          |

> **AgentXploit** uses fuzzed seed mutations, MCTS-based scoring, and was shown to double attack success rates vs. hand-crafted attacks.

---

## 🛡️ 6. Defense Principles

1. **Defense-in-Depth**
2. **Least Privilege & Separation**
3. **Safe-by-Design / Secure-by-Design**
4. **Formal Verification**

---

## 🧰 7. Defense Mechanisms (8-Part Framework)

| # | Mechanism                           | Description                                                |
| - | ----------------------------------- | ---------------------------------------------------------- |
| 1 | **Harden Models**                   | Robust training, alignment, data cleaning, unlearning      |
| 2 | **Input Guardrails**                | Validate, sanitize, normalize all prompts                  |
| 3 | **Policy Enforcement**              | Enforce least privilege on tool/API calls                  |
| 4 | **Privilege Management**            | Identity-based access control for users/agents             |
| 5 | **Privilege Separation**            | Modularize agents into sandboxes (e.g., via **Privtrans**) |
| 6 | **Monitoring & Detection**          | Logging, real-time anomaly detection                       |
| 7 | **Information Flow Tracking**       | Prevent unauthorized data propagation                      |
| 8 | **Secure-by-Design + Verification** | Prove system correctness under all input types             |

> **Progent**: A key tool enabling **programmable privilege control** for LLM agents, combining static (human) and dynamic (LLM-generated) security policies using a DSL.

---

## 🧩 Open Challenges Identified

* How to define **formal specifications** for LLMs?
* How to conduct **real-time monitoring** without prohibitive storage costs?
* How to **secure tool boundaries** during agent execution?
* How to manage **privileges and identities** in **multi-agent ecosystems**?

---

## 📌 Conclusion

Agentic AI systems are powerful yet vulnerable. The document calls for:

* **Stronger security architecture**
* **New red-teaming techniques**
* **Formal, provable guarantees**
* **Cross-disciplinary effort** bridging AI, security, and systems design

> The message is clear: to realize the promise of agentic AI, we must **design for safety in adversarial settings from day one**.


## Ref:

- https://llmagents-learning.org/slides/dawn-agentic-ai.pdf
- https://www.youtube.com/live/ti6yPE2VPZc

</details>


