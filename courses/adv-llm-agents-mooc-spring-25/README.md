# Advanced LLM Agents, MOOC, Spring 2025

https://llmagents-learning.org/sp25

## Lecture 1: Inference-Time Techniques for LLM Reasoning

<details>

### Overview:

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


### Briefing:

# Briefing Document: Inference-Time Techniques for LLM Reasoning

## Introduction

This document summarises a lecture on inference-time techniques for enhancing the reasoning capabilities of Large Language Models (LLMs). The lecture highlights the significant advancements in LLM reasoning, particularly with models like OpenAI's "o1" and "o3", which demonstrate impressive performance on complex tasks like math, coding, and STEM. However, these high levels of performance are often achieved by using substantial inference-time computation. The lecture explores various strategies to optimise this, categorising them into three main areas: using more tokens for a single solution, searching and selecting from multiple candidates, and iterative self-improvement.

## Part 1: Basic Prompting Techniques - Increasing Token Budget for Single Solution

### Standard Prompting Limitations
- Prior to advanced post-training techniques, standard prompting struggled with reasoning benchmarks.
- Few-shot examples only provided the format of the final solution, not the reasoning behind it.

### Chain-of-Thought (CoT) Prompting
- Prompts the model to generate reasoning steps before arriving at the final solution.
- Can be implemented via few-shot examples or instructions.
- **Scaling with Model Size:** CoT performance improves significantly with larger models.
- **Zero-Shot CoT:** Using instructions like "Let's think step by step" can elicit CoT without needing few-shot examples, though it is less effective.

### Analogical Prompting
- Enhances CoT performance by instructing the model to recall relevant exemplars before solving the test problem.
- Outperforms both zero-shot CoT and manual few-shot CoT, particularly with stronger models.

### LLM-Driven Prompt Optimisation
- Leverages LLMs to automatically design and optimise prompts.
- Uses past optimisation trajectories to generate improved instructions.
- A meta-prompt enables the LLM to propose new instructions based on previous ones.
- Optimised prompts can outperform standard zero-shot and few-shot prompts.

### CoT & Reasoning Strategies
- CoT allows variable computation based on task complexity.
- **Least-to-Most Prompting:** Decomposes complex problems into simpler sub-tasks solved sequentially.
- **Dynamic Least-to-Most Prompting:** Customises prompts for each sub-problem.
- **Self-Discover:** Instructs the LLM to autonomously compose reasoning structures without manually created demonstrations.

## Part 2: Search and Selection from Multiple Candidates - Increasing Width of Solution Space

### Rationale
- Exploring multiple branches allows the model to recover from single-generation errors.

### Self-Consistency
- Generates multiple candidate solutions and selects the most consistent final answer.
- Effective across models and tasks, scaling well with the number of samples.
- **Diversity in Sampling:** Ensures response variety using high-temperature sampling instead of beam search.

### Clustering by Execution (AlphaCode)
- In code generation, predicted code is clustered based on execution consistency.
- Improves performance by selecting a program from the largest semantically equivalent clusters.

### Universal Self-Consistency (USC)
- Extends self-consistency to free-form generation tasks, where consistency is evaluated within the LLM itself.

### LLM Rankers
- Training verifiers or reward models enhances solution selection.
- **Outcome-Supervised Reward Model (ORM):** Evaluates final solutions.
- **Process-Supervised Reward Model (PRM):** Evaluates step-by-step reasoning.

### Tree-of-Thoughts (ToT)
- Combines LLMs with tree search.
- Generates possible next reasoning steps, evaluates each, and prioritises promising solutions.
- Scales well with increased token budget.

## Part 3: Iterative Self-Improvement - Increasing Depth of Solution Search

### Rationale
- Mistakes occur even in strong LLMs.
- Sampling multiple solutions is insufficient without a feedback loop for error correction.

### Reflexion and Self-Refine
- The LLM generates feedback on its own output and refines it.
- Effective when external evaluation is available.

### Self-Debugging (Code)
- Uses execution feedback, such as unit tests, to improve generated code.
- More informative feedback yields better results.

### Limitations of Self-Correction (QA)
- Self-correction without an oracle verifier can reduce accuracy.
- General-purpose feedback prompts and multi-agent debates are often ineffective.

### Budget Optimisation
- Optimal inference budget depends on the task and model.
- Smaller models may generate more solutions within the same computational budget.

## Key Takeaways and General Principles

- **Adaptability:** Best practices for LLM interaction should evolve with model capabilities.
- **Chain-of-Thought:** Fundamental for reasoning enhancement.
- **Consistency-Based Selection:** A simple yet effective principle for better response selection.
- **Search:** Exploring multiple solution paths improves accuracy.
- **The "Bitter Lesson":** Emphasises general-purpose methods that scale well with computation.

## Conclusion

The lecture provides a comprehensive overview of inference-time techniques for improving LLM reasoning. These techniques focus on:
1. **Using more token budget for better single-solution generation.**
2. **Searching multiple branches in the solution space.**
3. **Iterative self-improvement of responses.**

They range from basic CoT prompting to advanced methods like tree-of-thought and self-debugging. The lecture underscores the importance of continuous adaptation, scalable general-purpose methods, and fostering models capable of independent discovery rather than pre-programmed intelligence.

</details>

## Lecture 2: Self-Improving and Reasoning with Large Language Models

<details>

## Introduction
This document outlines the research and development of self-improving and reasoning Large Language Models (LLMs), which aim to create AI that trains itself, evaluates its performance, and updates itself based on its understanding. The ultimate goal is to achieve superhuman performance through these methods.

## System 1 vs System 2
The document introduces two systems for how LLMs function, System 1 and System 2:
*   **System 1**: This is reactive, relies on associations, has fixed compute per token, directly outputs answers, and is prone to failures like hallucinations and spurious correlations. Standard LLMs are considered System 1.
*   **System 2**: This is more deliberate and effortful, involving multiple "calls" to the System 1 LLM. It uses planning, search, verification, and reasoning with dynamic computation. Techniques like Chain-of-Thought (CoT) and Tree-of-Thoughts (ToT) fall under System 2.

## Historical Context and Evolution of LLMs
The document provides a brief history of LLMs and related technologies:
*   **Early 2000s:** Support Vector Machines were prevalent.
*   **2014:** The LLM attention mechanism was developed.
*   **2019-2023:** A rapid evolution of LLMs occurred, from GPT-2 to GPT-4, including models like T5, Jurassic-1, Megatron-Turing NLG, Gopher, Chinchilla, PaLM, OPT, BLOOM, and LLaMA.
*   **Pre-2020:** Language models were trained by predicting the next token on "positive examples" of language.
*   **Post-2020:**  LLMs began using techniques like supervised fine-tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF).
*   **2022:** InstructGPT was developed using SFT and RLHF on GPT3.
*   **2023:** Models like Claude and GPT-4 began using extensive RLHF for safety and accuracy, and Direct Preference Optimization (DPO) was introduced.

## Improving Reasoning with System 2
*   **Prompting Approaches:** Early attempts to improve reasoning focused on prompting techniques.
*   **Chain-of-Verification (CoVe):** This method aims to reduce hallucinations by adding verification steps to the generation process. It includes variants like joint left-to-right generation, factored attention, and a factored-revise approach.
*   **System 2 Attention (S2A):** This method focuses on making attention more explicit and effortful by prompting the LLM to rewrite inputs, removing irrelevant or biased content, to improve the relevance of answers.
*   **Branch-Solve-Merge:** This approach breaks down complex tasks into subproblems, solves them individually, and merges the solutions to improve complex tasks where instructions are hard.

## Self-Improvement and Self-Rewarding LLMs
*   **Self-Training:** LLMs improve by assigning rewards to their own outputs and optimizing accordingly.
*   **Self-Rewarding LMs:** These models are trained to have both instruction-following and evaluation capabilities. They can generate responses to instructions and judge the quality of those responses, creating an iterative process of improvement.
*   **Iterative Training:** This involves two steps: self-instruction creation (generating prompts, responses, and self-rewards) and instruction training (using DPO on selected preference pairs).
*   **The "Superalignment challenge":** As LLMs improve they will become harder for humans to supervise.
*   **Initial Model:** Experiments start with a pre-trained LLaMA-2-70B model (M0) which is multitask trained using seed instruction following (IFT) and evaluation data (EFT). The model then goes through iterative training.
*    **Evaluation:** The self-rewarding models are evaluated on their ability to follow instructions and their ability to act as a reward model. The models are tested using internal instruction following test sets, AlpacaEval 2.0, and MT-Bench. They are also evaluated using the OpenAssistant validation set.
*    **Improvements:** The models show continuous improvement through iterative training.

## Iterative Reasoning and Meta-Rewarding
*   **Iterative Reasoning Preference Optimisation:** This technique uses self-rewarding techniques for reasoning tasks by generating multiple Chain-of-Thoughts (CoTs) and selecting preferences based on answer correctness.
*    **Thinking LLMs:**  This approach trains LLMs to think and respond for all instruction following tasks, not just math, using Thought Preference Optimization (TPO). It has achieved strong results on benchmarks like AlpacaEval and ArenaHard.
*   **Meta-Rewarding LLMs:** These models improve their judgments by meta-judging them. The LLM acts as an actor, judge, and meta-judge. Meta-rewards provide an additional training signal.
*   **LLM-as-a-Meta-Judge:** This is used to assess judgments. The method involves generating multiple judgments for pairs of responses and calculating pairwise meta-judgments.
*   **EvalPlanner:** This method trains LLMs to generate planning and reasoning CoTs for evaluation, converting evaluation tasks into verifiable tasks by generating similar prompts with high and low quality responses.

## Future Directions
*   **Latent System 2 Thoughts:** Explores the use of latent thoughts rather than tokens, with research into self-evaluation and learning from interaction.
*   **Improved System 1:** Research is needed to improve the fundamental architecture of System 1, such as better attention mechanisms and world models.

## Conclusion
The document highlights the significant progress in developing self-improving and reasoning LLMs. By using techniques like self-rewarding, iterative training, and meta-reasoning, LLMs are approaching and potentially surpassing human-level performance. Further research is needed to address limitations, improve reasoning, and explore the potential of more advanced approaches.
</details>



