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



