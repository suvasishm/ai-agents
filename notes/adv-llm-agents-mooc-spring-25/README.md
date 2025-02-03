# Advanced LLM Agents, MOOC, Spring 2025

https://llmagents-learning.org/sp25

## Lecture 1: Inference-Time Techniques for LLM Reasoning

<details>
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
</details>



