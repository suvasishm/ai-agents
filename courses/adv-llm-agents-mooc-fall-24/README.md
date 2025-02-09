# Advanced LLM Agents, MOOC, Fall 2024

https://llmagents-learning.org/f24

## Lecture 1: LLM Reasoning

<details>

### **Executive Summary**

This document explores the reasoning capabilities of Large Language Models (LLMs), moving beyond their basic functionality as text predictors. The core argument is that the ability to generate intermediate reasoning steps, mimicking human thought processes, is crucial for achieving effective problem-solving in LLMs. The document outlines various techniques to elicit this reasoning, such as Chain-of-Thought prompting and Self-Consistency, while also acknowledging limitations like sensitivity to irrelevant information and issues with self-correction.

---

### **Key Themes and Ideas**

#### **The Need for Reasoning in AI**
- The author questions whether current machine learning (ML), which often requires vast amounts of data, truly reflects artificial intelligence. They propose that a key missing element is reasoning, stating: *"Humans can learn from just a few examples because humans can reason."*
- A simple “last letter concatenation” task is used to highlight the deficiency of standard ML approaches and how LLMs can overcome this.
- The document points out a core concept: *"AI should be able to learn from just a few examples, like what humans usually do."*

#### **LLMs as Language Mimics and the Power of Intermediate Steps**
- LLMs are described as *"transformer"* models trained to predict the next word based on vast amounts of text, comparing their training to *"training parrots to mimic human languages."* This highlights their capacity for language manipulation.
- The paper emphasizes that LLMs can generate reasoning if prompted or trained correctly. The key concept is to *"derive the final answer through intermediate steps."*
- This point is illustrated with the *"last-letter-concatenation"* task by prompting the model to show its working, and with more complex math-based questions.

#### **Techniques for Eliciting Reasoning**
- **Chain-of-Thought (CoT) Prompting:**
  - *"Regardless of training, fine-tuning, or prompting, when provided with examples that include intermediate steps, LLMs will respond with intermediate steps."*
  - Demonstrates that prompting with step-by-step reasoning significantly improves accuracy.

- **Least-to-Most Prompting:**
  - Breaks down complex problems into smaller, more manageable sub-problems, allowing for easier generalization.
  - Example: *"Let's break down this problem: 1. How many apples does Anna have? 2. How many apples do Elsa and Anna have together?”*

- **Analogical Reasoning:**
  - LLMs can be prompted to recall related problems before attempting the current task.
  - *"A mathematician is a person who can find analogies between theorems; a better mathematician is one who can see analogies between proofs and the best mathematician can notice analogies between theories."* — Stefan Banach

- **Chain-of-Thought Decoding (Without Explicit Prompting):**
  - A less greedy decoding strategy allows LLMs to reveal step-by-step reasoning even without direct prompts.

- **Self-Consistency:**
  - Improves step-by-step reasoning by sampling multiple responses and selecting the most frequent answer.
  - *"More consistent, more likely to be correct."*

- **Universal Self-Consistency (USC):**
  - LLMs self-select the most consistent response, but *"the most consistent response"* is not always the most frequent.

#### **Theoretical Basis**
- *"There is nothing more practical than a good theory.”* — Kurt Lewin
- Argues that constant-depth transformers can solve inherently serial problems by generating sufficiently long intermediate steps.

---

### **Limitations of LLM Reasoning**

#### **Distraction by Irrelevant Context**
- LLMs can be easily distracted by irrelevant information in prompts, similar to humans.
- A significant performance drop is observed when irrelevant information is added to problems from the GSM8k dataset.

#### **Lack of Self-Correction**
- While LLMs can be prompted to review their responses, self-correction sometimes worsens the answer.
- Indicates that self-correction is not robust without access to an "oracle" answer.

#### **Importance of Premise Order**
- The order of information presented in a problem affects LLM reasoning capabilities.
- Random ordering results in a performance drop, showing that *"Premise Order Matters in Reasoning."*

---

### **Future Directions and Open Questions**
- The author calls for a model that can autonomously learn reasoning techniques and overcome identified limitations.
- Emphasizes the need to better understand these challenges before attempting to fix them.
- *"If I were given one hour to save the planet, I would spend 59 minutes defining the problem and one minute resolving it."* — Albert Einstein

---

### **Key Quotes**
- *"Humans can learn from just a few examples because humans can reason."*
- *"Make things as simple as possible but no simpler."* — Albert Einstein
- *"Derive the final answer through a series of small steps."*
- *"Regardless of training, fine-tuning, or prompting, when provided with examples that include intermediate steps, LLMs will respond with intermediate steps."*
- *"The truth always turns out to be simpler than you thought.”* — Richard P. Feynman
- *"The best way to predict the future is to invent it."* — Alan Kay

---

### **Conclusion**

This document highlights the critical role of reasoning in the development of advanced AI. It demonstrates that LLMs can be coaxed into reasoning using various techniques but acknowledges current limitations in areas such as self-correction and handling irrelevant information. Future LLM research must address these limitations to unlock their full potential for solving complex problems.


Ref:
- https://www.youtube.com/live/QL-FS_Zcmyo
- https://llmagents-learning.org/slides/llm-reasoning.pdf

</details>

## Lecture 2: LLM Agent History

<details>

#### **Overview of LLM Agents**
The document provides a historical and technical overview of **Large Language Model (LLM) agents**, their evolution, and their role in reasoning and acting within environments.

#### **Key Concepts**
1. **Definition of Agents** – Intelligent systems that interact with an environment, including physical (robots, autonomous cars) and digital environments (chatbots, DQN for games, etc.).
2. **LLM Agents** – These agents use **LLMs to process text-based observations and take actions**. They evolve through:
   - **Text Agents** (e.g., ELIZA)
   - **LLM-based Agents** (e.g., SayCan, Language Planner)
   - **Reasoning Agents** (e.g., ReAct, AutoGPT)

#### **Historical Evolution**
1. **Early AI Agents**
   - **ELIZA (1966)** – Simple rule-based text agent.
   - **LSTM-DQN (2015)** – Used reinforcement learning for text-based environments.
2. **LLMs and Reasoning**
   - **Few-shot learning** (NeurIPS 2020) showed LLMs could generalize with minimal examples.
   - **Chain-of-Thought (CoT)** and **Tree-of-Thought (ToT)** approaches improved stepwise reasoning.
3. **Modern LLM Agents**
   - **ReAct (Reasoning + Acting)** – Introduces structured reasoning combined with external actions (e.g., using tools, APIs).
   - **Retrieval-Augmented Generation (RAG)** – Enhances factual accuracy by retrieving external knowledge.

#### **Capabilities and Use Cases**
1. **Question Answering (QA)**
   - Requires different methods for **knowledge-based** (RAG), **computational** (PoT), and **reasoning-intensive** (CoT) queries.
2. **Tool Use**
   - LLMs interact with tools (calculators, APIs, search engines) to extend functionality.
3. **Long-Term Memory**
   - Concepts like **Reflexion**, **Voyager**, and **Generative Agents** enable LLMs to remember and refine knowledge over multiple interactions.

#### **Future Challenges & Research Areas**
1. **Enhancing Memory & Learning** – Moving beyond short-term context windows to **persistent knowledge retention**.
2. **Scaling Multi-Agent Systems** – Exploring **collaborative AI agents** that can interact efficiently.
3. **Improving Robustness & Accuracy** – Avoiding hallucinations and ensuring reliable decision-making.

#### **Conclusion**
The document highlights the **progression from rule-based AI to modern LLM agents**, emphasizing the synergy between reasoning and acting (ReAct), the importance of **retrieval-enhanced learning**, and the need for **long-term memory** in AI systems. Future advancements will focus on improving **generalization, robustness, and adaptability**.

Would you like a more detailed breakdown of a specific section?

</details>



