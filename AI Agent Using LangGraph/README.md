## 2_AI_Agent_LangGraph

This code sets up an intelligent agent using a combination of tools (LangChain, LangGraph), state management, and a language model (OpenAI's GPT-3.5 Turbo) to interact with users in a structured way.

- StateGraph: A directed state graph for defining workflows.
- END: A predefined constant to indicate the end of the workflow.
- TypedDict, Annotated: Used for type annotations, allowing better clarity and enforcing structure.
- ChatOpenAI: Wrapper around OpenAI’s GPT-3.5 Turbo, providing a language model interface.
- TavilySearchResults: A tool for searching information, simulating the behavior of a search engine.

Invoking the Agent

```python
messages = [HumanMessage(content="What is the weather in sf?")]
result = abot.graph.invoke({"messages": messages})
```

- Starts the agent with an initial user message.
- The StateGraph processes the message through the workflow:
1. Calls the language model.
2. Checks for tool calls.
3. Executes tool calls (e.g., search queries) if necessary.
4. Returns results and continues until no further actions are required.

Workflow in Action:

1. User Query: "What is the weather in SF?"
2. Step 1: Language Model (call_openai):
  - GPT-3.5 processes the query.
  - Determines that a search tool is needed.
3. Step 2: Action (take_action):
  - The search tool is invoked with the query.
  - Results are returned as tool messages.
4. Step 3: Back to LLM:
  - The LLM processes the results and generates a response.
End:
  - The agent completes the interaction once no further actions are required.