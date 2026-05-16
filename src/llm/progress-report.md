1.  **Graph-Enhanced Context**:
    - **Context Expansion**: (remove)
        - If an *Article* or *Clause* is retrieved, it automatically pulls child nodes.
        - If a *Point* is retrieved, it pulls parent node.
2.  **Evaluation Workflow**:
    - **Online Phase**: Fetches law texts and prepares payloads (Dataset -> Context).
    - **Offline Phase**: Runs inference on kaggle.
3.  **Generation**:
    - **Few-shot Prompting**: Currently using 3-example prompts for better structure.
