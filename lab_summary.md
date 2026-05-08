# Lab Summary: Relevance Scoring and Rerankers

Reranking helped most when the baseline retrieval returned broadly related chunks but did not put the most directly answerable evidence first. In this lab, that is most likely for queries where both documents discuss similar themes such as transparency, safety, risk, harmful outputs, or bias.

Use the cross-encoder reranker when you want a fast local reranking step after retrieval and want to avoid extra LLM calls. Use the LLM reranker when interpretive relevance judgment matters more than speed or cost, especially when the wording of the query needs deeper semantic judgment.

## Best Answer By Query

| Query | Best method |
| --- | --- |
| What AI systems are prohibited under the EU AI Act? | baseline |
| How does Anthropic test for harmful outputs? | llm_rerank |
| What transparency obligations apply to AI providers? | cross_encoder |
| What is the definition of a high-risk AI system? | llm_rerank |
| How should AI systems handle bias? | llm_rerank |
