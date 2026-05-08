# Lab Summary: Relevance Scoring and Rerankers

Reranking helped most when the baseline retrieval returned broadly related chunks but did not put the most directly answerable evidence first. In this lab, that is most likely for queries where both documents discuss similar themes such as transparency, safety, risk, harmful outputs, or bias.

Use the cross-encoder reranker when you want a fast local reranking step after retrieval and want to avoid extra LLM calls. Use the LLM reranker when interpretive relevance judgment matters more than speed or cost, especially when the wording of the query needs deeper semantic judgment.

## Best Answer By Query

| Query | Best method |
| --- | --- |
| What AI systems are prohibited under the EU AI Act? | llm_rerank |
| How does Anthropic test for harmful outputs? | baseline |
| What transparency obligations apply to AI providers? | llm_rerank |
| What is the definition of a high-risk AI system? | llm_rerank |
| How should AI systems handle bias? | cross_encoder |


============================================================
QUERY: What AI systems are prohibited under the EU AI Act?
============================================================

  [BASELINE]
  Sources: eu_ai_act p17, eu_ai_act p45, eu_ai_act p21
  Answer:  # AI Systems Prohibited Under the EU AI Act

Based on the provided context, I cannot give you a complete answer to this question. The context excerpts provided do not contain information about which AI systems are prohibited under the EU AI Act.

The context discusses:
- General requirements for AI ...

  [LLM_RERANK]
  Sources: eu_ai_act p17, eu_ai_act p0, eu_ai_act p45
  Answer:  # AI Systems Prohibited Under the EU AI Act

Based on the provided context, I cannot give you a complete answer to this question. The context excerpts provided do not contain information about which specific AI systems are prohibited under the EU AI Act.

The context discusses:
- The general objecti...

  [CROSS_ENCODER]
  Sources: eu_ai_act p0, eu_ai_act p45, eu_ai_act p17
  Answer:  Based on the provided context, I cannot give you a complete answer about which AI systems are prohibited under the EU AI Act.

The context provided (from pages 0, 45, and 17 of the EU AI Act) discusses:
- The general objectives of the regulation (human-centric and trustworthy AI)
- Exemptions from t...

============================================================
QUERY: How does Anthropic test for harmful outputs?
============================================================

  [BASELINE]
  Sources: anthropic p25, anthropic p24, anthropic p3
  Answer:  # How Anthropic Tests for Harmful Outputs

According to the context provided, Anthropic tests for harmful outputs through several methods:

1. **Red Teaming**: Anthropic invests in red teaming their models to identify potential safety vulnerabilities (Source: anthropic, page: 3).

2. **Biological Ev...

  [LLM_RERANK]
  Sources: anthropic p25, anthropic p3, anthropic p10
  Answer:  # How Anthropic Tests for Harmful Outputs

According to the provided context, Anthropic tests for harmful outputs through several methods:

1. **Red Teaming**: Anthropic invests in "red teaming our models to identify potential safety vulnerabilities" (Source: anthropic, page 3). This involves advers...

  [CROSS_ENCODER]
  Sources: anthropic p25, anthropic p3, anthropic p25
  Answer:  # How Anthropic Tests for Harmful Outputs

According to the provided context, Anthropic tests for harmful outputs through the following methods:

1. **Red Teaming**: Anthropic invests in "red teaming our models" to identify potential safety vulnerabilities. (Source: anthropic, page: 25)

2. **Compre...

============================================================
QUERY: What transparency obligations apply to AI providers?
============================================================

  [BASELINE]
  Sources: eu_ai_act p33, eu_ai_act p30, anthropic p2
  Answer:  # Transparency Obligations for AI Providers

Based on the provided context, the EU AI Act specifies the following transparency obligation:

## Deep Fakes Disclosure Requirement

According to the EU AI Act (page 33), **deployers who use an AI system to generate or manipulate image, audio or video con...

  [LLM_RERANK]
  Sources: eu_ai_act p20, eu_ai_act p30, eu_ai_act p34
  Answer:  # Transparency Obligations for AI Providers

Based on the EU AI Act context provided, the following transparency obligations apply to AI providers:

## For High-Risk AI Systems

According to page 20 of the EU AI Act, transparency is required for **high-risk AI systems before they are placed on the m...

  [CROSS_ENCODER]
  Sources: eu_ai_act p33, eu_ai_act p81, eu_ai_act p34
  Answer:  # Transparency Obligations for AI Providers

Based on the provided context from the EU AI Act, the following transparency obligations apply to AI providers:

## For Deepfakes and Manipulated Content

**Deployers** (not specifically providers) who use AI systems to generate or manipulate image, audio...

============================================================
QUERY: What is the definition of a high-risk AI system?
============================================================

  [BASELINE]
  Sources: anthropic p2, anthropic p41, eu_ai_act p21
  Answer:  Based on the provided context, I cannot find a complete definition of a high-risk AI system.

The EU AI Act excerpt (page 21) discusses **requirements for** high-risk AI systems, such as cybersecurity measures and robustness standards, but it does not provide the actual definition of what constitute...

  [LLM_RERANK]
  Sources: eu_ai_act p16, eu_ai_act p12, eu_ai_act p21
  Answer:  # Definition of High-Risk AI System

Based on the provided context, a complete formal definition of "high-risk AI system" is not explicitly stated. However, the context indicates that **the extent of adverse impact on fundamental rights is a key factor in classifying an AI system as high-risk**.

Ac...

  [CROSS_ENCODER]
  Sources: eu_ai_act p21, eu_ai_act p45, eu_ai_act p40
  Answer:  Based on the provided context, a complete definition of "high-risk AI system" is not explicitly stated. 

The context references "high-risk AI systems" multiple times (eu_ai_act, pages 21, 40, and 45) and mentions various requirements that apply to them, such as:

- Cybersecurity measures and securi...

============================================================
QUERY: How should AI systems handle bias?
============================================================

  [BASELINE]
  Sources: eu_ai_act p19, eu_ai_act p19, eu_ai_act p3
  Answer:  # How AI Systems Should Handle Bias

According to the EU AI Act, AI systems should handle bias through the following approaches:

## Data Quality Requirements
- Datasets should be "to the best extent possible complete and free of errors" (eu_ai_act, page 19)
- Providers should be aware that biases c...

  [LLM_RERANK]
  Sources: eu_ai_act p19, eu_ai_act p19, eu_ai_act p16
  Answer:  # How AI Systems Should Handle Bias

According to the EU AI Act, AI systems should handle bias through the following approaches:

## Data Quality Requirements
- Datasets should be "to the best extent possible complete and free of errors" (page 19)
- Biases can be inherent in underlying datasets, esp...

  [CROSS_ENCODER]
  Sources: eu_ai_act p19, eu_ai_act p19, anthropic p3
  Answer:  # How AI Systems Should Handle Bias

Based on the provided context, here are the key approaches to handling bias in AI systems:

## Detection and Correction
According to the EU AI Act (page 19), providers should "subject to appropriate safeguards for the fundamental rights," work to ensure "bias det...

