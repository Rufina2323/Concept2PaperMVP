"""LLM service: calls Groq API to generate a paper draft outline."""

import logging

from app.config import settings

logger = logging.getLogger(__name__)

_DRAFT_TEMPLATE = """# Research Draft: {concept_a} × {concept_b}

## Proposed Title
"Exploring the Intersection of {concept_a} and {concept_b}"

## Abstract (template)
This paper investigates the relationship between **{concept_a}** and **{concept_b}**,
two concepts that frequently appear in recent machine learning literature but whose
direct connection remains underexplored.

## Proposed Sections
1. Introduction — motivation for combining {concept_a} with {concept_b}
2. Related Work — prior work on each concept independently
3. Methodology — how to leverage {concept_a} techniques within {concept_b} context
4. Experiments — benchmark datasets and evaluation metrics
5. Conclusion — expected contributions and future directions

## Action Plan
- [ ] Survey recent papers (2020–2024) mentioning both concepts
- [ ] Identify open problems at the intersection
- [ ] Design initial experiment combining both approaches
- [ ] Benchmark against existing baselines
- [ ] Write and submit to target venue

*(Generated without LLM — set GROQ_API_KEY to enable AI-generated drafts)*
"""


def generate_draft(concept_a: str, concept_b: str) -> str:
    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY not set — returning template draft")
        return _DRAFT_TEMPLATE.format(concept_a=concept_a, concept_b=concept_b)

    try:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        prompt = (
            f"You are a research scientist at a top ML lab. "
            f"Generate a concise research paper outline (title, abstract, 5-section structure, "
            f"and a 5-step action plan) for a paper that combines the concepts: "
            f"'{concept_a}' and '{concept_b}'. "
            f"Format your response in Markdown. Be specific about methods and datasets."
        )
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.7,
        )
        return response.choices[0].message.content or _DRAFT_TEMPLATE.format(
            concept_a=concept_a, concept_b=concept_b
        )
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return _DRAFT_TEMPLATE.format(concept_a=concept_a, concept_b=concept_b)
