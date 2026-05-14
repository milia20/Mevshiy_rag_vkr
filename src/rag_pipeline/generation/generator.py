"""
Answer generation with source citations.

Generates answers based on retrieved context using LLMs.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for answer generation."""

    model_name: str = "qwen/qwen3.5-9b"
    max_tokens: int = 512
    temperature: float = 0.7
    system_prompt: str = (
        "You are a helpful assistant that answers questions based on the provided context. "
        "Always cite your sources by mentioning the document/source when possible. "
        "If you cannot find the answer in the context, say so honestly."
    )


@dataclass
class GeneratedAnswer:
    """Represents a generated answer with citations."""

    answer: str
    citations: list[dict]
    confidence: float = 0.0
    sources: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "confidence": self.confidence,
            "sources": self.sources,
        }


class AnswerGenerator:
    """
    Answer generator using LLMs.

    Takes retrieved context and generates answers with source citations.
    Supports multiple LLM backends (OpenAI, Ollama, LM Studio).
    """

    def __init__(self, config: GenerationConfig | None = None):
        """
        Initialize generator.

        Args:
            config: Generation configuration
        """
        self.config = config or GenerationConfig()
        self._client = None

    def _get_client(self):
        """Get LLM client (lazy initialization)."""
        if self._client is None:
            try:
                from openai import OpenAI

                # Try to initialize OpenAI client
                # Will work with OpenAI API, LM Studio, Ollama (with compatible endpoint)
                self._client = OpenAI(
                    base_url=None,  # Use default or set custom URL
                )
            except ImportError:
                logger.warning("OpenAI not installed, using mock generator")
                self._client = "mock"

        return self._client

    def _format_context(self, contexts: list[dict]) -> str:
        """Format retrieved contexts into a single string."""
        formatted = []
        for i, ctx in enumerate(contexts, 1):
            text = ctx.get("text", "")
            source = ctx.get("metadata", {}).get("source_id", f"Source {i}")
            formatted.append(f"[{i}] ({source}): {text}")
        return "\n\n".join(formatted)

    def generate(
        self,
        query: str,
        contexts: list[dict],
        max_contexts: int = 5,
    ) -> GeneratedAnswer:
        """
        Generate answer based on retrieved contexts.

        Args:
            query: User query
            contexts: List of retrieved context dictionaries
            max_contexts: Maximum number of contexts to use

        Returns:
            GeneratedAnswer with answer and citations
        """
        if not contexts:
            return GeneratedAnswer(
                answer="I could not find any relevant information to answer this question.",
                citations=[],
                confidence=0.0,
            )

        # Limit contexts
        contexts = contexts[:max_contexts]

        # Format context
        context_text = self._format_contexts(contexts)

        # Build prompt
        prompt = f"""Based on the following context, please answer the question.

Context:
{context_text}

Question: {query}

Answer:"""

        # Generate using LLM
        client = self._get_client()

        if client == "mock":
            # Mock response for testing without API key
            return self._generate_mock(query, contexts)

        try:
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            answer = response.choices[0].message.content.strip()

            # Extract citations
            citations = self._extract_citations(answer, contexts)

            return GeneratedAnswer(
                answer=answer,
                citations=citations,
                confidence=self._estimate_confidence(answer, contexts),
                sources=list({ctx.get("metadata", {}).get("source_id", "") for ctx in contexts}),
            )

        except Exception as e:
            logger.error("Generation failed: %s", e)
            return self._generate_mock(query, contexts)

    def _generate_mock(self, query: str, contexts: list[dict]) -> GeneratedAnswer:
        """Generate mock answer for testing."""
        # Simple extractive approach as fallback
        best_context = contexts[0] if contexts else {}
        text = best_context.get("text", "")

        # Return first sentence as mock answer
        answer = text.split(".")[0] + "." if text else "No answer available."

        return GeneratedAnswer(
            answer=answer,
            citations=[{"text": text, "source": best_context.get("metadata", {}).get("source_id", "")}],
            confidence=0.5,
            sources=[best_context.get("metadata", {}).get("source_id", "")],
        )

    def _extract_citations(self, answer: str, contexts: list[dict]) -> list[dict]:
        """Extract citations from answer."""
        citations = []
        for ctx in contexts:
            text = ctx.get("text", "")
            if text and len(text) > 10:  # Only cite substantial texts
                citations.append(
                    {
                        "text": text[:200] + "..." if len(text) > 200 else text,
                        "source": ctx.get("metadata", {}).get("source_id", ""),
                    }
                )
        return citations[:3]  # Limit to 3 citations

    def _estimate_confidence(self, answer: str, contexts: list[dict]) -> float:
        """Estimate answer confidence (simple heuristic)."""
        if not answer or len(answer) < 10:
            return 0.1
        if "cannot" in answer.lower() or "unknown" in answer.lower():
            return 0.2
        if len(contexts) >= 3:
            return 0.8
        return 0.5
