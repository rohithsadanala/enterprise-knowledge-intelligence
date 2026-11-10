"""LLM providers.

One interface — ``generate(prompt) -> str`` — with three backends:

``mock``
    Offline, deterministic: an *extractive* answerer. It pulls the
    sentences from the supplied context that best overlap the question
    and returns them with ``[n]`` citation markers. No model, no API
    key — used by the tests, the demo and CI.

``openai``
    LangChain ``ChatOpenAI`` (default ``gpt-4o-mini``). Needs
    ``OPENAI_API_KEY``.

``huggingface``
    A local instruct model via ``transformers`` (default
    ``HuggingFaceH4/zephyr-7b-beta`` is only a suggestion — any
    text-generation pipeline model works). Needs
    ``requirements-optional.txt`` and a GPU or patience; the code
    degrades gracefully to the mock provider if loading fails.
"""

from __future__ import annotations

import re


class MockLLM:
    """Extractive, deterministic answerer for offline use.

    Given a prompt that contains a ``Context:`` block with numbered
    sources (``[1] ...``, ``[2] ...``), it returns the sentences from
    the context that share the most content words with the question,
    each followed by its source marker. This keeps the offline demo
    honest: the answer is always grounded in retrieved text, and the
    citations are real.
    """

    def __init__(self, max_sentences: int = 4) -> None:
        self.max_sentences = max_sentences

    def generate(self, prompt: str) -> str:
        question = self._section(prompt, "Question:", "Answer")
        context = self._section(prompt, "Context:", "Question:")
        if not context.strip():
            return "I don't have enough information in the knowledge base to answer that question."
        sentences = self._rank_sentences(question, context)
        if not sentences:
            return "I don't have enough information in the knowledge base to answer that question."
        picked = sentences[: self.max_sentences]
        return " ".join(s.text for s in picked)

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _section(prompt: str, start_marker: str, end_marker: str) -> str:
        """Text between ``start_marker`` and ``end_marker`` (exclusive)."""
        start = prompt.find(start_marker)
        if start == -1:
            return ""
        start += len(start_marker)
        end = prompt.find(end_marker, start)
        if end == -1:
            end = len(prompt)
        return prompt[start:end].strip()

    @staticmethod
    def _norm_token(token: str) -> str:
        """Light normalisation for matching: strip a trailing plural 's'."""
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token

    # Words that carry no matching signal (articles, prepositions,
    # auxiliaries, question words, and "must", which appears in almost
    # every policy sentence). Filtered from both sides of the overlap.
    _STOPWORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "am", "what", "which", "who", "whom", "how", "when", "where", "why",
        "does", "do", "did", "can", "could", "should", "would", "will",
        "shall", "may", "might", "must", "of", "in", "on", "at", "to", "for",
        "with", "by", "from", "and", "or", "but", "not", "no", "if", "then",
        "than", "as", "it", "its", "this", "that", "these", "those", "there",
        "here", "about", "into", "over", "under", "between", "per", "via",
        "have", "has", "had", "i", "you", "he", "she", "we", "they",
    })

    @classmethod
    def _norm_tokens(cls, text: str) -> set[str]:
        from eki.retrieval import tokenize

        return {
            cls._norm_token(t) for t in tokenize(text) if t not in cls._STOPWORDS
        }

    @classmethod
    def _rank_sentences(cls, question: str, context: str):
        """Score context sentences by content-word overlap with the question.

        Source labels (``Title:`` / ``---`` lines), Markdown headers and
        duplicate sentences are removed first, so the answer is built
        only from real sentences in the chunk text.
        """
        q_words = cls._norm_tokens(question)
        # Context blocks look like "[1]\\nTitle: ...\\n---\\n<text>" — split
        # into (marker, block) pairs so citations survive.
        pattern = re.compile(r"\[(\d+)\]\s*")
        segments = pattern.split(context)
        # segments: ['', '1', ' block ', '2', ' block ', ...]
        scored = []
        seen_texts: set[str] = set()
        for i in range(1, len(segments) - 1, 2):
            marker = segments[i]
            block = segments[i + 1]
            # Drop the source-label lines and Markdown headers.
            lines = [
                line.strip()
                for line in block.splitlines()
                if line.strip()
                and not line.strip().startswith("Title:")
                and line.strip() != "---"
                and not line.strip().startswith("#")
            ]
            # Split sentences line by line: a line with no sentence-ending
            # punctuation (a metadata line, a heading) is not a sentence
            # and is skipped, instead of being merged into its neighbour.
            for line in lines:
                for sentence in re.split(r"(?<=[.!?])\s+", line):
                    sentence = sentence.strip()
                    if not sentence or len(sentence) < 20:
                        continue
                    # Skip fragments that are not complete sentences.
                    if sentence[-1] not in ".!?":
                        continue
                    key = sentence.lower()
                    if key in seen_texts:
                        continue
                    seen_texts.add(key)
                    overlap = len(q_words & cls._norm_tokens(sentence))
                    scored.append(_ScoredSentence(sentence, marker, overlap))
        scored.sort(key=lambda s: (-s.overlap, s.text))
        # Keep only sentences that share vocabulary with the question,
        # and only the ones close to the best match — this keeps a
        # factual answer focused instead of pulling in every chunk that
        # happens to share a word like "days" or "employees".
        with_overlap = [s for s in scored if s.overlap > 0]
        if not with_overlap:
            return scored
        best = max(s.overlap for s in with_overlap)
        threshold = max(2, best - 1)
        focused = [s for s in with_overlap if s.overlap >= threshold]
        return focused or with_overlap

    def invoke(self, prompt: str) -> str:
        """LangChain-compatible alias."""
        return self.generate(prompt)


class _ScoredSentence:
    def __init__(self, text: str, marker: str, overlap: int) -> None:
        self.marker = marker
        self.overlap = overlap
        # Attach the citation marker to the sentence if it lacks one.
        self.text = text if text.rstrip().endswith(f"[{marker}]") else f"{text} [{marker}]"


class OpenAILLM:
    """OpenAI chat model via LangChain."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = "", temperature: float = 0.0) -> None:
        from langchain_openai import ChatOpenAI

        self.model = model
        self.client = ChatOpenAI(model=model, api_key=api_key or None, temperature=temperature)

    def generate(self, prompt: str) -> str:
        from langchain_core.messages import HumanMessage

        return str(self.client.invoke([HumanMessage(content=prompt)]).content)

    def invoke(self, prompt: str) -> str:
        return self.generate(prompt)


class HuggingFaceLLM:
    """Local Hugging Face instruct model via a text-generation pipeline."""

    def __init__(self, model: str = "HuggingFaceH4/zephyr-7b-beta") -> None:
        from transformers import pipeline

        self.model = model
        self.pipe = pipeline("text-generation", model=model, max_new_tokens=512)

    def generate(self, prompt: str) -> str:
        out = self.pipe(prompt, return_full_text=False)
        text = out[0]["generated_text"]
        return str(text).strip()

    def invoke(self, prompt: str) -> str:
        return self.generate(prompt)


def get_llm(provider: str, model: str = "", api_key: str = "", temperature: float = 0.0):
    """Factory: build the LLM named in the settings."""
    provider = provider.lower()
    if provider == "openai":
        return OpenAILLM(model or "gpt-4o-mini", api_key=api_key, temperature=temperature)
    if provider == "huggingface":
        return HuggingFaceLLM(model or "HuggingFaceH4/zephyr-7b-beta")
    if provider == "mock":
        return MockLLM()
    raise ValueError(
        f"Unknown LLM provider: {provider!r} (expected 'mock', 'openai' or 'huggingface')"
    )
