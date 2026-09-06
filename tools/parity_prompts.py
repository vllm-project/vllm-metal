# SPDX-License-Identifier: Apache-2.0
"""Shared input corpus for parity tools."""

# Shared by every model; no random sampling or model-specific prompt selection.
# Upstream vLLM prompts (Apache-2.0, Copyright contributors to vLLM):
# https://github.com/vllm-project/vllm/tree/1970f3ed4be7fa8620e4ddc4a12c36a8384cfc27/tests
PROMPTS = [
    # Existing arithmetic, completion, multilingual, and technical cases.
    "The capital of France is",
    "The weather today is not",
    "One plus one equals",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
    "Machine learning is",
    "Machine learning is a branch of",
    "Two plus two equals",
    "Monday, Tuesday, Wednesday,",
    "서울은 대한민국의",
    "인공지능은",
    "The speed of light is approximately",
    "Write a short essay about the history of computing.",
    "Explain how a B-tree works and why databases use it.",
    "Describe the CAP theorem and its practical consequences.",
    # Verbatim tests/prompts/example.txt, including newlines kept by its fixture.
    "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.\n",
    "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.\n",
    "Compare and contrast artificial intelligence with human intelligence in terms of processing information.\n",
    "Describe the basic components of a neural network and how it can be trained.\n",
    "Write a short story about a robot that dreams for the first time.\n",
    "Analyze the impact of the COVID-19 pandemic on global economic structures and future business models.\n",
    "Explain the cultural significance of the Mona Lisa painting, and how its perception might vary in Western versus Eastern societies.\n",
    "Translate the following English sentence into Japanese, French, and Swahili: 'The early bird catches the worm.'\n",
    # Verbatim prompt_templates in tests/v1/determinism/utils.py.
    "Question: What is the capital of France?\nAnswer: The capital of France is",
    "Q: How does photosynthesis work?\nA: Photosynthesis is the process by which",
    "User: Can you explain quantum mechanics?\nAssistant: Quantum mechanics is",
    "Once upon a time in a distant galaxy, there lived",
    "The old man walked slowly down the street, remembering",
    "In the year 2157, humanity finally discovered",
    "To implement a binary search tree in Python, first we need to",
    "The algorithm works by iterating through the array and",
    "Here's how to optimize database queries using indexing:",
    "The Renaissance was a period in European history that",
    "Climate change is caused by several factors including",
    "The human brain contains approximately 86 billion neurons which",
    "I've been thinking about getting a new laptop because",
    "Yesterday I went to the store and bought",
    "My favorite thing about summer is definitely",
    # Fixed longer inputs using the padding text from the same upstream helper.
    *[
        " This is an interesting topic that deserves more explanation. " * repeats
        + "Question: What is the capital of France?\nAnswer: The capital of France is"
        for repeats in (8, 24)
    ],
]
