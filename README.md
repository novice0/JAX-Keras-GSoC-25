# Building LLMs from Scratch: JAX/Flax Internals, Performance, and Education

Welcome to the repository for my GSoC 2025 project proposal focused on **building LLM components from scratch** while **demystifying the internals of JAX/Flax**. This repo is aimed at learners, researchers, and practitioners who want to understand and implement the building blocks of large language models—not just use them.

---

## Note for the Viewers

**A "work in progress repo"** - Over the course of the community bonding period, a lot of notebook content will be appended to this repo regularly

**Call to my community!!** - I have had a difficult time locating JAX and Keras open source community for the GSoC problem statements. If anyone coming across this repo knows about where I can find the community page, I would be grateful if you'd contact me via mail or on LinkedIn :)

**I want more advice** - I am facing a lot of questions - whether to include graphical representations, diagrams, illustrations, gifs, interactive dashboards into jupyter notebook for creating LLM Examples. Suggestions along these lines would be greatly appreciated!!

**If you're reviewing this for GSoC** — Thank you for your time and consideration! This repo will become a living, growing, open-source knowledge base!

---

## Current Contents

### 1. MiniGPT from Scratch in Keras (TensorFlow)

A pedagogical implementation of a GPT-like architecture using **pure Keras** (TensorFlow backend). This includes:

- Tokenization & embeddings
- Self-attention layers
- Position embeddings
- Causal masking
- End-to-end training on small datasets

This acts as a **baseline**, helping users familiar with Keras transition toward JAX/Flax.

### 2. Profiling Speed/Memory Tradeoffs in `jax.jit`

This section contains a Jupyter Notebook that:

- Explains the purpose of `jax.jit` (Just-in-Time compilation)
- Benchmarks its overhead vs performance gains
- Shows memory profiling and compilation caching behavior
- Provides real-world tips for debugging and optimization

---

## Planned Deliverables (If Selected for GSoC)

If selected, this repository will evolve into a **complete curriculum** of JAX/Flax-based notebooks tailored to LLMs. Here's the proposed timeline and deliverables:

| **Phase**        | **Deliverable**                        | **Details**                                                        |
| ---------------- | -------------------------------------- | ------------------------------------------------------------------ |
| May 8 – June 1   | Community Bonding                      | Explore codebases, finalize scope, read JAX/Flax/TPU documentation |
| Week 1 (June 2)  | Notebook 1: JAX Basics                 | Functional programming, jit/grad/vmap, immutable arrays            |
| Week 2           | Notebook 2: Flax Basics                | linen.Module, param management, RNG use in Flax                    |
| Week 3           | Notebook 3: Self-Attention             | Implement GPT-style attention from scratch in Flax                 |
| Week 4           | Notebook 4: Rotary Embeddings          | Theory, math, and vectorized JAX implementation                    |
| Week 5           | Notebook 5: Key-Value Caching          | KV cache in Flax, enabling faster inference                        |
| Week 6 (Midterm) | Benchmarks + Debugging Tips            | TPU/GPU profiling, jit vs no-jit, XLA tips                         |
| Week 7           | Notebook 6: Model Parallelism          | pmap, shard_map, partitioning layers for large models              |
| Week 8           | Notebook 7: Checkpointing & LoRA       | Fit bigger models via gradient checkpointing & LoRA                |
| Week 9           | Notebook 8: Saving/Loading Flax Models | Serialization best practices                                       |
| Week 10          | Optional: Chat UI for Testing          | Minimal Gradio-based UI for inference demo                         |
| Final Week       | Documentation                          | Organize content, fix typos, polish markdown & tests               |

---

## Vision

While most tutorials focus on classification tasks like MNIST or ImageNet, this project focuses on:

- Real LLM components like GPT attention, rotary embeddings, KV caching
- Deep dives into JAX internals (e.g., how jit/vmap really work)
- Profiling for GPU/TPU performance
- Debugging edge cases with Flax in real-world settings

---

## Who is this for?

- ML practitioners switching from PyTorch/Keras to JAX
- Researchers building custom LLMs
- Students who want a hands-on but deep experience with LLM internals
- Anyone who’s asked: “Why is my Flax model so slow on TPU?”

---

## Contact

**Name:** Chinmay Mulmule

**Preferred Name:** Chinmay

**Email:** [chinmaymulmule@gmail.com]

**GitHub:** [novice0](https://github.com/novice0)

**LinkedIn / Website:** [chinmay-mulmule](https://www.linkedin.com/in/chinmay-mulmule-996195254/)
