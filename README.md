# TLDR OSInteraction: Agentic AI Project (CSE598)

This codebase contains the comprehensive implementation and evaluation framework for multimodal Vision-Language Models (VLMs) on computer-use tasks, specifically targeting the **Qwen3-VL model family** (2B, 4B, 8B). The project was conducted by **Team TLDR** and evaluates the models across two complementary benchmarks:
- **OmniACT**: 9,262 desktop and web automation tasks focusing on multi-step tool orchestration and UI interaction.
- **AgentHARM**: 176 multi-step harmful scenarios focusing on safety, refusal behavior, and blind goal-directedness.

All experiments were executed on the SOL supercomputer using a heterogeneous accelerator environment comprising **Intel Gaudi HPU** and **Nvidia A100 GPU** nodes.

## Project Structure & Phases

The project is structured into two main phases, evolving from single-agent baselines to a robust multi-agent architecture.

### Phase 1: Baseline Single-Agent Evaluation
Located in the `phase1/` directory, this phase established performance and safety baselines using a monolithic single-agent approach. 
- **Setup & Infrastructure:** Included setting up dynamic hardware auto-detection (`hl-smi` and `nvidia-smi`) and building a native PyTorch bridge (`GaudiProvider`) to bypass API limitations for Qwen3-VL inference on HPUs.
- **Key Limitations Discovered:**
  - **OmniACT:** Models suffered from Repetition Looping (E1), Geometric Drift / Missed Targets (E2), and Action-Type Confusion (E3).
  - **AgentHARM:** Models exhibited False Refusals (E4), Capability Failure Without Refusal (E5), and Incremental Escalation of harmful workflows (E6).

### Phase 2: Multi-Agent Tri-Agent Architecture
Located in the `phase2/` directory, this phase addressed the limitations of the single-agent baseline by introducing a **Tri-Agent Interaction Layer**:
- **The Manager (Planner):** Decomposes high-level goals into discrete steps and validates tool names.
- **The Executor (Actioner):** Generates coordinates and tool calls.
- **The Auditor (Critic/Safety):** Checks for repetition loops, validates state changes, and evaluates cumulative action sequences against safety policies.

**Key Innovations in Phase 2:**
1. **The "Multiple Guesses" Method (OmniACT):** Generating 5 different guesses for click locations concurrently, clustering them, and selecting the most reliable centroid to eliminate geometric drift and improve click accuracy.
2. **Immediate Stop & Risk Categorization (AgentHARM):** The Manager halts harmful instructions immediately (`HARD_REFUSAL`), while the Auditor categorizes the risk type (e.g., `[File_Access]`) before approving the action, dropping the Attack Success Rate (ASR) to under 3%.

## Getting Started

### Prerequisites
The codebase is designed to run on HPC environments (e.g., SOL Supercomputer) with either Intel Gaudi or Nvidia A100 nodes.
```bash
# Clone the benchmarks and set up the environment
cd phase1
./setup_benchmarks.sh
```
This script will configure cache directories, create a standard python environment, install hardware-specific PyTorch dependencies (Habana vs. CUDA), and clone the required repositories (e.g., OSWorld).

### Running Evaluations
You can run the sequential evaluation scripts directly. The `run_experiments.sh` script automatically detects your active hardware.
```bash
cd phase1
./run_experiments.sh
```

## Results Summary
By migrating to the specialized Multi-Agent setup in Phase 2, we achieved significant improvements:
- **OmniACT:** The Qwen3-VL-8B model improved its Success Score from 0.2184 to 0.2396, and Action Score from 40.17% to 51.46%.
- **AgentHARM:** The Qwen3-VL-8B model's Attack Success Rate plummeted from 32.19% to a highly secure 2.90%, and its Refusal Rate improved from 52.84% to 88.07%.

## Contributors (Team TLDR)
1. **Vedang**: Architecture design, infrastructure setup (SOL), Gaudi provider implementation, report authorship.
2. **Vidya**: Phase 2 iterations, "Multiple Guesses" method, risk categorization, safety stop-rules.
3. **Harshith**: Failure analysis, experimental result evaluation, and quantitative reporting.
4. **Shravan**: Environment setup, hardware cross-compatibility (Gaudi + A100), OSWorld deployment.
5. **Gouri**: Safety testing via adversarial prompts, vulnerability identification.
