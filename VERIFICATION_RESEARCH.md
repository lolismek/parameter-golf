# Verifiable ML Training Without Hardware Attestation: Research Report

## Context
A competition where users fork a repo, modify ONE training script, train a model, and submit model + score. Users are untrusted internet users (e.g., on RunPod). We need software-only verification that they actually ran the claimed training script and produced the claimed model. No TEE/SGX/TPM available.

---

## 1. Cryptographic Training Verification (Proof-of-Learning)

### 1.1 Proof-of-Learning (Jia et al., 2021) — IEEE S&P
**Paper**: arxiv.org/abs/2103.05633

**Mechanism**: Exploits the inherent stochasticity of SGD. During training, the prover periodically saves:
- Model weight checkpoints at specific intervals
- The data indices (which mini-batches were used at each step)
- Random seeds used

The verifier re-executes selected segments of training using the same data indices and seeds, then checks if the resulting weight updates match the claimed checkpoints within a tolerance.

**Key insight**: An adversary trying to forge a proof must perform at least as much computation as legitimate training, because they need to produce a consistent trajectory of (weights, data indices, gradients) that passes spot-check verification.

**Security**: Claimed to be robust to hardware/software variance. However...

### 1.2 "Proof-of-Learning is Currently More Broken Than You Think" (Fang et al., 2023) — IEEE EuroS&P
**Paper**: arxiv.org/abs/2208.03567

**Critical finding**: Demonstrates reproducible spoofing strategies that work across different PoL configurations at reduced computational cost. Identifies key vulnerabilities and argues that developing provably robust PoL verification "without further understanding of optimization in deep learning" is not feasible.

**Implication**: Pure PoL based on checkpoint trajectories is not sufficient alone against a motivated adversary.

### 1.3 PoLO: Proof-of-Learning and Proof-of-Ownership (2025)
**Paper**: arxiv.org/abs/2505.12296

**Mechanism**: Divides training into fine-grained "shards" and embeds a dedicated watermark in each shard. Each watermark is generated using the hash of the preceding shard, creating a cryptographic chain.

**Performance**:
- Verification costs reduced to 1.5-10% of traditional methods
- Forging requires 1.1-4x more resources than honest proof generation
- 99% watermark detection accuracy for ownership verification
- Preserves data privacy (unlike gradient-trajectory PoL)

**Assessment**: More practical than original PoL. The chained watermark approach is harder to spoof. Good candidate for your use case.

### 1.4 Data Forging Attacks — Re-evaluation (2024)
**Paper**: arxiv.org/abs/2411.05658

**Finding**: Current data forging attacks "cannot produce sufficiently identical gradients" in practice, especially when constrained to valid domains (e.g., pixel values 0-255). This suggests PoL may be more robust than the 2023 attack paper implied, at least for practical attacks.

---

## 2. Zero-Knowledge Proofs for ML Training (zkML)

### 2.1 Zero-Knowledge Proofs of Training (Abbaszadeh et al., CCS 2024)
**Paper**: eprint.iacr.org/2024/162

**Best current work for ZK training verification.**

**Mechanism**: Combines an optimized GKR-style proof system for gradient descent computation with recursive composition across training iterations.

**Performance**:
- Handles models up to VGG-11 (10M parameters)
- Prover runtime: ~15 minutes per training iteration
- Proof size: 1.63 MB
- Verifier runtime: 130 milliseconds

**Assessment**: The 15min/iteration prover overhead is significant. For a training run of thousands of iterations, this multiplies training time dramatically. Practical only for small models and short training runs. NOT practical for large-scale competitions yet.

### 2.2 zkDL: Efficient ZK Proofs of Deep Learning Training (Sun et al., 2023)
**Paper**: arxiv.org/abs/2307.16273

**Mechanism**: Custom proof system with zkReLU (for ReLU activations and backprop) and FAC4DNN (custom arithmetic circuits for neural networks).

**Performance**:
- Generates proofs in <1 second per batch update
- Scales to 8-layer networks with 10M parameters, batch size 64

**Assessment**: Faster than CCS 2024 paper per iteration, but still limited to ~10M parameter models. Not suitable for modern transformer training but could work for smaller competition models.

### 2.3 EZKL
**Website**: ezkl.xyz, github.com/zkonduit/ezkl

**Status**: Production-ready library (v23+, audited by Trail of Bits). Uses Halo2 proof system.
**Limitation**: Inference verification ONLY. Cannot verify training. Useful if you want to verify that a submitted model produces a claimed score on test data, but does not verify HOW the model was produced.

### 2.4 Survey: ZK-Based Verifiable ML (2025)
**Paper**: arxiv.org/abs/2502.18535

Comprehensive survey of ZKML from 2017-2024. Key finding: verifiable training is significantly underexplored compared to verifiable inference. Most practical systems handle inference only.

### 2.5 ZKMLOps Framework (2025)
**Paper**: arxiv.org/abs/2505.20136

Identifies 5 key properties for ZKPs in ML: non-interactivity, transparent setup, standard representations, succinctness, post-quantum security. Notes that training verification is the least developed area.

---

## 3. Checkpoint-Based Verification

### 3.1 Periodic Checkpoint Submission
**Approach**: Require participants to submit intermediate checkpoints (e.g., every N steps) along with:
- Model weights at that checkpoint
- Loss values
- Learning rate schedule
- Data ordering (batch indices)

**Verification**: Spot-check by re-running a few segments between consecutive checkpoints.

**Security analysis**:
- **Pro**: Catches lazy attacks (submitting a pre-trained model without any trajectory)
- **Pro**: Loss curve should show expected patterns (monotonic decrease with noise)
- **Con**: A sophisticated adversary can train a different model and then construct a plausible trajectory for the claimed script by actually running the script with carefully chosen seeds
- **Con**: Requires significant storage (many checkpoints for large models)

### 3.2 Gradient Commitment via Merkle Trees
**Approach** (used in PoGO — arxiv.org/abs/2504.07540):
- Use 4-bit quantized gradients to reduce storage
- Build Merkle trees over the full 32-bit model parameters
- Verifiers perform random leaf checks with minimal data
- Enable positive/negative attestations aggregated at finalization

**Assessment**: Good for blockchain contexts. The Merkle proof approach allows efficient selective verification. Quantization reduces overhead but introduces approximation.

### 3.3 Practical Checkpoint Protocol for Competitions
**Recommended approach**:
1. Training script saves checkpoints every K steps with: weights, optimizer state, loss, step number, wall-clock time, hash of previous checkpoint
2. Each checkpoint is hashed and committed (e.g., to a server or blockchain)
3. Verifier randomly selects 2-3 checkpoint pairs and re-runs the training between them
4. Verify that the weight delta matches within tolerance

**Overhead**: ~2-5% of training time for checkpointing. Verification cost = (number of spot-checks) * (steps between checkpoints) / (total steps) of the full training cost.

---

## 4. Reproducibility-Based Verification (Deterministic Training)

### 4.1 PyTorch Deterministic Mode
From PyTorch docs (docs.pytorch.org/docs/stable/notes/randomness.html):

Required settings:
```python
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)
```

**Critical limitations**:
- "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms"
- Results differ between CPU and GPU, and between GPU architectures
- Some operations have no deterministic implementation (will raise RuntimeError)
- Deterministic operations are slower than nondeterministic ones
- DataLoader workers need explicit seeding

### 4.2 Practical Implications for Competition Verification
**Challenge**: Users on RunPod may have different GPU models (A100 vs H100 vs 4090), different CUDA versions, different PyTorch builds. Even with identical seeds, floating-point non-associativity means results diverge.

**Mitigation strategies**:
1. **Fix the exact Docker image** (pins PyTorch, CUDA, cuDNN versions)
2. **Fix GPU architecture** (e.g., require A100 only) — reduces pool of participants
3. **Allow tolerance in verification** (weights match within epsilon after N steps)
4. **Verify short segments only** (divergence accumulates over steps; short segments stay close)

**Assessment**: Partial reproducibility is achievable and very useful as one layer of verification. Full bit-exact reproducibility across hardware is not practical.

---

## 5. Model Fingerprinting / Watermarking

### 5.1 Radioactive Data (Sablayrolles et al., 2020)
**Paper**: arxiv.org/abs/2002.00937

**Mechanism**: Makes imperceptible changes to training data such that any model trained on it bears an identifiable statistical mark.
- Detection works with high confidence (p < 10^-4) even when only 1% of training data is radioactive
- Survives different architectures, optimizers, and data augmentation
- Higher signal-to-noise ratio than data poisoning/backdoor methods

**Application to competition**: Embed radioactive markers in the training data. If a submitted model doesn't carry the markers, it wasn't trained on the correct data.

### 5.2 Chained Watermarks (PoLO approach)
As described in Section 1.3, watermarks can be embedded at each training shard and chained via hashing.

### 5.3 Backdoor-Based Fingerprinting
**Approach**: Embed specific input-output trigger patterns in the training data. After training, test if the model responds correctly to the triggers.

**Pros**: Simple to implement, hard to remove without retraining
**Cons**: Can degrade model quality; adversary who knows the triggers can train to include them while using a different base approach

### 5.4 Assessment for Competition Use
**Radioactive data is the strongest approach here**. The marks are statistical, imperceptible, and difficult to forge. An adversary would need to either:
- Actually train on the radioactive data (which means actually running training), OR
- Reverse-engineer the exact statistical perturbation and apply it to a differently-trained model (very difficult)

**Limitation**: Requires control over the training data, which you have in this competition setup.

---

## 6. Statistical Verification

### 6.1 Loss Curve Analysis
**Approach**: Require logging of per-step loss values. Analyze for:
- Expected convergence patterns for the given architecture/data
- Noise profile consistent with the batch size and learning rate
- No suspicious jumps or discontinuities that suggest checkpoint splicing

**Security**: Low — a motivated adversary can generate plausible fake loss curves.

### 6.2 Weight Distribution Analysis
**Approach**: Compare the statistical properties of submitted weights against expected distributions for models trained with the claimed script:
- Weight norms per layer
- Gradient statistics
- Distribution of activations on reference inputs
- Singular value spectra of weight matrices

**Security**: Medium — useful for catching crude fakes (e.g., submitting a much larger fine-tuned model), but sophisticated adversaries can match statistics.

### 6.3 Training Dynamics Fingerprinting
**Approach**: Certain training scripts produce characteristic patterns in how weights evolve. For example:
- Specific optimizer (Adam vs SGD) leaves different signatures in weight distributions
- Batch normalization statistics encode information about the training data
- The ratio of weight magnitudes across layers reflects the specific initialization + training duration

**Assessment**: Good as a secondary signal. Can catch adversaries who distill/fine-tune a larger model and claim it was trained from scratch.

---

## 7. Existing Platform Approaches

### 7.1 Kaggle: Code Competitions (Notebooks-Only)
**Approach**:
- Submissions MUST be made from Kaggle Notebooks
- Code runs in Kaggle's controlled environment
- Organizers can view the full notebook source
- Fixed compute resources and time limits
- No internet access during submission inference

**Key insight**: Kaggle's strongest verification is **running user code in a controlled environment**. This is the gold standard but requires hosting the compute.

### 7.2 MLPerf/MLCommons
**Approach**:
- Mandatory seed logging via mllog system
- No duplicate seeds across runs
- Reference Convergence Points (RCPs) with required precision
- Dataset verification via checksums
- Strict hyperparameter constraints
- Same system/framework required for entire submission
- Manual review of winning submissions

**Key insight**: MLPerf relies heavily on mandatory logging + manual review + reputation. They do NOT cryptographically verify training.

### 7.3 Gensyn (Decentralized Training)
**Paper**: docs.gensyn.ai/litepaper

**Three-layer verification**:
1. **Probabilistic Proof-of-Learning**: Periodic checkpoints with data indices; verifiers replicate selected segments
2. **Graph-Based Pinpoint Protocol**: When disputes arise, progressively narrows down the disputed computation to a single operation that can be verified on-chain
3. **Incentive Game**: Staking/slashing + periodic forced errors with jackpot payouts to overcome the verifier's dilemma

**Claimed efficiency**: 1,350% more efficient than full replication.

**Assessment**: Most sophisticated practical system. The combination of spot-check verification + economic incentives + dispute resolution is the state of the art for decentralized training verification.

### 7.4 Gensyn RL-Swarm (Practical Implementation)
**Repository**: github.com/gensyn-ai/rl-swarm

Current practical implementation uses:
- On-chain node registration for identity
- Frozen evaluator models for reward assessment
- Peer-to-peer rollout sharing
- Smart contract-based tracking

### 7.5 AIcrowd
Uses code submission + containerized execution. Participants submit Docker containers or Git repos that are re-executed in AIcrowd's infrastructure.

---

## 8. Container/Environment-Based Approaches

### 8.1 Docker Hash Verification
**Approach**:
1. Provide a locked Docker image with all dependencies
2. User modifies only the allowed training script
3. User runs training inside the container
4. Container logs all operations and produces a deterministic hash of the execution environment

**Limitations**: Docker itself doesn't prevent users from modifying the container or running different code. The user controls the machine.

### 8.2 Remote Execution in Controlled Environment
**Approach** (Kaggle-style):
- User submits code, not models
- Platform runs the code on its own infrastructure
- Score is computed server-side

**Pros**: Strongest possible verification — you run the code yourself
**Cons**: You pay for all compute. For GPU-intensive training, this is very expensive.

### 8.3 Sandboxed Execution with Logging
**Approach**:
1. Provide a Docker container with a tamper-evident logging agent
2. Agent logs: all file I/O, network calls, GPU utilization, memory patterns
3. Agent periodically sends signed heartbeats to verification server
4. Training script hash is verified before execution begins

**Security**: Medium — a sufficiently motivated adversary can modify the agent or run outside the container. Without hardware attestation, you can't guarantee the agent wasn't tampered with.

### 8.4 Deterministic Container Builds (Nix/Guix)
**Approach**: Use Nix or Guix for fully reproducible builds. Every dependency is pinned to exact hashes. The entire software environment is deterministic.

**Benefit**: Eliminates one source of non-reproducibility (software versions)
**Limitation**: Does not address hardware-level floating-point differences

---

## 9. Practical Hybrid Approaches (Recommended)

### 9.1 Recommended Architecture for Your Competition

Given the constraints (random internet users, RunPod GPUs, software-only), here is a layered defense approach ranked by implementation complexity:

#### Layer 1: Controlled Seed + Checkpoint Commitments (Easy, High Value)
- Assign each participant a unique random seed (server-generated)
- Training script must save checkpoints every K steps
- Each checkpoint includes: weights, loss, step, hash of previous checkpoint
- Participant commits checkpoint hashes to server in real-time during training
- Server verifies the chain is temporally consistent (timestamps must be monotonically increasing with plausible intervals)

**What it catches**: Pre-computed models, models trained offline with different code, trivially faked submissions

#### Layer 2: Spot-Check Verification (Medium complexity, High Value)
- After submission, randomly select 2-3 checkpoint intervals
- Re-run that segment of training using the participant's seed and the checkpoint as starting point
- Verify weights at the end match the next checkpoint within tolerance
- Run on YOUR infrastructure (costs ~5-15% of one full training run per verification)

**What it catches**: Any model that wasn't produced by the actual training script. The adversary would need to have actually run the training script to produce consistent checkpoints.

#### Layer 3: Radioactive Training Data (Medium complexity, High Value)
- Embed statistical markers in the training data before distributing
- Each participant could get slightly different markers (enables identifying who trained what)
- After submission, test the model for the presence of markers

**What it catches**: Models trained on different data, models distilled from other sources, pre-trained models

#### Layer 4: Statistical Consistency Checks (Easy, Medium Value)
- Verify loss curve follows expected patterns
- Check weight distribution statistics match expected profiles
- Verify model architecture matches the allowed script
- Test model on "canary" inputs that have characteristic behavior for correctly-trained models

**What it catches**: Crude fakes, distilled models, architecture mismatches

#### Layer 5: Timing/Resource Analysis (Easy, Low-Medium Value)
- Require real-time telemetry during training (heartbeats, GPU utilization logs)
- Training duration should be consistent with the model size, dataset, and hardware
- Flag submissions where training completed impossibly fast or used suspiciously low GPU memory

**What it catches**: Adversaries who skip training entirely and submit pre-made models

### 9.2 Security Assessment of the Hybrid Approach

**Against casual cheaters** (most participants): Layers 1-2 are sufficient. Very hard to fake consistent checkpoint chains that pass spot-check verification.

**Against motivated adversaries**: The combination of Layers 1-3 is strong. To beat it, an adversary must:
1. Actually run the training script (to produce consistent checkpoints) — Layer 2 forces this
2. Use the correct training data (to carry radioactive markers) — Layer 3 forces this
3. But they COULD modify the script in unauthorized ways while still producing valid checkpoints

**Against sophisticated adversaries**: The only remaining attack surface is modifying the training script in ways that don't alter the checkpoint structure but somehow produce a better model. This is actually hard to do — the script IS the thing they're supposed to modify. The real threat is:
- Pre-training on additional data and using the script for fine-tuning only (radioactive data catches this)
- Using a larger model and distilling into the correct architecture (statistical checks + radioactive data catch this)
- Ensembling multiple runs and selecting the best (this is legitimate unless rules prohibit it)

### 9.3 Cost Analysis

| Layer | Implementation Cost | Verification Cost per Submission | Security Value |
|-------|-------------------|--------------------------------|----------------|
| 1: Checkpoint Commits | Low (modify training script) | Near zero | High |
| 2: Spot-Check | Medium (verification infra) | 5-15% of one training run | Very High |
| 3: Radioactive Data | Medium (data preprocessing) | Low (run inference tests) | High |
| 4: Statistical Checks | Low (analysis scripts) | Low (automated) | Medium |
| 5: Timing Analysis | Low (logging) | Near zero | Low-Medium |

**Total verification cost**: ~5-20% of one training run per submission, plus fixed infrastructure costs.

---

## 10. Key Papers and Resources

### Must-Read Papers
1. **Proof-of-Learning** (Jia et al., 2021) — arxiv.org/abs/2103.05633 — Foundational PoL mechanism
2. **PoL is More Broken Than You Think** (Fang et al., 2023) — arxiv.org/abs/2208.03567 — Attacks on PoL
3. **Zero-Knowledge Proofs of Training** (Abbaszadeh et al., CCS 2024) — eprint.iacr.org/2024/162 — Best ZK training proof
4. **zkDL** (Sun et al., 2023) — arxiv.org/abs/2307.16273 — Efficient ZK training proofs
5. **PoLO** (2025) — arxiv.org/abs/2505.12296 — Chained watermarks for PoL + ownership
6. **PoGO** (2025) — arxiv.org/abs/2504.07540 — Merkle proofs over quantized gradients
7. **Radioactive Data** (Sablayrolles et al., 2020) — arxiv.org/abs/2002.00937 — Training data fingerprinting
8. **Data Forging Re-evaluation** (2024) — arxiv.org/abs/2411.05658 — PoL more robust than thought
9. **ZKML Survey** (2025) — arxiv.org/abs/2502.18535 — Comprehensive ZKML landscape
10. **ZKMLOps** (2025) — arxiv.org/abs/2505.20136 — Framework for ZK in ML operations

### Practical Tools and Platforms
- **EZKL** (ezkl.xyz) — Production ZK inference verification (NOT training)
- **Gensyn** (gensyn.ai) — Decentralized verifiable training with economic incentives
- **MLPerf/MLCommons** — Logging standards and verification approaches for benchmarks

### Key Implementations
- **VerifBFL** (arxiv.org/abs/2501.04319) — zk-SNARKs for federated learning (<81s proof generation)
- **ZKBoost** (arxiv.org/abs/2602.04113) — First zkPoT for XGBoost

---

## 11. Bottom Line Recommendations

### For your specific competition:

**If you can afford to run submissions yourself** (Kaggle model):
- This is the gold standard. Accept code submissions, run them on your infrastructure, compute the score. No verification needed because you control execution.
- Cost: You pay for all training compute.

**If participants must train on their own hardware** (your scenario):
1. **Mandatory**: Checkpoint commitment chain (Layer 1) — costs almost nothing
2. **Strongly recommended**: Spot-check verification (Layer 2) — costs ~10% of one training run per participant you verify
3. **Recommended**: Radioactive training data (Layer 3) — one-time setup cost
4. **Nice to have**: Statistical checks + timing analysis (Layers 4-5) — cheap automated filters

**What NOT to pursue** (for this use case):
- Full ZK proofs of training: Too expensive (15min overhead per iteration) and limited to small models
- Blockchain-based consensus: Overkill for a competition; designed for decentralized networks
- Full deterministic reproducibility: Not achievable across different GPU hardware

**The practical sweet spot** is: checkpoint chains + spot-check verification + radioactive data. This combination makes cheating require actually running the training script on the correct data, which is essentially what you want to verify.
