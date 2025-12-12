================================================================================
REMAINING EXPERIMENTS EXECUTION PLAN
================================================================================
Date: December 12, 2025
Status: 42 experiments remaining (14 activations × 3 trials)
Estimated Time: ~60-90 minutes on GPU via Docker

================================================================================
PRIORITY 1: HIGHEST IMPACT (Quick validation - 500 iterations)
================================================================================

**1. TinyLSTM - Complete remaining 4 activations** [TIME: ~15-20 min]
   Why: Validate exceptional reproducibility (PD=0.049 with ReLU)
   - GELU (3 trials)
   - Swish (3 trials)
   - SwiGLU (3 trials)
   - SmeLU-1 (3 trials)
   Expected outcome: Confirm if LSTM architecture is universally stable
   Research value: HIGH - Could establish LSTM as reproducibility baseline

**2. NanoTransformer - Complete remaining 4 activations** [TIME: ~15-20 min]
   Why: Complete transformer baseline comparison
   - GELU (3 trials)
   - Swish (3 trials)
   - SwiGLU (3 trials)
   - SmeLU-1 (3 trials)
   Expected outcome: Validate transformer sensitivity pattern
   Research value: MEDIUM - Confirms transformer architectural trends

**3. CharLM - Add SmeLU-1** [TIME: ~3-5 min]
   Why: Complete the most sensitive model (CV=21.89%)
   - SmeLU-1 (3 trials)
   Expected outcome: Test if SmeLU matches SwiGLU's excellence
   Research value: MEDIUM - Completes high-sensitivity case study

PRIORITY 1 TOTAL: 13 activations × 3 trials = 39 experiments (~35-45 minutes)

================================================================================
PRIORITY 2: COMPLETION (Quick validation - 500 iterations)
================================================================================

**4. ConvLM - Add remaining 2 activations** [TIME: ~6-10 min]
   - SwiGLU (3 trials)
   - SmeLU-1 (3 trials)
   Expected outcome: Test if SwiGLU improves CNN reproducibility
   Research value: LOW - ConvLM already shows moderate sensitivity

**5. HybridLM - Add remaining 2 activations** [TIME: ~6-10 min]
   - SwiGLU (3 trials)
   - SmeLU-1 (3 trials)
   Expected outcome: Complete least-sensitive model
   Research value: LOW - Hybrid already shows low sensitivity (CV=10.32%)

PRIORITY 2 TOTAL: 4 activations × 3 trials = 12 experiments (~12-20 minutes)

================================================================================
PRIORITY 3: OPTIONAL (Full scale - 5000 iterations)
================================================================================

**6. MiniGPT - Add SwiGLU** [TIME: ~4-5 hours on GPU]
   - SwiGLU (3 trials)
   Why: Test if SwiGLU's 45% advantage on CharLM generalizes to large models
   Expected outcome: Potentially best activation for 10.8M param models
   Research value: HIGH - But time-intensive
   Decision: DEFER unless significant research interest

================================================================================
RECOMMENDED EXECUTION STRATEGY
================================================================================

**Option A: Complete All Quick Validations (RECOMMENDED)**
Timeline: 60-90 minutes
Command sequence:
1. Fix the GPU run scripts (result access bug)
2. Create single script to run all remaining quick validations
3. Execute: TinyLSTM → NanoTransformer → CharLM → ConvLM → HybridLM
4. Process results and regenerate plots
5. Update report with complete findings

Deliverables:
✓ Complete 5/6 models (all except MiniGPT-SwiGLU)
✓ Definitive LSTM reproducibility characterization
✓ Complete transformer taxonomy
✓ Publication-ready dataset

**Option B: Critical Path Only (FASTEST)**
Timeline: 20-30 minutes
Execute: TinyLSTM + CharLM only (5 activations × 3 trials = 15 experiments)
Rationale: Completes highest-value experiments
- TinyLSTM: Validates exceptional LSTM stability
- CharLM: Completes most sensitive model

Deliverables:
✓ Answer key question: "Are LSTMs universally reproducible?"
✓ Complete CharLM SwiGLU discovery validation
✓ Sufficient for academic publication

**Option C: Defer All (CURRENT STATE)**
Timeline: 0 minutes
Rationale: Current results already support strong conclusions
- 4/6 models with 3+ activations tested
- SwiGLU breakthrough discovered
- Architecture taxonomy established

Publication readiness: ACCEPTABLE (with limitations noted)

================================================================================
TECHNICAL EXECUTION PLAN (Option A)
================================================================================

**Step 1: Fix GPU run scripts** [5 min]
Issue: Scripts access result['reproducibility_metrics']['relative_pd']
Fix: Change to result['avg_relative_pd']
Files: run_nanotransformer_gpu.py, run_tinylstm_gpu.py, run_convlm_gpu.py, 
       run_hybridlm_gpu.py

**Step 2: Create comprehensive run script** [5 min]
File: run_remaining_experiments.sh
Models: TinyLSTM, NanoTransformer, CharLM (SmeLU-1), ConvLM, HybridLM
Activations: Varies by model (see PRIORITY 1 & 2 above)

**Step 3: Execute via Docker GPU** [60-90 min]
Command: ./run_remaining_experiments.sh
Monitor: tail -f remaining_experiments.log
GPU check: watch -n 5 nvidia-smi

**Step 4: Process and visualize** [10 min]
Command: python process_all_results.py
Output: Updated plots in plots/ directory

**Step 5: Update report** [15 min]
- Add TinyLSTM complete results
- Add NanoTransformer complete results  
- Update conclusions with LSTM findings
- Regenerate statistics tables

Total time: 90-120 minutes for complete study

================================================================================
EXPECTED RESEARCH OUTCOMES
================================================================================

**If TinyLSTM remains stable across all activations:**
→ MAJOR FINDING: "LSTM architecture provides inherent reproducibility 
   independent of activation function choice"
→ Publication angle: "Architecture matters more than activation"
→ Practical impact: Use LSTMs for reproducibility-critical applications

**If NanoTransformer matches CharLM sensitivity:**
→ Confirms: "All transformers are activation-sensitive"
→ Strengthens: GELU/SwiGLU universal recommendation
→ Validates: Transformer taxonomy finding

**If SwiGLU improves ConvLM/HybridLM:**
→ MAJOR FINDING: "SwiGLU is universally superior activation"
→ Publication angle: "Gating mechanisms stabilize all architectures"
→ Practical impact: Replace GELU with SwiGLU everywhere

**If patterns hold:**
→ Publication-ready complete study
→ Definitive activation function guidelines by architecture
→ Industry-applicable recommendations

================================================================================
RECOMMENDATION
================================================================================

Execute **Option A: Complete All Quick Validations**

Rationale:
1. Only 60-90 minutes total time (reasonable investment)
2. Answers critical LSTM reproducibility question
3. Completes 5/6 models for comprehensive dataset
4. Enables publication without major limitations
5. Provides definitive practitioner guidelines

Alternative:
If time-constrained, execute **Option B** (TinyLSTM + CharLM only)
- 20-30 minutes
- Answers highest-value questions
- Acceptable for publication with noted limitations

Defer MiniGPT-SwiGLU (Priority 3) unless:
- Reviewer requests it
- Follow-up study planned
- GPU time available at low cost

================================================================================
NEXT IMMEDIATE ACTIONS
================================================================================

1. [ ] Fix result access bug in 4 GPU run scripts
2. [ ] Create run_remaining_experiments.sh combining all models
3. [ ] Test single activation to verify GPU execution works
4. [ ] Launch full remaining experiments via Docker
5. [ ] Monitor progress and verify GPU utilization
6. [ ] Process results when complete
7. [ ] Update report with final findings

Estimated start-to-finish: 2-3 hours including setup and analysis
