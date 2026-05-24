-- imports MUST precede all content in Lean 4
import SureGPU.Types
import SureGPU.Lifespan
import SureGPU.Stream
import SureGPU.Transfer

/-!
# SURE GPU Schedule — Main Integration Proofs

This file ties together the lifespan, stream, and transfer results into
a single, self-contained statement about the SURE GPU schedule.

## What is proved here

| Theorem                              | Meaning                                                  |
|--------------------------------------|----------------------------------------------------------|
| `sure_dep_graph_acyclic`             | The data-dependency DAG has no cycles.                   |
| `sure_schedule_valid`                | All dependencies are satisfied (producer before consumer).|
| `sure_parallel_groups`               | Two pairs of ops run concurrently; schedule has latency 7.|
| `sure_b_stays_on_device`             | Hutchinson probe never crosses the PCIe bus.             |
| `sure_grad_stays_on_device`          | SURE gradient is used only on-device (Langevin update).  |
| `sure_only_one_scalar_d2h`           | The sole D2H is the 4-byte SURE scalar.                  |
| `sure_async_d2h_free`                | That transfer is hidden behind `computeGrad`.            |
| `sure_tight_lifespans`               | All buffers are freed right after their last use.        |
| `sure_memory_sharing_possible`       | Three buffer pairs can share one allocation.             |
| `sure_full_schedule_theorem`         | Master theorem collecting all guarantees.                |
-/


namespace SureGPU

/-!
## §1  Dependency Graph Is Acyclic

We encode the edges of the dependency DAG explicitly and check that there is
no path from any node back to itself.  With 8 nodes the check is exhaustive.
-/

/-- Index the 8 SURE operations as natural numbers 0–7 for cycle detection. -/
def opIdx_genB        : ℕ := 0
def opIdx_denoiser1   : ℕ := 1
def opIdx_perturb     : ℕ := 2
def opIdx_denoiser2   : ℕ := 3
def opIdx_jacTrace    : ℕ := 4
def opIdx_computeSURE : ℕ := 5
def opIdx_computeGrad : ℕ := 6
def opIdx_d2hSURE     : ℕ := 7

/-- Adjacency list: `depEdges` maps each op to the list of ops it depends on. -/
def depEdges : List (ℕ × ℕ) :=
  [ (opIdx_perturb,     opIdx_genB)
  , (opIdx_perturb,     opIdx_denoiser1)
  , (opIdx_denoiser2,   opIdx_perturb)
  , (opIdx_jacTrace,    opIdx_denoiser2)
  , (opIdx_jacTrace,    opIdx_genB)
  , (opIdx_jacTrace,    opIdx_denoiser1)
  , (opIdx_computeSURE, opIdx_jacTrace)
  , (opIdx_computeSURE, opIdx_denoiser1)
  , (opIdx_computeGrad, opIdx_jacTrace)
  , (opIdx_computeGrad, opIdx_denoiser1)
  , (opIdx_d2hSURE,     opIdx_computeSURE)
  ]

/-- Every edge (consumer, producer) satisfies consumer > producer.
    This is the key no-cycle certificate: since indices strictly increase along
    every edge, no directed cycle can exist. -/
theorem sure_dep_graph_acyclic :
    ∀ e ∈ depEdges, e.1 > e.2 := by decide

/-- Corollary: the dependency graph is a DAG (no self-loops, no cycles). -/
theorem sure_dep_graph_dag :
    ∀ e ∈ depEdges, e.1 ≠ e.2 := by
  intro e he
  have h := sure_dep_graph_acyclic e he
  omega

/-!
## §2  Schedule Validity

All producer steps < consumer steps; combined with the barrier at step 2 that
synchronises streams S1 and S2, the schedule is correct.
-/

/-- Full schedule validity — all dependency ordering constraints are satisfied. -/
theorem sure_schedule_valid :
    op_genB.execStep      < op_perturb.execStep    ∧
    op_denoiser1.execStep < op_perturb.execStep    ∧
    op_perturb.execStep   < op_denoiser2.execStep  ∧
    op_denoiser2.execStep < op_jacTrace.execStep   ∧
    op_genB.execStep      < op_jacTrace.execStep   ∧
    op_jacTrace.execStep  < op_computeSURE.execStep ∧
    op_computeSURE.execStep < op_d2hSURE.execStep  ∧
    op_computeSURE.execStep < op_computeGrad.execStep :=
  sure_schedule_all_deps_honoured

/-- The step-2 barrier ensures genB (S2, step 1) is visible to perturb (S1, step 3). -/
theorem step2_barrier_provides_genB_visibility :
    -- genB finishes at step 1; the barrier is at step 2; perturb starts at step 3.
    -- So: genB.step (1) < barrier.step (2) < perturb.step (3).
    op_genB.execStep < STEP_SYNC ∧ STEP_SYNC < op_perturb.execStep := by decide

/-!
## §3  Parallel Groups

Two independent pairs of operations execute concurrently.
-/

/-- Group A: genB ∥ denoiser1 at step 1. -/
theorem sure_parallel_group_A : concurrent op_genB op_denoiser1 :=
  genB_denoiser1_concurrent

/-- Group B: computeGrad ∥ d2hSURE at step 7. -/
theorem sure_parallel_group_B : concurrent op_computeGrad op_d2hSURE :=
  computeGrad_d2hSURE_concurrent

/-- The schedule has latency 7 (parallel) vs. 8 (serial lower bound). -/
theorem sure_parallel_groups :
    concurrent op_genB op_denoiser1 ∧
    concurrent op_computeGrad op_d2hSURE ∧
    parallel_schedule_latency < serial_schedule_lower_bound :=
  ⟨sure_parallel_group_A, sure_parallel_group_B, parallel_is_strictly_faster⟩

/-!
## §4  Transfer Optimality
-/

/-- The Hutchinson probe b is never transferred to/from the host. -/
theorem sure_b_stays_on_device : SURETensor.needsH2D .b = false := b_no_h2d_flag

/-- The SURE gradient is never transferred to the host. -/
theorem sure_grad_stays_on_device : SURETensor.needsD2H .sureGrad = false :=
  sureGrad_no_d2h_flag

/-- The only D2H transfer is the 4-byte SURE scalar (sureVal). -/
theorem sure_only_one_scalar_d2h :
    allSURETensors.filter SURETensor.needsD2H = [.sureVal] :=
  sure_d2h_is_sureVal

/-- That D2H runs asynchronously on stream S3. -/
theorem sure_d2h_is_async : sureVal_d2h.isAsync := sureVal_d2h_is_async

/-- The async D2H is hidden inside the shadow of computeGrad for typical timings. -/
theorem sure_async_d2h_free (T_d2h T_grad : ℕ) (h : T_d2h ≤ T_grad) :
    Nat.max T_grad T_d2h = T_grad := async_d2h_hidden T_d2h T_grad h

/-!
## §5  Lifespan Tightness
-/

/-- All SURE tensor buffers are freed immediately after their last use. -/
theorem sure_tight_lifespans :
    span_b.waste STEP_JAC_TRACE         = 0 ∧
    span_r1.waste STEP_SURE              = 0 ∧
    span_xp.waste STEP_DENOISER2         = 0 ∧
    span_r2.waste STEP_JAC_TRACE          = 0 ∧
    span_jt.waste STEP_SURE               = 0 ∧
    span_sureVal.waste STEP_GRAD_D2H      = 0 :=
  all_sure_spans_tight

/-!
## §6  Memory-Sharing Opportunities
-/

/-- Three pairs of SURE tensors have non-overlapping lifespans and can share
    one GPU memory allocation. -/
theorem sure_memory_sharing_possible :
    span_xp.disjoint span_jt ∧
    span_r2.disjoint span_sureVal ∧
    span_jt.disjoint span_sureGrad :=
  sure_memory_sharing_pairs

/-!
## §7  Master Theorem
-/

/-- **sure_full_schedule_theorem** — a single statement collecting all guarantees
    of the SURE GPU schedule:

    (A) The computation graph is acyclic.
    (B) All data-dependency ordering constraints are satisfied.
    (C) Two pairs of independent operations execute concurrently (latency = 7).
    (D) No H2D transfer is needed for b (cuRAND on-device generation).
    (E) No D2H transfer is needed for the gradient (Langevin update on-device).
    (F) The sole D2H is the 4-byte SURE scalar, run asynchronously on S3.
    (G) Every buffer is freed immediately after its last use (tight lifespan).
    (H) Three buffer pairs can share one device allocation. -/
theorem sure_full_schedule_theorem :
    -- (A) acyclic dependency graph
    (∀ e ∈ depEdges, e.1 > e.2) ∧
    -- (B) all ordering constraints satisfied
    (op_genB.execStep      < op_perturb.execStep    ∧
     op_denoiser1.execStep < op_perturb.execStep    ∧
     op_perturb.execStep   < op_denoiser2.execStep  ∧
     op_denoiser2.execStep < op_jacTrace.execStep   ∧
     op_jacTrace.execStep  < op_computeSURE.execStep ∧
     op_computeSURE.execStep < op_d2hSURE.execStep) ∧
    -- (C) two concurrent pairs, latency strictly less than serial
    (concurrent op_genB op_denoiser1 ∧
     concurrent op_computeGrad op_d2hSURE ∧
     parallel_schedule_latency < serial_schedule_lower_bound) ∧
    -- (D) b does not need H2D
    SURETensor.needsH2D .b = false ∧
    -- (E) sureGrad does not need D2H
    SURETensor.needsD2H .sureGrad = false ∧
    -- (F) sole D2H = sureVal; it is async
    (allSURETensors.filter SURETensor.needsD2H = [.sureVal] ∧
     sureVal_d2h.isAsync) ∧
    -- (G) tight lifespans (zero waste)
    (span_b.waste STEP_JAC_TRACE      = 0 ∧
     span_r1.waste STEP_SURE           = 0 ∧
     span_xp.waste STEP_DENOISER2      = 0 ∧
     span_r2.waste STEP_JAC_TRACE       = 0 ∧
     span_jt.waste STEP_SURE            = 0 ∧
     span_sureVal.waste STEP_GRAD_D2H   = 0) ∧
    -- (H) three buffer pairs can share memory
    (span_xp.disjoint span_jt ∧
     span_r2.disjoint span_sureVal ∧
     span_jt.disjoint span_sureGrad) :=
  ⟨ sure_dep_graph_acyclic
  , ⟨by decide, by decide, by decide, by decide, by decide, by decide⟩
  , sure_parallel_groups
  , sure_b_stays_on_device
  , sure_grad_stays_on_device
  , ⟨sure_only_one_scalar_d2h, sure_d2h_is_async⟩
  , sure_tight_lifespans
  , sure_memory_sharing_possible
  ⟩

/-!
## §8  Connection to the Algebraic SURE Proofs

The GPU schedule above is *semantically neutral* — it says nothing about whether
the denoiser D_θ correctly estimates the score, whether SURE is unbiased, or
whether the Jacobian trace is well-approximated.  Those concerns live in the
`SGPSFull` namespace (SURE_sgps_full.lean / lean_proofs/SGPSProofs/).

The formal connection is:

  `SGPSFull.sure_mse_ring_identity` — algebraic SURE formula is correct.
  `SGPSFull.hutchinson_diagonal`    — Hutchinson trace estimator is unbiased.
  `SureGPU.sure_full_schedule_theorem` — the GPU schedule that computes these
                                         formulas is valid, parallel, and transfer-optimal.

Together they certify: the GPU implementation of SURE is both *mathematically
sound* (SGPSFull) and *computationally efficient* (SureGPU).
-/

end SureGPU
