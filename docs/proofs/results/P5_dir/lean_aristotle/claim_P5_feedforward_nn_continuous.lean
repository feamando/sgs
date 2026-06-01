import Mathlib

set_option maxHeartbeats 800000

/-!
# Claim P5: Feedforward Neural Networks are Continuous

A feedforward neural network `f = Wₙ ∘ σₙ₋₁ ∘ ⋯ ∘ σ₁ ∘ W₁`, where each `σₖ` is a
continuous activation function (ReLU, sigmoid, tanh, or LayerNorm) and each `Wₖ` is an
affine map (linear + bias), is continuous as a composition of continuous functions.

This is a standard result in analysis: the composition of finitely many continuous functions
is continuous.

## Formalization

We prove this in three parts:

1. **General composition theorem**: The composition of a list of continuous endomorphisms on
   a topological space is continuous (`continuous_composeList`).

2. **Neural network theorem (uniform dimension)**: A feedforward neural network with `n`
   layers on a single topological space, where each layer applies a continuous affine map
   `Wₖ` followed by a continuous activation `σₖ`, is continuous
   (`feedforward_nn_continuous`).

3. **Neural network theorem (heterogeneous dimensions)**: A feedforward neural network
   where each layer maps between (potentially different) topological spaces is continuous,
   proved by induction on the number of layers (`feedforward_nn_hetero_continuous`).
-/

open scoped Topology

-- ============================================================================
-- Part 1: General Composition Theorem
-- ============================================================================

/-- Compose a list of endomorphisms (applied right-to-left, head is outermost). -/
def composeList {α : Type*} : List (α → α) → (α → α)
  | [] => id
  | f :: fs => f ∘ composeList fs

/-
The composition of finitely many continuous endomorphisms is continuous.
-/
theorem continuous_composeList {α : Type*} [TopologicalSpace α]
    (fs : List (α → α)) (hfs : ∀ f ∈ fs, Continuous f) :
    Continuous (composeList fs) := by
  induction' fs with f fs ih;
  · exact continuous_id;
  · exact Continuous.comp ( hfs _ ( List.mem_cons_self ) ) ( ih fun g hg => hfs _ ( List.mem_cons_of_mem _ hg ) )

-- ============================================================================
-- Part 2: Feedforward Neural Network (uniform dimension)
-- ============================================================================

/-- A feedforward neural network with `n` layers on a topological space `X`.
    Each layer consists of a continuous map `W k` (affine transformation)
    followed by a continuous map `σ_act k` (activation function).
    The full network computes `(σ_act (n-1) ∘ W (n-1)) ∘ ⋯ ∘ (σ_act 0 ∘ W 0)`. -/
noncomputable def feedforwardNet {X : Type*} [TopologicalSpace X]
    {n : ℕ} (W σ_act : Fin n → (X → X)) : X → X :=
  composeList ((List.finRange n).reverse.map (fun k => σ_act k ∘ W k))

/-
A feedforward neural network, where each affine map `Wₖ` and each activation
    function `σₖ` is continuous, is itself continuous.
-/
theorem feedforward_nn_continuous {X : Type*} [TopologicalSpace X]
    {n : ℕ}
    (W σ_act : Fin n → (X → X))
    (hW : ∀ k, Continuous (W k))
    (hσ : ∀ k, Continuous (σ_act k)) :
    Continuous (feedforwardNet W σ_act) := by
  convert continuous_composeList _ _;
  exact fun f hf => by rw [ List.mem_map ] at hf; obtain ⟨ k, _, rfl ⟩ := hf; exact Continuous.comp ( hσ k ) ( hW k ) ;

/-
============================================================================
Part 3: Feedforward Neural Network (heterogeneous dimensions)
============================================================================

A feedforward neural network where layer `k` maps between topological spaces
    `X k` and `X (k+1)` is continuous. This models networks with varying hidden
    dimensions: input ∈ X₀, hidden layers in X₁, …, Xₙ₋₁, output ∈ Xₙ.

    Each layer function `layer k` represents `σₖ ∘ Wₖ : X k → X (k+1)`.
-/
theorem feedforward_nn_hetero_continuous
    (n : ℕ)
    (X : Fin (n + 1) → Type*)
    [inst : ∀ i, TopologicalSpace (X i)]
    (layer : ∀ k : Fin n, X (Fin.castSucc k) → X (Fin.succ k))
    (hlayer : ∀ k, Continuous (layer k)) :
    ∃ (f : X (⟨0, Nat.zero_lt_succ n⟩) → X (Fin.last n)),
      Continuous f := by
  induction' n with n ih;
  · exact ⟨ id, continuous_id ⟩;
  · obtain ⟨ f, hf ⟩ := ih ( fun i => X i.castSucc ) ( fun k => layer k.castSucc ) fun k => hlayer k.castSucc;
    exact ⟨ fun x => layer ( Fin.last _ ) ( f x ), hlayer _ |> Continuous.comp <| hf ⟩