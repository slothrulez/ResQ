# Q-Rescue mathematical formulation

Q-Rescue solves the **ambulance-to-incident assignment** subproblem with QAOA.
Road routing (shortest travel times on the road network) is a separate,
classical step (Dijkstra). QAOA never computes routes.

## 1. Inputs

- Ambulances `a ∈ A = {0..nA-1}`, each parked at a road-network node `s_a`.
- Incidents `i ∈ I = {0..nI-1}` at nodes `v_i`, with severity weight `w_i > 0`
  (critical = 3, serious = 2, minor = 1). Requires `nA ≥ nI`.
- Road network `G = (V, E)`, undirected, edge travel time `t_e > 0` minutes.
  A road closure removes edges from `G`.
- `d(u, v)`: shortest travel time from `u` to `v` in `G`, computed classically
  with Dijkstra. `d = ∞` if unreachable.

Dispatch cost of sending ambulance `a` to incident `i`:

    c[a,i] = w_i · d(s_a, v_i)          (severity-weighted minutes)

## 2. Decision variables

Binary `x[a,i] ∈ {0,1}`, `x[a,i] = 1` iff ambulance `a` is sent to incident `i`.
Only **reachable** pairs (`d(s_a, v_i) < ∞`) get a variable; the set of
variables is `K = {(a,i) : d(s_a,v_i) < ∞}` and `n = |K|` qubits.
Qubit index `k` enumerates `K` in row-major order (ambulance, then incident).

## 3. Constraints

- (C1) Every incident gets exactly one ambulance: `Σ_a x[a,i] = 1  ∀ i`.
- (C2) Every ambulance serves at most one incident: `Σ_i x[a,i] ≤ 1  ∀ a`.

If some incident has no reachable ambulance, or a perfect assignment of all
incidents does not exist in `K`, the instance is infeasible and is reported as
such before any quantum execution.

## 4. Objective

    minimise  C(x) = Σ_{(a,i)∈K} ĉ[a,i] · x[a,i]

where `ĉ = c / c_max`, `c_max = max_{(a,i)∈K} c[a,i]`, so `ĉ ∈ (0, 1]`.
Normalisation does not change the argmin; it keeps Ising coefficients O(1),
which matters for QAOA angle optimisation. All reported objective values are
converted back to minutes (`C · c_max`).

This minimises **total** severity-weighted response time. It does not minimise
the worst single response (a makespan / bottleneck objective would need a
different model).

## 5. QUBO with penalties

    E(x) = C(x) + P · H1(x) + P · H2(x)

    H1(x) = Σ_i ( 1 − Σ_a x[a,i] )²                  (C1, one-hot per incident)
    H2(x) = Σ_a Σ_{i<j} x[a,i] · x[a,j]              (C2, at most one per ambulance)

Sums only range over pairs in `K`.

`H2` encodes "at most one" exactly without slack qubits: for binary variables,
`Σ_{i<j} x_i x_j = m(m−1)/2` where `m` is the number of ones, which is 0 iff
`m ≤ 1`.

Expanded with `x² = x`:

    (1 − Σ_a x_a)² = 1 − Σ_a x_a + 2 Σ_{a<b} x_a x_b

so the QUBO has a constant `P · nI`, linear terms `ĉ[a,i] − P`, and quadratic
terms `2P` (same incident, different ambulances) and `P` (same ambulance,
different incidents).

### Penalty weight

    P = 1 + Σ_i max_a ĉ[a,i]

Claim: every feasible `x` has lower energy than every infeasible `x`.

- `H1` and `H2` are non-negative integers, and at least one is ≥ 1 when `x`
  violates a constraint. So infeasible `E(x) ≥ P + C(x) ≥ P` (costs ≥ 0).
- A feasible `x` picks one ambulance per incident, so
  `E(x) = C(x) ≤ Σ_i max_a ĉ[a,i] = P − 1 < P`.

Hence the QUBO ground state is the optimal feasible assignment. The downside
(a known QAOA risk) is that a large `P` compresses the cost differences between
feasible solutions relative to the penalty landscape.

## 6. Ising Hamiltonian

Substitute `x_k = (1 − z_k)/2` where `z_k ∈ {+1, −1}` is the eigenvalue of
Pauli `Z` on qubit `k` (`|0⟩ → x=0`, `|1⟩ → x=1`). For QUBO
`E = const + Σ_k q_k x_k + Σ_{k<l} Q_kl x_k x_l`:

    x_k        = 1/2 − z_k/2
    x_k x_l    = 1/4 − z_k/4 − z_l/4 + z_k z_l/4

giving `H = offset·I + Σ_k h_k Z_k + Σ_{k<l} J_kl Z_k Z_l` with

    h_k     = −q_k/2 − Σ_{l≠k} Q_kl/4
    J_kl    =  Q_kl/4
    offset  = const + Σ_k q_k/2 + Σ_{k<l} Q_kl/4

`H` is diagonal and `⟨b|H|b⟩ = E(x(b))` for every computational basis state;
this identity is checked by the tests for all `2^n` states.

## 7. QAOA

    |ψ(γ, β)⟩ = Π_{l=1..p} e^{−iβ_l Σ_k X_k} e^{−iγ_l H} |+⟩^{⊗n}

Angles are optimised classically (COBYLA, fixed multi-start grid, fixed seed)
to minimise `⟨ψ|H|ψ⟩` evaluated on Qiskit Aer. The optimised circuit is then
sampled on Aer; each measured bitstring is decoded to `x`, checked against C1
and C2, and scored with `C(x)`. The reported QAOA candidate is the
lowest-cost **valid** sampled bitstring. If no sample is valid, Q-Rescue
reports that and does not substitute a classical answer.

## 8. Decoding

Qiskit bitstrings are little-endian: character `n−1−k` of the string is qubit
`k`. `x[K[k]] = int(bitstring[n−1−k])`. The decoded assignment is the list of
`(ambulance, incident)` pairs with `x = 1`.

## 9. Validation (classical, independent of QAOA)

- **Exact optimum**: brute-force enumeration of all `2^n` bitstrings of the
  QUBO (feasible only for small `n`).
- **Classical baseline**: Hungarian algorithm
  (`scipy.optimize.linear_sum_assignment`) on the original cost matrix `c`
  (unreachable pairs disallowed). It solves the assignment problem exactly
  without the QUBO, so agreement with brute force cross-checks the QUBO.
