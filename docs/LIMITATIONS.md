# Limitations

- **Simulator only:** QAOA runs on Qiskit Aer using a classical statevector
  expectation and simulator sampling. No quantum hardware is used.
- **Tiny instances:** the prototype's demonstrations contain only a few
  ambulances and incidents. They do not establish scalability.
- **Exponential exact validation:** brute-force validation enumerates binary
  QUBO states and is deliberately limited to small problems.
- **Invalid samples:** QAOA uses an unconstrained X mixer, so measured samples
  can violate the assignment constraints. On the new 3-by-2 demo, the
  seed-7, 4096-shot reps 1/2/3 sweep measured valid fractions of
  **0.341064, 0.311523, 0.315430** for the open case and
  **0.182617, 0.306396, 0.367188** for the closure case. The actual
  measurements are recorded in `artifacts/m2_sweep.txt`.
- **Penalty scaling:** the QUBO uses a penalty to discourage constraint
  violations. Penalty scale affects the energy landscape and may affect
  optimization and sampling.
- **Objective simplification:** the objective minimizes the sum of
  severity-weighted travel minutes, not response-time makespan, fairness, or
  total system delay.
- **Synthetic, static network:** roads and travel times are fixed synthetic
  inputs. There is no live traffic, time-dependent routing, road capacity,
  dispatch latency, or empirical calibration.
- **No quantum-advantage claim:** the prototype provides no evidence that QAOA
  outperforms classical methods.
- **Assignment is not vehicle routing:** the model assigns at most one
  incident to each ambulance for this dispatch step. It does not plan vehicle
  tours, return-to-base trips, or follow-up dispatches.
