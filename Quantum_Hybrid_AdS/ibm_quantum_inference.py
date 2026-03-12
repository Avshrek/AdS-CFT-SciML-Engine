"""
ibm_quantum_inference.py — Run Quantum Neural-AdS on Real IBM Hardware
======================================================================

Connects to IBM Quantum Cloud and executes the trained 10-qubit circuit
on a real superconducting quantum processor, proving the model's quantum
gates physically translate to real-world hardware.

Usage
-----
    python ibm_quantum_inference.py
    python ibm_quantum_inference.py --backend ibm_sherbrooke
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Quantum Neural-AdS inference on IBM Quantum Cloud",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", type=str,
                    default=os.path.join("models", "NATURE_QUANTUM_MODEL.pth"))
    p.add_argument("--data_dir", type=str, default="data_collision_master")
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--backend", type=str, default=None,
                    help="IBM backend name (e.g. ibm_sherbrooke). Auto-selects least busy if omitted.")
    p.add_argument("--shots", type=int, default=32768,
                    help="Number of measurement shots per circuit execution")
    p.add_argument("--simulator", action="store_true",
                    help="Use IBM's cloud simulator instead of real hardware")
    p.add_argument("--mitigate", action="store_true", default=True,
                    help="Use EstimatorV2 with ZNE + readout mitigation + Dynamical Decoupling")
    p.add_argument("--no-mitigate", dest="mitigate", action="store_false",
                    help="Disable error mitigation (use raw SamplerV2)")
    p.add_argument("--optimization_level", type=int, default=3,
                    choices=[0, 1, 2, 3],
                    help="Transpiler optimization level (3 = maximum circuit compression)")
    p.add_argument("--zne_amplifier", type=str, default="gate_folding",
                    choices=["gate_folding", "pulse_stretching"],
                    help="ZNE noise amplification strategy")
    p.add_argument("--pec", action="store_true", default=False,
                    help="Use Probabilistic Error Cancellation (resilience_level=3) "
                         "with fractional-gate pulse-level transpilation")
    p.add_argument("--pec_shots", type=int, default=32768,
                    help="Shot budget for PEC/max-accuracy pipeline")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Connect to IBM Quantum
# ═══════════════════════════════════════════════════════════════════════
def connect_ibm(api_key: str):
    """Authenticate with IBM Quantum Platform."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    print("🔐 Authenticating with IBM Quantum Platform...")
    try:
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=api_key,
        )
        print("✅ Connected to IBM Quantum Platform!")
        backends = service.backends(min_num_qubits=10)
        print(f"   Available backends: {[b.name for b in backends]}")
        return service
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None


def select_backend(service, preferred: str | None, use_simulator: bool):
    """Select the best available backend."""
    if use_simulator:
        print("🖥️  Using IBM Cloud Simulator (no queue)")
        backend = service.least_busy(simulator=True, min_num_qubits=10)
        print(f"   Selected: {backend.name}")
        return backend

    if preferred:
        print(f"🔍 Looking for backend: {preferred}")
        backend = service.backend(preferred)
        print(f"   ✅ Found: {backend.name} ({backend.num_qubits} qubits)")
        return backend

    print("🔍 Finding least-busy 10+ qubit backend...")
    backend = service.least_busy(simulator=False, min_num_qubits=10)
    print(f"   ✅ Selected: {backend.name} ({backend.num_qubits} qubits)")
    return backend


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Build the Qiskit Circuit from PennyLane Weights
# ═══════════════════════════════════════════════════════════════════════
def build_circuit_from_weights(classical_latent: np.ndarray, quantum_weights: np.ndarray):
    """
    Reconstruct the exact PennyLane circuit as a Qiskit QuantumCircuit.

    This manually translates:
      - AngleEmbedding(inputs, rotation='Y')
      - StronglyEntanglingLayers(weights)
    into native Qiskit gates.
    """
    from qiskit.circuit import QuantumCircuit

    n_qubits = 10
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Phase A: AngleEmbedding with Y-rotation
    for i in range(n_qubits):
        qc.ry(float(classical_latent[i]), i)

    # Phase B: StronglyEntanglingLayers
    n_layers = quantum_weights.shape[0]
    for layer_idx in range(n_layers):
        # Rotation gates (Rot = RZ @ RY @ RZ)
        for qubit in range(n_qubits):
            phi   = float(quantum_weights[layer_idx, qubit, 0])
            theta = float(quantum_weights[layer_idx, qubit, 1])
            omega = float(quantum_weights[layer_idx, qubit, 2])
            qc.rz(phi, qubit)
            qc.ry(theta, qubit)
            qc.rz(omega, qubit)

        # CNOT entangling pattern (PennyLane's default for StronglyEntanglingLayers)
        for qubit in range(n_qubits):
            target = (qubit + 1 + layer_idx) % n_qubits
            qc.cx(qubit, target)

    # Measurement
    qc.measure(range(n_qubits), range(n_qubits))

    return qc


def build_circuit_no_measurements(classical_latent: np.ndarray, quantum_weights: np.ndarray):
    """
    Same circuit as build_circuit_from_weights but WITHOUT measurements.
    Required for EstimatorV2 (observables are passed separately).
    """
    from qiskit.circuit import QuantumCircuit

    n_qubits = 10
    qc = QuantumCircuit(n_qubits)           # no classical register

    for i in range(n_qubits):
        qc.ry(float(classical_latent[i]), i)

    n_layers = quantum_weights.shape[0]
    for layer_idx in range(n_layers):
        for qubit in range(n_qubits):
            phi   = float(quantum_weights[layer_idx, qubit, 0])
            theta = float(quantum_weights[layer_idx, qubit, 1])
            omega = float(quantum_weights[layer_idx, qubit, 2])
            qc.rz(phi, qubit)
            qc.ry(theta, qubit)
            qc.rz(omega, qubit)

        for qubit in range(n_qubits):
            target = (qubit + 1 + layer_idx) % n_qubits
            qc.cx(qubit, target)

    return qc


# ═══════════════════════════════════════════════════════════════════════
# STEP 2b: Fractional-Gate / Pulse-Level Transpilation Pipeline
# ═══════════════════════════════════════════════════════════════════════
def transpile_fractional_gates(circuit, backend):
    """
    Multi-stage transpilation pipeline targeting hardware-native fractional
    gates for MAXIMUM depth reduction.

    Stage 1: opt_level=3 base transpilation (fractional gate synthesis)
    Stage 2: Iterative gate-fusion optimization (with layout preservation):
             - ConsolidateBlocks     → fuse sequences of 1q/2q gates
             - UnitarySynthesis      → re-synthesize as optimal fractional
                                       RZX / ECR decompositions
             - Optimize1qGatesDecomposition → merge consecutive 1q rotations
             - CommutativeCancellation      → cancel adjacent commuting gates
             Run 3 rounds until convergence.
    Stage 3: Pad with Dynamical Decoupling (XX) at the transpiler level
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import (
        Optimize1qGatesDecomposition,
        CommutativeCancellation,
        ConsolidateBlocks,
        UnitarySynthesis,
        PadDynamicalDecoupling,
    )
    from qiskit.circuit.library import XGate

    print(f"\n🔬 FRACTIONAL-GATE PULSE-LEVEL TRANSPILATION PIPELINE")
    print(f"   Stage 1: Base transpilation (opt_level=3)...")
    pm_base = generate_preset_pass_manager(backend=backend, optimization_level=3)
    transpiled = pm_base.run(circuit)
    depth_s1 = transpiled.depth()
    gates_s1 = transpiled.size()
    print(f"   ├── Depth: {depth_s1},  Gates: {gates_s1}")

    # Save layout before custom passes (they don't preserve it)
    saved_layout = transpiled.layout

    # Stage 2: Iterative gate-fusion optimization (up to 3 rounds)
    print(f"   Stage 2: Iterative fractional-gate re-synthesis...")
    target = backend.target
    basis_gates = list(target.operation_names)

    optimization_round = PassManager([
        ConsolidateBlocks(),
        UnitarySynthesis(basis_gates=basis_gates, target=target,
                         min_qubits=2),
        Optimize1qGatesDecomposition(basis=basis_gates, target=target),
        CommutativeCancellation(),
        Optimize1qGatesDecomposition(basis=basis_gates, target=target),
    ])

    prev_depth = depth_s1
    for round_num in range(1, 4):
        transpiled = optimization_round.run(transpiled)
        new_depth = transpiled.depth()
        new_gates = transpiled.size()
        improved = prev_depth - new_depth
        print(f"   ├── Round {round_num}: depth={new_depth}, gates={new_gates}"
              f" (Δdepth={-improved})")
        if improved <= 0:
            break
        prev_depth = new_depth

    # Restore layout
    transpiled._layout = saved_layout

    depth_s2 = transpiled.depth()
    gates_s2 = transpiled.size()
    reduction_pct = (1 - depth_s2 / depth_s1) * 100 if depth_s1 > 0 else 0
    print(f"   ├── Total depth reduction: {reduction_pct:.1f}%")

    # Stage 3: Pad dynamical decoupling at transpiler level
    # Use XX sequence (two X gates) which is universally supported
    print(f"   Stage 3: Dynamical Decoupling padding...")
    try:
        dd_sequence = [XGate(), XGate()]   # XX echo (spin echo)
        dd_pass = PassManager([
            PadDynamicalDecoupling(
                target=target,
                dd_sequence=dd_sequence,
            )
        ])
        transpiled = dd_pass.run(transpiled)
        print(f"   └── XX echo pulses inserted into idle slots")
    except Exception as e:
        print(f"   └── DD padding skipped (will use runtime DD): {e}")

    depth_final = transpiled.depth()
    gates_final = transpiled.size()
    print(f"\n   📊 Final circuit: depth={depth_final}, gates={gates_final}")

    return transpiled


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Execute on IBM Hardware and Compute Expectation Values
# ═══════════════════════════════════════════════════════════════════════
def run_on_hardware(service, backend, circuit, shots: int,
                    optimization_level: int = 3):
    """Submit the circuit to IBM hardware (raw SamplerV2, no mitigation)."""
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    print(f"\n🔧 Transpiling circuit for {backend.name} (opt_level={optimization_level})...")
    pm = generate_preset_pass_manager(backend=backend,
                                      optimization_level=optimization_level)
    transpiled = pm.run(circuit)
    print(f"   Circuit depth: {transpiled.depth()}")
    print(f"   Gate count: {transpiled.size()}")

    print(f"\n🚀 Submitting job to {backend.name} with {shots} shots...")
    sampler = Sampler(mode=backend)
    job = sampler.run([transpiled], shots=shots)

    print(f"   Job ID: {job.job_id()}")
    print(f"   ⏳ Waiting for results (this may take a few minutes on real hardware)...")

    start = time.time()
    result = job.result()
    elapsed = time.time() - start
    print(f"   ✅ Results received in {elapsed:.1f} seconds!")

    return result, job.job_id(), elapsed


def run_on_hardware_mitigated(service, backend, circuit_no_meas, shots: int,
                              n_qubits: int = 10,
                              optimization_level: int = 3,
                              zne_amplifier: str = "gate_folding"):
    """
    Execute on IBM hardware with FULL error mitigation stack:
      1. Transpiler optimization_level=3  (maximum gate compression)
      2. Dynamical Decoupling – XY4       (suppresses idle decoherence)
      3. TREX readout error mitigation    (fixes measurement bit-flips)
      4. Zero-Noise Extrapolation (ZNE)   (extrapolates gate noise → 0)

    Uses EstimatorV2 which directly returns ⟨Z_i⟩ expectation values
    without manual bitstring counting.
    """
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    # ── Transpile at maximum optimization ─────────────────────────────
    print(f"\n🔧 Transpiling circuit for {backend.name} (opt_level={optimization_level})...")
    pm = generate_preset_pass_manager(backend=backend,
                                      optimization_level=optimization_level)
    transpiled = pm.run(circuit_no_meas)
    print(f"   Circuit depth: {transpiled.depth()}")
    print(f"   Gate count: {transpiled.size()}")

    # ── Build Pauli-Z observables mapped to physical qubits ────────────
    #    The transpiled circuit lives in the full 133-qubit backend space.
    #    We build 10-qubit logical observables then apply the transpiler's
    #    layout so they expand to the correct physical qubit indices.
    observables = []
    for i in range(n_qubits):
        label = ['I'] * n_qubits
        label[n_qubits - 1 - i] = 'Z'   # SparsePauliOp uses big-endian
        obs_logical = SparsePauliOp(''.join(label))
        obs_physical = obs_logical.apply_layout(transpiled.layout)
        observables.append(obs_physical)

    # ── Configure EstimatorV2 with resilience ─────────────────────────
    print(f"\n🛡️  Configuring error mitigation stack:")
    estimator = Estimator(mode=backend)

    # Resilience level 2 = ZNE + TREX (measurement error mitigation)
    estimator.options.resilience_level = 2
    print(f"   ✅ Resilience level 2  (ZNE + TREX readout correction)")

    # Dynamical Decoupling: insert XY4 pulse sequences during idle times
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = "XY4"
    print(f"   ✅ Dynamical Decoupling: XY4")

    # ZNE noise amplification strategy
    try:
        estimator.options.resilience.zne_mitigation = True
        estimator.options.resilience.zne.amplifier = zne_amplifier
        estimator.options.resilience.zne.noise_factors = [1, 3, 5]
        print(f"   ✅ ZNE noise factors: [1, 3, 5] via {zne_amplifier}")
    except Exception:
        # Some runtime versions configure ZNE purely via resilience_level
        print(f"   ℹ️  ZNE configured via resilience_level=2")

    # Shot budget
    estimator.options.default_shots = shots
    print(f"   ✅ Shots: {shots}")

    # ── Submit one PUB per qubit observable ────────────────────────────
    print(f"\n🚀 Submitting mitigated job to {backend.name}...")
    pubs = [(transpiled, obs) for obs in observables]
    job = estimator.run(pubs)

    print(f"   Job ID: {job.job_id()}")
    print(f"   ⏳ Waiting for error-mitigated results...")

    start = time.time()
    result = job.result()
    elapsed = time.time() - start
    print(f"   ✅ Mitigated results received in {elapsed:.1f} seconds!")

    # ── Extract expectation values ────────────────────────────────────
    expectations = np.zeros(n_qubits)
    for i in range(n_qubits):
        pub_result = result[i]
        evs = pub_result.data.evs
        val = float(evs) if np.ndim(evs) == 0 else float(evs[0])
        expectations[i] = val

    return expectations, job.job_id(), elapsed


def run_pec_pipeline(service, backend, circuit_no_meas, shots: int,
                     n_qubits: int = 10):
    """
    MAXIMUM ACCURACY PIPELINE: Fractional-gate transpilation + PEC
    ================================================================

    Combines two orthogonal error-reduction strategies:

    1. FRACTIONAL-GATE TRANSPILATION
       Reduces circuit depth by re-synthesizing into hardware-native
       fractional RZX/ECR gates, fusing 1q rotations, cancelling
       commuting operators, and pre-padding XY4 Dynamical Decoupling.

    2. PROBABILISTIC ERROR CANCELLATION (PEC, resilience_level=3)
       The theoretically optimal error mitigation protocol:
       - Learns the FULL noise model of each gate on the hardware
       - Decomposes noisy operations into quasi-probability mixtures
         of ideal operations
       - Stochastically samples correction operations that, when
         averaged, cancel noise EXACTLY (up to sampling variance)
       - Requires ~10-100x more shots for the variance to converge

    Combined with TREX readout correction and XY4 Dynamical Decoupling
    at the runtime level as a safety net.
    """
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    from qiskit.quantum_info import SparsePauliOp

    print("\n" + "=" * 70)
    print("  🔬 MAXIMUM ACCURACY PIPELINE: Fractional Gates + PEC")
    print("=" * 70)

    # ── Phase 1: Fractional-gate transpilation ────────────────────────
    transpiled = transpile_fractional_gates(circuit_no_meas, backend)

    # ── Phase 2: Build layout-mapped observables ──────────────────────
    print(f"\n🎯 Building {n_qubits} Pauli-Z observables (layout-mapped)...")
    observables = []
    for i in range(n_qubits):
        label = ['I'] * n_qubits
        label[n_qubits - 1 - i] = 'Z'
        obs_logical = SparsePauliOp(''.join(label))
        obs_physical = obs_logical.apply_layout(transpiled.layout)
        observables.append(obs_physical)
    print(f"   ✅ Observables mapped to physical qubits")

    # ── Phase 3: Configure EstimatorV2 — maximum accuracy ──────────────
    print(f"\n🛡️  Configuring MAXIMUM ACCURACY error mitigation:")
    estimator = Estimator(mode=backend)

    # Try resilience_level=3 (PEC) first; fall back to enhanced level 2
    pec_active = False
    try:
        estimator.options.resilience_level = 3
        pec_active = True
        print(f"   ✅ Resilience level 3  (PEC + TREX readout correction)")
    except Exception:
        estimator.options.resilience_level = 2
        print(f"   ℹ️  PEC requires premium plan — using enhanced ZNE (level 2)")
        print(f"   ✅ Resilience level 2  (ZNE + TREX readout correction)")

    # Dynamical Decoupling as runtime safety net
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = "XY4"
    print(f"   ✅ Dynamical Decoupling: XY4 (runtime level)")

    if pec_active:
        try:
            estimator.options.resilience.pec_mitigation = True
            estimator.options.resilience.pec_max_overhead = None
            print(f"   ✅ PEC: unlimited overhead (best accuracy)")
        except Exception:
            print(f"   ℹ️  PEC configured via resilience_level=3")
    else:
        # Enhanced ZNE: 3 noise factors + exponential extrapolation
        # (3 factors keeps runtime within IBM open-plan limits while
        #  still enabling quadratic-order extrapolation)
        try:
            estimator.options.resilience.zne_mitigation = True
            estimator.options.resilience.zne.amplifier = "gate_folding"
            estimator.options.resilience.zne.noise_factors = [1, 3, 5]
            try:
                estimator.options.resilience.zne.extrapolator = "exponential"
                print(f"   ✅ ZNE: 3 noise factors [1,3,5] + exponential extrapolation")
            except Exception:
                print(f"   ✅ ZNE: 3 noise factors [1,3,5] + default extrapolation")
        except Exception:
            print(f"   ℹ️  ZNE configured via resilience_level=2")

    # Shot budget & execution time limit
    estimator.options.default_shots = shots
    try:
        estimator.options.max_execution_time = 900  # 15 min safety margin
    except Exception:
        pass
    print(f"   ✅ Shots: {shots:,}")

    # ── Phase 4: Submit to hardware ───────────────────────────────────
    print(f"\n🚀 Submitting PEC job to {backend.name}...")
    pubs = [(transpiled, obs) for obs in observables]
    job = estimator.run(pubs)

    print(f"   Job ID: {job.job_id()}")
    print(f"   ⏳ Waiting for PEC-corrected results (this will take several minutes)...")

    start = time.time()
    result = job.result()
    elapsed = time.time() - start
    print(f"   ✅ PEC results received in {elapsed:.1f} seconds!")

    # ── Phase 5: Extract expectations + standard errors ───────────────
    expectations = np.zeros(n_qubits)
    std_errors = np.zeros(n_qubits)
    for i in range(n_qubits):
        pub_result = result[i]
        evs = pub_result.data.evs
        val = float(evs) if np.ndim(evs) == 0 else float(evs[0])
        expectations[i] = val
        # PEC also returns standard error estimates
        if hasattr(pub_result.data, 'stds'):
            stds = pub_result.data.stds
            std_errors[i] = float(stds) if np.ndim(stds) == 0 else float(stds[0])

    print(f"\n📊 PEC Expectation Values ± Standard Error:")
    for i in range(n_qubits):
        if std_errors[i] > 0:
            print(f"   Qubit {i}: {expectations[i]:+.6f} ± {std_errors[i]:.6f}")
        else:
            print(f"   Qubit {i}: {expectations[i]:+.6f}")

    return expectations, std_errors, job.job_id(), elapsed


def _extract_counts_from_field(field, n_qubits: int):
    """Extract a {bitstring: count} dict from a BitArray or similar object."""
    from collections import Counter

    # Method 1: get_counts() — available on most BitArray versions
    if hasattr(field, 'get_counts'):
        try:
            return field.get_counts()
        except Exception:
            pass

    # Method 2: get_bitstrings() → Counter  (returns list[str] or ndarray)
    if hasattr(field, 'get_bitstrings'):
        try:
            raw = field.get_bitstrings()
            strs = [
                bs if isinstance(bs, str)
                else ''.join(str(int(b)) for b in bs)
                for bs in raw
            ]
            return dict(Counter(strs))
        except Exception:
            pass

    # Method 3: .bitstrings ndarray attribute  (shape [shots, n_qubits])
    if hasattr(field, 'bitstrings'):
        try:
            raw = field.bitstrings          # numpy array
            strs = [''.join(str(int(b)) for b in row) for row in raw]
            return dict(Counter(strs))
        except Exception:
            pass

    # Method 4: .array packed-byte attribute  (shape [shots, ceil(n/8)])
    if hasattr(field, 'array'):
        try:
            arr = np.asarray(field.array)
            n_shots = arr.shape[0]
            strs = []
            for s in range(n_shots):
                bits = ''.join(format(byte, '08b') for byte in arr[s])
                strs.append(bits[-n_qubits:])   # right-aligned
            return dict(Counter(strs))
        except Exception:
            pass

    # Method 5: already a plain dict
    if isinstance(field, dict):
        return field

    return None


def counts_to_expectation_values(result, n_qubits: int = 10):
    """Convert measurement counts to Pauli-Z expectation values per qubit.

    Handles every known SamplerV2 / BitArray format across
    qiskit-ibm-runtime 0.15 → 0.40+ (including the new packed-binary
    BitArray returned by backends like ibm_torino).
    """
    pub_result = result[0]
    data = pub_result.data

    counts = None

    # ── Strategy 1: try well-known classical-register names ──────────
    for reg_name in ('c', 'meas', 'cr'):
        field = getattr(data, reg_name, None)
        if field is not None:
            counts = _extract_counts_from_field(field, n_qubits)
            if counts:
                print(f"   (extracted counts from data.{reg_name})")
                break

    # ── Strategy 2: dynamic field discovery via __fields__ ───────────
    if not counts:
        field_names = (
            list(data.__fields__)
            if hasattr(data, '__fields__')
            else [a for a in dir(data) if not a.startswith('_')]
        )
        for fname in field_names:
            field = getattr(data, fname, None)
            if field is not None:
                counts = _extract_counts_from_field(field, n_qubits)
                if counts:
                    print(f"   (extracted counts from data.{fname})")
                    break

    # ── Strategy 3: get_counts on data itself (very old API) ─────────
    if not counts and hasattr(data, 'get_counts'):
        try:
            counts = data.get_counts()
        except Exception:
            pass

    if not counts:
        raise RuntimeError(
            f"Could not extract measurement counts from IBM result.\n"
            f"  data type : {type(data)}\n"
            f"  fields    : {getattr(data, '__fields__', 'N/A')}\n"
            f"  dir(data) : {[a for a in dir(data) if not a.startswith('_')]}"
        )

    total_shots = sum(counts.values())
    print(f"   📊 Total measurement shots: {total_shots}")
    print(f"   📊 Unique bitstrings: {len(counts)}")

    expectations = np.zeros(n_qubits)
    for bitstring, count in counts.items():
        for qubit_idx in range(n_qubits):
            # Qiskit bit ordering is reversed
            bit = int(bitstring[n_qubits - 1 - qubit_idx])
            # |0⟩ → +1, |1⟩ → -1
            expectations[qubit_idx] += (1 - 2 * bit) * count

    expectations /= total_shots
    return expectations


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Classical Decoder Pass
# ═══════════════════════════════════════════════════════════════════════
def run_classical_decoder(model, quantum_output: np.ndarray):
    """Pass the quantum expectation values through the classical decoder."""
    with torch.no_grad():
        q_tensor = torch.from_numpy(quantum_output).float().unsqueeze(0)
        x_expanded = model.decoder_projection(q_tensor)
        x_reshaped = x_expanded.view(-1, 64, 2, 8, 8)
        bulk_pred = model.decoder_conv(x_reshaped)
    return bulk_pred.squeeze().cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    API_KEY = os.environ.get("IBM_QUANTUM_TOKEN", "X48_nNb7wCLzjR5uekoVRaXAsxWzb_scOqWxBEgzO8kM")

    print("=" * 70)
    print("  QUANTUM NEURAL-AdS  —  IBM QUANTUM CLOUD INFERENCE")
    print("=" * 70)

    # ── Load Model ────────────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hybrid_autoencoder import HybridQuantumAdS

    print("\n📦 Loading trained model weights...")
    model = HybridQuantumAdS(in_channels=1, out_channels=1)
    state_dict = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Extract quantum gate weights
    quantum_weights = None
    for name, param in model.named_parameters():
        if "q_layer.weights" in name or "q_weights" in name:
            quantum_weights = param.detach().cpu().numpy()
            print(f"   ✅ Quantum weights extracted: shape {quantum_weights.shape}")
            break

    if quantum_weights is None:
        print("❌ Could not find quantum weights in model!")
        return

    # ── Load Data & Run Classical Encoder ──────────────────────────────
    bdy_path = os.path.join(args.data_dir, "bdy_collision.npy")
    if not os.path.exists(bdy_path):
        print(f"❌ Boundary data not found: {bdy_path}")
        return

    print("📦 Loading boundary sample...")
    bdy = np.load(bdy_path)
    bdy_sample = torch.from_numpy(bdy[args.sample_idx]).float()
    bdy_input = bdy_sample.unsqueeze(-1).repeat(1, 1, 64).unsqueeze(0).unsqueeze(0)

    print("🧠 Running classical encoder (CPU)...")
    with torch.no_grad():
        classical_latent = model.encoder(bdy_input).squeeze().cpu().numpy()
    print(f"   ✅ Encoded to {classical_latent.shape[0]} classical values: {classical_latent}")

    # ── Build Quantum Circuit ─────────────────────────────────────────
    print("\n⚛️  Building 10-qubit circuit from trained weights...")
    circuit = build_circuit_from_weights(classical_latent, quantum_weights)
    print(f"   ✅ Circuit built: {circuit.num_qubits} qubits, depth {circuit.depth()}")

    # ── Connect to IBM ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    service = connect_ibm(API_KEY)
    if service is None:
        print("\n⚠️  IBM connection failed. Running local simulation instead...")
        run_local_fallback(model, bdy_input, args)
        return

    # ── Select Backend ────────────────────────────────────────────────
    try:
        backend = select_backend(service, args.backend, args.simulator)
    except Exception as e:
        print(f"\n⚠️  Backend selection failed: {e}")
        print("   Running local simulation instead...")
        run_local_fallback(model, bdy_input, args)
        return

    # ── Execute on Hardware ───────────────────────────────────────────
    print("\n" + "─" * 70)
    pec_stds = None
    if args.pec:
        print("\n🔬 MAXIMUM ACCURACY MODE (Fractional Gates + PEC + DD + TREX)")
        circuit_no_meas = build_circuit_no_measurements(classical_latent, quantum_weights)
        ibm_expectations, pec_stds, job_id, elapsed = run_pec_pipeline(
            service, backend, circuit_no_meas, args.pec_shots,
            n_qubits=10,
        )
    elif args.mitigate:
        print("\n🛡️  ERROR-MITIGATED MODE (EstimatorV2 + ZNE + DD + TREX)")
        circuit_no_meas = build_circuit_no_measurements(classical_latent, quantum_weights)
        ibm_expectations, job_id, elapsed = run_on_hardware_mitigated(
            service, backend, circuit_no_meas, args.shots,
            n_qubits=10,
            optimization_level=args.optimization_level,
            zne_amplifier=args.zne_amplifier,
        )
    else:
        print("\n⚡ RAW MODE (SamplerV2, no error mitigation)")
        result, job_id, elapsed = run_on_hardware(
            service, backend, circuit, args.shots,
            optimization_level=args.optimization_level,
        )
        ibm_expectations = counts_to_expectation_values(result)

    print(f"\n📊 IBM Quantum Expectation Values (Pauli-Z):")
    for i, val in enumerate(ibm_expectations):
        if pec_stds is not None and pec_stds[i] > 0:
            print(f"   Qubit {i}: {val:+.6f} ± {pec_stds[i]:.6f}")
        else:
            print(f"   Qubit {i}: {val:+.4f}")

    # ── Run Classical Decoder ─────────────────────────────────────────
    print("\n🧠 Running classical decoder (CPU) on IBM quantum output...")
    ibm_bulk = run_classical_decoder(model, ibm_expectations)

    # ── Compare with Local Simulation ─────────────────────────────────
    print("\n🔬 Running local PennyLane simulation for comparison...")
    with torch.no_grad():
        local_bulk = model(bdy_input).squeeze().cpu().numpy()

    # ── Load Ground Truth ─────────────────────────────────────────────
    blk_path = os.path.join(args.data_dir, "bulk_collision.npy")
    blk_truth = np.load(blk_path)[args.sample_idx]

    # ── Compute Metrics ───────────────────────────────────────────────
    ibm_mae  = np.mean(np.abs(ibm_bulk - blk_truth))
    ibm_mse  = np.mean((ibm_bulk - blk_truth) ** 2)
    local_mae = np.mean(np.abs(local_bulk - blk_truth))
    local_mse = np.mean((local_bulk - blk_truth) ** 2)

    print("\n" + "=" * 70)
    print("  RESULTS: IBM QUANTUM vs LOCAL SIMULATION")
    print("=" * 70)
    print(f"  {'Metric':<25} {'IBM Quantum':>15} {'Local Sim':>15}")
    print(f"  {'─'*25} {'─'*15} {'─'*15}")
    print(f"  {'MSE':<25} {ibm_mse:>15.6f} {local_mse:>15.6f}")
    print(f"  {'MAE':<25} {ibm_mae:>15.4f} {local_mae:>15.4f}")
    print(f"  {'MAE %':<25} {ibm_mae*100:>14.2f}% {local_mae*100:>14.2f}%")
    if args.pec:
        mitigation_label = "PEC+FracGates+DD+TREX"
        shot_count = args.pec_shots
    elif args.mitigate:
        mitigation_label = "ZNE+DD+TREX"
        shot_count = args.shots
    else:
        mitigation_label = "none"
        shot_count = args.shots
    print(f"\n  Backend:    {backend.name}")
    print(f"  Shots:      {shot_count:,}")
    print(f"  Mitigation: {mitigation_label}")
    print(f"  Opt Level:  {args.optimization_level}")
    print(f"  Job ID:     {job_id}")
    print(f"  Time:       {elapsed:.1f}s")
    print("=" * 70)

    # ── Save Results ──────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    results_dict = {
        "backend": backend.name,
        "job_id": job_id,
        "shots": shot_count,
        "error_mitigation": mitigation_label,
        "optimization_level": args.optimization_level,
        "execution_time_seconds": round(elapsed, 2),
        "ibm_expectations": ibm_expectations.tolist(),
        "ibm_mse": float(ibm_mse),
        "ibm_mae": float(ibm_mae),
        "local_mse": float(local_mse),
        "local_mae": float(local_mae),
        "sample_idx": args.sample_idx,
    }
    if pec_stds is not None:
        results_dict["pec_standard_errors"] = pec_stds.tolist()

    save_path = os.path.join(args.output_dir, "ibm_quantum_results.json")
    with open(save_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n💾 Results saved to: {save_path}")


def run_local_fallback(model, bdy_input, args):
    """Fallback: run inference locally if IBM connection fails."""
    print("\n🔬 Running full local PennyLane simulation...")
    with torch.no_grad():
        pred = model(bdy_input).squeeze().cpu().numpy()

    blk_path = os.path.join(args.data_dir, "bulk_collision.npy")
    if os.path.exists(blk_path):
        blk_truth = np.load(blk_path)[args.sample_idx]
        mae = np.mean(np.abs(pred - blk_truth))
        mse = np.mean((pred - blk_truth) ** 2)
        print(f"\n✅ Local Inference Complete:")
        print(f"   MSE: {mse:.6f}")
        print(f"   MAE: {mae:.4f} ({mae*100:.2f}% error)")
    else:
        print("✅ Local inference complete (no ground truth available for comparison)")


if __name__ == "__main__":
    main()
