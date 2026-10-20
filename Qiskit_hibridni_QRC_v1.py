#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Qiskit QRC - Quantum Reservoir Computing

Hibridni Quantum Reservoir Computing za Loto 7/39.

Najnoviji rad koji sam uspeo da potvrdim za ovaj zadatak objavljen je 4. oktobra 2026:
An enhanced quantum reservoir computing framework for hydrological time series modeling — Liu Zhen i Alina Bărbulescu.

Sačuvana osnova najnovijeg rada:
- sedam četvorokubitnih kvantnih rezervoara;
- prenošenje kvantnog stanja kroz celu istoriju;
- 18 kvantnih opservabli po rezervoaru;
- zajednički, L2-regularizovan linearni izlaz.

Distribucioni izlaz:
- nema uslovljavanja stvarnim prethodnim brojem ciljne kombinacije;
- nema presecanja CDF-a i prebacivanja mase na njegove granice;
- sedam izlaznih potencijala definišu JEDNU raspodelu:

    P(b1,...,b7 | H) =
        proizvod_r w_r(br | H) / Z(H)

    za 1 <= b1 < ... < b7 <= 39.

- Z obuhvata sve legalne kombinacije;
- učenje minimizuje kvadratnu grešku CDF-ova marginala
  te zajedničke raspodele;
- L2 kazna regularizuje koeficijente linearnog izlaza;
- tačan maksimum modela nalazi dinamičko programiranje.

linearni izlaz se sada optimizuje kroz zajedničku raspodelu 

Nema slučajne inicijalizacije, uzorkovanja, klasifikatora,
nametnutog oblika raspodele ili estetskih filtera kombinacija.
"""


import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import itertools
import re
import sys
from pathlib import Path

try:
    import numpy as np
    import scipy
    from scipy.optimize import minimize
    import torch
    import torch.nn.functional as F
    import qiskit
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, Pauli, Statevector
except (ImportError, OSError) as exc:
    raise SystemExit(
        "Zavisnosti se ne mogu učitati.\n"
        "Instalacija:\n"
        "python3 -m pip install numpy scipy torch qiskit\n"
        f"Detalj: {exc}"
    )

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

DEFAULT_FILES = (
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83_loto_2974.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83_loto_plus_1730.csv",
)

BALLS = 39
PICKS = 7
QUBITS = 4
LAYERS = 3

OUTPUT_SIZE = PICKS * BALLS
BATCH_SIZE = 192

MAX_ITERATIONS = 60
PATIENCE = 8

REGULARIZATIONS = (1e-4, 1e-3, 1e-2, 1e-1)

REAL = torch.float64

support = np.zeros((PICKS, BALLS), dtype=np.float64)

for rank in range(PICKS):
    # Isključivo podrška koja sledi iz pravila rastućeg 7/39.
    support[rank, rank:BALLS - PICKS + rank + 1] = 1.0

SUPPORT = torch.tensor(support, dtype=REAL)[None, :, :]


def read_csv(path):
    path = Path(path).expanduser()

    if not path.is_file():
        raise ValueError(f"CSV ne postoji: {path}")

    records = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()

            if not line:
                continue

            fields = re.split(r"[,;\s]+", line)

            if len(fields) != PICKS:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    f"očekivano 7 polja, pronađeno {len(fields)}."
                )

            numbers = []

            for field in fields:
                token = field.strip().strip('"').strip("'")

                if not re.fullmatch(r"[+-]?\d+", token):
                    raise ValueError(
                        f"{path.name}, red {line_number}: "
                        f"{field!r} nije ceo broj."
                    )

                numbers.append(int(token))

            if any(n < 1 or n > BALLS for n in numbers):
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "broj izvan raspona 1–39."
                )

            if len(set(numbers)) != PICKS:
                raise ValueError(
                    f"{path.name}, red {line_number}: ponovljen broj."
                )

            records.append(sorted(numbers))

    if not records:
        raise ValueError(f"Prazan CSV: {path}")

    return np.asarray(records, dtype=np.int64)


def format_numbers(numbers):
    return ", ".join(str(int(n)) for n in numbers)


def longest_run(numbers):
    best = current = 1

    for previous, number in zip(numbers[:-1], numbers[1:]):
        current = current + 1 if number == previous + 1 else 1
        best = max(best, current)

    return best


class QuantumReservoir:
    def __init__(self):
        reservoir = QuantumCircuit(QUBITS)

        for _ in range(LAYERS):
            for qubit in range(QUBITS):
                reservoir.rx(np.pi / 4.0, qubit)

            for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
                reservoir.cz(first, second)

            for qubit in range(QUBITS):
                reservoir.ry(np.pi / 6.0, qubit)

        transitions = []

        for number in range(1, BALLS + 1):
            value = (number - 1.0) / (BALLS - 1.0)
            circuit = QuantumCircuit(QUBITS)

            for qubit in range(QUBITS):
                circuit.ry(np.pi * value, qubit)
                circuit.rz(np.pi * value / 2.0, qubit)

            circuit.compose(reservoir, inplace=True)

            matrix = np.asarray(
                Operator(circuit).data,
                dtype=np.complex128,
            )

            if not np.allclose(
                matrix.conj().T @ matrix,
                np.eye(2 ** QUBITS),
                atol=1e-12,
                rtol=1e-12,
            ):
                raise ArithmeticError(
                    "Matrica kvantnog kola nije unitarna."
                )

            transitions.append(matrix)

        self.transitions = np.stack(transitions)
        self.observables = self.make_observables()

    @staticmethod
    def make_observables():
        matrices = []

        for symbol in ("X", "Y", "Z"):
            for qubit in range(QUBITS):
                label = ["I"] * QUBITS
                label[QUBITS - 1 - qubit] = symbol
                matrices.append(Pauli("".join(label)).to_matrix())

        for first in range(QUBITS):
            for second in range(first + 1, QUBITS):
                label = ["I"] * QUBITS
                label[QUBITS - 1 - first] = "Z"
                label[QUBITS - 1 - second] = "Z"
                matrices.append(Pauli("".join(label)).to_matrix())

        return np.asarray(matrices, dtype=np.complex128)

    def transform(self, draws):
        initial = np.asarray(
            Statevector.from_label("0000").data,
            dtype=np.complex128,
        )

        # Reset postoji samo na početku obrade svakog CSV-a.
        states = np.tile(initial, (PICKS, 1))
        feature_count = PICKS * len(self.observables) + PICKS

        features = np.empty(
            (len(draws), feature_count),
            dtype=np.float64,
        )

        for index, draw in enumerate(draws):
            operators = self.transitions[draw - 1]

            states = np.einsum(
                "bij,bj->bi",
                operators,
                states,
                optimize=True,
            )

            norms = np.sum(np.abs(states) ** 2, axis=1)

            if (
                not np.all(np.isfinite(norms))
                or not np.allclose(
                    norms, 1.0, atol=1e-9, rtol=1e-9
                )
            ):
                raise ArithmeticError(
                    f"Neispravna norma stanja u kolu {index + 1}."
                )

            # Samo korekcija numeričke greške norme.
            states = states / np.sqrt(norms)[:, None]

            expectations = np.einsum(
                "bi,kij,bj->bk",
                states.conj(),
                self.observables,
                states,
                optimize=True,
            )

            if np.max(np.abs(expectations.imag)) > 1e-9:
                raise ArithmeticError(
                    "Očekivana vrednost opservable nije realna."
                )

            direct_input = (draw - 1.0) / (BALLS - 1.0)

            features[index] = np.concatenate(
                (expectations.real.reshape(-1), direct_input)
            )

            if (index + 1) % 1000 == 0:
                print(
                    f"  Kvantna obrada: {index + 1}/{len(draws)}",
                    flush=True,
                )

        if not np.all(np.isfinite(features)):
            raise ArithmeticError(
                "Kvantne karakteristike nisu konačne."
            )

        return features


def normalize_vector(values):
    normalizer = values.sum(dim=-1, keepdim=True)

    if (
        not bool(torch.isfinite(normalizer).all())
        or bool((normalizer <= 0.0).any())
    ):
        raise ArithmeticError(
            "Zajednička raspodela nema pozitivnu konačnu masu."
        )

    return values / normalizer, normalizer


def exclusive_prefix_sum(values):
    cumulative = torch.cumsum(values, dim=-1)

    return torch.cat(
        (
            torch.zeros_like(cumulative[:, :1]),
            cumulative[:, :-1],
        ),
        dim=-1,
    )


def exclusive_suffix_sum(values):
    cumulative = torch.flip(
        torch.cumsum(torch.flip(values, dims=(-1,)), dim=-1),
        dims=(-1,),
    )

    return torch.cat(
        (
            cumulative[:, 1:],
            torch.zeros_like(cumulative[:, :1]),
        ),
        dim=-1,
    )


def joint_marginals(weights):
    """
    Tačno sabiranje svih rastućih putanja.

    Numeričko skaliranje međurezultata je zajedničko za ceo
    uzorak, a ne zasebno za prethodno izabrane brojeve.
    """
    positions = weights.shape[1]

    forward = []

    current, scale = normalize_vector(weights[:, 0, :])
    forward.append(current)
    log_partition = torch.log(scale[:, 0])

    for rank in range(1, positions):
        current = (
            weights[:, rank, :] * exclusive_prefix_sum(current)
        )
        current, scale = normalize_vector(current)

        forward.append(current)
        log_partition = log_partition + torch.log(scale[:, 0])

    backward = [None] * positions
    backward[-1] = torch.ones_like(weights[:, -1, :])

    for rank in range(positions - 2, -1, -1):
        current = exclusive_suffix_sum(
            weights[:, rank + 1, :] * backward[rank + 1]
        )
        current, _ = normalize_vector(current)
        backward[rank] = current

    marginals = []

    for rank in range(positions):
        marginal, _ = normalize_vector(
            forward[rank] * backward[rank]
        )
        marginals.append(marginal)

    return torch.stack(marginals, dim=1), log_partition


def maximum_path(weights):
    """Tačan maksimum proizvoda težina pod pravilom strogog rasta."""
    weights = np.asarray(weights, dtype=np.float64)
    positions, numbers = weights.shape

    logarithms = np.full_like(weights, -np.inf)
    np.log(weights, out=logarithms, where=weights > 0.0)

    scores = logarithms[0].copy()
    predecessors = np.full(
        (positions, numbers), -1, dtype=np.int64
    )

    for rank in range(1, positions):
        updated = np.full(numbers, -np.inf)
        best_score = -np.inf
        best_previous = -1

        for number in range(numbers):
            if number > 0 and scores[number - 1] > best_score:
                best_score = scores[number - 1]
                best_previous = number - 1

            if (
                np.isfinite(best_score)
                and np.isfinite(logarithms[rank, number])
            ):
                updated[number] = (
                    best_score + logarithms[rank, number]
                )
                predecessors[rank, number] = best_previous

        scores = updated

    last = int(np.argmax(scores))

    if not np.isfinite(scores[last]):
        raise ArithmeticError("Nema legalne putanje modela.")

    result = [last]

    for rank in range(positions - 1, 0, -1):
        last = int(predecessors[rank, last])

        if last < 0:
            raise ArithmeticError(
                "Neispravna rekonstrukcija maksimuma."
            )

        result.append(last)

    return tuple(reversed(result))


def verify_distribution():
    """Poređenje DP-a sa potpunim nabrajanjem malog prostora."""
    positions, numbers = 3, 8

    index = np.arange(
        1, positions * numbers + 1, dtype=np.float64
    ).reshape(positions, numbers)

    weights = 0.2 + np.cos(index * np.sqrt(2.0)) ** 2

    partition = 0.0
    reference = np.zeros_like(weights)
    best_path = None
    best_weight = -1.0

    for combination in itertools.combinations(
        range(numbers), positions
    ):
        weight = float(
            np.prod(
                [
                    weights[rank, number]
                    for rank, number in enumerate(combination)
                ]
            )
        )

        partition += weight

        for rank, number in enumerate(combination):
            reference[rank, number] += weight

        if weight > best_weight:
            best_weight = weight
            best_path = combination

    reference /= partition

    with torch.no_grad():
        marginals, log_partition = joint_marginals(
            torch.tensor(weights[None, :, :], dtype=REAL)
        )

    if not np.allclose(
        marginals[0].numpy(),
        reference,
        atol=1e-12,
        rtol=1e-11,
    ):
        raise ArithmeticError(
            "Provera zajedničkih marginala nije prošla."
        )

    if not np.isclose(
        log_partition[0].item(),
        np.log(partition),
        atol=1e-12,
        rtol=1e-11,
    ):
        raise ArithmeticError(
            "Provera zajedničke normalizacije nije prošla."
        )

    if maximum_path(weights) != best_path:
        raise ArithmeticError("Provera maksimuma nije prošla.")


class EarlyStop(Exception):
    pass


class JointQRCReadout:
    """
    L2-regularizovan linearni izlaz sa pozitivnom veznom funkcijom.

    Nema skrivenih neuronskih slojeva.
    Softplus obezbeđuje pozitivne potencijale; ne nameće
    raspodelu brojeva.

    Koeficijenti se uče kroz marginalne CDF-ove jedne zajedničke
    raspodele, a ne kroz sedam odvojenih uslovnih CDF-ova.
    """

    def __init__(self, context, outcomes, train_stop):
        training = context[:train_stop]

        self.mean = training.mean(axis=0)
        scale = training.std(axis=0)
        self.active = scale > 1e-10

        if not np.any(self.active):
            raise ValueError(
                "Nema promenljivih regresionih karakteristika."
            )

        self.scale = scale[self.active]
        self.design = torch.from_numpy(self.transform(context))

        self.shape = (self.design.shape[1], OUTPUT_SIZE)
        self.parameter_count = self.shape[0] * self.shape[1]

        targets = (
            outcomes[:, :, None]
            <= np.arange(1, BALLS)[None, None, :]
        ).astype(np.float64)

        self.targets = torch.from_numpy(targets)
        self.parameters = None

    def transform(self, context):
        context = np.asarray(context, dtype=np.float64)

        if context.ndim == 1:
            context = context[None, :]

        normalized = (
            context[:, self.active] - self.mean[self.active]
        ) / self.scale

        # Poslednja kolona je intercept.
        design = np.column_stack(
            (normalized, np.ones(len(normalized)))
        )

        if not np.all(np.isfinite(design)):
            raise ArithmeticError(
                "Nekonačne standardizovane karakteristike."
            )

        return np.ascontiguousarray(design, dtype=np.float64)

    def initial_parameters(self):
        row = np.arange(1, self.shape[0] + 1)[:, None]
        column = np.arange(1, OUTPUT_SIZE + 1)[None, :]

        # Analitička, mala, nejednaka inicijalizacija.
        coefficients = (
            0.01
            * np.sin(
                np.sqrt(2.0) * row * column
                + 0.13 * row
                + 0.07 * column
            )
            / np.sqrt(self.shape[0])
        )

        return coefficients.reshape(-1)

    def potentials(self, design, coefficients):
        linear_output = design @ coefficients

        potentials = F.softplus(linear_output).reshape(
            -1, PICKS, BALLS
        )

        return potentials * SUPPORT

    def objective(
        self,
        flat_parameters,
        start,
        stop,
        regularization=0.0,
        gradient=False,
    ):
        count = stop - start

        if count <= 0:
            raise ValueError("Prazan skup za učenje ili proveru.")

        parameters = torch.tensor(
            np.asarray(flat_parameters),
            dtype=REAL,
            requires_grad=gradient,
        )

        total = 0.0

        with torch.set_grad_enabled(gradient):
            for first in range(start, stop, BATCH_SIZE):
                last = min(first + BATCH_SIZE, stop)
                fraction = (last - first) / count

                coefficients = parameters.reshape(self.shape)

                weights = self.potentials(
                    self.design[first:last], coefficients
                )

                marginals, _ = joint_marginals(weights)

                cumulative = torch.cumsum(
                    marginals, dim=-1
                )[:, :, :BALLS - 1]

                loss = (
                    (
                        cumulative - self.targets[first:last]
                    ).square().mean()
                    * fraction
                )

                if not bool(torch.isfinite(loss)):
                    raise ArithmeticError(
                        "Nekonačna distribuciona regresiona greška."
                    )

                total += float(loss.detach())

                if gradient:
                    loss.backward()

            if regularization > 0.0:
                coefficients = parameters.reshape(self.shape)

                # L2 kazna po izlazu; intercept ima blažu kaznu.
                penalty = regularization * (
                    coefficients[:-1].square().sum()
                    + 0.01 * coefficients[-1].square().sum()
                ) / OUTPUT_SIZE

                total += float(penalty.detach())

                if gradient:
                    penalty.backward()

        if gradient:
            if parameters.grad is None:
                raise ArithmeticError("Nedostaje gradijent.")

            derivative = parameters.grad.detach().numpy().copy()

            if not np.all(np.isfinite(derivative)):
                raise ArithmeticError("Nekonačan gradijent.")

            return total, derivative

        return total

    def verify_gradient(self):
        parameters = self.initial_parameters()
        stop = min(2, len(self.targets))
        regularization = 1e-3

        _, analytic = self.objective(
            parameters,
            0,
            stop,
            regularization=regularization,
            gradient=True,
        )

        indices = (
            0,
            BALLS - 1,
            self.parameter_count // 2,
            self.parameter_count - 1,
        )

        epsilon = 1e-6

        for index in indices:
            plus = parameters.copy()
            minus = parameters.copy()

            plus[index] += epsilon
            minus[index] -= epsilon

            numerical = (
                self.objective(
                    plus, 0, stop, regularization
                )
                - self.objective(
                    minus, 0, stop, regularization
                )
            ) / (2.0 * epsilon)

            if not np.isclose(
                analytic[index],
                numerical,
                atol=2e-7,
                rtol=3e-5,
            ):
                raise ArithmeticError(
                    f"Provera gradijenta nije prošla: {index}."
                )

    def fit(
        self,
        train_stop,
        regularization,
        iterations,
        validation=None,
    ):
        initial = self.initial_parameters()

        state = {
            "current": 0,
            "best_loss": np.inf,
            "best_parameters": None,
            "best_iteration": 0,
            "stale": 0,
        }

        def objective(parameters):
            return self.objective(
                parameters,
                0,
                train_stop,
                regularization=regularization,
                gradient=True,
            )

        def callback(parameters):
            state["current"] += 1
            iteration = state["current"]

            if validation is not None:
                start, stop = validation

                # Validaciona greška bez regularizacione kazne.
                error = self.objective(
                    parameters, start, stop
                )

                if error < state["best_loss"] - 1e-10:
                    state["best_loss"] = error
                    state["best_parameters"] = parameters.copy()
                    state["best_iteration"] = iteration
                    state["stale"] = 0
                else:
                    state["stale"] += 1

                if iteration == 1 or iteration % 10 == 0:
                    print(
                        f"    Iteracija {iteration}/{iterations}: "
                        f"validaciona CDF-MSE={error:.8f}",
                        flush=True,
                    )

                if state["stale"] >= PATIENCE:
                    raise EarlyStop()

            elif iteration == 1 or iteration % 10 == 0:
                print(
                    f"    Iteracija {iteration}/{iterations}",
                    flush=True,
                )

        try:
            result = minimize(
                objective,
                initial,
                method="L-BFGS-B",
                jac=True,
                callback=callback,
                options={
                    "maxiter": int(iterations),
                    "maxls": 30,
                    "maxcor": 10,
                    "ftol": 1e-12,
                    "gtol": 1e-9,
                },
            )
        except EarlyStop:
            if validation is None:
                raise

            if state["best_parameters"] is None:
                raise ArithmeticError(
                    "Nema prihvaćenih parametara modela."
                )

            self.parameters = state["best_parameters"]

            return (
                state["best_iteration"],
                state["best_loss"],
            )

        if result.status not in (0, 1):
            raise ArithmeticError(
                f"Optimizacija nije završena uredno: {result.message}"
            )

        if not np.all(np.isfinite(result.x)):
            raise ArithmeticError(
                "Neispravni regresioni koeficijenti."
            )

        if validation is not None:
            if state["best_parameters"] is None:
                raise ArithmeticError(
                    "Optimizator nije napravio korak učenja."
                )

            self.parameters = state["best_parameters"]

            return (
                state["best_iteration"],
                state["best_loss"],
            )

        self.parameters = result.x.copy()

        print(
            f"    Prihvaćenih koraka: {int(result.nit)}",
            flush=True,
        )

        return int(result.nit), float(result.fun)

    def mse(self, start, stop):
        if self.parameters is None:
            raise ValueError("Model nije obučen.")

        return self.objective(self.parameters, start, stop)

    def predict(self, context):
        if self.parameters is None:
            raise ValueError("Model nije obučen.")

        design = torch.from_numpy(self.transform(context))
        coefficients = torch.tensor(
            self.parameters.reshape(self.shape),
            dtype=REAL,
        )

        with torch.no_grad():
            weights = self.potentials(design, coefficients)
            marginals, log_partition = joint_marginals(weights)

        weights = weights[0].numpy()
        indices = maximum_path(weights)
        combination = tuple(number + 1 for number in indices)

        if (
            len(combination) != PICKS
            or len(set(combination)) != PICKS
            or tuple(sorted(combination)) != combination
            or combination[0] < 1
            or combination[-1] > BALLS
        ):
            raise ArithmeticError("Nelegalna NEXT kombinacija.")

        log_mass = float(
            sum(
                np.log(weights[rank, number])
                for rank, number in enumerate(indices)
            )
            - log_partition[0].item()
        )

        return combination, marginals[0].numpy(), log_mass


def process_file(path, reservoir, validation_size, test_size):
    draws = read_csv(path)

    transitions = len(draws) - 1
    train_stop = transitions - validation_size - test_size
    test_start = transitions - test_size

    if train_stop < 128:
        raise ValueError(
            f"{Path(path).name}: potrebno najmanje "
            f"{129 + validation_size + test_size} kola."
        )

    fingerprint = hashlib.sha256(
        np.asarray(draws, dtype="<i8").tobytes()
    ).hexdigest()

    print("\n" + "=" * 70)
    print(Path(path).name)
    print(f"Kola: {len(draws)}")
    print(f"SHA256: {fingerprint}")
    print(
        "Kontrola: prethodno izvučeno kolo: "
        + format_numbers(draws[-1])
    )
    print(
        f"Prelazi: trening {train_stop}, "
        f"validacija {validation_size}, test {test_size}",
        flush=True,
    )

    features = reservoir.transform(draws)
    context = features[:-1]
    outcomes = draws[1:]

    # Standardizacija se u ovoj fazi uči samo na trening prefiksu.
    selection = JointQRCReadout(
        context, outcomes, train_stop
    )
    selection.verify_gradient()

    print("  Provera gradijenta prošla.", flush=True)

    best_choice = None

    for regularization in REGULARIZATIONS:
        print(
            f"  Validacija: lambda={regularization:g}",
            flush=True,
        )

        iterations, error = selection.fit(
            train_stop,
            regularization,
            MAX_ITERATIONS,
            validation=(train_stop, test_start),
        )

        print(
            f"    Najbolja CDF-MSE={error:.8f}; "
            f"iteracija={iterations}",
            flush=True,
        )

        choice = (error, regularization, iterations)

        if best_choice is None or error < best_choice[0]:
            best_choice = choice

    _, regularization, iterations = best_choice

    print(
        f"Izabrano: lambda={regularization:g}, "
        f"iteracija={iterations}",
        flush=True,
    )

    del selection

    # Novo učenje na treningu + validaciji.
    # Test nije korišćen za izbor regularizacije ili trajanja.
    print("  Učenje za nezavisnu test proveru:", flush=True)

    test_model = JointQRCReadout(
        context, outcomes, test_start
    )
    test_model.fit(
        test_start, regularization, iterations
    )

    test_error = test_model.mse(test_start, transitions)

    hit_counts = np.zeros(PICKS + 1, dtype=np.int64)
    previous_hits = []

    for index in range(test_start, transitions):
        combination, _, _ = test_model.predict(context[index])
        actual = set(int(n) for n in outcomes[index])

        hits = len(set(combination) & actual)
        hit_counts[hits] += 1

        previous_hits.append(
            len(set(int(n) for n in draws[index]) & actual)
        )

    average_hits = float(
        np.dot(np.arange(PICKS + 1), hit_counts) / test_size
    )

    print(f"Test: zajednička marginalna CDF-MSE={test_error:.8f}")
    print(f"Test: prosečan broj pogodaka={average_hits:.4f}")
    print(
        "Test: raspodela pogodaka: "
        + ", ".join(
            f"{hits} → {int(value)}"
            for hits, value in enumerate(hit_counts)
        )
    )
    print(
        "Dijagnostika, ponavljanje prethodnog kola: "
        f"prosečno {np.mean(previous_hits):.4f} pogodaka"
    )

    del test_model

    # Završni model i standardizacija koriste sve istorijske prelaze.
    print(
        f"  Završno učenje: svih {transitions} prelaza",
        flush=True,
    )

    final_model = JointQRCReadout(
        context, outcomes, transitions
    )
    final_model.fit(
        transitions, regularization, iterations
    )

    combination, _, log_mass = final_model.predict(
        features[-1]
    )

    print("NEXT: " + format_numbers(combination))
    print(
        "Log-masa NEXT unutar naučene raspodele: "
        f"{log_mass:.8f}"
    )
    print(
        f"Dijagnostika: najduži uzastopni niz={longest_run(combination)}; "
        f"brojeva 1–19={sum(n <= 19 for n in combination)}; "
        f"brojeva 20–39={sum(n >= 20 for n in combination)}"
    )

    repetitions = int(
        np.sum(np.all(draws == np.asarray(combination), axis=1))
    )
    print(f"Ista kombinacija u istoriji: {repetitions}")

    return str(Path(path).expanduser()), combination, fingerprint


def parse_args():
    parser = argparse.ArgumentParser(
        description="QRC — regresija zajedničke raspodele Loto 7/39."
    )

    parser.add_argument(
        "csv",
        nargs="*",
        help="Tačno tri CSV putanje; inače DEFAULT_FILES.",
    )
    parser.add_argument("--validation", type=int, default=64)
    parser.add_argument("--test", type=int, default=64)

    args = parser.parse_args()

    if args.csv and len(args.csv) != 3:
        parser.error("Navedi tačno tri CSV putanje.")

    if args.validation < 16 or args.test < 16:
        parser.error(
            "Validacija i test moraju imati najmanje 16 kola."
        )

    return args


def main():
    args = parse_args()
    files = tuple(args.csv) if args.csv else DEFAULT_FILES

    print("QISKIT QRC — ZAJEDNIČKA RASPODELA LOTO 7/39")
    print(
        f"Qiskit {qiskit.__version__}; "
        f"NumPy {np.__version__}; "
        f"SciPy {scipy.__version__}; "
        f"PyTorch {torch.__version__}"
    )
    print(
        "Sedam četvorokubitnih rezervoara; "
        "regularizovani linearni distribucioni izlaz.",
        flush=True,
    )

    verify_distribution()

    print(
        "Provera zajedničke normalizacije, marginala "
        "i tačnog maksimuma prošla.",
        flush=True,
    )

    reservoir = QuantumReservoir()
    results = []
    failed = False

    for path in files:
        try:
            results.append(
                process_file(
                    path,
                    reservoir,
                    args.validation,
                    args.test,
                )
            )
        except Exception as exc:
            failed = True
            print(
                f"\nGREŠKA — {Path(path).name}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    if results:
        print("\n" + "=" * 70)
        print("ZAVRŠNE NEXT KOMBINACIJE")

        for path, combination, _ in results:
            print(
                f"{Path(path).name}: {format_numbers(combination)}"
            )

        for first in range(len(results)):
            for second in range(first + 1, len(results)):
                path_a, next_a, hash_a = results[first]
                path_b, next_b, hash_b = results[second]

                if hash_a == hash_b:
                    print(
                        "Dijagnostika: isti numerički sadržaj CSV-a: "
                        f"{Path(path_a).name} / {Path(path_b).name}"
                    )
                elif next_a == next_b:
                    print(
                        "Dijagnostika: različiti CSV sadržaji imaju "
                        "isti maksimum naučenog modela: "
                        f"{Path(path_a).name} / {Path(path_b).name}"
                    )

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())



"""
QISKIT QRC — ZAJEDNIČKA RASPODELA LOTO 7/39
Qiskit 1.4.4; NumPy 1.26.4; SciPy 1.17.1; PyTorch 2.12.0
Sedam četvorokubitnih rezervoara; regularizovani linearni distribucioni izlaz.
Provera zajedničke normalizacije, marginala i tačnog maksimuma prošla.


======================================================================
loto7_4704_k83.csv
Kola: 4704
SHA256: 1badcf2632a9f9669e5a46a612df4e4513ab4bb0c3d488b7d746e25d458ab22c
Kontrola: prethodno izvučeno kolo: 7, 10, 13, 27, 28, 31, 38
Prelazi: trening 4575, validacija 64, test 64
  Kvantna obrada: 1000/4704
  Kvantna obrada: 2000/4704
  Kvantna obrada: 3000/4704
  Kvantna obrada: 4000/4704
  Provera gradijenta prošla.
  Validacija: lambda=0.0001
    Iteracija 1/60: validaciona CDF-MSE=0.07364855
    Iteracija 10/60: validaciona CDF-MSE=0.07336044
    Najbolja CDF-MSE=0.07271265; iteracija=8
  Validacija: lambda=0.001
    Iteracija 1/60: validaciona CDF-MSE=0.07364827
    Iteracija 10/60: validaciona CDF-MSE=0.07332791
    Najbolja CDF-MSE=0.07271681; iteracija=8
  Validacija: lambda=0.01
    Iteracija 1/60: validaciona CDF-MSE=0.07364551
    Iteracija 10/60: validaciona CDF-MSE=0.07312603
    Najbolja CDF-MSE=0.07276104; iteracija=8
  Validacija: lambda=0.1
    Iteracija 1/60: validaciona CDF-MSE=0.07362048
    Iteracija 10/60: validaciona CDF-MSE=0.07308750
    Najbolja CDF-MSE=0.07298425; iteracija=9
Izabrano: lambda=0.0001, iteracija=8
  Učenje za nezavisnu test proveru:
    Iteracija 1/8
    Prihvaćenih koraka: 8
Test: zajednička marginalna CDF-MSE=0.07640386
Test: prosečan broj pogodaka=1.3906
Test: raspodela pogodaka: 0 → 10, 1 → 28, 2 → 18, 3 → 7, 4 → 1, 5 → 0, 6 → 0, 7 → 0
Dijagnostika, ponavljanje prethodnog kola: prosečno 1.1406 pogodaka
  Završno učenje: svih 4703 prelaza
    Iteracija 1/8
    Prihvaćenih koraka: 8
NEXT: 3, 8, 18, 22, 24, 33, 34
Log-masa NEXT unutar naučene raspodele: -15.52991157
Dijagnostika: najduži uzastopni niz=2; brojeva 1–19=3; brojeva 20–39=4
Ista kombinacija u istoriji: 0



======================================================================
loto7_4704_k83_loto_2974.csv
Kola: 2974
SHA256: 992bc400ed1af4f4d5ce2395fcf52f92b086402a37ae6aff96c153c96e4a33e5
Kontrola: prethodno izvučeno kolo: 9, 14, 17, 19, 24, 28, 35
Prelazi: trening 2845, validacija 64, test 64
  Kvantna obrada: 1000/2974
  Kvantna obrada: 2000/2974
  Provera gradijenta prošla.
  Validacija: lambda=0.0001
    Iteracija 1/60: validaciona CDF-MSE=0.07326552
    Iteracija 10/60: validaciona CDF-MSE=0.07490185
    Najbolja CDF-MSE=0.07310020; iteracija=4
  Validacija: lambda=0.001
    Iteracija 1/60: validaciona CDF-MSE=0.07326480
    Iteracija 10/60: validaciona CDF-MSE=0.07478304
    Najbolja CDF-MSE=0.07309800; iteracija=4
  Validacija: lambda=0.01
    Iteracija 1/60: validaciona CDF-MSE=0.07325770
    Iteracija 10/60: validaciona CDF-MSE=0.07395679
    Najbolja CDF-MSE=0.07307762; iteracija=4
  Validacija: lambda=0.1
    Iteracija 1/60: validaciona CDF-MSE=0.07319248
    Iteracija 10/60: validaciona CDF-MSE=0.07317157
    Najbolja CDF-MSE=0.07296903; iteracija=4
Izabrano: lambda=0.1, iteracija=4
  Učenje za nezavisnu test proveru:
    Iteracija 1/4
    Prihvaćenih koraka: 4
Test: zajednička marginalna CDF-MSE=0.07644547
Test: prosečan broj pogodaka=1.0938
Test: raspodela pogodaka: 0 → 18, 1 → 26, 2 → 17, 3 → 2, 4 → 1, 5 → 0, 6 → 0, 7 → 0
Dijagnostika, ponavljanje prethodnog kola: prosečno 1.3125 pogodaka
  Završno učenje: svih 2973 prelaza
    Iteracija 1/4
    Prihvaćenih koraka: 4
NEXT: 1, 5, 7, 12, 34, 37, 39
Log-masa NEXT unutar naučene raspodele: -15.71090462
Dijagnostika: najduži uzastopni niz=1; brojeva 1–19=4; brojeva 20–39=3
Ista kombinacija u istoriji: 0



======================================================================
loto7_4704_k83_loto_plus_1730.csv
Kola: 1730
SHA256: fc57041695a80b3c4e88c747acabf06e40b83a4914e21669bef89bcf469eef52
Kontrola: prethodno izvučeno kolo: 7, 10, 13, 27, 28, 31, 38
Prelazi: trening 1601, validacija 64, test 64
  Kvantna obrada: 1000/1730
  Provera gradijenta prošla.
  Validacija: lambda=0.0001
    Iteracija 1/60: validaciona CDF-MSE=0.07755194
    Najbolja CDF-MSE=0.07755194; iteracija=1
  Validacija: lambda=0.001
    Iteracija 1/60: validaciona CDF-MSE=0.07755104
    Najbolja CDF-MSE=0.07755104; iteracija=1
  Validacija: lambda=0.01
    Iteracija 1/60: validaciona CDF-MSE=0.07754209
    Najbolja CDF-MSE=0.07754209; iteracija=1
  Validacija: lambda=0.1
    Iteracija 1/60: validaciona CDF-MSE=0.07746075
    Najbolja CDF-MSE=0.07746075; iteracija=1
Izabrano: lambda=0.1, iteracija=1
  Učenje za nezavisnu test proveru:
    Iteracija 1/1
    Prihvaćenih koraka: 1
Test: zajednička marginalna CDF-MSE=0.07532594
Test: prosečan broj pogodaka=1.1562
Test: raspodela pogodaka: 0 → 14, 1 → 29, 2 → 18, 3 → 3, 4 → 0, 5 → 0, 6 → 0, 7 → 0
Dijagnostika, ponavljanje prethodnog kola: prosečno 1.2812 pogodaka
  Završno učenje: svih 1729 prelaza
    Iteracija 1/1
    Prihvaćenih koraka: 1
NEXT: 1, 4, 6, 29, 32, 36, 38
Log-masa NEXT unutar naučene raspodele: -16.37781416
Dijagnostika: najduži uzastopni niz=1; brojeva 1–19=3; brojeva 20–39=4
Ista kombinacija u istoriji: 0


======================================================================
ZAVRŠNE NEXT KOMBINACIJE
loto7_4704_k83.csv: 3, 8, 18, 22, 24, 33, 34
loto7_4704_k83_loto_2974.csv: 1, 5, 7, 12, 34, 37, 39
loto7_4704_k83_loto_plus_1730.csv: 1, 4, 6, 29, 32, 36, 38
"""



"""
Hibridni QRC: isti kvantni rezervoar i klasični linearni izlaz sa L2 regularizacijom. 
Izlaz se uči kroz zajedničku normalizaciju cele kombinacije.
"""



"""
Najnoviji rad koji sam uspeo da potvrdim za ovaj zadatak objavljen je 4. oktobra 2026:
An enhanced quantum reservoir computing framework for hydrological time series modeling — Liu Zhen i Alina Bărbulescu.

Rad koristi Qiskit, kvantni rezervoar i regresiju za predikciju sledećeg koraka vremenske serije. 
Ovo je hibridna kvantna regresija: kvantno kolo obrađuje istoriju, a izlazna ridge regresija uči predikciju. 

Njegova korisna ideja je prenošenje kvantnog stanja kroz istoriju. 
Sistem sa četiri kubita zadržava stanje između koraka. 
Fiksne rotacije i CZ sprezanje obrađuju ulaze; očekivane vrednosti kvantnih opservabli postaju ulazi regresora. 
Autori koriste tačnu simulaciju Statevector, bez šuma od konačnog broja merenja. 

Prilagođavanje:
- Sačuvati kvantni rezervoar, prenošenje stanja i regresioni izlaz.
- Sedam brojeva svakog kola obraditi uz očuvanje granica između kola.
- Regresijom učiti raspodelu naredne kombinacije, umesto zaokruživanja sedam prosečnih vrednosti.
- Za svaki od tri CSV-a napraviti zaseban model; hronološki izabrati regularizaciju, zatim završno obučiti na celoj istoriji.
- Koristiti fiksno određene uglove i tačne kvantne proračune, bez slučajne inicijalizacije, generisanja kombinacija i naknadnog nametanja grupa ili razmaka.


Zadržava kvantni rezervoar i prenošenje stanja iz rada. 
Adaptacija koristi sedam rezervoara i zajedničku regresiju uslovnih raspodela. 
Završnu kombinaciju bira tačno dinamičko programiranje prema tom modelu.
"""
