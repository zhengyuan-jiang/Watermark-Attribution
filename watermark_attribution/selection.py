"""Watermark selection methods used in the paper."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


BitArray = NDArray[np.uint8]


def _validate_size(num_users: int, n: int) -> None:
    if isinstance(num_users, bool) or not isinstance(num_users, int) or num_users <= 0:
        raise ValueError("num_users must be a positive integer")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if num_users > 1 << n:
        raise ValueError("num_users exceeds the number of unique n-bit watermarks")


def _random_watermark(rng: np.random.Generator, n: int) -> BitArray:
    return rng.integers(0, 2, size=n, dtype=np.uint8)


def _match_counts(existing: BitArray, candidate: BitArray) -> NDArray[np.int64]:
    return np.count_nonzero(existing == candidate, axis=1)


def _starting_threshold(pool: BitArray, n: int) -> int:
    if pool.shape[0] < 2:
        return n // 2
    return int(np.max(_match_counts(pool[:-1], pool[-1])))


def _bounded_search_tree(
    existing: BitArray,
    candidate: BitArray,
    depth: int,
    max_matches: int,
    rng: np.random.Generator,
) -> BitArray | None:
    """Run BSTA for one candidate (Algorithm 1 in the paper)."""
    if depth < 0:
        return None

    matches = _match_counts(existing, candidate)
    closest_index = int(np.argmax(matches))
    closest_matches = int(matches[closest_index])

    if closest_matches > max_matches + depth:
        return None
    if closest_matches <= max_matches:
        return candidate.copy()

    matching_positions = np.flatnonzero(candidate == existing[closest_index])
    branch_positions = rng.permutation(matching_positions)[: max_matches + 1]
    for position in branch_positions:
        next_candidate = candidate.copy()
        next_candidate[position] ^= np.uint8(1)
        result = _bounded_search_tree(
            existing,
            next_candidate,
            depth - 1,
            max_matches,
            rng,
        )
        if result is not None:
            return result
    return None


def _non_redundant_guess(
    existing: BitArray,
    candidate: BitArray,
    max_matches: int,
    rng: np.random.Generator,
) -> BitArray | None:
    """Run NRG for one candidate (Algorithm 2 in the paper)."""
    candidate = candidate.copy()
    forbidden = np.zeros(candidate.shape[0], dtype=bool)
    remaining_budget = max_matches

    while remaining_budget > 0:
        matches = _match_counts(existing, candidate)
        closest_index = int(np.argmax(matches))
        closest_matches = int(matches[closest_index])

        if closest_matches > 2 * max_matches:
            return None
        if closest_matches <= max_matches:
            return candidate.copy()

        flips_needed = closest_matches - max_matches
        available = np.flatnonzero(
            (candidate == existing[closest_index]) & ~forbidden
        )
        if available.size < flips_needed:
            return None
        selected = rng.choice(available, size=flips_needed, replace=False)
        candidate[selected] ^= np.uint8(1)
        forbidden[selected] = True
        remaining_budget -= flips_needed

    return None


def generate_random(
    num_users: int,
    n: int = 64,
    seed: int = 0,
) -> BitArray:
    """Select unique watermarks uniformly at random."""
    _validate_size(num_users, n)
    rng = np.random.default_rng(seed)
    selected: list[BitArray] = []
    seen: set[bytes] = set()

    while len(selected) < num_users:
        candidate = _random_watermark(rng, n)
        key = candidate.tobytes()
        if key not in seen:
            selected.append(candidate)
            seen.add(key)
    return np.stack(selected)


def _generate_sequential(
    num_users: int,
    n: int,
    seed: int,
    candidate_factory: Callable[[BitArray, np.random.Generator, int], BitArray],
    search: Callable[
        [BitArray, BitArray, int, np.random.Generator],
        BitArray | None,
    ],
) -> BitArray:
    _validate_size(num_users, n)
    rng = np.random.default_rng(seed)
    pool = _random_watermark(rng, n).reshape(1, n)

    while pool.shape[0] < num_users:
        max_matches = _starting_threshold(pool, n)
        selected: BitArray | None = None
        while selected is None and max_matches < n:
            candidate = candidate_factory(pool, rng, n)
            selected = search(pool, candidate, max_matches, rng)
            if selected is None:
                max_matches += 1

        if selected is None:
            raise RuntimeError("failed to find another unique watermark")
        pool = np.vstack((pool, selected))

    return pool.astype(np.uint8, copy=False)


def generate_nrg(
    num_users: int,
    n: int = 64,
    seed: int = 0,
) -> BitArray:
    """Select watermarks with Non-Redundant Guess (NRG)."""

    def complement_first(
        pool: BitArray,
        _rng: np.random.Generator,
        _n: int,
    ) -> BitArray:
        return np.uint8(1) - pool[0]

    def search(
        pool: BitArray,
        candidate: BitArray,
        max_matches: int,
        rng: np.random.Generator,
    ) -> BitArray | None:
        return _non_redundant_guess(pool, candidate, max_matches, rng)

    return _generate_sequential(
        num_users,
        n,
        seed,
        complement_first,
        search,
    )


def generate_absta(
    num_users: int,
    n: int = 64,
    depth: int = 8,
    seed: int = 0,
) -> BitArray:
    """Select watermarks with the paper's approximate BSTA (A-BSTA)."""
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError("depth must be a non-negative integer")

    def random_candidate(
        _pool: BitArray,
        rng: np.random.Generator,
        length: int,
    ) -> BitArray:
        return _random_watermark(rng, length)

    def search(
        pool: BitArray,
        candidate: BitArray,
        max_matches: int,
        rng: np.random.Generator,
    ) -> BitArray | None:
        return _bounded_search_tree(
            pool,
            candidate,
            depth,
            max_matches,
            rng,
        )

    return _generate_sequential(
        num_users,
        n,
        seed,
        random_candidate,
        search,
    )


def select_watermarks(
    method: str,
    num_users: int,
    n: int = 64,
    seed: int = 0,
    depth: int = 8,
) -> BitArray:
    """Select a watermark pool using ``random``, ``nrg``, or ``absta``."""
    normalized = method.lower().replace("-", "")
    if normalized == "random":
        return generate_random(num_users=num_users, n=n, seed=seed)
    if normalized == "nrg":
        return generate_nrg(num_users=num_users, n=n, seed=seed)
    if normalized == "absta":
        return generate_absta(
            num_users=num_users,
            n=n,
            depth=depth,
            seed=seed,
        )
    raise ValueError("method must be one of: random, nrg, absta")
