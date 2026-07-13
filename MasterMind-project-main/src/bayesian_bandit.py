"""Bayesian linear bandit: the pure mathematical core of the Ouroboros
policy engine.

Phase 1 of docs/whitepapers/ouroboros-policy-engine.md, sections 2.3-2.5
and 2.9. No event bus, no LLM calls, no I/O of any kind - this module
knows only about vectors and matrices. Genomes are referenced purely by
an opaque `genome_id: str`; this module never imports `genome.py`. That
decoupling is deliberate: everything here is provable against synthetic
data with a known ground truth, independent of whatever the rest of the
system eventually wires it to (see tests/test_bayesian_bandit.py).

The model, per arm g:

    r = xᵗβ_g + ε,   ε ~ N(0, observation_variance)
    β_g | data ~ N(mean, cov)

`update()` is the standard Bayesian-linear-regression sequential update -
which is *identical to* (not merely analogous to) the Kalman filter
measurement-update equation, with x playing the role of the observation
matrix H and observation_variance playing the role of the measurement
noise R. See the whitepaper's section 2.4 for the full argument; this
module is the equations from that section, executable.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

__all__ = ["BayesianLinearBandit", "thompson_select"]


@dataclasses.dataclass(slots=True)
class BayesianLinearBandit:
    """Posterior belief for ONE arm (one genome), over a shared context space.

    Stored in covariance form (not precision/information form): Thompson
    Sampling needs to *draw a sample* from the posterior on every
    decision, and `numpy`'s multivariate normal sampler wants a covariance
    matrix - keeping it already inverted avoids paying for a matrix
    inversion on every single decision, at the cost of a cheap rank-1
    (Sherman-Morrison-style) update in `update()` instead of a full
    re-inversion.
    """

    dimension: int
    prior_variance: float = 1.0
    observation_variance: float = 0.25
    discount: float = 1.0  # 1.0 = stationary (no forgetting); <1.0 = non-stationary

    mean: npt.NDArray[np.float64] = dataclasses.field(init=False)
    cov: npt.NDArray[np.float64] = dataclasses.field(init=False)
    n_observations: int = dataclasses.field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError(f"dimension must be >= 1, got {self.dimension}")
        if self.prior_variance <= 0:
            raise ValueError(f"prior_variance must be > 0, got {self.prior_variance}")
        if self.observation_variance <= 0:
            raise ValueError(f"observation_variance must be > 0, got {self.observation_variance}")
        if not (0.0 < self.discount <= 1.0):
            raise ValueError(f"discount must be in (0, 1], got {self.discount}")
        self.mean = np.zeros(self.dimension, dtype=np.float64)
        self.cov = np.eye(self.dimension, dtype=np.float64) * self.prior_variance

    def update(self, context: npt.ArrayLike, reward: float) -> None:
        """Fuse one new (context, reward) observation into the posterior.

        This is the Kalman filter measurement-update equation. If
        `discount < 1.0`, the covariance is inflated first (exponential
        forgetting) so old evidence never fully "freezes" the belief -
        the mechanism by which this bandit stays responsive to a
        drifting reward function instead of converging and then refusing
        to move (whitepaper 2.5).
        """
        x = self._as_vector(context)

        if self.discount < 1.0:
            # Inflating covariance by 1/discount here is the covariance-form
            # equivalent of multiplying *precision* by `discount` - the
            # standard exponential-forgetting recursive-least-squares device.
            self.cov = self.cov / self.discount

        cov_x = self.cov @ x
        innovation_variance = self.observation_variance + float(x @ cov_x)
        kalman_gain = cov_x / innovation_variance
        residual = float(reward) - float(x @ self.mean)

        self.mean = self.mean + kalman_gain * residual
        self.cov = self.cov - np.outer(kalman_gain, cov_x)
        # Numerical hygiene: repeated rank-1 updates can accumulate enough
        # floating-point asymmetry over thousands of rounds to make the
        # matrix an invalid covariance for the sampler below. Symmetrizing
        # after every update is cheap (O(d^2)) insurance against that.
        self.cov = (self.cov + self.cov.T) * 0.5
        self.n_observations += 1

    def sample_predicted_reward(
        self, context: npt.ArrayLike, *, rng: np.random.Generator | None = None
    ) -> float:
        """Thompson Sampling primitive: draw ONE belief from the current
        posterior (not the mean belief) and predict the reward under that
        sampled belief. Exploration is a side effect of this sampling,
        not a separate mechanism - see `thompson_select` below.
        """
        rng = rng if rng is not None else np.random.default_rng()
        x = self._as_vector(context)
        beta_sample = rng.multivariate_normal(self.mean, self.cov)
        return float(x @ beta_sample)

    def predict_mean(self, context: npt.ArrayLike) -> float:
        """The posterior *mean* prediction (no sampling) - useful for
        inspection/testing and for any future consumer that wants a
        stable point estimate rather than an exploratory sample.
        """
        x = self._as_vector(context)
        return float(x @ self.mean)

    def lower_confidence_bound(self, context: npt.ArrayLike, *, z: float = 1.96) -> float:
        """A pessimistic point estimate (whitepaper 2.9): use this for
        decisions that are expensive or irreversible to undo (e.g.
        retiring a genome from a population). Never use this for action
        selection, which should stay optimistic via Thompson Sampling -
        conflating the two is a real failure mode, not a style choice.
        """
        x = self._as_vector(context)
        mean = float(x @ self.mean)
        variance = float(x @ self.cov @ x)
        return mean - z * float(np.sqrt(max(variance, 0.0)))

    def _as_vector(self, context: npt.ArrayLike) -> npt.NDArray[np.float64]:
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.dimension:
            raise ValueError(f"context has dimension {x.shape[0]}, expected {self.dimension}")
        return x


def thompson_select(
    candidates: Mapping[str, BayesianLinearBandit],
    context: npt.ArrayLike,
    *,
    rng: np.random.Generator | None = None,
) -> str:
    """Thompson Sampling action-selection policy (whitepaper 2.3).

    Draws one sampled reward per candidate arm and returns the genome_id
    of the argmax. This is the entire "intelligence" of the policy - no
    epsilon-greedy, no explicit UCB exploration bonus. Exploration and
    exploitation are both consequences of the same line of code: an arm
    with a wide, uncertain posterior occasionally samples optimistically
    high and wins, purely because its posterior is wide.
    """
    if not candidates:
        raise ValueError("thompson_select requires at least one candidate arm")
    rng = rng if rng is not None else np.random.default_rng()
    sampled = {
        genome_id: bandit.sample_predicted_reward(context, rng=rng)
        for genome_id, bandit in candidates.items()
    }
    return max(sampled, key=lambda genome_id: sampled[genome_id])
