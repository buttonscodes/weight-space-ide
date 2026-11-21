"""The sampler of the automatic search: Heretic's TPE sampler, built exactly as Heretic builds it, on an in-memory study.

This is the only module that imports Optuna, and only when a ``Sampler`` is made: listing, reading and the
Pareto front of a search need no Optuna. Heretic (64e9991, ``main.py``) creates

    TPESampler(n_startup_trials=..., n_ei_candidates=128, multivariate=True, seed=...)

on a study named "heretic" whose directions are the scorers' optimizations. The search keeps its trials as its
own records (wsp.directions_search) and restores the sampler's state by replay: a fresh study, then for each
finished trial in order ``ask`` with Heretic's suggest calls, check that the draws equal the recorded ones bit for
bit, and ``tell`` the recorded outcome. Both of the sampler's random generators (its own and its random
sampler's, each ``numpy.random.RandomState(seed)``) then stand where they stood after the last recorded trial,
so a stopped and resumed search draws what an uninterrupted one draws.

Only public Optuna calls are used: ``create_study``, ``Study.ask``, ``Study.tell``, ``Trial.suggest_categorical``,
``suggest_float`` and ``suggest_int`` (Heretic's calls), and ``distributions.json_to_distribution``.
"""
from __future__ import annotations

import importlib.metadata
import json
import re
import warnings

OPTUNA_RANGE = ((5, 0), (6, 0))        # accepted: >=5.0,<6 (Heretic's optuna~=5.0)
VALIDATED = "5.0.0"                    # the version the draws were checked against Heretic's journals with
SAMPLER = {"class": "TPESampler", "n_ei_candidates": 128, "multivariate": True, "study_name": "heretic"}
REQUIREMENTS = "requirements-search.txt"


def version() -> str | None:
    """The installed Optuna's version, read from its metadata (Optuna is not imported), or None."""
    try:
        return importlib.metadata.version("optuna")
    except importlib.metadata.PackageNotFoundError:
        return None


def _major_minor(v: str) -> tuple[int, int] | None:
    m = re.match(r"(\d+)\.(\d+)", v)
    return (int(m[1]), int(m[2])) if m else None


def require() -> str:
    """The installed Optuna's version when it is one the search accepts (5.x), else SearchError (C26)."""
    from .directions_search import SearchError
    v = version()
    if v is None:
        raise SearchError("the automatic search needs Optuna 5 (the sampler Heretic uses), which is not installed here: "
                          f"python -m pip install -r {REQUIREMENTS}")
    mm = _major_minor(v)
    if mm is None or not OPTUNA_RANGE[0] <= mm < OPTUNA_RANGE[1]:
        raise SearchError(f"the automatic search needs Optuna 5 (Heretic's optuna~=5.0); this environment has Optuna {v}")
    return v


def versions() -> dict:
    import numpy
    return {"optuna": version(), "numpy": numpy.__version__, "validated": VALIDATED}


class ReplayMismatch(Exception):
    """A recorded trial's draws are not what the sampler draws now (``index``: the trial's 1-based index)."""

    def __init__(self, index, recorded, now):
        super().__init__(f"trial {index}: recorded {recorded!r}, drawn now {now!r}")
        self.index, self.recorded, self.now = index, recorded, now


class Sampler:
    """Heretic's sampler on an in-memory study. ``directions``: "minimize" / "maximize" per objective."""

    def __init__(self, *, seed, n_startup_trials, directions):
        import optuna
        from optuna.exceptions import ExperimentalWarning
        from optuna.samplers import TPESampler
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._optuna = optuna
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            sampler = TPESampler(n_startup_trials=n_startup_trials, n_ei_candidates=SAMPLER["n_ei_candidates"],
                                 multivariate=SAMPLER["multivariate"], seed=seed)
            self.study = optuna.create_study(sampler=sampler, directions=list(directions),
                                             study_name=SAMPLER["study_name"])
        self._warning = ExperimentalWarning

    def ask(self, space) -> tuple[object, dict]:
        """One trial: Heretic's suggest call per row of ``space`` (in order) -> (the trial, its draws)."""
        from optuna.distributions import json_to_distribution
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", self._warning)
            trial = self.study.ask()
            for row in space:
                name, dist = row["name"], row["distribution"]
                a = dist["attributes"]
                if dist["name"] == "CategoricalDistribution":
                    trial.suggest_categorical(name, list(a["choices"]))
                elif dist["name"] == "FloatDistribution":
                    trial.suggest_float(name, a["low"], a["high"], log=a["log"])
                elif dist["name"] == "IntDistribution":
                    trial.suggest_int(name, a["low"], a["high"])
                else:
                    raise ValueError(f"{name}: unknown distribution {dist['name']!r}")
                if trial.distributions[name] != json_to_distribution(json.dumps(dist)):
                    raise ValueError(f"{name}: the sampler's distribution {trial.distributions[name]!r} is not the "
                                     f"recorded {dist!r}")
        return trial, dict(trial.params)

    def tell(self, trial, *, values=None, failed=False) -> None:
        """COMPLETE with the objective values, or FAIL (a trial that raised)."""
        if failed:
            self.study.tell(trial, state=self._optuna.trial.TrialState.FAIL)
        else:
            self.study.tell(trial, list(values))

    def replay(self, records, space) -> int:
        """Ask each finished record's trial again, check its draws (==), tell its outcome -> how many were replayed."""
        n = 0
        for rec in records:
            trial, draws = self.ask(space)
            if draws != rec["draws"]:
                raise ReplayMismatch(rec["index"], rec["draws"], draws)
            if rec["state"] == "complete":
                self.tell(trial, values=rec["values"])
            elif rec["state"] == "failed":
                self.tell(trial, failed=True)
            else:
                raise ValueError(f"trial {rec['index']} is {rec['state']}; only finished trials are replayed")
            n += 1
        return n

    @staticmethod
    def versions() -> dict:
        return versions()
