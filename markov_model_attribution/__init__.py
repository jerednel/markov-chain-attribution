"""First-order Markov chain attribution modeling.

Reallocates conversions across marketing channels using the removal-effect
approach described in Anderl, Becker, Wangenheim & Schumann, "Mapping the
Customer Journey: A Graph-Based Framework for Online Attribution Modeling".
"""

import numpy as np
import pandas as pd

__all__ = ['run_model', 'first_order', 'calculate_removals']

START = 'start'
CONV = 'conv'
NULL = 'null'

_ABSORBING = [NULL, CONV]
_RESERVED = (START, CONV, NULL)
_DELIMITER = '>'


def run_model(paths):
    """Run a first-order Markov attribution model over a column of paths.

    Args:
        paths: A single-column pandas DataFrame (or a Series) of journey
            strings. Each path must begin with 'start', end with 'conv' or
            'null', and delimit touchpoints with ' > '.

    Returns:
        A dict with keys 'markov_conversions', 'last_touch_conversions',
        'removal_effects', 'base_cvr', 'transition_matrix' and
        'absorption_matrix'.
    """
    return first_order(paths)


def first_order(paths):
    """Build the transition matrix and attribute conversions to each channel.

    Accepts the same input as :func:`run_model`. The caller's DataFrame is
    never modified.
    """
    journeys = _parse_paths(paths)

    states = {state for journey in journeys for state in journey}
    # Both absorbing states must exist even if the sample never reaches one.
    states.update(_ABSORBING)
    ordered_states = sorted(states)

    # Last-touch conversion counts, for comparison against the Markov result.
    conv_dict = {state: 0 for state in states}
    total_conversions = 0
    for journey in journeys:
        if journey[-1] == CONV:
            total_conversions += 1
            conv_dict[journey[-2]] += 1

    if total_conversions == 0:
        raise ValueError(
            'No converting paths found. At least one path must end with '
            '{0!r} to allocate conversions.'.format(CONV)
        )

    transition_counts = {}
    outgoing_counts = {}
    for journey in journeys:
        for source, target in zip(journey, journey[1:]):
            transition_counts[(source, target)] = transition_counts.get((source, target), 0) + 1
            outgoing_counts[source] = outgoing_counts.get(source, 0) + 1

    transition_matrix = pd.DataFrame(0.0, index=ordered_states, columns=ordered_states)
    for (source, target), count in transition_counts.items():
        transition_matrix.loc[source, target] = count / outgoing_counts[source]
    for state in _ABSORBING:
        transition_matrix.loc[state, state] = 1.0

    absorption = _absorption_matrix(transition_matrix)
    base_cvr = float(absorption.loc[START, CONV])

    removal_effects = calculate_removals(transition_matrix, base_cvr)

    denominator = float(np.sum(list(removal_effects.values())))
    if denominator == 0:
        raise ValueError(
            'All removal effects are zero, so conversions cannot be allocated. '
            'This usually means no channel influences conversion in this sample.'
        )
    markov_conversions = {
        channel: (effect / denominator) * total_conversions
        for channel, effect in removal_effects.items()
    }

    for state in _RESERVED:
        conv_dict.pop(state, None)

    return {'markov_conversions': markov_conversions,
            'last_touch_conversions': conv_dict,
            'removal_effects': removal_effects,
            'base_cvr': base_cvr,
            'transition_matrix': transition_matrix,
            'absorption_matrix': absorption,
            }


def calculate_removals(df, base_cvr):
    """Return each channel's removal effect given a transition matrix."""
    removal_effects = {}
    channels = [state for state in df.columns if state not in _RESERVED]
    for channel in channels:
        removal_df = df.drop(channel, axis=1).drop(channel, axis=0)

        # Probability mass that used to flow into the removed channel is
        # redirected to the null state so each row stays stochastic.
        leaked = (1.0 - removal_df.sum(axis=1)).clip(lower=0.0)
        removal_df[NULL] = removal_df[NULL] + leaked
        for state in _ABSORBING:
            removal_df.loc[state, state] = 1.0

        removal_cvr = float(_absorption_matrix(removal_df).loc[START, CONV])
        removal_effects[channel] = 1 - removal_cvr / base_cvr
    return removal_effects


def _absorption_matrix(matrix):
    """Absorption probabilities for the transient states of ``matrix``."""
    transient = [state for state in matrix.index if state not in _ABSORBING]
    R = matrix.loc[transient, _ABSORBING].to_numpy(dtype=float)
    Q = matrix.loc[transient, transient].to_numpy(dtype=float)
    try:
        M = np.linalg.solve(np.identity(len(transient)) - Q, R)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            'The transition matrix is singular and cannot be solved. This '
            'usually means a group of channels forms a closed loop that never '
            'reaches {0!r} or {1!r}.'.format(CONV, NULL)
        ) from exc
    return pd.DataFrame(M, index=transient, columns=_ABSORBING)


def _parse_paths(paths):
    """Validate the input and split it into a list of touchpoint lists."""
    if isinstance(paths, pd.Series):
        column = paths
    elif isinstance(paths, pd.DataFrame):
        if paths.shape[1] != 1:
            raise ValueError(
                'Expected a single-column DataFrame of paths but got {0} columns '
                '({1}). Select the path column first, e.g. df[["Paths"]].'.format(
                    paths.shape[1], list(paths.columns))
            )
        column = paths.iloc[:, 0]
    else:
        raise TypeError(
            'paths must be a pandas DataFrame or Series, got {0}.'.format(
                type(paths).__name__)
        )

    if len(column) == 0:
        raise ValueError('paths contains no rows.')

    journeys = []
    for label, value in column.items():
        if not isinstance(value, str):
            raise ValueError(
                'Row {0!r}: expected a path string, got {1!r}.'.format(label, value))

        touchpoints = [part.strip() for part in value.split(_DELIMITER)]
        if any(not part for part in touchpoints):
            raise ValueError(
                'Row {0!r}: path {1!r} contains an empty touchpoint.'.format(label, value))
        if len(touchpoints) < 2:
            raise ValueError(
                'Row {0!r}: path {1!r} has no transitions.'.format(label, value))
        if touchpoints[0] != START:
            raise ValueError(
                'Row {0!r}: path must begin with {1!r} but begins with {2!r}.'.format(
                    label, START, touchpoints[0]))
        if touchpoints[-1] not in (CONV, NULL):
            raise ValueError(
                'Row {0!r}: path must end with {1!r} or {2!r} but ends with {3!r}.'.format(
                    label, CONV, NULL, touchpoints[-1]))
        for part in touchpoints[1:-1]:
            if part in _RESERVED:
                raise ValueError(
                    'Row {0!r}: reserved state {1!r} may only appear at the start or '
                    'end of a path.'.format(label, part))

        journeys.append(touchpoints)
    return journeys
