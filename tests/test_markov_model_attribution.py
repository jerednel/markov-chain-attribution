import numpy as np
import pandas as pd
import pytest

import markov_model_attribution as mma


README_PATHS = ['start > cone > ctwo > cthree > conv',
                'start > cone > null',
                'start > ctwo > cthree > null']


def frame(paths, column='Paths'):
    return pd.DataFrame({column: paths})


class TestDocumentedExample:
    """The numbers published in the readme are the contract."""

    def test_markov_conversions(self):
        model = mma.run_model(paths=frame(README_PATHS))
        assert model['markov_conversions'] == pytest.approx(
            {'cone': 0.2, 'ctwo': 0.4, 'cthree': 0.4})

    def test_removal_effects(self):
        model = mma.run_model(paths=frame(README_PATHS))
        assert model['removal_effects'] == pytest.approx(
            {'cone': 0.5, 'ctwo': 1.0, 'cthree': 1.0})

    def test_last_touch_conversions(self):
        model = mma.run_model(paths=frame(README_PATHS))
        assert model['last_touch_conversions'] == {'cone': 0, 'ctwo': 0, 'cthree': 1}

    def test_base_cvr(self):
        model = mma.run_model(paths=frame(README_PATHS))
        assert model['base_cvr'] == pytest.approx(1 / 3)

    def test_returned_keys(self):
        model = mma.run_model(paths=frame(README_PATHS))
        assert set(model) == {'markov_conversions', 'last_touch_conversions',
                              'removal_effects', 'base_cvr', 'transition_matrix',
                              'absorption_matrix'}


class TestTransitionMatrix:
    """Regression tests for the chained-assignment bug (all-NaN on pandas >= 2.0)."""

    def test_matrix_is_populated(self):
        matrix = mma.run_model(paths=frame(README_PATHS))['transition_matrix']
        assert matrix.to_numpy().sum() > 0
        assert not matrix.isna().any().any()

    def test_rows_are_stochastic(self):
        matrix = mma.run_model(paths=frame(README_PATHS))['transition_matrix']
        assert matrix.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(matrix)))

    def test_absorbing_states_are_self_looping(self):
        matrix = mma.run_model(paths=frame(README_PATHS))['transition_matrix']
        assert matrix.loc['conv', 'conv'] == 1.0
        assert matrix.loc['null', 'null'] == 1.0

    def test_known_transition_probabilities(self):
        matrix = mma.run_model(paths=frame(README_PATHS))['transition_matrix']
        # 'start' fans out evenly across its three journeys.
        assert matrix.loc['start', 'cone'] == pytest.approx(2 / 3)
        assert matrix.loc['start', 'ctwo'] == pytest.approx(1 / 3)
        # 'cone' converts onward once and drops out once.
        assert matrix.loc['cone', 'ctwo'] == pytest.approx(0.5)
        assert matrix.loc['cone', 'null'] == pytest.approx(0.5)

    def test_no_nan_results(self):
        model = mma.run_model(paths=frame(README_PATHS))
        assert not any(np.isnan(v) for v in model['markov_conversions'].values())
        assert not any(np.isnan(v) for v in model['removal_effects'].values())


class TestChannelNames:
    """Regression tests for the regex that stripped digits and punctuation."""

    def test_digits_are_preserved(self):
        model = mma.run_model(paths=frame(['start > c1 > c2 > conv',
                                           'start > c1 > null',
                                           'start > c2 > null']))
        assert set(model['markov_conversions']) == {'c1', 'c2'}

    def test_underscores_and_hyphens_are_preserved(self):
        model = mma.run_model(paths=frame(['start > email_promo_2024 > fb-retarget > conv',
                                           'start > email_promo_2024 > null',
                                           'start > fb-retarget > null']))
        assert set(model['markov_conversions']) == {'email_promo_2024', 'fb-retarget'}

    def test_similar_names_are_not_merged(self):
        model = mma.run_model(paths=frame(['start > ch1 > ch2 > conv',
                                           'start > ch1 > null',
                                           'start > ch2 > null']))
        assert len(model['markov_conversions']) == 2

    def test_names_with_spaces_and_slashes(self):
        model = mma.run_model(paths=frame(['start > paid search > google/cpc > conv',
                                           'start > paid search > null',
                                           'start > google/cpc > null']))
        assert set(model['markov_conversions']) == {'paid search', 'google/cpc'}

    def test_irregular_delimiter_spacing(self):
        spaced = mma.run_model(paths=frame(['start > a > b > conv',
                                            'start > a > null',
                                            'start > b > null']))
        tight = mma.run_model(paths=frame(['start>a>b>conv',
                                           'start>a>null',
                                           'start>b>null']))
        assert spaced['markov_conversions'] == pytest.approx(tight['markov_conversions'])


class TestInputIsNotMutated:
    """The caller's DataFrame used to be renamed and overwritten in place."""

    def test_column_name_is_preserved(self):
        df = frame(['start > a_1 > conv', 'start > b_2 > null'], column='my_col')
        mma.run_model(paths=df)
        assert list(df.columns) == ['my_col']

    def test_values_are_preserved(self):
        original = ['start > a_1 > conv', 'start > b_2 > null']
        df = frame(original, column='my_col')
        mma.run_model(paths=df)
        assert df['my_col'].tolist() == original


class TestValidation:
    def test_path_without_terminal_state(self):
        with pytest.raises(ValueError, match="must end with"):
            mma.run_model(paths=frame(['start > a > b > conv', 'start > b']))

    def test_path_without_start(self):
        with pytest.raises(ValueError, match="must begin with"):
            mma.run_model(paths=frame(['a > b > conv', 'start > a > null']))

    def test_null_value_in_column(self):
        with pytest.raises(ValueError, match="expected a path string"):
            mma.run_model(paths=frame(['start > a > conv', None]))

    def test_multi_column_dataframe(self):
        df = pd.DataFrame({'Paths': ['start > a > conv'], 'count': ['start > b > null']})
        with pytest.raises(ValueError, match="single-column"):
            mma.run_model(paths=df)

    def test_non_dataframe_input(self):
        with pytest.raises(TypeError, match="DataFrame or Series"):
            mma.run_model(paths=['start > a > conv'])

    def test_empty_input(self):
        with pytest.raises(ValueError, match="no rows"):
            mma.run_model(paths=frame([]))

    def test_reserved_state_mid_path(self):
        with pytest.raises(ValueError, match="reserved state"):
            mma.run_model(paths=frame(['start > a > conv > b > null']))

    def test_empty_touchpoint(self):
        with pytest.raises(ValueError, match="empty touchpoint"):
            mma.run_model(paths=frame(['start > a >  > conv']))

    def test_no_converting_paths(self):
        with pytest.raises(ValueError, match="No converting paths"):
            mma.run_model(paths=frame(['start > a > null', 'start > b > null']))

    def test_error_message_identifies_the_row(self):
        with pytest.raises(ValueError, match="Row 1"):
            mma.run_model(paths=frame(['start > a > conv', 'start > b']))


class TestEdgeCases:
    def test_all_paths_convert(self):
        """Used to raise KeyError('null') because the state was never created."""
        model = mma.run_model(paths=frame(['start > a > conv', 'start > b > conv']))
        assert set(model['markov_conversions']) == {'a', 'b'}
        assert model['base_cvr'] == pytest.approx(1.0)

    def test_series_input(self):
        series = pd.Series(README_PATHS)
        assert mma.run_model(paths=series)['markov_conversions'] == pytest.approx(
            mma.run_model(paths=frame(README_PATHS))['markov_conversions'])

    def test_repeated_channel_in_path(self):
        model = mma.run_model(paths=frame(['start > a > a > b > conv',
                                           'start > a > null',
                                           'start > b > null']))
        assert set(model['markov_conversions']) == {'a', 'b'}

    def test_direct_conversion(self):
        model = mma.run_model(paths=frame(['start > conv', 'start > a > conv',
                                           'start > a > null']))
        assert set(model['markov_conversions']) == {'a'}

    def test_conversions_are_fully_allocated(self):
        model = mma.run_model(paths=frame(README_PATHS))
        assert sum(model['markov_conversions'].values()) == pytest.approx(1)

    def test_allocation_matches_conversion_count(self):
        paths = ['start > a > b > conv', 'start > b > conv', 'start > a > null',
                 'start > b > a > conv', 'start > a > b > null']
        model = mma.run_model(paths=frame(paths))
        assert sum(model['markov_conversions'].values()) == pytest.approx(3)
        assert sum(model['last_touch_conversions'].values()) == 3

    def test_reserved_states_excluded_from_output(self):
        model = mma.run_model(paths=frame(README_PATHS))
        for key in ('start', 'conv', 'null'):
            assert key not in model['markov_conversions']
            assert key not in model['last_touch_conversions']
            assert key not in model['removal_effects']

    def test_results_are_order_independent(self):
        forward = mma.run_model(paths=frame(README_PATHS))['markov_conversions']
        reverse = mma.run_model(paths=frame(list(reversed(README_PATHS))))['markov_conversions']
        assert forward == pytest.approx(reverse)

    def test_transition_matrix_ordering_is_deterministic(self):
        first = mma.run_model(paths=frame(README_PATHS))['transition_matrix']
        second = mma.run_model(paths=frame(README_PATHS))['transition_matrix']
        assert list(first.index) == list(second.index) == sorted(first.index)


class TestAbsorptionMatrix:
    def test_rows_sum_to_one(self):
        absorption = mma.run_model(paths=frame(README_PATHS))['absorption_matrix']
        assert absorption.sum(axis=1).to_numpy() == pytest.approx(
            np.ones(len(absorption)))

    def test_labelled_by_state(self):
        absorption = mma.run_model(paths=frame(README_PATHS))['absorption_matrix']
        assert list(absorption.columns) == ['null', 'conv']
        assert 'start' in absorption.index
        assert 'conv' not in absorption.index

    def test_base_cvr_matches_absorption(self):
        model = mma.run_model(paths=frame(README_PATHS))
        assert model['base_cvr'] == pytest.approx(
            model['absorption_matrix'].loc['start', 'conv'])


class TestRemovalEffects:
    def test_channel_on_every_converting_path_has_effect_one(self):
        model = mma.run_model(paths=frame(['start > a > b > conv',
                                           'start > a > null',
                                           'start > b > null']))
        assert model['removal_effects']['b'] == pytest.approx(1.0)

    def test_effects_are_bounded(self):
        paths = ['start > a > b > conv', 'start > b > c > conv', 'start > a > null',
                 'start > c > a > conv', 'start > b > null', 'start > c > null']
        effects = mma.run_model(paths=frame(paths))['removal_effects']
        assert all(0 <= v <= 1 for v in effects.values())

    def test_calculate_removals_is_reusable(self):
        model = mma.run_model(paths=frame(README_PATHS))
        recomputed = mma.calculate_removals(model['transition_matrix'],
                                            model['base_cvr'])
        assert recomputed == pytest.approx(model['removal_effects'])
