import pandas
import numpy
import random
import io
import base64
from perf_tools.principal_surprise import *

from scipy.stats import lognorm


def gen_rows(row_templates, k=1, weights=None, random_seed=1):

    rng = random.Random(random_seed)
    numpy.random.seed(random_seed)

    def gen_value(values):
        if hasattr(values, "rvs"):
            return values.rvs(1)[0]
        else:
            return rng.choices(values)[0]

    def gen_row(row_template):
        return list(map(gen_value, row_template))

    return [
        gen_row(row_template) for row_template in rng.choices(row_templates, k=k, weights=weights)
    ]


def test_basic():

    #
    # Generate a history of tasks that take different lengths of time
    #

    row_templates = [
        [["2019-01-01"], ["alice", "bob", "carl"], ["bike"], lognorm(s=0.5, loc=1, scale=1000)],
        [["2019-01-01"], ["alice", "bob", "carl"], ["drive"], lognorm(s=0.5, loc=1, scale=100)],
        [["2019-01-01"], ["alice", "bob", "carl"], ["fly"], lognorm(s=0.5, loc=1, scale=10)],
    ]

    # Make there be a lot more bike data than drive data than fly data
    row_weights = [16, 4, 1]

    # Add one normal and one abnormal task completion per person, where the abnormal task is more
    # abnormal for biking than driving than flying
    test_rows = [
        ["2019-02-01", "alice", "bike", 1500.0],
        ["2019-02-01", "bob", "drive", 150.0],
        ["2019-02-01", "carl", "fly", 15.0],
        ["2019-02-01", "alice", "drive", 1.0],
        ["2019-02-01", "bob", "fly", 1.0],
        ["2019-02-01", "carl", "bike", 1.0],
    ]

    task_df = pandas.DataFrame(
        gen_rows(row_templates, k=1000, weights=row_weights, random_seed=99) + test_rows,
        columns=["date", "worker", "activity", "secs"],
    )

    # Do the analysis
    analysis = AbnormalTaskPerfAnalyzer(
        task_df,
        ["2019-01-01", "2019-01-31"],
        ["2019-02-01", "2019-02-28"],
        start_dt_col="date",
        worker_group_cols=["worker"],
        task_group_cols=["activity"],
        elapsed_time_col="secs",
        max_model_err_pct=1.0,
    )

    result_df = analysis.analyze()

    assert len(analysis.historical_task_df) > len(analysis.test_task_df)
    assert len(analysis.task_group_models) == 3

    bike_m = analysis.task_group_models[0]
    drive_m = analysis.task_group_models[1]
    fly_m = analysis.task_group_models[2]

    assert bike_m.mean() < 10000
    assert drive_m.mean() < 1000
    assert fly_m.mean() < 100

    # We have the most bike data, second most drive data, and least fly data
    assert bike_m.expected_err_pct < drive_m.expected_err_pct
    assert drive_m.expected_err_pct < fly_m.expected_err_pct

    surprise_df = analysis.surprise_df
    print(surprise_df)

    # The expected error above controls how much surprise we get from our test data point
    assert (
        surprise_df.loc["alice"][(bike_m.group_def, "mean")]
        > surprise_df.loc["bob"][(drive_m.group_def, "mean")]
    )
    assert (
        surprise_df.loc["bob"][(drive_m.group_def, "mean")]
        > surprise_df.loc["carl"][(fly_m.group_def,"mean")]
    )

    # Make sure that the 1.0 data point gets weirder as the models have higher means and more data
    assert (
        surprise_df.loc["bob"][(fly_m.group_def, "mean")]
        < surprise_df.loc["alice"][(drive_m.group_def, "mean")]
    )
    assert (
        surprise_df.loc["alice"][(drive_m.group_def, "mean")]
        < surprise_df.loc["carl"][(bike_m.group_def, "mean")]
    )

    assert numpy.isnan(surprise_df.loc["alice"][(fly_m.group_def, "mean")])
    assert numpy.isnan(surprise_df.loc["bob"][(bike_m.group_def, "mean")])
    assert numpy.isnan(surprise_df.loc["carl"][(drive_m.group_def, "mean")])

    metrics_df = analysis.metrics_df
    print(metrics_df)

    normal_model = analysis.normal_model

    # Since we have only three workers, we'll always have equal abnormality based on normal
    # distribution so the ordering will only depend on total surprise.  The max total surprise is
    # carl's bike performance and the min total surprise is bob's fly performance
    assert result_df.index[0] == "bob"
    assert result_df.index[-1] == "carl"


def test_minimal():

    #
    # Test what happens if we only have a small number of workers, we should default to
    # max surprise vs abnormality
    #

    row_templates = [
        [["2019-01-01"], ["carl"], ["bike"], lognorm(s=0.5, loc=1, scale=1000)],
    ]

    # We've got one data point per person, not enough to give a normal distribution
    test_rows = [["2019-02-01", "alice", "bike", 1500.0], ["2019-02-01", "bob", "bike", 1.0]]

    task_df = pandas.DataFrame(
        gen_rows(row_templates, k=1000, weights=[1], random_seed=99) + test_rows,
        columns=["date", "worker", "activity", "secs"],
    )

    analysis = AbnormalTaskPerfAnalyzer(
        task_df,
        ["2019-01-01", "2019-01-31"],
        ["2019-02-01", "2019-02-28"],
        start_dt_col="date",
        worker_group_cols=["worker"],
        task_group_cols=["activity"],
        elapsed_time_col="secs",
        max_model_err_pct=1.0,
    )

    result_df = analysis.analyze()

    assert len(analysis.historical_task_df) > len(analysis.test_task_df)
    assert len(analysis.task_group_models) == 1

    # The most abnormal is bob who was way off the normal task time
    assert result_df.index[0] == "alice"
    assert result_df.index[-1] == "bob"

    assert result_df['dist_from_mean'].iloc[0] > 0.7 and result_df['dist_from_mean'].iloc[0] < 0.8

def test_figures():

    #
    # Make sure we can generate figures
    #

    row_templates = [
        [["2019-01-01"], ["alice", "bob", "carl"], ["bike"], lognorm(s=0.5, loc=1, scale=1000)],
        [["2019-01-01"], ["alice", "bob", "carl"], ["drive"], lognorm(s=0.5, loc=1, scale=100)],
        [["2019-01-01"], ["alice", "bob", "carl"], ["fly"], lognorm(s=0.5, loc=1, scale=10)],
    ]

    test_rows = [
        ["2019-02-01", "alice", "bike", 1500.0],
        ["2019-02-01", "bob", "drive", 150.0],
        ["2019-02-01", "carl", "fly", 15.0],
        # ["2019-02-01", "alice", "drive", 1.0],
        # ["2019-02-01", "bob", "fly", 1.0],
        # ["2019-02-01", "carl", "bike", 1.0],
    ]

    task_df = pandas.DataFrame(
        gen_rows(row_templates, k=1000, weights=[1, 1, 1], random_seed=99) + test_rows,
        columns=["date", "worker", "activity", "secs"],
    )

    analysis = AbnormalTaskPerfAnalyzer(
        task_df,
        ["2019-01-01", "2019-01-31"],
        ["2019-02-01", "2019-02-28"],
        start_dt_col="date",
        worker_group_cols=["worker"],
        task_group_cols=["activity"],
        elapsed_time_col="secs",
        max_model_err_pct=1.0,
        with_figures=True,
    )

    result_df = analysis.analyze()

    assert len(result_df.figures) == 3

    for key, figures in sorted(list((k, v) for k, v in result_df.figures.items())):
        assert key == "alice"

        buf = io.BytesIO()
        figures.bullseye.savefig(buf, format="pdf")

        buf.seek(0)
        b64 = base64.standard_b64encode(buf.getvalue())
        assert len(b64) > 1000

        break
