import os

import pandas
from scipy.stats import lognorm

from tests.test_abnormal_task_perf_analyzer import gen_rows

WORKERS = ["alice", "bob", "carla", "dan", "ella", "felix"]

TASK_DEFS = [
    ("register person", 10, 16),
    ("register group", 100, 4),
    ("register community", 1000, 1),
    ("survey person", 20, 16),
    ("survey group", 200, 4),
    ("survey community", 2000, 1),
    ("vacc person", 30, 16),
    ("vacc group", 300, 4),
    ("vacc community", 3000, 1),
]


def _gen_distributed_ps_task_df(dates, num_rows, random_seed, extra_rows=[]):

    """
    Generate a history of tasks that take different lengths of time
    """

    row_templates = [
        [dates, WORKERS, [task_type], lognorm(s=0.5, loc=1, scale=est_task_time)]
        for task_type, est_task_time, _ in TASK_DEFS
    ]

    # Make there be a lot more bike data than drive data than fly data
    row_weights = [task_weight for _, _, task_weight in TASK_DEFS]

    task_df = pandas.DataFrame(
        gen_rows(row_templates, k=num_rows, weights=row_weights, random_seed=random_seed)
        + extra_rows,
        columns=["date", "worker", "activity", "secs"],
    )

    task_df.sort_values(["date"], inplace=True)
    task_df.reset_index(drop=True, inplace=True)

    return task_df


def gen_ps_task_2019_data(
    filename=os.path.join(os.path.dirname(__file__), "tasks_2019.csv"),
    num_rows=10000,
    random_seed=1,
):

    task_df = _gen_distributed_ps_task_df(
        ["2019-%02d-01" % month for month in range(1, 12 + 1)], num_rows, random_seed
    )

    task_df.to_csv(filename, index_label="id")


def gen_ps_task_2020_jan_data(
    filename=os.path.join(os.path.dirname(__file__), "tasks_2020_jan.csv"),
    num_rows=1000,
    random_seed=1,
):

    # Add abnormal task completions
    abnormal_rows = (
        # Very abnormal
        (
            [
                ["2020-01-02", "felix", "register_person", 100],
                ["2020-01-03", "felix", "register person", 100],
                ["2020-01-04", "felix", "register person", 100],
            ]
            * 10
        )
        +
        # Moderately abnormal
        (
            [
                ["2020-01-05", "dan", "survey group", 1000],
                ["2020-01-06", "dan", "survey group", 1000],
                ["2020-01-07", "dan", "survey group", 1000],
            ]
            * 10
        )
        +
        # Somewhat abnormal
        (
            [
                ["2020-01-08", "bob", "vacc community", 10000],
                ["2020-01-09", "bob", "vacc community", 10000],
                ["2020-01-10", "bob", "vacc community", 10000],
            ]
            * 10
        )
    )

    task_df = _gen_distributed_ps_task_df(["2020-01-01"], num_rows, random_seed, abnormal_rows)
    task_df.to_csv(filename, index_label="id")


def main():
    gen_ps_task_2019_data()
    gen_ps_task_2020_jan_data()


if __name__ == "__main__":
    main()
