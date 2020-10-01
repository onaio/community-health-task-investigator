# Principal Surprise Synthetic Examples

This module contains code to generate sample data for analysis:

```bash
$ PYTHONPATH=. poetry run python examples/ps/gen_ps_task_data.py
```

After running the above, there should be two files created:

* `tasks_2019.csv` - generated task data for 6 different workers doing 9 different tasks in 2019, where the worker task times are pulled from a `lognorm` random distribution per-task
* `tasks_2020_jan.csv` - generated task data for the same workers and tasks in 2020, but with some abnormal task completions added for the PS algorithm to identify (the male named workers `bob`, `dan`, `felix` have abnormal task completions added)

## Reports

After generating the sample data, the PS algorithm can be used to generate two reports - one for `2019_dec` and one for `2020_jan`:

```bash
$ PYTHONPATH=. poetry run python examples/ps/analyze_ps_task_data.py
```

For each report the worker abnormality ordering (along with all metrics) is output (as a `.csv` and at the command line) as well as, for each worker:

* a `bullseye` chart where the distance from center indicates how (relatively) abnormal the workers' performance was on particular task types
* a set of`histogram` charts directly comparing the workers' task completion times to the historical models

The colors on the two charts match, so by first viewing the `bullseye` chart you can get a sense of which `histograms` are most interesting.

The `2019_dec` report has no abnormal task completions added, so the output field worker ordering is just due to random noise in the data set (overall `dist_from_mean` varies relatively little).  The `2020_jan` report is ordered by most-abnormal to least abnormal, with the male names `felix`, `dan`, and `bob` at the top of the list.