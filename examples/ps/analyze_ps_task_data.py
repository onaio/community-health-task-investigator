import os
import pandas
import dask
import dask.dataframe

from chti.abnormal_task_perf_analyzer import AbnormalTaskPerfAnalyzer

def ps_task_2020_jan_report(
    hist_filename=os.path.join(os.path.dirname(__file__), "tasks_2019.csv"),
    latest_filename=os.path.join(os.path.dirname(__file__), "tasks_2020_jan.csv"),
    results_dir=os.path.join(os.path.dirname(__file__), "2020_jan_report")):
    
    task_ddf = dask.dataframe.read_csv([hist_filename, latest_filename])
    task_df = pandas.DataFrame(task_ddf.compute())
    task_df.set_index('id', inplace=True)

    # Do the analysis
    analysis = AbnormalTaskPerfAnalyzer(
        task_df,
        ["2019-01-01", "2019-12-31"],
        ["2020-01-01", "2020-01-31"],
        start_dt_col="date",
        worker_group_cols=["worker"],
        task_group_cols=["activity"],
        elapsed_time_col="secs",
        with_figures=True)
        
    result_df = analysis.analyze()
    result_df.sort_values(["logpdf_bucket", "total_surprise"], inplace=True, ascending=[True, False])

    pandas.set_option("display.max_rows", 1000, "display.max_columns", 1000)
    print(result_df)

    os.makedirs(results_dir, exist_ok=True)

    result_df.to_csv(os.path.join(results_dir, f"results_2020_jan.csv"))

    for name, figures in analysis.figures.items():
        for fig_type, figure in dict(figures._asdict()).items():
            figure.savefig(os.path.join(results_dir, f"{name}_{fig_type}.png"), format="png")


def ps_task_2019_dec_report(
    hist_filename=os.path.join(os.path.dirname(__file__), "tasks_2019.csv"),
    results_dir=os.path.join(os.path.dirname(__file__), "2019_dec_report")):
    
    task_ddf = dask.dataframe.read_csv([hist_filename])
    task_df = pandas.DataFrame(task_ddf.compute())
    task_df.set_index('id', inplace=True)

    pandas.set_option("display.max_rows", 1000, "display.max_columns", 1000)
    print(task_df)

    # Do the analysis
    analysis = AbnormalTaskPerfAnalyzer(
        task_df,
        ["2019-01-01", "2019-11-30"],
        ["2019-12-01", "2019-12-31"],
        start_dt_col="date",
        worker_group_cols=["worker"],
        task_group_cols=["activity"],
        elapsed_time_col="secs",
        with_figures=True)
        
    result_df = analysis.analyze()
    result_df.sort_values(["logpdf_bucket", "total_surprise"], inplace=True, ascending=[True, False])

    pandas.set_option("display.max_rows", 1000, "display.max_columns", 1000)
    print(result_df)

    os.makedirs(results_dir, exist_ok=True)
    
    result_df.to_csv(os.path.join(results_dir, f"results_2019_dec.csv"))

    for name, figures in analysis.figures.items():
        for fig_type, figure in dict(figures._asdict()).items():
            figure.savefig(os.path.join(results_dir, f"{name}_{fig_type}.png"), format="png")


def main():
    ps_task_2019_dec_report()
    ps_task_2020_jan_report()

if __name__ == "__main__":
    main()