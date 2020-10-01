import math
from collections import namedtuple

import colorhash
import dask
import dask.dataframe
import numpy
import pandas
from matplotlib import pyplot
from scipy.stats import kstest, lognorm, multivariate_normal
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


class AbnormalTaskPerfAnalyzer:

    """
    Most of this analysis is cribbed from the jivita_worker_perf.ipynb notebook - this code is now
    newer, so that notebook should be converted over to manually test that the analysis here works
    as-expected. (Notebooks have access to this code too.)

    Note that this class *must be serializable* after __init__ - it may get passed to a local or
    remote process for execution.  This means we can't pass local dask data frames as inputs, for
    example, as they rely on local state (distributed dataframes we probably can).
    """

    def __init__(
        self,
        task_window,
        history_dt_interval,
        test_dt_interval,
        *,
        worker_group_cols=None,
        task_group_cols=None,
        start_dt_col=None,
        elapsed_time_col=None,
        outlier_pct=0.10,
        max_model_err_pct=0.25,
        min_event_time_resolution=1.0,
        max_exvar_in_metrics_pct=0.90,
        logpdf_bucket_size=0.01,
        random_seed=1,
        with_figures=False,
    ):

        self.task_window = task_window
        self.history_dt_interval = to_dt(history_dt_interval)
        self.test_dt_interval = to_dt(test_dt_interval)

        self.worker_group_cols = worker_group_cols
        self.task_group_cols = task_group_cols
        self.start_dt_col = start_dt_col
        self.elapsed_time_col = elapsed_time_col

        self.outlier_pct = outlier_pct
        self.min_event_time_resolution = min_event_time_resolution
        self.max_model_err_pct = max_model_err_pct
        self.max_exvar_in_metrics_pct = max_exvar_in_metrics_pct
        self.logpdf_bucket_size = logpdf_bucket_size

        self.random_seed = random_seed

        self.with_figures = with_figures

    def _to_task_ddf(self, task_window):

        if isinstance(task_window, pandas.DataFrame):
            return dask.dataframe.from_pandas(task_window, npartitions=1)

        if isinstance(task_window, dask.dataframe.DataFrame):
            return task_window

        task_window = task_window.clone_lazy(
            start_dt=min(self.history_dt_interval[0], self.test_dt_interval[0])
        )
        return task_window.to_dask_frame()

    def analyze(self):

        """
        The core of the analysis, step-by-step
        """

        # If we needed to do anything distributed because of a huge number of tasks, we could
        # do that here as a ddf.  Right now we aren't.
        task_ddf = self._to_task_ddf(self.task_window)
        task_df = pandas.DataFrame(task_ddf.compute())

        numpy.random.seed(self.random_seed)

        self.historical_task_df = tasks_in_dt_interval(
            task_df, self.start_dt_col, self.history_dt_interval
        )
        self.test_task_df = tasks_in_dt_interval(task_df, self.start_dt_col, self.test_dt_interval)

        # Build statistical task-time models for the different groups of historical tasks
        self.task_group_models = build_task_group_models(
            self.historical_task_df,
            self.task_group_cols,
            self.elapsed_time_col,
            outlier_pct=self.outlier_pct,
            figs=False,
        )

        # Compute surprise of test data based on the model
        self.full_surprise_df = build_worker_group_surprise_df(
            self.test_task_df,
            self.worker_group_cols,
            self.elapsed_time_col,
            self.task_group_models,
            max_model_err_pct=self.max_model_err_pct,
            min_event_time_resolution=self.min_event_time_resolution,
        )

        # For now, reduce dimensions to only mean surprisal, but save the full surprisal for sorting
        # degenerate cases of 1, 2, 3 workers later.
        self.surprise_df = self.full_surprise_df.loc[
            :, filter(lambda c: len(c[0]) > 0 and c[-1] == "mean", self.full_surprise_df.columns)
        ]

        self.metrics_df = build_pca_df(self.surprise_df, n_components=self.max_exvar_in_metrics_pct)

        self.normal_model = build_mv_normality_model(self.metrics_df, outlier_pct=self.outlier_pct)

        logpdf = self.normal_model.logpdf(self.metrics_df.values)
        logpdf_bucket = numpy.floor(logpdf / self.logpdf_bucket_size) * self.logpdf_bucket_size
        dist_from_mean = numpy.linalg.norm(self.metrics_df.values - self.normal_model.mean, axis=1)
        total_surprise = self.full_surprise_df.loc[:, ""]["sum"]

        # ... and sort our groups by how weird they look in the lower-d space
        self.normal_df = pandas.DataFrame(
            {
                "dist_from_mean": dist_from_mean,
                "logpdf_bucket": logpdf_bucket,
                "total_surprise": total_surprise,
            },
            index=self.surprise_df.index,
        )
        self.normal_df.sort_values(
            ["logpdf_bucket", "total_surprise"], ascending=[False, True], inplace=True
        )

        # Determine the (surprise) features which contributed most to each group's abnormality
        self.abnormal_surprise_df = build_feature_abnormality_df(
            self.surprise_df, self.metrics_df, self.normal_model
        )
        self.abnormal_surprise_df = self.abnormal_surprise_df.reindex(self.normal_df.index)

        self.result_df = pandas.concat(
            [self.normal_df, self.abnormal_surprise_df, self.metrics_df], axis=1
        )

        # Plot some figures if asked
        if self.with_figures:
            self.figures = build_abnormal_task_model_figures(
                self.test_task_df,
                self.elapsed_time_col,
                self.task_group_models,
                pandas.concat(
                    [self.abnormal_surprise_df.iloc[:3, :], self.abnormal_surprise_df.iloc[-3:, :]]
                ),
            )
            self.result_df.figures = self.figures

        return self.result_df


def build_task_group_models(
    task_df, task_group_cols, elapsed_time_col, *, outlier_pct=0.0, figs=None
):

    group_models = []

    for group, group_df in task_df.groupby(task_group_cols):

        group = to_tuple(group)
        group_df = group_df.copy()
        group_df.reset_index(drop=True, inplace=True)

        model = build_task_time_model(
            group_df, elapsed_time_col, outlier_pct=outlier_pct, fig=str(group) if figs else None
        )
        model.group_def = tuple(zip(task_group_cols, group))

        group_models.append(model)

    return group_models


def bootstrap_estimates(samples, estimator, num_tests=50):
    estimates = []
    for i in range(0, num_tests):
        estimates.append(estimator(numpy.random.choice(samples, len(samples))))
    return estimates


def build_task_time_model(
    task_df, elapsed_time_col, *, outlier_pct=0.0, fig=None, fig_max_sample_ksd_median=0.2
):

    outlier_test = task_df[elapsed_time_col].isnull()
    if len(task_df) > 5:
        outlier_min, outlier_max = numpy.percentile(
            task_df[elapsed_time_col], [(outlier_pct / 2.0) * 100, 100 - (outlier_pct / 2.0) * 100]
        )
        outlier_test = (
            outlier_test
            | (task_df[elapsed_time_col] < outlier_min)
            | (task_df[elapsed_time_col] > outlier_max)
        )

    nonoutliers = task_df.loc[~outlier_test]
    outliers = task_df.loc[outlier_test]

    # model = lognorm(*lognorm.fit(nonoutliers[elapsed_time_col]))
    model = lognorm(
        s=numpy.std(numpy.log(nonoutliers[elapsed_time_col])),
        scale=numpy.exp(numpy.mean(numpy.log(nonoutliers[elapsed_time_col]))),
    )
    model.nonoutliers = nonoutliers
    model.outliers = outliers

    model.kstest = kstest(nonoutliers[elapsed_time_col], model.cdf)

    # Here we want to run some tests on how close a model is compared to the data.  There are a number of statistical tests, Chi^2 for
    # discrete and KS for continuous are pretty standard options.  The metric of the KS test is the maximum distance between the cdfs of
    # the sample and model distributions.
    #
    # The output of this test also gives us a p-value, like Chi^2, which is a probability that the distributions are exactly the same.
    # For our purposes we pretty much know this is an estimate so that particular p-value isn't helpful as a "goodness of fit"
    # indicator.  For example, we get low p-values for very closely fit distributions of thousands of points (max cdf diff is < 0.05)
    # because we have so much data that "we're sure" that even that close approximation isn't exactly right.  This is true, but not
    # helpful in this context.  Other tests with p-values would have the same issue.
    #
    # What we arguably want to threshold on is the KS metric distance (D-value) but we also need a sense of how robust that metric is -
    # data sets with small numbers of points can sometimes give us low D-values but in some sense we'd like to be less sure of that
    # number.  (Tried to use high/low p-value cutoffs, this didn't seem to give consistent results but that's a TODO).  As a more-direct
    # heuristic measure of how robust the D-value is, we instead bootstrap the sampled data to get a measure of how good the fit "usually" is.

    ksds = bootstrap_estimates(nonoutliers[elapsed_time_col], lambda s: kstest(s, model.cdf)[0])
    model.sample_ksd_mean = numpy.mean(ksds)
    model.sample_ksd_median = numpy.median(ksds)
    model.sample_ksd_std = numpy.std(ksds)
    model.expected_err_pct = model.sample_ksd_mean + model.sample_ksd_std

    if numpy.isnan(model.expected_err_pct):
        model.expected_err_pct = 1.0

    if fig and model.sample_ksd_median <= fig_max_sample_ksd_median:

        title = None
        if isinstance(fig, str):
            title = fig
            fig = pyplot.figure()

        axs = fig.add_subplot(1, 1, 1)
        axs_pdf = axs.twinx()

        axs.hist(nonoutliers[elapsed_time_col], alpha=0.5)

        pdf_range = model.interval(0.99)
        pdf_x = numpy.linspace(pdf_range[0], pdf_range[1], 100)
        axs_pdf.scatter(pdf_x, model.pdf(pdf_x), alpha=0.5)

        axs.set_title(
            (title + " " if title else "")
            + f"n: {len(model.nonoutliers)} ks: {model.kstest} {model.sample_ksd_median}"
        )
        axs.set_ylabel("# of Tasks")
        axs_pdf.set_ylabel("Modeled Probability")
        axs.set_xlabel("Task Time")

        axs_pdf.set_ylim(0, axs_pdf.get_ylim()[1])

        fig.set_size_inches(5, 5)
        model.fig = fig

    return model


#
# Compute surprise of a measure based on a probability model of that measure.  The model need not
# be anything in particular but it does need to be continuous.
#


def build_worker_group_surprise_df(
    task_df, worker_group_cols, elapsed_time_col, task_group_models, **kwargs
):

    surprise_df = build_surprise_df(task_df, elapsed_time_col, task_group_models, **kwargs)
    combined_df = pandas.concat([task_df[worker_group_cols], surprise_df], axis=1)
    grouped_df = combined_df.groupby(worker_group_cols).agg(["mean", "sum", "count"])

    surprise_total_df = pandas.DataFrame(surprise_df.sum(axis=1), columns=[("",)])
    combined_total_df = pandas.concat([task_df[worker_group_cols], surprise_total_df], axis=1)
    grouped_total_df = combined_total_df.groupby(worker_group_cols).agg(["mean", "sum", "count"])

    return pandas.concat([grouped_df, grouped_total_df], axis=1)


def build_surprise_df(
    task_df,
    elapsed_time_col,
    task_group_models,
    *,
    max_model_err_pct=1.0,
    min_event_time_resolution=1.0,
):

    surprise_series = {}

    for model in task_group_models:

        if model.expected_err_pct > max_model_err_pct:
            continue

        # A measure of surprise when using a continuous distribution has a scaling problem - if we take our samples as
        # one or more delta functions compared with a distribution the integral formula gives us either infinities or
        # negative information - there's some discussion here:
        # https://stats.stackexchange.com/questions/211175/kullback-leibler-divergence-for-two-samples
        # for example.
        #
        # The workaround is to choose a bucket size - we do this by using the expected %pct error (uncertainty) in the model.
        # Using uncertainty has the nice feature that tight models give more information and loose models give much less, so
        # the threshold above reduces unhelpful computation but naturally removes models with relatively low information.
        #

        sample_interval = (
            min(model.nonoutliers[elapsed_time_col]),
            max(model.nonoutliers[elapsed_time_col]),
        )

        assert not numpy.isnan(sample_interval[0])
        assert not numpy.isnan(sample_interval[1])
        assert sample_interval[1] != sample_interval[0]

        sample_interval_pct = model.cdf(sample_interval[1]) - model.cdf(sample_interval[0])

        assert not numpy.isnan(sample_interval_pct)

        bucket_size = max(
            (sample_interval[1] - sample_interval[0])
            * (model.expected_err_pct / sample_interval_pct),
            min_event_time_resolution,
        )

        def compute_surprise_bits(row):

            for group_col, group_val in model.group_def:
                if row.loc[group_col] != group_val:
                    return numpy.NaN

            task_time = row.loc[elapsed_time_col]

            start_time, end_time = task_time - (bucket_size / 2.0), task_time + (bucket_size / 2.0)
            if start_time < 0:
                start_time, end_time = 0, bucket_size

            # Use log version of cdf to compute log probability in a (potentially) more numerically
            # stable way.
            u, v = model.logcdf(end_time), model.logcdf(start_time)
            log_prob = (u + numpy.log1p(-numpy.exp(v - u))) / numpy.log(2)

            assert not numpy.isnan(log_prob)
            return -1.0 * log_prob

            # task_time_prob = model.cdf(task_time + (bucket_size / 2.0)) - model.cdf(
            #    task_time - (bucket_size / 2.0)
            # )
            # return -1.0 * numpy.log2(task_time_prob)

        surprise_series[(model.group_def,)] = task_df.apply(compute_surprise_bits, axis=1)

    if len(surprise_series) == 0:
        surprise_series[(task_group_models[0].group_def,)] = [numpy.NaN] * len(task_df)

    return pandas.DataFrame(surprise_series)


def impute_missing_cols(features_df, value=0.0):
    features_df = features_df.copy()
    for col in features_df:
        if features_df[col].isnull().all():
            features_df[col].fillna(value, inplace=True)
    return features_df


def build_pca_df(features_df, *, n_components="mle", fig=None):

    # DRAGONS - the imputer works well, but removes features entirely when there's no data. We
    # don't want this - we just the PCA to discard them since otherwise it screws up our
    # dimensionalities of components later on.
    features_df = impute_missing_cols(features_df)

    features = features_df.values
    features = SimpleImputer(missing_values=numpy.NaN, strategy="median").fit_transform(features)
    # features = StandardScaler().fit_transform(features)

    pca = PCA(n_components=n_components, whiten=True)
    metrics = pca.fit_transform(features)

    metrics_df = pandas.DataFrame(
        metrics, index=features_df.index, columns=[f"pca{i}" for i in range(0, len(metrics[0]))]
    )
    metrics_df.pca = pca

    if fig:

        title = None
        if isinstance(fig, str):
            title = fig
            fig = pyplot.figure()
        if isinstance(fig, bool):
            title = "PCA Values"
            fig = pyplot.figure()

        axs = fig.add_subplot(1, 1, 1)

        axs.scatter(metrics_df.iloc[:, 0], metrics_df.iloc[:, 1])

        # zip joins x and y coordinates in pairs
        for i, (x, y) in enumerate(zip(metrics_df.iloc[:, 0], metrics_df.iloc[:, 1])):

            # this method is called for each point
            axs.annotate(
                metrics_df.index[i],  # this is the text
                (x, y),  # this is the point to label
                textcoords="offset points",  # how to position the text
                xytext=(0, 10),  # distance from text to points (x,y)
                ha="center",
            )  # horizontal alignment can be left, right or center

        axs.set_title(title + " " if title else "")
        axs.set_ylabel(metrics_df.columns[0])
        axs.set_xlabel(metrics_df.columns[1])

        fig.set_size_inches(8, 8)
        metrics_df.fig = fig

    return metrics_df


# HACK since we don't really have a nice way to convert back to a probability density from a percentile
def est_logpdf_for_percentile(model, outer_pct):

    n_samples = 100 * 1000 if len(model.mean) > 1 else 1000
    logpdf_for_percentile = numpy.percentile(model.logpdf(model.rvs(n_samples)), outer_pct * 100)

    return logpdf_for_percentile


def build_mv_normality_model(metrics_df, *, outlier_pct=0.0, fig=None, fig_pct_cutoffs=[1, 50, 99]):

    metrics = metrics_df.values
    mean = numpy.mean(metrics, axis=0)
    cov = numpy.cov(metrics, rowvar=0) if len(metrics_df) > 1 else [1.0]

    dist_model = multivariate_normal(mean, cov, allow_singular=True)

    outlier_logpdf_cutoff = est_logpdf_for_percentile(
        dist_model, outlier_pct if len(metrics_df) > 3 else 0.0
    )

    # TODO: Figure out how to compute pdf boundary directly for N% of results
    outlier_test = pandas.Series([(dist_model.logpdf(m) < outlier_logpdf_cutoff) for m in metrics])
    nonoutliers_df = metrics_df.iloc[list(~outlier_test), :]
    outliers_df = metrics_df.iloc[list(outlier_test), :]

    nonoutliers = nonoutliers_df.values
    mean = numpy.mean(nonoutliers, axis=0)
    cov = numpy.cov(nonoutliers, rowvar=0) if len(metrics_df) > 1 else 1.0

    normal_model = multivariate_normal(mean, cov, allow_singular=True)
    normal_model.nonoutliers = nonoutliers_df
    normal_model.outliers = outliers_df

    if fig:

        title = None
        if isinstance(fig, str):
            title = fig
            fig = pyplot.figure()
        if isinstance(fig, bool):
            title = "Normality Model"
            fig = pyplot.figure()

        axs = fig.add_subplot(1, 1, 1)

        metrics_x, metrics_y = list(zip(*metrics_df.iloc[:, 0:2].values))
        mean_metrics_other = list(mean[2:])

        def pretty_range(metrics, n_steps=500, overfill=1.1):
            m_min, m_max = min(metrics), max(metrics)
            m_center = (m_max + m_min) / 2.0
            m_radius = abs((m_max - m_min) / 2.0) * overfill
            m_step = (m_radius * 2.0) / float(n_steps)
            m_min = m_center - m_radius
            return m_min, m_min + m_step * n_steps, m_step

        x_min, x_max, x_step = pretty_range(metrics_x)
        y_min, y_max, y_step = pretty_range(metrics_y)

        x, y = numpy.mgrid[x_min:x_max:x_step, y_min:y_max:y_step]
        x_y = numpy.dstack((x, y))

        def xy_to_metric(x, y):
            return tuple([x, y] + mean_metrics_other)

        metrics_x_y = numpy.apply_along_axis(lambda xy: xy_to_metric(*xy), -1, x_y)

        def mark_pdf_range(v_arr):
            return numpy.array(list(map(lambda v: -1 if False else v, v_arr)))

        def marked_pdf(metrics):
            return numpy.apply_along_axis(mark_pdf_range, -1, normal_model.pdf(metrics))

        values_x_y = marked_pdf(metrics_x_y)

        min_value = min(min(r) for r in values_x_y)
        max_value = max(max(r) for r in values_x_y)

        contours = axs.contourf(x, y, values_x_y, levels=numpy.linspace(min_value, max_value, 5))
        # axs.contour(contours, levels=[numpy.exp(outlier_logpdf_cutoff)], colors='red')
        for fig_pct_cutoff in fig_pct_cutoffs:
            fig_logpdf_cutoff = est_logpdf_for_percentile(normal_model, fig_pct_cutoff)
            axs.contour(
                contours, levels=[numpy.exp(fig_logpdf_cutoff)], colors="orange", alpha=0.75
            )
        axs.scatter(metrics_x, metrics_y, alpha=0.5, color="white")

        axs.set_title(title + " " if title else "")
        # axs.clabel(contours, [outlier_pdf_cutoff, fig_pdf_cutoff], inline=True)
        axs.set_ylabel(metrics_df.columns[1])
        axs.set_xlabel(metrics_df.columns[0])

        fig.set_size_inches(8, 8)
        normal_model.fig = fig

    return normal_model


def impute_zero_sparse_rows(arr):
    def impute_zero_sparse_row(row):
        if numpy.sum(numpy.nan_to_num(row)) > 0:
            return row
        return row + 1.0

    return numpy.apply_along_axis(impute_zero_sparse_row, 1, arr)


def build_feature_abnormality_df(features_df, metrics_df, normal_model):

    abnormal_vs = metrics_df.values - normal_model.mean
    feature_vs = features_df - metrics_df.pca.mean_

    # Compute the total proportion of abnormality distance that each feature vector component is responsible for
    feature_sensitivity = numpy.matmul(
        numpy.abs(abnormal_vs), numpy.abs(metrics_df.pca.components_)
    )

    feature_contributions = numpy.multiply(
        feature_sensitivity, numpy.nan_to_num(impute_zero_sparse_rows(abs(feature_vs)))
    )

    col_index = pandas.MultiIndex.from_tuples(
        list(map(lambda t: t + ("value",), features_df.columns.to_flat_index()))
        + list(map(lambda t: t + ("abn_cont",), features_df.columns.to_flat_index())),
        names=features_df.columns.names + ("meta",),
    )

    contribution_df = pandas.DataFrame(feature_contributions, index=metrics_df.index)

    abnormal_df = pandas.concat([features_df, contribution_df], axis=1)

    abnormal_df.columns = col_index

    return abnormal_df


def build_abnormal_task_model_figures(task_df, elapsed_time_col, task_time_models, abnormal_df):

    abnormal_cols = list(
        filter(
            lambda mc: isinstance(mc, (list, tuple)) and mc[-1] == "abn_cont", abnormal_df.columns
        )
    )

    model_feature_map = {}
    for abnormal_col in abnormal_cols:

        group_def = abnormal_col[0]
        value_cols, abnormal_cols = model_feature_map.setdefault(group_def, ([], []))

        value_cols.append(abnormal_col[0:-1] + ("value",))
        abnormal_cols.append(abnormal_col)

    model_by_def = {}
    # model_value = {}
    model_abnormality = {}
    for model in task_time_models:
        model_by_def[model.group_def] = model
        if model.group_def not in model_feature_map:
            continue

        value_cols, abnormal_cols = model_feature_map[model.group_def]
        model_abnormality[model.group_def] = abnormal_df[abnormal_cols].sum(axis=1)
        # model_value[model.group_def] = abnormal_df[value_cols].sum(axis=1)

    model_abnormality_df = pandas.DataFrame(model_abnormality)
    model_abnormality_df.columns = model_abnormality_df.columns.to_flat_index()

    def to_tuple(v):
        if isinstance(v, tuple):
            return v
        if isinstance(v, list):
            return tuple(v)
        return (v,)

    class Comparisons:
        def __init__(self, task_group_def):
            self.task_group_def = task_group_def
            self.model_comparisons = []

    task_group_cols = to_tuple(abnormal_df.index.names)

    all_comparisons = []
    for i in range(0, len(abnormal_df)):

        task_group_def = tuple(zip(task_group_cols, to_tuple(abnormal_df.iloc[i, :].name)))

        comparisons = Comparisons(task_group_def)
        all_comparisons.append(comparisons)

        task_group_task_mask = None
        for col, value in task_group_def:
            next_mask = task_df[col] == value
            task_group_task_mask = (
                task_group_task_mask & next_mask if task_group_task_mask is not None else next_mask
            )

        abnormality_values = model_abnormality_df.iloc[i, :].values
        ordered_abnormality = sorted(
            zip(abnormality_values, list(model_abnormality_df.columns)), reverse=True
        )

        for abnormality, group_def in ordered_abnormality:

            model = model_by_def[group_def]

            model_task_mask = None
            for col, value in model.group_def:
                next_mask = task_df[col] == value
                model_task_mask = (
                    model_task_mask & next_mask if model_task_mask is not None else next_mask
                )

            model_task_df = task_df.loc[task_group_task_mask & model_task_mask]
            if len(model_task_df) == 0:
                continue

            comparisons.model_comparisons.append((abnormality, model, model_task_df))

    max_abnormality = max(model_abnormality_df.max())
    active_models = list(filter(lambda m: m.group_def in model_feature_map, task_time_models))

    AbnormalFigures = namedtuple("AbnormalFigures", ["bullseye", "histogram"])
    all_figures = {}

    for ci, comparisons in enumerate(all_comparisons):

        def pretty_vals(group_def):
            return " - ".join(str(v) for k, v in group_def)

        def pretty_spoke(group_def):
            return "Task " + "\n".join(str(v) for k, v in group_def)

        # if pretty_vals(comparisons.task_group_def) != 'protiva':
        #    continue

        fig = pyplot.figure()

        comparison_map = dict(
            (comparison[1].group_def, comparison) for comparison in comparisons.model_comparisons
        )

        theta = numpy.linspace(0, 2 * numpy.pi, len(active_models), endpoint=False)

        axs = fig.add_subplot(1, 1, 1, projection="polar")

        abnormality = numpy.array(
            list(
                map(
                    lambda m: comparison_map[m.group_def][0]
                    if m.group_def in comparison_map
                    else 0,
                    active_models,
                )
            )
        )

        # print("abnormality", abnormality)

        spoke_labels = list(map(lambda m: pretty_spoke(m.group_def), active_models))

        axs.set_thetagrids(numpy.degrees(theta), spoke_labels, weight="bold", size="medium")
        axs.tick_params(rotation="auto", pad=1.2)
        # axs.set_varlabels(spoke_labels, weight='bold', size='medium')

        alpha = (abnormality / max_abnormality) * 0.9 + 0.1
        rgb_colors = list(map(lambda m: colorhash.ColorHash(m.group_def).rgb, active_models))
        rgba_colors = numpy.zeros((len(abnormality), 4))
        rgba_colors[:, 0:3] = numpy.array(rgb_colors) / 255.0
        rgba_colors[:, 3] = alpha

        axs.scatter(
            theta,
            abnormality,
            c=rgba_colors,
            s=(((abnormality / max_abnormality) * 30.0) ** 2) * numpy.pi,
            marker="o",
        )

        axs.set_rlim(0, max_abnormality + 1.0)
        rgrids = numpy.arange(0, max_abnormality + 1.0, 1.0)
        axs.set_rgrids(rgrids, labels=[""] * len(rgrids))

        axs.set_title(
            f"Task Completion Time Normality for\n{pretty_vals(comparisons.task_group_def)}\n",
            weight="bold",
            size="large",
        )

        for spine in axs.spines.values():
            spine.set_edgecolor("lightgray")

        # fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
        fig.set_size_inches(8, 8)

        bullseye = fig

        fig = pyplot.figure()

        cols = math.ceil(math.sqrt(len(active_models)))
        rows = math.ceil(len(active_models) / cols)

        # for i, comparison in enumerate(comparisons.model_comparisons):
        for i, model in enumerate(active_models):

            axs = fig.add_subplot(rows, cols, i + 1)
            axs_pdf = axs.twinx()

            comparison = comparison_map.get(model.group_def, None)
            if comparison is None:
                continue

            abnormality, model, model_task_df = comparison

            # binsize = model.event_resolution
            # bins = numpy.arange(0, max(model.ppf(0.99), max(model_task_df[elapsed_time_col])) + binsize, binsize)

            binrange = max(model.ppf(0.99), max(model_task_df[elapsed_time_col]))
            bins = numpy.linspace(0, binrange, 10)

            #                 bins_99 = model.ppf(numpy.linspace(0.01, 0.99, 99))
            #                 last_bin_size = bins_99[-1] - bins_99[-2]
            #                 bins = [0] + list(bins_99) + [max(bins_99[-1] + last_bin_size, max(model_task_df[elapsed_time_col]))]

            rgb_color = numpy.array(colorhash.ColorHash(model.group_def).rgb) / 255.0

            # axs.hist(model_task_df[elapsed_time_col], bins=bins, alpha=0.05, color='purple')
            axs.hist(
                model_task_df[elapsed_time_col],
                bins=bins,
                alpha=(abnormality / max_abnormality) * 0.9 + 0.1,
                color=rgb_color,
            )

            pdf_range = model.interval(0.99)
            pdf_x = numpy.linspace(pdf_range[0], pdf_range[1], 100)
            axs_pdf.plot(
                pdf_x,
                model.pdf(pdf_x),
                alpha=(1.0 - model.expected_err_pct) * 0.9 + 0.1,
                color="black",
                linewidth=3.0,
            )

            axs.set_title(
                f"Task {pretty_vals(model.group_def)}\n{pretty_vals(comparisons.task_group_def)}\n(Ab:{abnormality:.2f} ErrPct:{model.expected_err_pct:.2f})\n"
            )
            axs.set_ylabel("# of Tasks Completed")
            axs_pdf.set_ylabel("Modeled Probability")
            axs.set_xlabel("Task Completion Time (mins)")

            # axs.set_xscale('log')
            # axs_pdf.set_xscale('log')

        fig.subplots_adjust(wspace=0.4, hspace=0.25)
        fig.set_size_inches(5 * cols, 10 * rows)

        histogram = fig

        _, task_group_index_values = zip(*comparisons.task_group_def)

        all_figures[maybe_tuple(task_group_index_values)] = AbnormalFigures(bullseye, histogram)

    return all_figures


def maybe_tuple(val):
    val = tuple(val)
    if len(val) == 1:
        return val[0]
    return val


def to_tuple(val):
    if isinstance(val, (tuple, list)):
        return tuple(val)
    return (val,)


def to_dt(val):
    if isinstance(val, (list, tuple)):
        return list(map(to_dt, val))
    return pandas.to_datetime(val, infer_datetime_format=True)


def to_dtoffset(val):
    if isinstance(val, pandas.DateOffset):
        return val
    if isinstance(val, dict):
        return pandas.DateOffset(**val)
    if isinstance(val, (int, float)):
        return pandas.DateOffset(days=val)
    return None


# 2. Divide data into historical and test time periods
def tasks_in_dt_interval(task_df, task_dt_col, dt_interval):
    dt_interval = (
        pandas.to_datetime(dt_interval[0], infer_datetime_format=True),
        pandas.to_datetime(dt_interval[1], infer_datetime_format=True),
    )
    dt_seq = pandas.to_datetime(task_df[task_dt_col], infer_datetime_format=True)
    dt_filter = (dt_seq >= dt_interval[0]) & (dt_seq < dt_interval[1])
    return task_df.loc[dt_filter]
