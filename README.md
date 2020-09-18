# Community Health Task Investigator

## Focusing field worker management with data

Organizations wanting to monitor or improve lives in places with limited infrastructure face the challenge of managing teams of field workers.  Examples include research teams monitoring pre- and post-natal health in developing countries, international orgs running malaria control programs, or NGOs administering vaccines to remote regions.  Teams of *field workers* are assigned and complete tasks for the organization directed by *field managers*, and the overall success of this *field team* can determine the success of a project.

Managing a field team well, and identifying problems as early as possible, is challenging.  Often workers do many different kinds of tasks and there are few established baselines for performance.  Our experiment here is to see if a data-driven approach can help - instead of starting from external performance targets, can we validate the performance of workers against other workers doing similar tasks and prioritize valuable field manager time?

## Toolset

### Worker Task-Time Abnormality

To compare a worker's performance at a task to the overall population of workers, a minimal starting point is to compare the completion times of tasks the worker was assigned against the overall distribution of completion times from some historical or hopefully-representative data set.  Nearly all systems record this information and it is easy to compare.

The output for a field manager, based on this input, is a listing of workers ranked by overall abnormality across all their tasks of different types (with accompanying task-time graphs, visualizations, etc.).  This allows the field manager to spend their time identifying problems and learning from high-performers and spend less time with workers who are "unsurprising" in their productivity.

#### I/O

More formally, the **input** to such an algorithm looks like:

* A set of task tuples of the form:

  ```
  (task_id, task_type, task_start_timestamp, task_duration, worker_id)
  ```

* A historical time range over which tasks contribute to a task-time model:

  ```
  [history_start_timestamp, history_end_timestamp)
  ```

* An analysis time range over which task-time is used to rank workers for abnormality:

  ```
  [analysis_start_timestamp, analysis_end_timestamp)
  ```

The **output** for the field manager is then:

* A list of workers ranked by an abnormality score:

  ```
  [(worker_id, worker_abnormality_score), ...]
  ```

* Task-time metrics for each worker and task type used to generate the overall abnormality score (this can be presented as graphs with comparison to the historical models to make it human-readable):

  ```
  [(worker_id, worker_metric_0_task_0, ..., worker_metric_n_task_m), ...]
  ```

### Ona "Principal Surprise" Algorithm

One basic approach to this worker abnormality problem is to compute the information gain (surprise) from a worker's task completion times based on historical models of task completion times.  Then, with a measure of surprise for each task type, we can determine the principal task types that differentiate workers and compute a weighted sum of the surprises as the overall abnormality.

The idea is straightforward, but there are a few complications that need to be managed:

* **Data varies greatly in quantity** - some task types will have much more historical data than others.  Also, in a week, for example, there is usually not enough data to directly build and compare a worker-specific model for a particular task type even with large data sets.
* **Field managers usually have workers doing many different kinds of tasks** - ideally the worker ranking would smoothly handle a single task type, workers that do some tasks but not others, and/or workers doing any combination of heterogeneous tasks.
* **There are lots of ways to build statistical models of task completion times** - the model building assumptions should ideally be a pluggable input to the overall algorithm.  Also, not all models will be equally "good" or predictive and our surprise metrics should reflect this.

#### High-level overview

Given the approach and requirements above, the Principal Surprise (PS) algorithm works in the following way:

1. ##### Compute task time surprise

    1.  Load the historical task completion times for all workers, for each task type:

        1. Generate a `lognorm` ([log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution)) model of historical task completion times

        2. Compute an `expected_(max_)err_pct` (percent) via resampling with a `kstest` ([Kolmogorov-Smirnov test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)) metric

    2. Load the analysis task completion times for all workers, for each worker, for each task type:
      
        1. Compute the sum and mean surprise (in bits) based on the historical task type model and expected error of that model.  
         
           High probability + low error ⇒ low surprise

           Low probability + low error ⇒ high surprise

           Low probability + high error ⇒ low surprise
   
2. ##### Project surprise metrics down to a ranking

   1. Project the worker surprise metrics down to the most informative axes via a `PCA` ([principal components analysis](https://en.wikipedia.org/wiki/Principal_component_analysis))

   2. Compute a `multivariate_normal` distribution / mean in the projected space, rank workers by distance from mean

      1. Also compute the `abs` individual contribution of each task-type surprise to the distance from mean for each worker to rank task-types by abnormality per-worker

#### Example Use Case

See `examples/transport.csv`

#### Extensions / TODO

As is, the PS algorithm seems to give interesting results, but there are a number of areas that haven't been explored fully yet:

* **More task-type metrics** - so far, we've only looked at task completion times, but there are many other possibilities here so long as a useful model can be generated.  For example, number of tasks completed in some time interval, task-specific metrics (for example, the number of questions completed in a survey), or geospatial distribution would be interesting possibilities.

* **More complex models and error** - the `lognorm` task-completion model is simple and supported by some research, but if more is known about a task a more appropriate model could be built.  For example, making very-low-time tasks unsurprising if field workers often fail to meet a client may make the field manager ranking more relevant.  Also, if there is enough data, the error in a model can be approximated in a more granular way than just overall expected maximum.

* **Integrating external targets into the field manager output** - for many tasks, completion time (or other targets) can be estimated and these are a standard way of monitoring field team performance.  By comparing a workers' performance on the most abnormal task-types to the pre-set targets, the ranking can be divided to show workers who are exceeding expectations (and by how much) and those who are not meeting them.

  Ideally this could also result in a virtuous cycle where external targets that are initially set inaccurately can be refined over time by identifying targets that are *unsurprisingly* not met.

* **Integrating field manager feedback into the rankings** - a ranking system based on task metadata will not be able to fully capture factors not in that task data.  For example, some worker tasks may be more experimental and expected to vary more in completion time, but should still be monitored for outliers.  By allowing field managers to mark particular abnormal workers as uninteresting, the next worker ranking could take this into account by weighting the uninteresting worker's surprising task types less heavily via some adaptive algorithm.

