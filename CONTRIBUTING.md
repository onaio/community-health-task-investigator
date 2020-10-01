# Contributing

## Quickstart

To set up your dev environment, you will need the following prerequisites:

* Python v3.7.x (recommended to manage with the `pyenv` tool)
* Poetry ^1.0.5 (dependency manager, wraps `virtualenv` and `pip`)

On Ubuntu, you can run:

```
$ # Install pyenv
$ [sudo] curl https://pyenv.run | bash
$ exec $SHELL
$ # Install Python+Poetry
$ [sudo] pyenv install 3.7.7
$ [sudo] pip install 'poetry==1.0.5'
```

Once these are installed, run:

```
$ poetry install
$ poetry run pytest
```

If these commands complete successfully, your environment is set up correctly!

To enter the project `venv`:

```
$ poetry shell
```

## PRs and Code Style

The autostyle tool `pre-commit` is installed by default by `poetry install` as a dev-dependency - to trigger it manually run the command:

```
$ poetry run pre-commit run --all-files
```

No PRs will be merged with style violations, but feel free to open PRs for unfinished code for review.

The `pre-commit` package supports triggers on various `git` events - to see the full set of options visit [pre-commit.com](https://pre-commit.com/).
