"""
Tests for example data scripts, if required
"""

from examples.ps.gen_ps_task_data import main as example_ps_main


def test_example_ps():
    # Just make sure we don't throw errors
    example_ps_main()
