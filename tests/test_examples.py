"""
Tests for example data scripts, if required
"""

from examples.ona.gen_ona_task_data import main as example_ona_main

def test_example_ona():
    # Just make sure we don't throw errors
    example_ona_main()

