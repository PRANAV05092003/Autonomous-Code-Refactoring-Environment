from .grader import grade_task
from .easy_task import EasyTask
from .medium_task import MediumTask
from .hard_task import HardTask

__all__ = [
    "EasyTask",
    "MediumTask",
    "HardTask",
    "grade_task",
]

from acre.tasks.task_registry import Task, TaskRegistry

__all__ = ["Task", "TaskRegistry"]
