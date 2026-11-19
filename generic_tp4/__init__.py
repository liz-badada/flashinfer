from .config import ExecutionConfig, ProjectionConfig

__all__ = ['ExecutionConfig', 'ProjectionConfig', 'BlockWeight', 'LocalWeights', 'Workspace']


def __getattr__(name):
    if name in ('BlockWeight', 'LocalWeights', 'Workspace'):
        from . import operator
        return getattr(operator, name)
    raise AttributeError(name)
