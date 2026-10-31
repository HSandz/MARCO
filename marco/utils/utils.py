from typing import TypeVar

T = TypeVar('T')

def get_rm(d: dict, key: str, value: T) -> T:
    ret = d.get(key, value)
    if key in d:
        del d[key]
    return ret

def task2name(task: str) -> str:
    if task == 'rp':
        return 'Rating Prediction'
    elif task == 'sr':
        return 'Sequential Recommendation'
    else:
        raise ValueError(f'Task {task} is not supported.')

def system2dir(system: str) -> str:
    assert 'system' in system.lower(), 'The system name should contain "system"!'
    return system.lower().replace('system', '')
