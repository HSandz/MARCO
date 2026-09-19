from marco.systems.base import System
from marco.systems.marco import MARCOSystem
from marco.systems.single import SingleSystem
SYSTEMS: list[type[System]] = [value for value in globals().values() if isinstance(value, type) and issubclass(value, System) and value != System]