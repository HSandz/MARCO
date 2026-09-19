from abc import ABC, abstractmethod

from marco.utils import read_json

class Tool(ABC):
    def __init__(self, config_path: str = None, config: dict = None, *args, **kwargs) -> None:
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = read_json(config_path)
        else:
            self.config = {}

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError("reset method not implemented")

class RetrievalTool(Tool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def search(self, query: str) -> str:
        raise NotImplementedError("search method not implemented")

    @abstractmethod
    def lookup(self, title: str, term: str) -> str:
        raise NotImplementedError("lookup method not implemented")
