from abc import ABC, abstractmethod

class Node(ABC):
    ## *=================================================
    ## *
    ## * __init__
    ## *
    ## *=================================================

    def __init__(self) -> None:
        self.value: float = 0.0
        self.__has_evaluated__: bool = False

    ## *=================================================
    ## *
    ## * reset
    ## *
    ## *=================================================

    def reset(self) -> None:
        self.value = 0.0
        self.__has_evaluated__ = False

    ## *=================================================
    ## *
    ## * getValue
    ## *
    ## *=================================================

    def getValue(self) -> float:
        return self.value
    
    ## *=================================================
    ## *
    ## * setValue
    ## *
    ## *=================================================

    def setValue(self, value: float) -> None:
        self.value = value

    ## *=================================================
    ## *
    ## * evaluate
    ## *
    ## *=================================================

    def evaluate(self) -> None:
        if self.__has_evaluated__:
            return
        
        self.__evaluate__()

    ## *=================================================
    ## *
    ## * __evaluate__
    ## *
    ## *=================================================

    @abstractmethod
    def __evaluate__(self) -> None:
        pass