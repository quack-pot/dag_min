from .node import Node

class InputNode(Node):
    ## *=================================================
    ## *
    ## * __init__
    ## *
    ## *=================================================

    def __init__(self) -> None:
        super().__init__()

    ## *=================================================
    ## *
    ## * __evaluate__
    ## *
    ## *=================================================

    def __evaluate__(self) -> None:
        return # ? This is effectively a no-op since the value is set directly