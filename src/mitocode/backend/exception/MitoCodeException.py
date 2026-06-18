import builtins

class MitoCodeBaseError(builtins.Exception):
    def __init__(self, message="Unknown Error") -> None:
        self.message = message
        super().__init__(self.message)

class MitoCodePathConvertError(MitoCodeBaseError):
    pass

class MitoCodeWorkspaceInitError(MitoCodeBaseError):
    pass
