import builtins


class MitoCodeBaseError(builtins.Exception):
    def __init__(self, message="Unknown Error") -> None:
        self.message = message
        super().__init__(self.message)


class MitoCodePathConvertError(MitoCodeBaseError):
    pass


class MitoCodeWorkspaceError(MitoCodeBaseError):
    pass


class MitoCodeWorkspaceInitError(MitoCodeWorkspaceError):
    pass


class MitoCodeWorkspaceCreateError(MitoCodeWorkspaceError):
    pass


class MitoCodeWorkspaceDumpError(MitoCodeWorkspaceError):
    pass


class MitoCodeWorkspaceTooOldError(MitoCodeWorkspaceError):
    pass


class MitoCodeAppTooOldError(MitoCodeBaseError):
    pass


class MitocodeMigrateError(MitoCodeWorkspaceError):
    pass
