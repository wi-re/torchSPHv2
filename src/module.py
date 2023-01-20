class Module():
    def getParameters(self):
        return None
    def initialize(self, config, state):
        return
    def finalize(self):
        return
    def __init__(self, identifier, moduleDescription):
        self.description = moduleDescription
        self.name = identifier
        return