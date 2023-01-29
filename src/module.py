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


class BoundaryModule():
    def getParameters(self):
        return None
    def initialize(self, config, state):
        return
    def finalize(self):
        return
    def __init__(self, identifier, moduleDescription):
        super().__init__(identifier, moduleDescription)
        return
    def dfsphPrepareSolver(self, simulationState, simulation):
        raise Exception('Operation dfsphPrepareSolver not implemented for ', self.identifier)
    def dfsphBoundaryAccelTerm(self, simulationState, simulation):
        raise Exception('Operation dfsphBoundaryAccelTerm not implemented for ', self.identifier)
    def dfsphBoundaryPressureSum(self, simulationState, simulation):
        raise Exception('Operation dfsphBoundaryPressureSum not implemented for ', self.identifier)
    def dfsphBoundaryAlphaTerm(self, simulationState, simulation):
        raise Exception('Operation dfsphBoundaryAlphaTerm not implemented for ', self.identifier)
    def dfsphBoundarySourceTerm(self, simulationState, simulation):
        raise Exception('Operation dfsphBoundarySourceTerm not implemented for ', self.identifier)
    def boundaryPressure(self, simulationState, simulation):
        raise Exception('Operation boundaryPressure not implemented for ', self.identifier)
    def boundaryDensity(self, simulationState, simulation):
        raise Exception('Operation boundaryDensity not implemented for ', self.identifier)
    def boundaryFriction(self, simulationState, simulation):
        raise Exception('Operation boundaryFriction not implemented for ', self.identifier)
    def boundaryNeighborsearch(self, simulationState, simulation):
        raise Exception('Operation boundaryNeighborsearch not implemented for ', self.identifier)
    def boundaryFilterNeighborhoods(self, simulationState, simulation):
        return # Default behavior here is do nothing so no exception needs to be thrown
        # raise Exception('Operation dfsphBoundaryAccelTerm not implemented for ', self.identifier)