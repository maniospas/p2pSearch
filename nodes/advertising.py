@DeprecationWarning
class AdvertisingNode:
    def __init__(self, name):
        self.name = name
        self.neighbors = dict()
        self.embedding = 0

    def set_personalization(self, personalization):
        self.personalization = personalization
        self.embedding = personalization

    def update(self):
        pass
        # if len(self.neighbors) == 0:
        #     return
        # embedding = 0
        # for neighbor_embedding in self.neighbors.values():
        #     embedding = embedding + neighbor_embedding
        # self.embedding = embedding / len(self.neighbors)**0.5 * 0.9 + self.personalization * 0.1

    def receive(self, neighbor, neighbor_embedding):
        self.neighbors[neighbor] = neighbor_embedding
        self.update()

    def send(self, _):
        return self.embedding
    