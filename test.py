
class Thing:


    counter = 0

    def __init__(self, name):
        self.name = name
        self.count = self.__class__.counter
        self.__class__.counter += 1

things = [Thing("test") for i in range(10)]

for thing in things:
    print(thing.count)
