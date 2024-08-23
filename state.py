
class State:
    def __init__(self, x, y, label, bag, damage, attack, last=None, sundown=False, final=False):
        self.x = x
        self.y = y
        self.label = label
        self.inventory = bag
        self.damage = damage
        self.attack = attack
        self.last_move = last
        self.sundown = sundown
        self.final = label == 'M' and not self.inventory

    def __key(self):
        return (self.x, self.y, self.label, self.attack, self.damage, tuple(sorted(self.inventory)),self.sundown)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, State):
            return self.__key() == other.__key()

    def copy(self):
        x = self.x
        y = self.y
        label = self.label
        bag = self.inventory
        damage = self.damage
        attack = self.attack
        last = self.last_move
        sundown = self.sundown
        final = self.final
        return State(x, y, label, bag, damage, attack, last, sundown, final)

    def sPrint(self):
        return "x="+str(self.x)+", y="+str(self.y)+", label="+self.label+", attack="+str(self.attack)+", bag="+str(sorted(self.inventory))+ ", damage="+ str(self.damage)+", sunset="+str(self.sundown)

    def get_value(self):
        value = 0
        for obj in self.inventory:
            if obj == 'W':
                value += 1
            elif obj == 'O':
                value += 1
        return value

    def __getattribute__(self, __name):
        return super().__getattribute__(__name)

    def __eq__(self, other):
        if isinstance(other, State):
            return self.x == other.x and self.y == other.y and self.label == other.label and sorted(self.inventory) == sorted(other.inventory) and self.damage == other.damage and self.sundown == other.sundown
        else:
            return False




