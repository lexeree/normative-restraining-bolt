import random
import copy
from state import State

class Environment:
    def __init__(self, map, risk=0.7, capacity=10, damage=20, sunset=20):
        self.mapfile = map
        self.map = self.parseMap(map)
        self.risk = risk
        self.capacity = capacity
        self.max_damage = damage
        self.stateno = 0
        self.sunset = sunset

    def copy(self):
        mapfile = self.mapfile
        risk = self.risk
        capacity = self.capacity
        damage = self.max_damage
        return Environment(mapfile, risk, capacity, damage)

    def parseMap(self, mapfile):
        map = {}
        with open(mapfile) as f:
            chars = list(f.read())
        i = 0
        j = 0
        for c in chars:
            if c == '\n':
                self.x = i
                i = 0
                j += 1
            else:
                if c == 'H':
                    self.home = (i,j)
                if c == 'M':
                    self.market = (i,j)
                map[(i, j)] = c
                i += 1
        self.y = j
        return map


    def distFromDest(self, x, y):
        fringe = [(x, y, 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if pos_x == self.market[0] and pos_y == self.market[1]:
                return dist
            # otherwise spread out from the location to its neighbours
            if self.map[(pos_x, pos_y + 1)] != 'X':
                fringe.append((pos_x, pos_y+1, dist + 1))
            if self.map[(pos_x, pos_y - 1)] != 'X':
                fringe.append((pos_x, pos_y - 1, dist + 1))
            if self.map[(pos_x + 1, pos_y)] != 'X':
                fringe.append((pos_x + 1, pos_y, dist + 1))
            if self.map[(pos_x - 1, pos_y)] != 'X':
                fringe.append((pos_x - 1, pos_y, dist + 1))
        return None

    def reset(self):
        self.map = self.parseMap(self.mapfile)
        self.stateno = 0


    def initialState(self):
        self.sundown = False
        return State(self.home[0], self.home[1], 'H', [], 0, False, None)

    def stateTransition(self, state, action):
        final = False
        attack = False
        self.stateno += 1
        sundown = self.stateno+1 >= self.sunset
        #print("STATENO="+str(self.stateno))
        reward = 0
        if state.attack:
            if state.damage + 1 >= self.max_damage:
                final = True
            if action == 'Fight':
                return State(state.x, state.y, state.label, state.inventory, state.damage, attack, state.last_move, sundown=sundown, final=final), reward
            elif action == 'Unload' and len(state.inventory) > 0:
                reward = -100 * state.get_value()
                return State(state.x, state.y, state.label, [], state.damage, attack, state.last_move, sundown=sundown, final=final), reward
            else:
                # TODO: change back to state.damage + 1
                return State(state.x, state.y, state.label, state.inventory, state.damage, state.attack,
                             state.last_move, sundown=sundown, final=final), reward
        else:
            x = copy.deepcopy(state.x)
            y = copy.deepcopy(state.y)
            label = copy.deepcopy(state.label)
            bag = copy.deepcopy(state.inventory)
            last = copy.deepcopy(state.last_move)
            if action == 'North':
                #if self.map[(x, y)] == 'C':
                #    self.map[(x, y)] = ''
                y -= 1
                label = self.map[(x, y)]
                last = action
            elif action == 'South':
                #if self.map[(x, y)] == 'C':
                #    self.map[(x, y)] = ''
                y += 1
                label = self.map[(x, y)]
                last = action
            elif action == 'East':
                #if self.map[(x, y)] == 'C':
                #    self.map[(x, y)] = ''
                x += 1
                label = self.map[(x, y)]
                last = action
            elif action == 'West':
                #if self.map[(x, y)] == 'C':
                #    self.map[(x, y)] = ''
                x -= 1
                label = self.map[(x, y)]
                last = action
            elif action == 'Unload':
                bag = []
                if (x, y) == self.market:
                    reward = 100 * state.get_value()
                    final = True
                else:
                    reward = -100 * state.get_value()
            elif action == 'Extract':
                if state.label == 'T':
                    bag.append('W')
                    self.map[(x, y)] = 'C'
                    label = 'C'
                    reward = 50
                elif state.label == 'R':
                    bag.append('O')
                    self.map[(x, y)] = 'C'
                    label = 'C'
                    reward = 50
            if self.map[(x,y)] == 'D':
                if random.random() <= self.risk:
                    attack = True
            if self.stateno > 50:
                final = True
                reward = -100
            return State(x, y, label, bag, state.damage, attack, last, sundown, final), reward

        def __getattr__(self, name: str):
            return self.__dict__[f"_{name}"]

