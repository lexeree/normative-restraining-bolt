from agents import Agent
import random
from collections import defaultdict
import log



class RestrainingBoltNormAgent(Agent):
    def __init__(self, env, dist=False, specs=None, alpha=0.5, epsilon=0.05, gamma=0.9, ntrain=10):
        Agent.__init__(self, env, dist)
        self.name = 'Restraining Bolt Norm Agent'
        self.logger = log.Log(self.name, self.env.map)
        self.qValues = {}
        self.qValues = defaultdict(lambda:0.0, self.qValues)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.ntrain = ntrain
        self. automata = specs

    def getQValue(self, state, action, aStates):
        ast = ','.join(str(i) for i in aStates)
        return self.qValues[(ast, state, action)]

    def computeValue(self, state, possible, aStates):
        qvals = [self.getQValue(state, a, aStates) for a in possible]
        return max(qvals)

    def policy(self, state, aStates, train):
        possible = self.getLegalActions(state, train)
        v = self.computeValue(state, possible, aStates)
        acts = []
        for act in possible:
            if self.getQValue(state, act, aStates) == v:
                acts.append(act)
        return random.choice(acts)

    def act(self, state, aStates, train=False):
        action = self.policy(state, aStates, train)
        if train:
            if random.random() <= self.epsilon:
                return random.choice(self.getLegalActions(state, train))
            else:
                return action
        else:
            return action

    def update(self, state0, action, state1, aStates0, aStates1, reward):
        curQ = self.getQValue(state0, action, aStates0)
        ast = ','.join(str(i) for i in aStates0)
        self.qValues[(ast, state0, action)] = (1 - self.alpha) * curQ + self.alpha * (
                reward + self.gamma * self.computeValue(state1, self.getPossibleActions(state1), aStates1))

    def test(self, rec=False):
        self.env.reset()
        state = self.env.initialState()
        steps = 1
        aStates = [a.state0 for a in self.automata]
        while not state.final:
            act = self.act(state, aStates)
            if rec:
                lst = []
                for p in self.getPossibleActions(state):
                    a = p + ' = ' + str(self.getQValue(state, p, aStates))
                    lst.append(a)
                self.logger.record_state(state, act, lst)
            nstate, r = self.env.stateTransition(state, act)
            oldaStates = aStates.copy()
            inpt = self.get_labels(state) + [act]
            for i in range(len(aStates)):
                aStates[i] = self.automata[i].transition(aStates[i], inpt)
                if aStates[i] in self.automata[i].final:
                    r += self.automata[i].reward
                    if self.automata[i].achievement:
                        aStates[i] = 0
                    else:
                        aStates[i] = oldaStates[i]
            steps += 1
            state = nstate
        if rec:
            self.logger.export_trace()
        return steps, len(state.inventory), state.get_value(), state.damage

    def run(self, n, rec=False):
        for i in range(n):
            time, mass, value, damage = self.test(rec=False)
            if rec:
                self.logger.add_summary(i, time, mass, value, damage)
        if rec:
            self.logger.export_summary()

    def train(self):
        for n in range(self.ntrain):
            self.env.reset()
            state = self.env.initialState()
            reward = 0
            aStates = [a.state0 for a in self.automata]
            while not state.final:
                act = self.act(state, aStates, train=True)
                inpt = self.get_labels(state) + [act]
                nstate, r = self.env.stateTransition(state, act)
                oldaStates = aStates.copy()
                #print(inpt)
                for i in range(len(aStates)):
                    aStates[i] = self.automata[i].transition(aStates[i], inpt)
                    if aStates[i] in self.automata[i].final:
                        r += self.automata[i].reward
                        #print("Automaton reward:")
                        #print(self.automata[i].reward)
                        if self.automata[i].achievement:
                            aStates[i] = 0
                        else:
                            aStates[i] = oldaStates[i]
                self.update(state, act, nstate, oldaStates, aStates, r)
                reward += r
                state = nstate
            print('Episode '+str(n + 1)+ ' complete!')
            print('Total rewards: '+str(reward))
            print('Damage taken: ' + str(state.damage))
            print('Inventory value: ' + str(state.get_value()))



