

class DFA:
    def __init__(self, states, reward, trans=None, final=None, achievement=False):
        self.states = states
        self.transitions = trans
        self.final = final
        self.state0 = 0
        self.reward = reward
        self.achievement = achievement
        self.sinkStates = []

    def transition(self, state, inpt):
        foo = self.transitions[state]
        return foo(inpt)

    def addSinkStates(self, s):
        self.sinkStates.append(s)




