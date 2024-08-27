from environment import Environment
import restraining_bolt as rb
from automaton import DFA
import os

if __name__ == '__main__':
    env = Environment('layouts/basic.txt', risk=1)
    automata = []

    # "The merchant is forbidden from entering the dangerous area"
    # viol: F(at_danger)
    def trans0a(l):
        if 'at_danger' in l:
            return 1
        else:
            return 0
    def trans1a(l):
        return 1
    transa = {}
    transa[0] = trans0a
    transa[1] = trans1a
    automatona = DFA([0,1], -5, trans=transa, final=[1])

    # "if the merchant enters the dangerous area, they ought to unload their inventory"
    # viol: F(at_danger & attacked & !(Unload))
    def trans0b(l):
        if 'at_danger' in l and 'attacked' in l and 'Unload' not in l:
            return 1
        else:
            return 0
    def trans1b(l):
        return 1

    transb = {}
    transb[0] = trans0b
    transb[1] = trans1b
    automatonb = DFA([0, 1], -120, trans=transb, final=[1])

    automata.append(automatona)
    automata.append(automatonb)
    agent = rb.RestrainingBoltNormAgent(env, dist=True, specs=automata, alpha=0.5, epsilon=0.25, gamma=0.9, ntrain=5000)
    #agent = q.RewardQAgent(env, dist=True, alpha=0.5, epsilon=0.25, gamma=0.9, ntrain=5000)
    agent.train()
    a, i, b, c = agent.test(rec=True)
