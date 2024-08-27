from environment import Environment
import restraining_bolt as rb
import qlearning as q
from automaton import DFA

if __name__ == '__main__':
    env = Environment('layouts/twist.txt', risk=1)
    automata = []

    # "From the time it is at home, the merchant ought to visit the market before sunset"
    # viol: F(at_home & !at_market U sundown)
    def trans0a(l):
        if 'sundown' in l and 'at_home' in l:
            return 2
        elif 'at_home' in l and 'sundown' not in l and 'at_market' not in l:
            return 1
        else:
            return 0


    def trans1a(l):
        if 'sundown' in l:
            return 2
        elif 'at_market' not in l:
            return 1
        else:
            return 0


    def trans2a(l):
        return 2


    transa = {}
    transa[0] = trans0a
    transa[1] = trans1a
    transa[2] = trans2a
    automatona = DFA([0, 1, 2], -50, trans=transa, final=[2])
    automata.append(automatona)

    agent = rb.RestrainingBoltNormAgent(env, specs=automata, alpha=0.5, epsilon=0.25, gamma=0.9, ntrain=5000)

    agent.train()
    a, i, b, c = agent.test(rec=True)