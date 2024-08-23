from environment import Environment
import restraining_bolt as rb
from automaton import DFA

if __name__ == '__main__':
    env = Environment('../layouts/cycle.txt', risk=1)
    automata = []

    # The merchant must not visit the market until the sun has set"
    # viol: F(at_home & !sundown U at_market)
    def trans0a(l):
        if 'at_market' in l and 'at_home' in l:
            return 2
        elif 'at_home' in l and 'sundown' not in l and 'at_market' not in l:
            return 1
        else:
            return 0
    def trans1a(l):
        if 'at_market' in l:
            return 2
        elif 'sundown' not in l:
            return 1
        else:
            return 0
    def trans2a(l):
        return 2


    transa = {}
    transa[0] = trans0a
    transa[1] = trans1a
    transa[2] = trans2a
    automatona = DFA([0, 1, 2], -500, trans=transa, final=[2])
    automata.append(automatona)

    agent = rb.RestrainingBoltNormAgent(env, specs=automata, alpha=0.5, epsilon=0.25, gamma=0.9, ntrain=5000)
    agent.train()
    a, i, b, c = agent.test(rec=True)