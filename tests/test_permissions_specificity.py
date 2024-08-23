from environment import Environment
import restraining_bolt as rb
from automaton import DFA

if __name__ == '__main__':
    env = Environment('../layouts/basic.txt', risk=1)
    automata = []

    # The merchant ought not extract wood
    # Viol: F(at_tree & Extract)
    def trans0a(l):
        if 'at_tree' in l and 'Extract' in l:
            return 1
        else:
            return 0

    def trans1a(l):
        return 1

    transa = {}
    transa[0] = trans0a
    transa[1] = trans1a
    automatona = DFA([0, 1], -100, trans=transa, final=[1])
    automata.append(automatona)

    # If the merchant has no wood in its inventory, it is permitted to extract wood.
    # Viol: F(at_tree & !has_wood & Extract)
    def trans0b(l):
        if 'at_tree' in l and 'has_wood' not in l and 'Extract' in l:
            return 1
        else:
            return 0

    def trans1b(l):
        return 1

    transb = {}
    transb[0] = trans0b
    transb[1] = trans1b
    automatonb = DFA([0, 1], 100, trans=transb, final=[1])
    automata.append(automatonb)

    agent = rb.RestrainingBoltNormAgent(env, dist=True, specs=automata, alpha=0.5, epsilon=0.25, gamma=0.9, ntrain=5000)
    agent.train()
    a, i, b, c = agent.test(rec=True)