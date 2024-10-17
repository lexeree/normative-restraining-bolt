## Normative Restraining Bolt Experiments

Below are instructions for running the case studies for "Enforcing Norms in Reinforcement Learning Agents via Restraining Bolts".

To generate the trajectory for the "pacifist" normative system run: `python3 test_ctd.py`

To generate the trajectory for the "environmentally friendly" normative system run: `python3 test_permissions_specificity.py`

To generate the trajectory for the "early" normative system run: `python3 test_achievement.py`

To generate the trajectory for the "late" normative system run: `python3 test_maintenance.py`

For all the cases, we have set $\alpha = 0.5$, $\gamma = 0.9$, and $\epsilon = 0.25$. We left the number of training episodes at 5000; for all these cases it is unnecessarily high but we found it necessary for some more complex combinations of norms.

The automata for the case studies were generated [here](http://ltlf2dfa.diag.uniroma1.it/).


