# Bandits-using-UCB-algorithm
Thompson Sampling for Bandits using UCB policy

Multi-armed bandits problem, see https://en.wikipedia.org/wiki/Multi-armed_bandit 

Suppose there are K slot machines, each of them provide reward amount specific to its own distribution. A gambler has to decide which arm to lift each time and at what order to maximize the total rewards he/she will recieve.

The UCB algorithm specifies at time t, we pull arm a_t that has the maximum value of (observed_mean reward of a + UCB confidence bound)

This program assumes K = 5, and the reward each arm gives subjects to Bernoulli distribution.  If we adopt a Bayes point of view, our prior belief is that the probability of each arm is distributed according to a Beta(1, 1) distribution (i.e. our prior is uniform for each arm).


