from pomegranate import *

# Generate synthetic data for BHMM training
numpy.random.seed(0)

# Create sequences of observations for training the BHMM
num_sequences = 5
seq_length = 50
sequences = []
for _ in range(num_sequences):
    seq = numpy.random.randn(seq_length, 2)
    sequences.append(seq)

# Define the states for the BHMM
states = [State(NormalDistribution(mu, sigma)) for mu, sigma in zip([-3, 0, 3], [1, 1, 1])]
start_probabilities = [1.0, 0.0, 0.0]  # Starting probabilities for the HMM states
transitions = numpy.array([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.9]])  # Transition probabilities

# Create the Bayesian Hidden Markov Model (BHMM) using the defined states and transition probabilities
bhmm = BayesianHMM.from_matrix(transitions, states, start_probabilities)

# Fit the BHMM to the generated sequences
bhmm.fit(sequences)

# Generate a new sequence using the trained BHMM
new_sequence = bhmm.sample(seq_length)

print("Generated sequence using Bayesian HMM:")
print(new_sequence)
