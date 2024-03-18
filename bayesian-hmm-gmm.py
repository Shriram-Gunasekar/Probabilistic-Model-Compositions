from pomegranate import *

# Generate synthetic data for Bayesian GMM training
numpy.random.seed(0)
data = numpy.random.randn(1000, 2)

# Create a Bayesian Gaussian Mixture Model (GMM) and fit it to the data
bgmm = BayesianGaussianMixture(n_components=3, n_init=10, max_iter=1000, tol=1e-6, weight_concentration_prior=1e-3)
bgmm.fit(data)

# Generate sequences of observations from the Bayesian GMM
seq_length = 50
num_sequences = 5
sequences = []
for _ in range(num_sequences):
    seq = []
    for _ in range(seq_length):
        seq.append(bgmm.sample()[0])
    sequences.append(seq)

# Define the states for the Bayesian Hidden Markov Model (HMM)
states = [State(NormalDistribution(mu, sigma)) for mu, sigma in zip([-3, 0, 3], [1, 1, 1])]
start_probabilities = [1.0, 0.0, 0.0]  # Starting probabilities for the HMM states
transitions = numpy.array([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.9]])  # Transition probabilities

# Create the Bayesian Hidden Markov Model (HMM) using the defined states and transition probabilities
bhmm = BayesianHMM.from_matrix(transitions, states, start_probabilities)

# Fit the Bayesian HMM to the generated sequences (using the Bayesian GMM outputs as observations)
bhmm.fit(sequences)

# Generate a new sequence using the trained Bayesian HMM
new_sequence = bhmm.sample(seq_length)

print("Generated sequence using Bayesian HMM:")
print(new_sequence)
