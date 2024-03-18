from pomegranate import *

# Generate synthetic data for GMM training
numpy.random.seed(0)
data = numpy.random.randn(1000, 2)

# Train a Gaussian Mixture Model (GMM) on the synthetic data
gmm = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=3, X=data)

# Generate sequences of observations from the trained GMM
seq_length = 50
num_sequences = 5
sequences = []
for _ in range(num_sequences):
    seq = []
    for _ in range(seq_length):
        seq.append(gmm.sample()[0])
    sequences.append(seq)

# Define the states for the Hidden Markov Model (HMM)
states = [State(NormalDistribution(mu, sigma)) for mu, sigma in zip([-3, 0, 3], [1, 1, 1])]
start_probabilities = [1.0, 0.0, 0.0]  # Starting probabilities for the HMM states
transitions = numpy.array([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.9]])  # Transition probabilities

# Create the Hidden Markov Model (HMM) using the defined states and transition probabilities
hmm = HiddenMarkovModel.from_matrix(transitions, states, start_probabilities)

# Fit the HMM to the generated sequences (using the GMM outputs as observations)
hmm.fit(sequences)

# Generate a new sequence using the trained HMM
new_sequence = hmm.sample(seq_length)

print("Generated sequence using HMM:")
print(new_sequence)
