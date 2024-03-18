from pomegranate import *

# Generate synthetic data for Bayesian GMM training
numpy.random.seed(0)
data = numpy.random.randn(1000, 2)

# Create a Bayesian Gaussian Mixture Model (BGMM) and fit it to the data
bgmm = BayesianGaussianMixture(n_components=3, n_init=10, max_iter=1000, tol=1e-6, weight_concentration_prior=1e-3)
bgmm.fit(data)

# Generate sequences of observations from the BGMM
seq_length = 50
num_sequences = 5
sequences = []
for _ in range(num_sequences):
    seq = []
    for _ in range(seq_length):
        seq.append(bgmm.sample()[0])
    sequences.append(seq)

# Create the Bayesian Hidden Markov Model (BHMM) using the BGMM outputs as observations
bhmm = BayesianHMM.from_samples(MultivariateGaussianDistribution, n_components=3, X=sequences)

# Generate a new sequence using the trained BHMM
new_sequence = bhmm.sample(seq_length)

print("Generated sequence using Bayesian HMM:")
print(new_sequence)
