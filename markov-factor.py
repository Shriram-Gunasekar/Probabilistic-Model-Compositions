from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import TabularCPD

# Define the factor graph (Bayesian network)
factor_graph = FactorGraph()

# Define the variables and their CPDs
A = ['A']
B = ['B']
C = ['C']
D = ['D']
E = ['E']

cpd_A = TabularCPD(variable=A, variable_card=2, values=[[0.6], [0.4]])
cpd_B = TabularCPD(variable=B, variable_card=2, values=[[0.3], [0.7]])
cpd_C = TabularCPD(variable=C, variable_card=2, values=[[0.8, 0.3], [0.2, 0.7]], evidence=[A], evidence_card=[2])
cpd_D = TabularCPD(variable=D, variable_card=2, values=[[0.9, 0.6], [0.1, 0.4]], evidence=[A], evidence_card=[2])
cpd_E = TabularCPD(variable=E, variable_card=2, values=[[0.7, 0.1], [0.3, 0.9]], evidence=[B, C], evidence_card=[2, 2])

factor_graph.add_nodes_from([A, B, C, D, E])
factor_graph.add_edges_from([(A, C), (A, D), (B, E), (C, E)])

factor_graph.add_cpds(cpd_A, cpd_B, cpd_C, cpd_D, cpd_E)

# Define a simple Markov chain transition matrix
transition_matrix = [
    [0.7, 0.3],
    [0.4, 0.6]
]

# Define the initial state probabilities
initial_state_probs = [0.6, 0.4]

# Function to perform a single step in the Markov chain
def step_markov_chain(current_state):
    next_state = numpy.random.choice([0, 1], p=transition_matrix[current_state])
    return next_state

# Simulate a sequence of Markov chain states and query the factor graph at each step
current_state = numpy.random.choice([0, 1], p=initial_state_probs)
for _ in range(5):  # Simulate 5 steps
    next_state = step_markov_chain(current_state)
    print(f"Current Markov Chain State: {current_state}")
    
    # Query the factor graph
    query_variables = [A, B, C, D, E]
    evidence = {A: current_state}
    inference = factor_graph.query(variables=query_variables, evidence=evidence)
    print(inference)
    
    current_state = next_state
