import numpy as np
import time as time
import cvxpy as cp
try:
    from Queue import PriorityQueue
except:
    from queue import PriorityQueue

class Kidney_Paired_Donation:
  def __init__(self, num_pairs, probs):
    # Initialize global variables for blood types, respective probabilities, compatibility, and the number of pairs to be matched
    self.num_pairs = num_pairs
    self.probs = probs
    self.types = ['A', 'B', 'AB', 'O']
    self.compats = {'A': ['A', 'AB'], 'B': ['B', 'AB'], 'AB': ['AB'], 'O': ['O', 'A', 'B', 'AB']}

    # Populate our randomly generated list of patient-donor pairs
    self.p_d_pairs = []
    for i in range(self.num_pairs):
      patient = np.random.choice(self.types, p=self.probs)
      donor = np.random.choice(self.types, p=self.probs)
      self.p_d_pairs.append([i, [patient, donor]])
    print(f"Patient-Donor Pairs: {self.p_d_pairs}")

    # Varify that user inputs are valid
    if len(self.probs) != len(self.types) or sum(self.probs) != 1 or num_pairs <= 1:
      raise ValueError

  # Define function for cycles with length bound 2
  def swaps_only(self):
    # Define edges between pairs who are compatible with one another
    edges = []
    count = 0
    for i in range(self.num_pairs):
      for j in range(i+1, self.num_pairs):
        pair1, pair2 = self.p_d_pairs[i], self.p_d_pairs[j]
        if (pair2[1][0] in self.compats[pair1[1][1]]) and (pair1[1][0] in self.compats[pair2[1][1]]):
          edges.append([count, [pair1, pair2]])
          count += 1

    # Now we define the variables, constraints, and objective for our IP and solve
    y = cp.Variable(len(edges), boolean=True)
    objective = cp.problems.objective.Maximize(cp.sum(y))
    constraints = []

    # Populate our constraints by iterating through each pair
    for pair in self.p_d_pairs:
      # Filter down to edges that include this pair
      potential_swaps = list(filter(lambda x: pair in x[1], edges))

      # Identify the indices of the edges that can be activated
      edge_ids = list(map(lambda x: x[0], potential_swaps))

      # Sum the activation variables for each edge and make sure they are no more than 1
      var_sum = 0
      for idx in edge_ids:
        var_sum += y[idx]
      constraints.append(var_sum <= 1)

    # Solve our LP
    prob = cp.Problem(objective, constraints)
    edges_activated = prob.solve()
    cardinality = 2 * edges_activated
    assignment = y.value
    assigned = []
    for i in range(len(edges)):
      if assignment[i] == 1:
        assigned.append([edges[i][1][0][0], edges[i][1][1][0]])

    return cardinality, assigned

  # Define function for cycles of unbounded length
  def unbound_cycles(self):
    # Define directed edges from one pair to another if they can make a donation to them
    edges = []
    count = 0
    for i in range(self.num_pairs):
      for j in range(self.num_pairs):
        pair1, pair2 = self.p_d_pairs[i], self.p_d_pairs[j]
        if (pair2[1][0] in self.compats[pair1[1][1]]) and (i != j):
          edges.append([count, [pair1, pair2]])
          count += 1

    # Now we define the variables, constraints, and objective for our LP and solve
    y = cp.Variable(len(edges), boolean=True)
    objective = cp.problems.objective.Maximize(cp.sum(y))
    constraints = []

    # Populate the constraints by iterating through each pair
    for pair in self.p_d_pairs:
      # Filter edges down to potential donations
      potential_donations = list(filter(lambda x: x[1][0] == pair, edges))

      # Filter the rest down to potential receptions
      potential_receptions = list(filter(lambda x: x[1][1] == pair, edges))

      # Identify the indices of the donations that can be made
      don_ids = list(map(lambda x: x[0], potential_donations))

      # Identify the indices of the donations that can be received
      rec_ids = list(map(lambda x: x[0], potential_receptions))

      # Sum the activation variables to ensure that only one donation is made, and if so they also receive a donation
      don_sum = 0
      for idx in don_ids:
        don_sum += y[idx]
      constraints.append(don_sum <= 1)

      rec_sum = 0
      for idx in rec_ids:
        rec_sum += y[idx]
      constraints.append(rec_sum <= 1)

      constraints.append(don_sum == rec_sum)

    # Solve our LP
    prob = cp.Problem(objective, constraints)
    cardinality = prob.solve()
    assignment = y.value
    assigned = []
    for i in range(len(edges)):
      if assignment[i] == 1:
        assigned.append([edges[i][1][0][0], edges[i][1][1][0]])

    return cardinality, assigned

    # Try approximating with branch and bound
    # Show varying runtimes for different amounts of patient-donor pairs
    # Branch and bound can be done using cvxpy


  # Define function for cycles of bound length 3
  def cycles_of_n(self, n):
    # We'll be using the networkx library to define our graph and identify cycles
    import networkx as nx

    # Define directed edges from one pair to another if they can make a donation to them
    edges = []
    count = 0
    for i in range(self.num_pairs):
      for j in range(self.num_pairs):
        pair1, pair2 = self.p_d_pairs[i], self.p_d_pairs[j]
        if (pair2[1][0] in self.compats[pair1[1][1]]) and (i != j):
          edges.append([count, [pair1, pair2]])
          count += 1

    # Define our graph and populate it with nodes and edges
    # Filter our edges down to the pair IDs and add them to our graph
    print(f"num of edges: {len(edges)}")
    edge_pair_ids = list(map(lambda x: (x[1][0][0], x[1][1][0]), edges))
    G = nx.DiGraph(edge_pair_ids)

    # Identify all the cycles in our grpah and filter them down to those of length n and less
    cycles = list(nx.simple_cycles(G))
    filt_cycles = list(filter(lambda x: len(x) <= n, cycles))

    lens = []
    for cyc in filt_cycles:
      lens.append(len(cyc))

    # Now we define the variables, constraints, and objective for our LP and solve
    y = cp.Variable(len(filt_cycles), boolean=True)
    objective = cp.problems.objective.Maximize(cp.sum(y @ lens))
    constraints = []

    # Iterate through each pair and make sure they are part of at most 1 cycle
    for pair in self.p_d_pairs:
      pair_id = pair[0]
      var_sum = 0
      for i in range(len(filt_cycles)):
        cycle = filt_cycles[i]
        if pair_id in cycle:
          var_sum += y[i]
      constraints.append(var_sum <= 1)

    # Solve our LP via Branch and Bound
    start = time.time()
    prob = cp.Problem(objective, constraints)
    prob.solve()
    print(f"Time: {time.time() - start}")
    cardinality = 0
    assignment = y.value
    assigned = []
    for i in range(len(filt_cycles)):
      if assignment[i] == 1:
        cardinality += len(filt_cycles[i])
        assigned.append(filt_cycles[i])

    return cardinality, assigned


if __name__ == "__main__":
  KPD = Kidney_Paired_Donation(10, [.1, .3, .4, .2])

  # Testing Swaps Only
  cardinality, assignment = KPD.swaps_only()

  print(f"SWAPS ONLY\n")
  print(f"----------------------\n")
  print(f"Maximum Cardinality of Swaps: {cardinality}\n")
  print(f"Swaps Made: {assignment}\n")

  # Testing Unbound
  cardinality, assignment = KPD.unbound_cycles()

  print(f"UNBOUND CYCLES\n")
  print(f"----------------------\n")
  print(f"Maximum Cardinality of Unbound Cycles Donations: {cardinality}\n")
  print(f"Donations Made: {assignment}\n")

  # Testing Bound at n = 3
  cardinality, assignment = KPD.cycles_of_n(4)

  print(f"CYCLES OF LENGTH N = 3\n")
  print(f"----------------------\n")
  print(f"Maximum Cardinality of Cycles Bound at N: {cardinality}\n")
  print(f"Cycles Traded On: {assignment}\n")
