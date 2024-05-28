import random
import matplotlib.pyplot as plt

def build_pattern_library(patterns):
  """
  Builds a dictionary to store patterns and their continuations.

  Args:
    patterns: A list of tuples, where each tuple represents a pattern
              and its valid continuations.

  Returns:
    A dictionary mapping patterns to a list of possible continuations.
  """
  pattern_library = {}
  for pattern, continuations in patterns:
    pattern_library[pattern] = continuations
  return pattern_library

def generate_sequence(pattern_library, max_length):
  """
  Generates a sequence by iteratively applying pattern rules.

  Args:
    pattern_library: A dictionary mapping patterns to continuations.
    max_length: The maximum desired length of the generated sequence.

  Returns:
    A list representing the generated sequence.
  """
  sequence = []
  current_pattern = tuple()
  while len(sequence) < max_length:
    if current_pattern not in pattern_library:
      break
    continuations = pattern_library[current_pattern]
    next_element = random.choice(continuations)
    sequence.append(next_element)
    current_pattern = tuple(sequence[-2:])  # Consider the last two elements
  return sequence

# Example usage
patterns = [
  ((0,), [0, 1]),
  ((1,), [0]),
]

pattern_library = build_pattern_library(patterns)
sequence = generate_sequence(pattern_library, 20)

# Plot the sequence
plt.plot(sequence)
plt.xlabel("Position in Sequence")
plt.ylabel("Value")
plt.title("Generated Sequence")
plt.show()
