# Parse the two blocks from the user's pasted summary and plot Euclidean Distance Error for RBPF vs LLM.
import re
import matplotlib.pyplot as plt
import pandas as pd

text = """
================== Entry 1 ==================
Ground Truth: {'blue': (4, 5), 'green': (6, 7), 'grey': (7, 10), 'purple': (8, 4), 'red': (8, 10)}
LLM's Map   : {'blue': (4, 5), 'green': (6, 7), 'grey': (7, 9), 'purple': (8, 4), 'red': (8, 9)}
----------------------------------------
Overall Accuracy (Jaccard): 42.86%
Goal Accuracy (Avg. Correctness): 60.00%
Euclidean Distance Error (for matched goals): 0.40 units

================== Entry 2 ==================
Ground Truth: {'grey': (1, 5), 'blue': (2, 2), 'green': (6, 1), 'red': (8, 1), 'purple': (9, 3)}
LLM's Map   : {'blue': (2, 2), 'green': (6, 1), 'grey': (1, 5), 'purple': (9, 3), 'red': (8, 1)}
----------------------------------------
Overall Accuracy (Jaccard): 100.00%
Goal Accuracy (Avg. Correctness): 100.00%
Euclidean Distance Error (for matched goals): 0.00 units

================== Entry 3 ==================
Ground Truth: {'grey': (3, 6), 'green': (3, 10), 'red': (6, 1), 'blue': (6, 10), 'yellow': (9, 2)}
LLM's Map   : {'blue': (6, 9), 'green': (2, 9), 'grey': (3, 6), 'red': (6, 1), 'yellow': (9, 2)}
----------------------------------------
Overall Accuracy (Jaccard): 42.86%
Goal Accuracy (Avg. Correctness): 60.00%
Euclidean Distance Error (for matched goals): 0.48 units

================== Entry 4 ==================
Ground Truth: {'green': (2, 9), 'red': (3, 3), 'blue': (6, 10), 'grey': (7, 2), 'purple': (7, 3)}
LLM's Map   : {'blue': (5, 9), 'green': (1, 9), 'grey': (7, 2), 'purple': (7, 3), 'red': (3, 3)}
----------------------------------------
Overall Accuracy (Jaccard): 42.86%
Goal Accuracy (Avg. Correctness): 60.00%
Euclidean Distance Error (for matched goals): 0.48 units

================== Entry 5 ==================
Ground Truth: {'green': (3, 2), 'red': (4, 9), 'purple': (7, 3), 'grey': (7, 9), 'blue': (8, 7)}
LLM's Map   : {'blue': (8, 7), 'green': (3, 2), 'grey': (7, 9), 'purple': (7, 3), 'red': (4, 9)}
----------------------------------------
Overall Accuracy (Jaccard): 100.00%
Goal Accuracy (Avg. Correctness): 100.00%
Euclidean Distance Error (for matched goals): 0.00 units

================== Entry 6 ==================
Ground Truth: {'blue': (1, 2), 'green': (5, 9), 'purple': (6, 2), 'yellow': (6, 6), 'red': (9, 2)}
LLM's Map   : {'blue': (1, 2), 'green': (5, 9), 'purple': (6, 2), 'red': (8, 2), 'yellow': (6, 6)}
----------------------------------------
Overall Accuracy (Jaccard): 66.67%
Goal Accuracy (Avg. Correctness): 80.00%
Euclidean Distance Error (for matched goals): 0.20 units

================== Entry 7 ==================
Ground Truth: {'purple': (1, 3), 'red': (1, 6), 'grey': (3, 3), 'yellow': (6, 7), 'blue': (7, 1)}
LLM's Map   : {'blue': (7, 1), 'green': (9, 8), 'grey': (3, 3), 'purple': (1, 3), 'red': (1, 6), 'yellow': (6, 7)}
----------------------------------------
Overall Accuracy (Jaccard): 83.33%
Goal Accuracy (Avg. Correctness): 100.00%
Euclidean Distance Error (for matched goals): 0.00 units

================== Entry 8 ==================
Ground Truth: {'red': (3, 5), 'yellow': (4, 2), 'blue': (5, 9), 'purple': (8, 5), 'grey': (9, 9)}
LLM's Map   : {'blue': (5, 9), 'green': (0, 0), 'grey': (9, 9), 'purple': (8, 5), 'red': (3, 5), 'yellow': (4, 2)}
----------------------------------------
Overall Accuracy (Jaccard): 83.33%
Goal Accuracy (Avg. Correctness): 100.00%
Euclidean Distance Error (for matched goals): 0.00 units

================== Entry 9 ==================
Ground Truth: {'red': (1, 5), 'yellow': (2, 10), 'green': (3, 3), 'purple': (6, 8), 'blue': (10, 9)}
LLM's Map   : {'blue': (5, 0), 'green': (2, 3), 'purple': (6, 7), 'red': (0, 5), 'yellow': (1, 9)}
----------------------------------------
Overall Accuracy (Jaccard): 0.00%
Goal Accuracy (Avg. Correctness): 0.00%
Euclidean Distance Error (for matched goals): 2.94 units

================== Entry 10 ==================
Ground Truth: {'blue': (2, 1), 'yellow': (3, 8), 'grey': (7, 5), 'purple': (9, 9), 'red': (10, 9)}
LLM's Map   : {'blue': (2, 1), 'green': (0, 9), 'grey': (7, 5), 'red': (9, 9), 'yellow': (3, 8)}
----------------------------------------
Overall Accuracy (Jaccard): 42.86%
Goal Accuracy (Avg. Correctness): 60.00%
Euclidean Distance Error (for matched goals): 0.25 units

================== FINAL SUMMARY ==================
Analyzed 10 entries.
Average Overall Accuracy (Jaccard): 60.48%
Average Goal-Only Accuracy        : 72.00%================== Entry 1 ==================
Ground Truth: {'blue': (4, 5), 'green': (6, 7), 'grey': (7, 10), 'purple': (8, 4), 'red': (8, 10)}
LLM's Map   : {'blue': (4, 5), 'green': (6, 5), 'purple': (9, 2), 'red': (9, 8), 'grey': (8, 8)}
----------------------------------------
Overall Accuracy (Jaccard): 11.11%
Goal Accuracy (Avg. Correctness): 20.00%
Euclidean Distance Error (for matched goals): 1.74 units

================== Entry 2 ==================
Ground Truth: {'grey': (1, 5), 'blue': (2, 2), 'green': (6, 1), 'red': (8, 1), 'purple': (9, 3)}
LLM's Map   : {'green': (6, 1), 'red': (8, 1), 'blue': (2, 2), 'purple': (9, 3), 'grey': (1, 5)}
----------------------------------------
Overall Accuracy (Jaccard): 100.00%
Goal Accuracy (Avg. Correctness): 100.00%
Euclidean Distance Error (for matched goals): 0.00 units

================== Entry 3 ==================
Ground Truth: {'grey': (3, 6), 'green': (3, 10), 'red': (6, 1), 'blue': (6, 10), 'yellow': (9, 2)}
LLM's Map   : {'blue': (6, 10), 'green': (4, 12), 'grey': (8, 12), 'red': (5, 7), 'yellow': (7, 8)}
----------------------------------------
Overall Accuracy (Jaccard): 11.11%
Goal Accuracy (Avg. Correctness): 20.00%
Euclidean Distance Error (for matched goals): 4.49 units

================== Entry 4 ==================
Ground Truth: {'green': (2, 9), 'red': (3, 3), 'blue': (6, 10), 'grey': (7, 2), 'purple': (7, 3)}
LLM's Map   : {'purple': (7, 3), 'grey': (7, 2), 'red': (3, 1), 'blue': (4, -2), 'green': (2, 1)}
----------------------------------------
Overall Accuracy (Jaccard): 25.00%
Goal Accuracy (Avg. Correctness): 40.00%
Euclidean Distance Error (for matched goals): 4.43 units

================== Entry 5 ==================
Ground Truth: {'green': (3, 2), 'red': (4, 9), 'purple': (7, 3), 'grey': (7, 9), 'blue': (8, 7)}
LLM's Map   : {'blue': (8, 7), 'purple': (7, 3), 'green': (3, 2), 'red': (4, 9)}
----------------------------------------
Overall Accuracy (Jaccard): 80.00%
Goal Accuracy (Avg. Correctness): 80.00%
Euclidean Distance Error (for matched goals): 0.00 units

================== Entry 6 ==================
Ground Truth: {'blue': (1, 2), 'green': (5, 9), 'purple': (6, 2), 'yellow': (6, 6), 'red': (9, 2)}
LLM's Map   : {'green': (5, 9), 'yellow': (6, 6), 'blue': (1, 2), 'purple': (6, 2), 'red': (9, 2)}
----------------------------------------
Overall Accuracy (Jaccard): 100.00%
Goal Accuracy (Avg. Correctness): 100.00%
Euclidean Distance Error (for matched goals): 0.00 units

================== Entry 7 ==================
Ground Truth: {'purple': (1, 3), 'red': (1, 6), 'grey': (3, 3), 'yellow': (6, 7), 'blue': (7, 1)}
LLM's Map   : {'red': (1, 6), 'purple': (1, 3), 'grey': (3, 3), 'blue': (7, 1), 'yellow': (6, 7)}
----------------------------------------
Overall Accuracy (Jaccard): 100.00%
Goal Accuracy (Avg. Correctness): 100.00%
Euclidean Distance Error (for matched goals): 0.00 units

================== Entry 8 ==================
Ground Truth: {'red': (3, 5), 'yellow': (4, 2), 'blue': (5, 9), 'purple': (8, 5), 'grey': (9, 9)}
LLM's Map   : {'yellow': (4, 2), 'purple': (8, 5), 'blue': (5, 9), 'grey': (9, 9), 'red': (3, 5)}
----------------------------------------
Overall Accuracy (Jaccard): 100.00%
Goal Accuracy (Avg. Correctness): 100.00%
Euclidean Distance Error (for matched goals): 0.00 units

================== Entry 9 ==================
Ground Truth: {'red': (1, 5), 'yellow': (2, 10), 'green': (3, 3), 'purple': (6, 8), 'blue': (10, 9)}
LLM's Map   : {'purple': (6, 8), 'blue': (10, 9), 'yellow': (2, 10), 'red': (1, 5), 'green': (3, 3)}
----------------------------------------
Overall Accuracy (Jaccard): 100.00%
Goal Accuracy (Avg. Correctness): 100.00%
Euclidean Distance Error (for matched goals): 0.00 units

================== Entry 10 ==================
Ground Truth: {'blue': (2, 1), 'yellow': (3, 8), 'grey': (7, 5), 'purple': (9, 9), 'red': (10, 9)}
LLM's Map   : {'yellow': (3, 7), 'grey': (7, 4), 'blue': (8, 4), 'purple': (9, 8), 'red': (10, 8)}
----------------------------------------
Overall Accuracy (Jaccard): 0.00%
Goal Accuracy (Avg. Correctness): 0.00%
Euclidean Distance Error (for matched goals): 2.14 units

================== FINAL SUMMARY ==================
Analyzed 10 entries.
Average Overall Accuracy (Jaccard): 62.72%
Average Goal-Only Accuracy        : 66.00%
"""

# Find the two "Entry 1" anchors to split into RBPF block and LLM block
anchors = [m.start() for m in re.finditer(r"^=+ Entry 1 =+\s*$", text, flags=re.M)]
if len(anchors) >= 2:
    rbpf_block = text[anchors[0]:anchors[1]]
    llm_block = text[anchors[1]:]
else:
    # Fallback: assume the first 10 entries are RBPF and the next 10 are LLM
    rbpf_block = text
    llm_block = ""

def extract_distances(block: str):
    # Capture "Euclidean Distance Error ...: X units"
    pat = re.compile(r"Euclidean Distance Error.*?:\s*([0-9.]+)\s*units", re.S)
    vals = [float(x) for x in pat.findall(block)]
    return vals

rbpf_dist = extract_distances(rbpf_block)
llm_dist = extract_distances(llm_block)

# Ensure same length (truncate to the shorter length if needed)
n = min(len(rbpf_dist), len(llm_dist))
rbpf_dist = rbpf_dist[:n]
llm_dist = llm_dist[:n]
x = list(range(1, n + 1))

# Build a small dataframe for reference
df = pd.DataFrame({
    "Case": x,
    "RBPF_Euclid": rbpf_dist,
    "LLM_Euclid": llm_dist
})
print(rbpf_dist)
print(llm_dist)

# Plot both series on one chart

plt.figure(figsize=(8, 4.5))
plt.plot(x, rbpf_dist, marker='o', label='RBPF')
plt.plot(x, llm_dist, marker='s', label='LLM')
plt.xlabel("Case #")
plt.ylabel("Euclidean Distance Error")
plt.title("Euclidean Distance per Case: RBPF vs LLM")
plt.xticks(x)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
out_path = "euclidean_distance_rbpf_vs_llm.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
