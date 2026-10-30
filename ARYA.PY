"""
Unit 1: Protein Sequence Analysis and Linux-based Biological Data Handling
Protein-Protein Interaction Network Analysis - Mini Demonstration
 
This beginner-friendly program reads protein sequences from a FASTA file,
calculates basic sequence properties, compares sequences by global
alignment (percent identity), reads a STRING-style interaction table and
draws a small protein network.
 
Requirements:
    pip install numpy matplotlib networkx
Run:
    python unit1_ppi_sequence_analysis.py
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
 
# Average residue masses (Da) and Kyte-Doolittle hydropathy values
MASS = {"A": 71.08, "R": 156.19, "N": 114.10, "D": 115.09, "C": 103.14,
        "E": 129.12, "Q": 128.13, "G": 57.05, "H": 137.14, "I": 113.16,
        "L": 113.16, "K": 128.17, "M": 131.19, "F": 147.18, "P": 97.12,
        "S": 87.08, "T": 101.10, "W": 186.21, "Y": 163.18, "V": 99.13}
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "E": -3.5,
      "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
      "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
      "Y": -1.3, "V": 4.2}
 
 
def read_fasta(path):
    """Read a FASTA file and return {id: sequence}."""
    seqs, name = {}, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = ""
            elif name:
                seqs[name] += line.upper()
    return seqs
 
 
def molecular_weight(seq):
    return sum(MASS[a] for a in seq) + 18.02          # add one water
 
 
def gravy(seq):
    """Grand average of hydropathy (Kyte-Doolittle)."""
    return sum(KD[a] for a in seq) / len(seq)
 
 
def global_identity(a, b, match=1, mismatch=-1, gap=-2):
    """Needleman-Wunsch global alignment; returns percent identity."""
    n, m = len(a), len(b)
    S = np.zeros((n + 1, m + 1))
    S[:, 0] = np.arange(n + 1) * gap
    S[0, :] = np.arange(m + 1) * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = S[i-1, j-1] + (match if a[i-1] == b[j-1] else mismatch)
            S[i, j] = max(d, S[i-1, j] + gap, S[i, j-1] + gap)
    i, j, same, length = n, m, 0, 0                  # traceback
    while i > 0 or j > 0:
        length += 1
        if i > 0 and j > 0 and S[i, j] == S[i-1, j-1] + (match if a[i-1] == b[j-1] else mismatch):
            same += a[i-1] == b[j-1]
            i, j = i - 1, j - 1
        elif i > 0 and S[i, j] == S[i-1, j] + gap:
            i -= 1
        else:
            j -= 1
    return 100.0 * same / length
 
 
def main():
    seqs = read_fasta("sample_protein.fasta")
    names = list(seqs)
 
    print("PROTEIN-PROTEIN INTERACTION NETWORK ANALYSIS")
    print("UNIT 1: PROTEIN SEQUENCE ANALYSIS")
    print("-" * 66)
    print(f"{'Protein':<16}{'Length':>7}{'MW (Da)':>11}{'GRAVY':>8}{'Top residue':>13}")
    for n in names:
        s = seqs[n]
        top = max(set(s), key=s.count)
        print(f"{n:<16}{len(s):>7}{molecular_weight(s):>11.1f}{gravy(s):>8.2f}"
              f"{top + ' (' + str(s.count(top)) + ')':>13}")
 
    # Pairwise identity matrix
    k = len(names)
    ident = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ident[i, j] = 100.0 if i == j else global_identity(seqs[names[i]], seqs[names[j]])
    print("\nMost similar pairs (global alignment % identity):")
    pairs = sorted(((ident[i, j], names[i], names[j])
                    for i in range(k) for j in range(i + 1, k)), reverse=True)[:3]
    for p, a, b in pairs:
        print(f"  {a} vs {b}: {p:.1f}%")
 
    # Read interaction table and build network
    G = nx.Graph()
    with open("interactions.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            G.add_edge(row["protein1"], row["protein2"], weight=float(row["score"]))
    deg = dict(G.degree())
    hub = max(deg, key=deg.get)
    print(f"\nNetwork: {G.number_of_nodes()} proteins, {G.number_of_edges()} interactions")
    print(f"Hub protein (highest degree): {hub} with {deg[hub]} interactions")
    print("Note: demo data only - not real experimental interactions.")
 
    # Figure: composition, identity heat map and network
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Unit 1 Output: Protein Sequence Analysis and Interaction Network",
                 fontsize=14, fontweight="bold")
    s0 = seqs[names[0]]
    letters = sorted(MASS)
    ax[0].bar(letters, [s0.count(a) for a in letters], color="#1F4E79")
    ax[0].set_title(f"Amino acid composition\n({names[0]})")
    ax[0].set_xlabel("Amino acid"); ax[0].set_ylabel("Count")
    im = ax[1].imshow(ident, cmap="Blues", vmin=0, vmax=100)
    ax[1].set_xticks(range(k)); ax[1].set_yticks(range(k))
    ax[1].set_xticklabels([n[:3] for n in names]); ax[1].set_yticklabels([n[:3] for n in names])
    ax[1].set_title("Pairwise % identity"); fig.colorbar(im, ax=ax[1], fraction=0.046)
    pos = nx.spring_layout(G, seed=3)
    nx.draw_networkx(G, pos, ax=ax[2], node_color="#9DC3E6",
                     node_size=[300 + 250 * deg[n] for n in G], font_size=7,
                     width=[3 * G[u][v]["weight"] for u, v in G.edges])
    ax[2].set_title("Interaction network (node size = degree)"); ax[2].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("unit1_ppi_output.png", dpi=160, bbox_inches="tight")
    plt.show()
 
 
if __name__ == "__main__":
     main()