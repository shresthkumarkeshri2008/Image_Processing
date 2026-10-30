"""
Unit 1: DNA Sequence Processing using Python and Linux Commands
Project 6: DNA Sequence Classification for Disease Prediction - Preprocessing Demo
 
What this program does (beginner level, no machine learning yet):
  1. Reads a multi-FASTA file
  2. Cleans each sequence (uppercase, remove spaces)
  3. Validates it (only A, C, G, T allowed; N bases are counted)
  4. Removes sequences that are too short, invalid, or duplicated
  5. Calculates length, nucleotide counts and GC content
  6. Shows reverse complement and transcription of one sequence
  7. Saves clean_sequences.fasta, sequence_summary.csv and a figure
 
Requirements:  pip install matplotlib
Run:           python3 unit1_dna_preprocessing.py
(If data/raw_sequences.fasta is missing, this program creates it automatically.)
"""
 
import csv
import os
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
RAW_FILE = "data/raw_sequences.fasta"
CLEAN_FILE = "output/clean_sequences.fasta"
SUMMARY_FILE = "output/sequence_summary.csv"
FIGURE_FILE = "output/unit1_python_output_figure.png"
MIN_LENGTH = 50
VALID_BASES = set("ACGT")
COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
 
 
def create_sample_data(path):
    """Create a small SYNTHETIC multi-FASTA file (with deliberate errors) if none exists.
    Sequences are random, not real patient or gene data."""
    random.seed(42)
 
    def rand_seq(n, gc=0.5):
        return "".join(random.choices("ACGT", weights=[(1-gc)/2, gc/2, gc/2, (1-gc)/2], k=n))
 
    def wrap(x, w=60):
        return "\n".join(x[i:i+w] for i in range(0, len(x), w))
 
    recs = []
    recs.append(("seq01 sample_gene_A", wrap(rand_seq(240, 0.55))))
    recs.append(("seq02 sample_gene_B", wrap(rand_seq(300, 0.42))))
    recs.append(("seq03 lowercase_example", wrap(rand_seq(180, 0.50).lower())))
    s4 = list(rand_seq(210, 0.48)); s4[40:46] = "NNNNNN"
    recs.append(("seq04 contains_N_bases", wrap("".join(s4))))
    recs.append(("seq05 too_short", rand_seq(25, 0.5)))
    s6 = rand_seq(200, 0.60); s6 = s6[:70] + "XZ" + s6[72:]
    recs.append(("seq06 invalid_characters", wrap(s6)))
    dup = rand_seq(260, 0.52)
    recs.append(("seq07 original_copy", wrap(dup)))
    recs.append(("seq08 duplicate_of_seq07", wrap(dup)))
    s9 = rand_seq(150, 0.45)
    recs.append(("seq09 has_spaces", wrap(s9[:50] + " " + s9[50:100] + " " + s9[100:])))
    recs.append(("seq10 sample_gene_C", wrap(rand_seq(320, 0.58))))
    with open(path, "w") as f:
        for h, seq in recs:
            f.write(f">{h}\n{seq}\n")
    print(f"Sample data was missing, so {path} was created ({len(recs)} records).")
 
 
def read_fasta(path):
    """Return a list of (header, sequence) tuples from a FASTA file."""
    records, header, parts = [], None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(parts)))
                header, parts = line[1:], []
            else:
                parts.append(line)
    if header is not None:
        records.append((header, "".join(parts)))
    return records
 
 
def clean_sequence(seq):
    """Uppercase and remove spaces/tabs."""
    return "".join(seq.split()).upper()
 
 
def gc_content(seq):
    """Percentage of G and C bases."""
    return (seq.count("G") + seq.count("C")) / len(seq) * 100
 
 
def reverse_complement(seq):
    return "".join(COMPLEMENT[b] for b in reversed(seq))
 
 
def transcribe(seq):
    """DNA coding strand -> mRNA (replace T with U)."""
    return seq.replace("T", "U")
 
 
def main():
    # Make sure the folders and sample data exist (works in Colab/Jupyter too)
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    if not os.path.exists(RAW_FILE):
        create_sample_data(RAW_FILE)
 
    records = read_fasta(RAW_FILE)
    print("UNIT 1: DNA SEQUENCE PROCESSING USING PYTHON AND LINUX COMMANDS")
    print("-" * 64)
    print(f"Sequences read from file : {len(records)}")
 
    kept, seen, report = [], set(), []
    for header, raw in records:
        name = header.split()[0]
        seq = clean_sequence(raw)
        n_count = seq.count("N")
        invalid = set(seq) - VALID_BASES - {"N"}
 
        if invalid:
            status = f"REMOVED (invalid letters: {''.join(sorted(invalid))})"
        elif len(seq) < MIN_LENGTH:
            status = f"REMOVED (shorter than {MIN_LENGTH} bp)"
        elif seq in seen:
            status = "REMOVED (duplicate sequence)"
        else:
            seen.add(seq)
            if n_count:
                # N means an unknown base; guessing it would add errors, so drop the sequence
                status = f"REMOVED ({n_count} ambiguous N bases)"
            else:
                status = "KEPT"
                kept.append((name, seq))
        report.append((name, len(seq), status))
        print(f"{name:<7} length={len(seq):<4} {status}")
 
    print("-" * 64)
    print(f"Sequences kept           : {len(kept)}")
    print(f"Sequences removed        : {len(records) - len(kept)}")
 
    # Save clean FASTA and summary table
    with open(CLEAN_FILE, "w") as f:
        for name, seq in kept:
            f.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + "\n")
 
    rows = []
    for name, seq in kept:
        counts = {b: seq.count(b) for b in "ACGT"}
        rows.append([name, len(seq), counts["A"], counts["C"], counts["G"],
                     counts["T"], round(gc_content(seq), 2)])
    with open(SUMMARY_FILE, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["sequence", "length", "A", "C", "G", "T", "GC_percent"])
        w.writerows(rows)
 
    print("\nCLEAN SEQUENCE SUMMARY")
    print(f"{'Sequence':<8}{'Length':>7}{'A':>6}{'C':>6}{'G':>6}{'T':>6}{'GC %':>8}")
    for r in rows:
        print(f"{r[0]:<8}{r[1]:>7}{r[2]:>6}{r[3]:>6}{r[4]:>6}{r[5]:>6}{r[6]:>8.2f}")
 
    # Demonstration of basic sequence operations on the first 30 bases
    demo_name, demo_seq = kept[0]
    piece = demo_seq[:30]
    print(f"\nDEMO ON {demo_name} (first 30 bases)")
    print("DNA               :", piece)
    print("Reverse complement:", reverse_complement(piece))
    print("mRNA (T -> U)     :", transcribe(piece))
 
    # Figure: GC content and length of kept sequences
    names = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("Unit 1 Output: DNA Sequence Preprocessing", fontsize=13, fontweight="bold")
    axes[0].bar(names, [r[6] for r in rows], color="#2E75B6")
    axes[0].axhline(50, color="grey", linestyle="--", linewidth=1)
    axes[0].set_title("GC content of cleaned sequences")
    axes[0].set_ylabel("GC (%)")
    axes[0].tick_params(axis="x", rotation=45)
    all_names = [r[0] for r in report]
    colors = ["#2E75B6" if r[2] == "KEPT" else "#C0504D" for r in report]
    axes[1].bar(all_names, [r[1] for r in report], color=colors)
    axes[1].axhline(MIN_LENGTH, color="grey", linestyle="--", linewidth=1)
    axes[1].set_title("Length of all sequences (red = removed)")
    axes[1].set_ylabel("Length (bp)")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(FIGURE_FILE, dpi=160, bbox_inches="tight")
    print(f"\nSaved: {CLEAN_FILE}, {SUMMARY_FILE}, {FIGURE_FILE}")
 
 
if __name__ == "__main__":
    main()