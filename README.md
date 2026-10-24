# Predicting Where a Protein Lives in the Cell with ESM-2

> Predict where a protein lives in the cell from its amino acid sequence, using ESM-2 embeddings and logistic regression.

This is a beginner-friendly machine learning project in the life sciences. It takes a protein sequence and predicts its **subcellular location** (nucleus, mitochondrion, and so on). Everything in this README is written for someone who is new to both machine learning and protein biology.

---

## Table of contents
1. [The goal in one picture](#1-the-goal-in-one-picture)
2. [Background for beginners](#2-background-for-beginners)
3. [The dataset: DeepLoc](#3-the-dataset-deeploc)
4. [The model: ESM-2](#4-the-model-esm-2)
5. [The pipeline, step by step](#5-the-pipeline-step-by-step)
6. [Results](#6-results)
7. [Limitations](#7-limitations)
8. [How to run it](#8-how-to-run-it)
9. [Beginner FAQ](#9-beginner-faq)
10. [Credits and references](#10-credits-and-references)

---

## 1. The goal in one picture

```
protein sequence  ->  ESM-2  ->  vector of numbers  ->  logistic regression  ->  "Mitochondrion"
 "MKTAYIAKQR..."     (translator)    (320 numbers)         (classifier)           (prediction)
```

- **Input:** a protein, written as a string of letters (one letter per amino acid).
- **Output:** a predicted location in the cell, with a probability for each possible location.

There are two stages, and it helps to keep them separate:

| Stage | What it is | Does it learn location? |
|---|---|---|
| **ESM-2** | A pretrained protein language model that turns a sequence into 320 numbers | No. It was never told what a nucleus is. It only learned patterns in protein sequences. |
| **Logistic regression** | A small classifier trained by us on top of those numbers | Yes. This is the part that learns which number patterns mean which location. |

---

## 2. Background for beginners

### What is a protein?
A protein is a chain of building blocks called **amino acids**. There are 20 standard amino acids, and each has a one-letter code (`A C D E F G H I K L M N P Q R S T V W Y`). Written out, a protein is just a string of letters, for example `MKTAYIAKQR...`. Order matters: the same letters in a different order are a different protein.

### What is "subcellular location"?
A cell has compartments (the nucleus, the mitochondria, the membrane, and so on). Each protein works in a particular place, so knowing where a protein lives gives clues about what it does.

### Why can a sequence predict location?
Cells sort proteins using signals written into the sequence itself, for example:
- A short stretch at the start of the protein can send it to the mitochondrion or out of the cell.
- A nuclear localization signal (a short run of positively charged amino acids such as lysine and arginine) gets a protein into the nucleus.
- Long hydrophobic ("oily") stretches mark proteins that sit in membranes.

A model that has read millions of protein sequences can pick up these patterns without being told about them.

### What is a classifier?
A classifier is a program that learns to sort things into categories by studying examples. It is like showing a child 100 photos labeled "cat" or "dog": after a while, they can label a new photo themselves. Here, the "photos" are rows of 320 numbers and the "labels" are cell locations.

---

## 3. The dataset: DeepLoc

We use the **DeepLoc 1.0** dataset from DTU Health Tech. It is a single FASTA file, and each entry is a protein sequence paired with its known location.

- **Size:** 14,004 eukaryotic proteins.
- **Source of labels:** UniProt proteins with experimental evidence of their location (not just computational guesses).
- **Train/test split:** about 80% training and about 20% test. The dataset provides its own split, and we use it instead of a random one. A random split can put near-identical proteins in both sets, which makes the accuracy look better than it really is.

### What one entry looks like

```
>Q9H400 Cell.membrane-M test
MGLPVSWAPPALWVLGCCALLLSLWALCTACRRPEDAVAPRKRARRQRARLQGSATAAEASLLRRTHLCS...
```

| Piece | Meaning |
|---|---|
| `>` | Marks the start of a new entry (FASTA format). Not part of the data. |
| `Q9H400` | The UniProt accession, the protein's ID. You can paste it into uniprot.org to look the protein up. |
| `Cell.membrane` | The **location label**. This is the answer the classifier learns to predict. |
| `-M` | Membrane flag: `M` = membrane-bound, `S` = soluble, `U` = unknown. |
| `test` | This protein is in the **held-out test set**. The classifier never trains on it. |
| The letters below | The protein sequence, one letter per amino acid. This one is 295 amino acids long. |

### The 10 location classes and their counts

| Location | Proteins |
|---|---|
| Nucleus | 4,043 |
| Cytoplasm | 2,688 |
| Extracellular | 1,973 |
| Mitochondrion | 1,510 |
| Cell membrane | 1,340 |
| Endoplasmic reticulum | 862 |
| Plastid (includes chloroplast) | 757 |
| Golgi apparatus | 356 |
| Lysosome/Vacuole | 321 |
| Peroxisome | 154 |

The classes are **very uneven**. Nucleus has about 26 times more proteins than peroxisome. A model that always answered "Nucleus" would be right about 29% of the time, which is why we always compare against a baseline (see [step 5.5](#55-classify-and-evaluate)).

### Why there is no "ribosome" class
These labels are organelle-level compartments. A ribosome is a protein-RNA machine, not a compartment, so ribosomal proteins fall under cytoplasm (or nucleolus in finer-grained schemes).

---

## 4. The model: ESM-2

**ESM-2** is a protein language model from Meta AI ("ESM" stands for Evolutionary Scale Modeling). It works like a text language model, but it reads protein sequences instead of sentences.

### How it was trained
It was trained with a "fill in the blank" game on millions of natural protein sequences: hide some amino acids and predict them from the surrounding ones. To get good at this, the model has to learn patterns left by evolution, such as which amino acids appear together and which positions are conserved. Nobody labeled any of this. It learned from raw sequences alone.

### What is a transformer?
A transformer is a neural network that processes a sequence by letting every element look at every other element and decide which ones matter. This mechanism is called **attention**. It is useful for proteins because two amino acids far apart in the sequence can end up next to each other once the protein folds, and attention can link them.

The basic recipe:
1. Break the input into tokens (for ESM-2, one token per amino acid) and turn each into a list of numbers.
2. Add information about each token's position in the sequence, because attention alone has no sense of order.
3. Pass through attention layers, where each token gathers information from the tokens that matter to it.
4. Pass through small feed-forward layers.
5. Repeat steps 3 and 4 many times (ESM-2 8M has 6 layers, and the biggest versions have 48).

### Is ESM-2 really a transformer if it has no decoder?
Yes. "Transformer" refers to the attention-based building block, and the encoder-decoder combination was just the original 2017 arrangement (built for translation). The same blocks are used in three arrangements:

| Type | Examples | What it does |
|---|---|---|
| Encoder-only | BERT, **ESM-2** | Reads the whole input at once in both directions and produces representations |
| Decoder-only | GPT-style chat models | Generates text one token at a time |
| Encoder-decoder | The original transformer, T5 | Reads one sequence and writes another |

ESM-2 is encoder-only. It also has a small "language-model head" on top, used only during pretraining to guess the hidden amino acids. That head is a simple prediction layer, not a decoder.

### What does ESM-2 output?
Like BERT, it produces a **contextual embedding for every token**: one vector per amino acid, where each vector reflects that amino acid in the context of the whole protein. We average these per-amino-acid vectors into **one vector per protein**.

### Model sizes

| Model | Layers | Parameters | Vector size per protein |
|---|---|---|---|
| `esm2_t6_8M_UR50D` | 6 | 8 million | 320 |
| `esm2_t12_35M_UR50D` | 12 | 35 million | 480 |
| `esm2_t30_150M_UR50D` | 30 | 150 million | 640 |

Bigger models have more layers and more room to store what they learned, so they usually give somewhat better embeddings, at the cost of speed and memory. The 8M model runs on a laptop CPU. The 150M model really wants a GPU.

---

## 5. The pipeline, step by step

### 5.1 Load the data
The notebook downloads the DeepLoc FASTA file and reads it into a table (a pandas DataFrame) with **4 columns** and one row per protein:

| Column | Meaning |
|---|---|
| `acc` | The UniProt ID, like `Q9H400` |
| `label` | The location, like `Cell.membrane` |
| `is_test` | True or False: is it in the test set? |
| `seq` | The amino acid string |

The full table has shape `(14004, 4)`: 14,004 rows (proteins) and 4 columns (fields).

### 5.2 Sample a subset
To keep things fast, the notebook drops sequences longer than 1,000 amino acids (ESM-2 reads at most 1,022) and samples 3,000 training and 1,000 test proteins. You can raise `N_TRAIN` and `N_TEST` to use more.

### 5.3 Embed every protein with ESM-2
This is the slow step. For each protein it:
1. **Tokenizes** the sequence: each letter becomes an integer ID, and two special tokens, `<cls>` (start) and `<eos>` (end), are added. Shorter proteins in a batch are padded so all rows have the same length.
2. Runs the tokens through **ESM-2**, getting one vector per position.
3. **Averages** the vectors of the real amino acids only (ignoring padding, `<cls>` and `<eos>`) to get a single vector per protein.

Practical details in the code:
- `@torch.no_grad()` tells PyTorch not to keep the bookkeeping needed for training. We are only using ESM-2, not teaching it, so this saves memory and time.
- Proteins are **sorted by length** before batching, so similar-sized proteins share a batch and little padding is wasted.
- On a GPU the model runs in half precision (`.half()`) for speed.
- Results are **cached** in a `.npz` file, so rerunning the cell is instant.

**ESM-2 does not learn anything in this step.** It is already trained. It just reads your proteins.

### 5.4 What the data looks like after embedding
Two objects are created for the training set:

- **`X_tr`**: the vectors. Shape `(3000, 320)`, meaning 3,000 proteins (rows) with 320 numbers each (columns). It contains *only numbers*. The protein letters are gone from this table.
- **`y_tr`**: the answers. A list of 3,000 location labels, for example `['Cytoplasm', 'Nucleus', 'Mitochondrion', ...]`.

```
X_tr[0] = [ 0.12, -0.80, 0.33, ... ]    y_tr[0] = 'Cytoplasm'
X_tr[1] = [-0.41,  0.27, 0.90, ... ]    y_tr[1] = 'Nucleus'
```

They are linked **by row position**: row 0 of `X_tr` belongs with entry 0 of `y_tr`, and so on. Never shuffle one without the other, or the labels will no longer match the proteins. The original `acc`, `label`, `is_test` and `seq` columns still exist in the `train` and `test` tables; they just are not part of `X_tr`.

No single one of the 320 columns has a human-readable meaning (such as "hydrophobicity"). The information is spread across all of them together.

### 5.5 Classify and evaluate
**The classifier** is a logistic regression. Despite the name, it is a classifier. It handles more than two classes (here, 10) by learning one set of 320 weights per location. For each protein it computes a score per location, converts the scores to probabilities, and picks the highest.

```python
clf = make_pipeline(StandardScaler(),
                    LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
clf.fit(X_tr, y_tr)        # learn from examples
pred = clf.predict(X_te)   # guess locations for unseen proteins
```

What each setting does:
- **`StandardScaler`** rescales each of the 320 columns so no single column dominates.
- **`C=0.1`** keeps the weights modest so the model does not memorize quirks of the training set (a small C means stronger protection against overfitting).
- **`max_iter=2000`** gives training enough rounds to settle.
- **`class_weight="balanced"`** makes mistakes on small classes count more, so the model does not just favor Nucleus.

**How we judge it:**
- **Majority-class baseline:** a dummy "classifier" that always answers with the most common location. Its accuracy is the number to beat.
- **Accuracy:** the fraction of test proteins guessed correctly.
- **Macro-F1:** a score that gives every class equal weight, so doing badly on small classes drags it down. It is harder to fool than accuracy on uneven data.
- **Classification report:** per-location precision (when it says "mitochondrion", how often is it right?), recall (of all true mitochondrial proteins, how many did it find?), and F1.
- **Confusion matrix:** a grid with true locations as rows and predicted locations as columns. The diagonal is correct guesses, and bright off-diagonal cells show which locations the model mixes up.

### 5.6 Visualize with UMAP
Each protein has 320 numbers, which cannot be plotted. **UMAP** squeezes them down to 2 numbers per protein so each protein becomes a dot on a flat chart, colored by its true location. It places proteins that were similar in the 320-number space near each other.

How to use the plot:
- UMAP never sees the labels. If dots of the same color still group together, the embeddings captured location information on their own.
- Heavily mixed colors show locations that are hard to separate (expect nucleus and cytoplasm to overlap).
- It is a sanity check and a way to build intuition, not a score. Distances and cluster sizes are only roughly meaningful, and overlap in 2D does not prove the classifier cannot separate the classes, since it uses all 320 numbers.

The x/y positions on this chart have nothing to do with the transformer. They are picked by UMAP, a separate tool, purely to draw similar proteins close together. (The "positional information" that a transformer uses is something else: it tells the model the order of amino acids within a sequence.)

### 5.7 Compare model sizes
The same pipeline is repeated for ESM-2 8M, 35M and 150M, holding everything else constant, and the results are shown in a table and bar chart. This tells you whether a bigger model actually helps for this task. (Note that `C=0.1` was picked with the 8M model in mind, so trying a few values of `C` per model would make the comparison fairer.)

### 5.8 Bonus: ask the model to fill in blanks
Using ESM-2's language-model head, the notebook hides one amino acid in a real protein and checks whether the model guesses it. Random guessing would be right about 5% of the time (1 in 20), so anything well above that shows the model has learned real protein patterns.

---

## 6. Results

All numbers are on a held-out test set of 1,000 proteins (the dataset's own test split, never used for training), with a classifier trained on 3,000 training proteins. The classifier settings are identical for all three models (`C=0.1`, balanced class weights).

### Model-size comparison

| Model | Vector size | Accuracy | Macro-F1 |
|---|---|---|---|
| Majority-class baseline (always "Nucleus") | n/a | 0.269 | n/a |
| ESM-2 8M + logistic regression | 320 | 0.636 | 0.503 |
| ESM-2 35M + logistic regression | 480 | 0.707 | 0.546 |
| ESM-2 150M + logistic regression | 640 | **0.733** | **0.601** |

**Takeaways**
- Even the smallest model scores about 2.4 times the baseline (0.636 vs. 0.269), so the embeddings clearly carry location information.
- **Bigger helped, consistently.** Going from 8M to 150M parameters raised accuracy by about 10 percentage points (0.636 to 0.733) and macro-F1 from 0.50 to 0.60. The gain from 8M to 35M (+7.1 points) was larger than from 35M to 150M (+2.6 points), a sign of diminishing returns.
- Macro-F1 improved even more than accuracy in relative terms, which suggests the larger models help the harder, smaller classes too.
- For context, the original DeepLoc paper reported about 78% accuracy on these 10 classes using a much more elaborate model. The 150M result (73.3%) gets reasonably close using only frozen embeddings, a linear classifier and about a quarter of the available training data.
- The classifier was configured while looking at the 8M model, and the settings were not retuned for the larger ones. Tuning `C` per model could shift the numbers a little.

### Detailed results for the 8M model

```
majority-class baseline accuracy: 0.269
ESM-2 8M + logistic regression: accuracy=0.636  macro-F1=0.503

                       precision    recall  f1-score   support

        Cell.membrane       0.71      0.60      0.65        92
            Cytoplasm       0.49      0.49      0.49       170
Endoplasmic.reticulum       0.37      0.44      0.40        66
        Extracellular       0.90      0.79      0.84       165
      Golgi.apparatus       0.12      0.27      0.17        22
     Lysosome/Vacuole       0.21      0.27      0.23        26
        Mitochondrion       0.68      0.68      0.68       122
              Nucleus       0.78      0.74      0.76       269
           Peroxisome       0.07      0.25      0.11         4
              Plastid       0.75      0.64      0.69        64

             accuracy                           0.64      1000
            macro avg       0.51      0.52      0.50      1000
         weighted avg       0.67      0.64      0.65      1000
```

**Reading the report**
- **Strongest classes:** Extracellular (F1 0.84), Nucleus (0.76), Plastid (0.69), Mitochondrion (0.68) and Cell membrane (0.65). These have clearer sequence signals and more training examples.
- **Middling:** Cytoplasm (0.49) and Endoplasmic reticulum (0.40). Cytoplasm is close to an "everything else" class and overlaps with the nucleus.
- **Weakest:** Golgi (0.17), Lysosome/Vacuole (0.23) and Peroxisome (0.11). These are the rarest classes, with only 22, 26 and 4 test proteins respectively. With just 4 peroxisome proteins in the test set, the peroxisome scores are very noisy, so they should not be over-interpreted.
- **Macro-F1 (0.50) is below accuracy (0.64)** because macro-F1 treats all 10 classes equally, so the weak small classes pull it down.
- **Precision is below recall for the rare classes** (for example Golgi: 0.12 vs. 0.27). `class_weight="balanced"` makes the model guess rare classes more often, so it finds more of them but is wrong more often.

### Confusion matrix (8M model)

The rows of the matrix are true locations and the columns are predicted ones. The biggest patterns:

| True location | Correct | Main mistake |
|---|---|---|
| Nucleus | 200 of 269 | 48 predicted as Cytoplasm |
| Cytoplasm | 84 of 170 | 43 predicted as Nucleus, 14 as Mitochondrion |
| Extracellular | 130 of 165 | 9 as Golgi, 8 as ER, 8 as Lysosome/Vacuole |
| Mitochondrion | 83 of 122 | 10 as Cytoplasm, 8 as Plastid, 6 as Peroxisome |
| Cell membrane | 55 of 92 | 14 predicted as ER |
| Plastid | 41 of 64 | 13 predicted as Mitochondrion |
| Endoplasmic reticulum | 29 of 66 | 16 as Golgi, 8 as Cell membrane |
| Golgi apparatus | 6 of 22 | spread across many classes |
| Lysosome/Vacuole | 7 of 26 | spread across many classes |
| Peroxisome | 1 of 4 | too few examples to say |

**What stands out**
- **Nucleus and Cytoplasm are confused with each other more than any other pair** (48 and 43 mistakes). This matches biology: many proteins move between the nucleus and the cytoplasm, so the "true" label is blurry.
- **Mitochondrion and Plastid are confused with each other** (13 and 8 mistakes). Both are organelles whose proteins are imported using similar kinds of start-of-sequence signals.
- **Compartments of the same secretory pathway blur together:** ER, Golgi and Cell membrane are all part of the route proteins take to the cell surface, and their sequences share similar features.
- **Extracellular is the cleanest class**, with most of its proteins correctly identified, probably because secreted proteins carry a distinctive signal at the start of the sequence.

These patterns are the kind of mistakes you would expect from a biologically sensible model, rather than random errors.

### UMAP plot

The UMAP of the 8M embeddings shows **some structure but a lot of overlap**: a few groups of proteins gather together and some small tight clusters stand out, but many locations are mixed in the middle of the plot. This fits the classifier's results: the embeddings carry location information, but the 8M model does not separate all ten locations cleanly. (UMAP is only a rough 2D picture of a 320-dimensional space, so overlap in the plot does not mean the classifier cannot separate the classes.)

### Notes on the loading messages

When ESM-2 loads, `transformers` prints a report listing `lm_head.*` weights as UNEXPECTED and `pooler.dense.*` weights as MISSING. This is harmless here. We load the model without its language-model head (those weights are simply not needed), and we use the averaged per-amino-acid vectors rather than the pooler, so the newly initialized pooler is never used.

---

## 7. Limitations

- **This is a learning project**, not a research-grade or clinical tool. Treat predictions as hypotheses to check, not facts.
- **One location per protein.** Many real proteins live in several places; the multi-label problem is handled by later tools such as DeepLoc 2.0.
- **Nucleus vs. cytoplasm** are often confused, because many proteins shuttle between them.
- **Small classes** (peroxisome, lysosome/vacuole, Golgi) have few examples and are predicted less reliably.
- **Sequence only.** The model sees no 3D structure, binding partners, cell type or conditions, all of which can affect where a protein goes.
- **Length limit:** sequences over 1,000 amino acids are skipped.
- **Subset of the data** is used by default (3,000 train and 1,000 test proteins) to keep runtimes short.
- **A prediction is not a measurement.** Experiments are what confirm where a protein actually sits.

---

## 8. How to run it

1. Open [Google Colab](https://colab.research.google.com) and upload `esm2_protein_localization.ipynb` (File > Upload notebook).
2. Choose **Runtime > Change runtime type > T4 GPU**. The 8M model also runs on CPU, just more slowly.
3. Run the cells from top to bottom. The first run downloads the DeepLoc dataset and the ESM-2 weights.
4. Optional: raise `N_TRAIN` and `N_TEST` to use more of the data.

Main libraries: `transformers`, `torch`, `scikit-learn`, `umap-learn`, `pandas`, `matplotlib`.

---

## 9. Beginner FAQ

**Can a protein language model really predict where a protein lives?**
To a useful degree, yes. Sequences contain location signals, and the model's embeddings carry them. It is not perfect (see [Limitations](#7-limitations)).

**So I give a protein and get a location?**
Yes. Under the hood, ESM-2 turns the sequence into numbers, and the classifier turns those numbers into a location with probabilities.

**Has someone already done this?**
Many times, and that is good news. DeepLoc, DeepLoc 2.0 and several ESM-2-based tools exist, so there are known benchmarks to compare against.

**What does the DeepLoc dataset contain?**
Protein sequences paired with their known locations, plus a flag for which proteins are in the test set and whether each is membrane-bound or soluble. That is what makes this supervised learning: the sequence is the question and the location is the answer.

**Why is the table shape `(14004, 4)`?**
14,004 proteins (rows) and 4 fields (columns): `acc`, `label`, `is_test`, `seq`.

**What is the shape `(3000, 320)`?**
3,000 training proteins, each described by 320 numbers from ESM-2 8M.

**Where did the 4 columns go after embedding?**
Nothing happened to them. They are still in the `train` and `test` tables. `seq` was used as ESM-2's input, `label` was copied into `y_tr`, `is_test` was used to split the data, and `acc` is just an ID.

**Does `X_tr` contain the proteins?**
No, only the vectors. The classifier never sees the letters, only the 320 numbers that represent each protein.

**Can logistic regression classify more than two classes?**
Yes. It learns one set of weights per class and converts the scores into probabilities that sum to 100%, so here it handles 10 classes.

**Why is the UMAP plot useful?**
It shows whether the embeddings group proteins by location without being told the labels, which classes are easy or hard to separate, and whether your results make sense. It is a sanity check, not a score.

**Does the transformer learn positions on the UMAP chart?**
No. Those positions are chosen by UMAP afterward. The only "position" a transformer uses is the order of amino acids within a sequence.

**What is the difference between the three model sizes?**
Same design and training data, but more layers, parameters and vector length as they get bigger (see the [table](#model-sizes)). Bigger usually means somewhat better embeddings, at the cost of speed and memory.

**Is 63.6% accuracy low?**
Not for this setup. The baseline (always guessing "Nucleus") scores 26.9%, so 63.6% is more than double that, and the 150M model reaches 73.3%. The published DeepLoc result (about 78%) used a purpose-built model, while this project uses frozen embeddings, a simple classifier and only a subset of the data. Accuracy alone also hides how uneven the classes are, which is why macro-F1 is reported too.

**Does a bigger model help?**
Yes, in this experiment: accuracy rose from 0.636 (8M) to 0.707 (35M) to 0.733 (150M), with the biggest jump between the first two.

**Is ESM-2 a "real" transformer without a decoder?**
Yes. It is encoder-only, like BERT.

---

## 10. Credits and references

- **ESM-2:** Lin et al., Meta AI. Model used: `facebook/esm2_t6_8M_UR50D` (plus the 35M and 150M versions for comparison).
- **DeepLoc 1.0 data and method:** Almagro Armenteros et al., "DeepLoc: prediction of protein subcellular localization using deep learning," *Bioinformatics*, 2017. Dataset page: https://services.healthtech.dtu.dk/services/DeepLoc-1.0/3-Data.php. The dataset is **not** redistributed here, so please download it from DTU and check their terms of use.
- **Protein database:** UniProt (https://www.uniprot.org).
- **Transformer architecture:** Vaswani et al., "Attention Is All You Need," 2017.

*This project is for learning and exploration only.*
