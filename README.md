# Movie Recommender System

A collaborative filtering recommender system built on the [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dataset, featuring a Probabilistic Matrix Factorization (PMF) model with SVD warm-start, bias terms, and momentum-based mini-batch gradient descent.

---

## Overview

This project implements and evaluates a matrix factorization approach to predicting user movie ratings. The core model (`CompetitionRecSys`) extends standard PMF with several improvements:

- **SVD warm-start** — latent user/item vectors are initialized from a truncated SVD of the sparse rating matrix instead of random Gaussian initialization, leading to faster and more stable convergence
- **User and item bias terms** — captures systematic patterns (e.g. users who consistently rate high, or movies that are consistently well-regarded)
- **Mini-batch gradient descent with momentum** — improves training efficiency and convergence speed
- **L2 regularization** — reduces overfitting

---

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — 100,000 ratings (1–5) from 943 users on 1,682 movies.

The dataset is downloaded automatically when the notebook is run.

---

## Model Details

### `CompetitionRecSys` (PMF with enhancements)

| Hyperparameter | Default | Description |
|---|---|---|
| `num_feat` | 35 | Number of latent features |
| `epsilon` | 1.75 | Learning rate |
| `_lambda` | 0.18 | L2 regularization strength |
| `momentum` | 0.75 | Momentum coefficient |
| `maxepoch` | 50 | Training epochs |
| `num_batches` | 160 | Mini-batches per epoch |
| `batch_size` | 1700 | Samples per mini-batch |

Hyperparameters were tuned via grid search.

### Rating prediction

```
r̂(u, i) = w_User[u] · w_Item[i] + global_mean + b_User[u] + b_Item[i]
```

---

## Evaluation

Models are evaluated using 5-fold cross-validation (`CrossValidation` class) with 95% confidence intervals. Supported metrics:

| Metric | Description |
|---|---|
| `RMSE` | Root Mean Squared Error |
| `P@K` | Precision at K |
| `R@K` | Recall at K |
| `RPrecision` | R-Precision |

Ratings ≥ 4 are treated as "liked" for ranking metrics.

---

## Requirements

```
numpy
pandas
scipy
scikit-learn
matplotlib
tqdm
wget
```

Install with:

```bash
pip install numpy pandas scipy scikit-learn matplotlib tqdm wget
```

---

## Usage

1. Clone the repo and open the notebook:
   ```bash
   git clone <your-repo-url>
   jupyter notebook RecommenderSystem.ipynb
   ```

2. Run all cells. The notebook will automatically download and unzip the MovieLens 100K dataset.

3. Cross-validation results are printed after training completes.

---

## Project Structure

```
RecommenderSystem.ipynb   # Main notebook
ml-100k/                  # Dataset directory (auto-generated)
```
