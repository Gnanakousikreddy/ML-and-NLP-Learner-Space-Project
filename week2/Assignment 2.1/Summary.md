# Assignment 2.1: Text Vectorization Implementation

## 📌 Objective

The goal of this assignment is to **manually implement the TF-IDF algorithm** from scratch and compare its output to that of **scikit-learn’s `CountVectorizer` and `TfidfVectorizer`**.

---

## 📚 Corpus Used

```python
corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]
```

---

## 🧠 Implementation Details

### ✅ Manual TF-IDF

- Built a custom tokenizer using regular expressions.
- Constructed vocabulary from the corpus with lowercase filtering.
- Computed **raw term frequencies (TF)** without normalization.
- Calculated **inverse document frequency (IDF)** using the smoothed formula:

**IDF(t)** = log((1 + N) / (1 + DF(t))) + 1

- Applied **L2 normalization** to the final TF-IDF vectors.

---

### 🧪 Vectorizer Comparisons

- Used **`CountVectorizer`** for raw term frequency.
- Used **`TfidfVectorizer`** with the following parameters:
  ```python
  TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)

---

## 🔍 Comparison of Word Scores

| Method         | Output Match              | Explanation                                       |
|----------------|---------------------------|---------------------------------------------------|
| **Manual TF**  | ✅ Matches `CountVectorizer`   | Raw frequency of terms in documents               |
| **Manual TF-IDF** | ✅ Matches `TfidfVectorizer` | Smoothing and L2 normalization applied correctly  |

---

## 🤔 Explanation: Why Common Words Differ

Words like `'the'` occur in every document, which gives them:

- **High frequency (TF)**, so `CountVectorizer` gives them **high scores**  
- **Low IDF**, since they're not unique, making their **TF-IDF score close to 0**

This behavior is intentional in TF-IDF:

> Common words provide less information and are therefore down-weighted.

---

## ✅ Final Conclusion

- ✅ My **manual TF-IDF implementation** exactly matches the output of `TfidfVectorizer` from scikit-learn.
- ✅ My **manual TF computation** exactly matches the output of `CountVectorizer`.

