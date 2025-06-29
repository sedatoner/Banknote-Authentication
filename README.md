# Fake Banknote Detection – Decision Tree Classification

Hey there! In this project, I built a machine learning model to detect whether a banknote is real or fake based on some statistical features. My main goal was to catch fake transactions with as few false negatives as possible.

## Problem Statement

The dataset contains four numerical features:
- variance
- skewness
- kurtosis
- entropy

And a target column:
- 0 = Authentic
- 1 = Fake

Note: The dataset is imbalanced (fakes are the minority). But in this first version, I didn’t apply SMOTE or any balancing technique. Just wanted to see how far I can go with raw modeling.

## Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- Google Colab (I fetched the dataset from Google Drive)

## Initial Analysis

First Glimpse:
When I loaded the data, I noticed that kurtosis and entropy had some extreme values.

Some features like variance, skewness, and kurtosis show clear separations between real and fake. entropy is more blended and seems less useful.

## Model Training – Decision Tree

I trained Decision Tree classifiers using two different splitting criteria:
- gini (Gini Impurity)
- entropy (Information Gain)

And for each, I tested three tree depths: max_depth = 3, 5, 7

My Goal:
Keep the tree simple but effective, minimizing false negatives (missed fake banknotes).

## Confusion Matrix Results

Gini Criterion:

| Depth | TN | FP | FN | TP | Notes |
|-------|----|----|----|----|-------|
| 3     | 139| 9  | 16 | 111| Weak recall |
| 5     | 148| 0  | 9  | 118| Much better |
| 7     | 148| 0  | 5  | 122| One of the best setups |

Entropy Criterion:

| Depth | TN | FP | FN | TP | Notes |
|-------|----|----|----|----|-------|
| 3     | 142| 6  | 8  | 119| Slightly better than Gini |
| 5     | 147| 1  | 5  | 122| Solid |
| 7     | 148| 0  | 4  | 123| Best overall performance |

Important: I focused heavily on minimizing false negatives, since missing a fake is worse than a false alarm.

## Decision Tree Visualization

Here’s a visual of the final decision tree (with entropy and max_depth = 7):

(tree.png)

You can see the tree splits mainly based on variance, kurtosis, and skewness. entropy shows up less often, which matches what we see below in feature importance.

## Feature Importance

Final model’s feature importance:

(feature_importance.png)

- variance: by far the most critical feature (around 60%)
- skewness and kurtosis: moderate impact
- entropy: barely contributes

## Summary & Takeaways

In this project:
- I trained decision tree models with different parameters
- Evaluated each using confusion matrices and classification reports
- Identified the most informative features
- Observed that deeper trees (up to depth 7) performed better without overfitting

What I Learned:
- Shallow trees underfit (miss important patterns)
- entropy as a splitting criterion performed more consistently
- Even with imbalanced data, decision trees can do a solid job

## Potential Next Steps

Here’s what I might add later:
- Apply SMOTE to balance the dataset
- Try other classifiers like RandomForest, XGBoost
- Use ROC/AUC scores for deeper evaluation
- Build a Streamlit or Flask app for interactive predictions


## Contributions

Pull requests are welcome! Especially if you're interested in balancing techniques or alternative models—I'd love to compare approaches.

## Contact

Feel free to reach out via GitHub if you have any questions or ideas.

Thanks for stopping by!
