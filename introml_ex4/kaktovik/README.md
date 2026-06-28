# Kaktovik Subset for Exercise 4

This folder contains a compact local Kaktovik dataset for the PCA/classification exercise.

Chosen setup:
- `8` classes
- `80` images per class
- images copied from `Kaktovik/processed`

Included classes:
- `CLASS_0`
- `CLASS_1`
- `CLASS_2`
- `CLASS_5`
- `CLASS_6`
- `CLASS_10`
- `CLASS_20`
- `CLASS_21`

Rationale:
- We do not use all `22` classes in the first version of the exercise.
- Several original Kaktovik classes are visually very close stroke variants.
- For this exercise, the main learning goal is the PCA pipeline plus a comparison of classifiers, not fine-grained symbol discrimination.
- This subset keeps the runtime small, stays balanced, and still provides clearly different symbol families.

Selection rule:
- Per class, `80` images were chosen deterministically from the sorted source list using evenly spaced indices.
