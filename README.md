# Overview
This project explores multi‑class classification of wine quality using the Wine Quality dataset, applying three different machine learning approaches: Random Forest, Support Vector Machines, and a Neural Network.
The goal is to predict wine quality (rated 3–9) based on chemical properties, evaluate model performance across seven classes, and analyse the challenges posed by class imbalance and limited class coverage.

This work demonstrates practical model development, comparison, and evaluation on real‑world tabular data, with insights relevant to wine buyers, merchants, and producers seeking data‑driven quality assessment.

# Results

## Random Forest
Strong baseline accuracy

Version 2 improved precision for some minority classes

Version 3 improved recall and captured more minority classes

Trade‑off: higher recall but lower precision and overall accuracy

## Support Vector Machine
Version 1 collapsed into majority‑class predictions

Version 2 improved minority recall but destabilised majority predictions

Version 3 (tuned) restored accuracy and weighted‑F1

Minority classes remained largely undetected

## Neural Network
Version 1 captured majority class but failed on minority classes

Version 2 improved minority recall but destabilised majority predictions

Version 3 improved weighted‑F1/precision for majority classes

Persistent overfitting: widening loss gap and lower validation accuracy
