#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Classification Project
# 
# This notebook includes three different Machine Learning models (Random Forest, Support Vector Machines, and Neural Network) to predict wine quality based on several chemical properties. For reproducability, the versions of Python and all key libraries used during development are as below:
# - Python version: 3.11.0 (main, Oct 24 2022, 18:26:48) [MSC v.1933 64 bit (AMD64)]
# - Pandas version: 2.3.3
# - NumPy version: 2.3.5
# - scikit-learn version: 1.7.2
# - TensorFlow version: 2.20.0

# ---

# # Random Forest Classifier - Version 1

# In[45]:


# Imports
import sys
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# In[46]:


# Run the below to print versions of all key libraries used.
print("Python version:", sys.version)
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("scikit-learn version:", sklearn.__version__)
print("TensorFlow version:", tf.__version__)


# In[47]:


# Load Dataset
dataset = pd.read_csv("wine_data.csv")


# In[48]:


# Class Distribution (just to demonstrate imbalance)
print(dataset['quality'].value_counts().sort_index())


# In[49]:


# Prepare Dataset
X = dataset.drop(columns=['wine ID', 'quality'])   # drop ID + target
y = dataset['quality']


# In[50]:


# Train-test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[51]:


# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced" # to handle class imbalance
)
rf.fit(X_train, y_train)


# In[52]:


# Predictions
y_pred = rf.predict(X_test)


# In[53]:


# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[54]:


# Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Greens",
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()


# In[55]:


# Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=X.columns[indices], color="green")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()


# ---

# # Random Forest Classifier - Version 2

# In[21]:


# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


# In[22]:


# Load Dataset
dataset = pd.read_csv("wine_data.csv")


# In[23]:


# Prepare Data
X = dataset.drop(columns=['wine ID', 'quality'])
y = dataset['quality']


# In[24]:


# Train-Test Split
# Here we have a larger training set: 85% train, 15% test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)


# In[25]:


# Compute Class Weights
# These are computed by inverse class frequencies.
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))


# In[26]:


# Train Random Forest with Class Weights
rf_v2 = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight=class_weights
)
rf_v2.fit(X_train, y_train)


# In[27]:


# Predictions
y_pred_v2 = rf_v2.predict(X_test)


# In[28]:


# Evaluation
print("Classification Report (Version 2):\n", classification_report(y_test, y_pred_v2))
print("Confusion Matrix (Version 2):\n", confusion_matrix(y_test, y_pred_v2))
print("Balanced Accuracy (Version 2):", balanced_accuracy_score(y_test, y_pred_v2))


# In[29]:


# Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred_v2), annot=True, fmt='d', cmap="Greens",
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix (Version 2)")
plt.show()


# In[30]:


# Feature Importance
importances = rf_v2.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=X.columns[indices], color="green")
plt.title("Feature Importance (Random Forest Version 2)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()


# ---

# # Random Forest Classifier - Version 3

# In[11]:


# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


# Load Dataset
dataset = pd.read_csv("wine_data.csv")


# In[13]:


# Prepare Data
X = dataset.drop(columns=['wine ID', 'quality'])
y = dataset['quality']


# In[14]:


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)


# In[15]:


# Compute Class Weights
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))


# In[16]:


# Train Random Forest with Tuned Hyperparameters
rf_v3 = RandomForestClassifier(
    n_estimators=500,        # more trees for stability
    max_depth=20,            # limit depth to prevent overfitting
    min_samples_leaf=5,      # ensure leaves have enough samples
    max_features="sqrt",     # diversify splits
    random_state=42,
    class_weight=class_weights
)
rf_v3.fit(X_train, y_train)


# In[17]:


# Predictions
y_pred_v3 = rf_v3.predict(X_test)


# In[18]:


# Evaluation
print("Classification Report (Version 3):\n", classification_report(y_test, y_pred_v3))
print("Confusion Matrix (Version 3):\n", confusion_matrix(y_test, y_pred_v3))
print("Balanced Accuracy (Version 3):", balanced_accuracy_score(y_test, y_pred_v3))


# In[19]:


# Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred_v3), annot=True, fmt='d', cmap="Greens",
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix (Version 3)")
plt.show()


# In[20]:


# Feature Importance
importances = rf_v3.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=X.columns[indices], color="green")
plt.title("Feature Importance (Random Forest Version 3)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()


# ---

# # Support Vector Machine Classifier - Version 1

# In[1]:


# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load Dataset
dataset = pd.read_csv("wine_data.csv")


# In[3]:


# Prepare Model
X = dataset.drop(columns=['wine ID', 'quality'])
y = dataset['quality']


# In[5]:


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=8, stratify=y
)


# In[6]:


# Train SVM
svm_v1 = SVC(kernel='linear', random_state=8)
svm_v1.fit(X_train, y_train)


# In[7]:


# Predictions
y_pred_v1 = svm_v1.predict(X_test)


# In[8]:


# Evaluation
print("Classification Report (SVM Version 1):\n", classification_report(y_test, y_pred_v1))
print("Confusion Matrix (SVM Version 1):\n", confusion_matrix(y_test, y_pred_v1))
print("Balanced Accuracy (SVM Version 1):", balanced_accuracy_score(y_test, y_pred_v1))


# In[9]:


# Confusion Matrix 
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred_v1), annot=True, fmt='d', cmap="Blues",
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix (Version 1)")
plt.show()


# ---

# # Support Vector Machine Classifier - Version 2

# In[57]:


# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[58]:


# Load Dataset
dataset = pd.read_csv("wine_data.csv")


# In[59]:


# Prepare Model
X = dataset.drop(columns=['wine ID', 'quality'])
y = dataset['quality']


# In[60]:


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=8, stratify=y
)


# In[61]:


# Train SVM (with RBF Kernel and balanced class weights)
svm_v2 = SVC(kernel='rbf', class_weight='balanced', random_state=8)
svm_v2.fit(X_train, y_train)


# In[62]:


# Predictions
y_pred_v2 = svm_v2.predict(X_test)


# In[63]:


# Evaluation
print("Classification Report (SVM Version 2):\n", classification_report(y_test, y_pred_v2))
print("Confusion Matrix (SVM Version 2):\n", confusion_matrix(y_test, y_pred_v2))
print("Balanced Accuracy (SVM Version 2):", balanced_accuracy_score(y_test, y_pred_v2))


# In[64]:


# Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred_v2), annot=True, fmt='d', cmap="Blues",
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix (Version 2)")
plt.show()


# ---

# # Support Vector Machine - Version 3

# In[65]:


# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[66]:


# Load Dataset
dataset = pd.read_csv("wine_data.csv")


# In[67]:


# Prepare Model
X = dataset.drop(columns=['wine ID', 'quality'])
y = dataset['quality']


# In[68]:


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=8, stratify=y
)


# In[70]:


# Define Parameter Grid for Tuning
param_grid = {
    'C': [0.1, 1, 10],          # regularisation strength
    'gamma': [0.01, 0.1, 1],    # kernel coefficient
    'kernel': ['rbf']           # non-linear kernel
}


# In[72]:


# Grid-Search with Cross-Valildation
grid_search = GridSearchCV(
    SVC(class_weight='balanced', random_state=8),
    param_grid,
    cv=3,            # 3-fold cross-validation
    scoring='f1_weighted',
    verbose=2,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)


# In[75]:


# Best Model
best_svm = grid_search.best_estimator_
print("Best Parameters (Version 3):", grid_search.best_params_)


# In[76]:


# Predictions
y_pred_v3 = best_svm.predict(X_test)


# In[77]:


# Evaluation 
print("Classification Report (SVM Version 3):\n", classification_report(y_test, y_pred_v3))
print("Confusion Matrix (SVM Version 3):\n", confusion_matrix(y_test, y_pred_v3))
print("Balanced Accuracy (SVM Version 3):", balanced_accuracy_score(y_test, y_pred_v3))


# In[78]:


# Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred_v3), annot=True, fmt='d', cmap="Blues",
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix (Version 3)")
plt.show()


# ---

# # Neural Network Classifier - Version 1

# In[88]:


# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[89]:


# Load Dataset
dataset = pd.read_csv("wine_data.csv")


# In[90]:


# Prepare Model
X = dataset.drop(columns=['wine ID', 'quality'])
y = dataset['quality']


# In[91]:


# Encode target (wine quality classes 3-9)
y_encoded = to_categorical(y - y.min())


# In[92]:


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=8, stratify=y
)


# In[93]:


# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[94]:


# Build NN Model
model_v1 = Sequential([
    Input(shape=(X_train.shape[1],)),              # Input layer
    Dense(64, activation='relu'),                  # Hidden layer
    Dense(y_encoded.shape[1], activation='softmax') # Output layer
])


# In[95]:


# Compile Model
model_v1.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[96]:


# Train Model
history_v1 = model_v1.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)


# In[103]:


# Evaluate Model
y_pred_probs = model_v1.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred, zero_division=0))
print("Confusion Matrix (NN Version 1):\n", confusion_matrix(y_true, y_pred))
print("Balanced Accuracy (NN Version 1):", balanced_accuracy_score(y_true, y_pred))


# In[104]:


# Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap="Reds",
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("NN Confusion Matrix (Version 1)")
plt.show()


# ---

# # Neural Network Classifier - Version 2

# In[118]:


# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[119]:


# Load Dataset
dataset = pd.read_csv("wine_data.csv")


# In[120]:


# Prepare Model
X = dataset.drop(columns=['wine ID', 'quality'])
y = dataset['quality']


# In[121]:


# Encode target (wine quality classes 3-9)
y_encoded = to_categorical(y - y.min())


# In[122]:


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.15, random_state=8, stratify=y
)


# In[123]:


# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[124]:


# Compute Class Weights
classes = np.unique(y)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y
)
class_weight_dict = {i: w for i, w in zip(range(len(classes)), class_weights)}


# In[125]:


# Build NN Model
model_v2 = Sequential([
    Input(shape=(X_train.shape[1],)),              # Input layer
    Dense(128, activation='relu'),                 # Hidden layer 1
    Dense(64, activation='relu'),                  # Hidden layer 2
    Dense(y_encoded.shape[1], activation='softmax') # Output layer
])


# In[126]:


# Compile Model
model_v2.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[127]:


# Train Model with Class Weights
history_v2 = model_v2.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    verbose=1
)


# In[128]:


# Evaluate Model
y_pred_probs = model_v2.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Classification Report (NN Version 2):\n", classification_report(y_true, y_pred, zero_division=0))
print("Confusion Matrix (NN Version 2):\n", confusion_matrix(y_true, y_pred))
print("Balanced Accuracy (NN Version 2):", balanced_accuracy_score(y_true, y_pred))


# In[129]:


# Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap="Reds",
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("NN Confusion Matrix (Version 2)")
plt.show()


# ---

# # Neural Network Classifier - Version 3

# In[130]:


# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[131]:


# Load Dataset
dataset = pd.read_csv("wine_data.csv")


# In[132]:


# Prepare Model
X = dataset.drop(columns=['wine ID', 'quality'])
y = dataset['quality']


# In[133]:


# Encode target (wine quality classes 3-9)
y_encoded = to_categorical(y - y.min())


# In[134]:


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.15, random_state=8, stratify=y
)


# In[135]:


# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[136]:


# Compute Class Weights
classes = np.unique(y)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y
)
class_weight_dict = {i: w for i, w in zip(range(len(classes)), class_weights)}


# In[139]:


# Build NN Model
model_v3 = Sequential([
    Input(shape=(X_train.shape[1],)), # Input layer
    Dense(256, activation='relu'), # Hidden layer 1
    Dense(128, activation='relu'), # Hidden layer 2
    Dense(64, activation='relu'), # Hidden layer 3
    Dense(y_encoded.shape[1], activation='softmax') # Output layer
])


# In[140]:


# Compile Model
model_v3.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[141]:


# Add Early Stopping Callback
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10, # Stop if there is no improvement for 10 epochs.
    restore_best_weights=True
)


# In[142]:


# Train Model
history_v3 = model_v3.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100, # Higher cap, but early stopping prevents overfitting.
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)


# In[143]:


# Evaluate Model
y_pred_probs = model_v3.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Classification Report (NN Version 3):\n", classification_report(y_true, y_pred, zero_division=0))
print("Confusion Matrix (NN Version 3):\n", confusion_matrix(y_true, y_pred))
print("Balanced Accuracy (NN Version 3):", balanced_accuracy_score(y_true, y_pred))


# In[147]:


# Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap="Reds",
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("NN Confusion Matrix (Version 3)")
plt.show()


# In[146]:


# Training vs Validation Accuracy/Loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history_v3.history['accuracy'], label='Train Accuracy')
plt.plot(history_v3.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy (V3)')

plt.subplot(1,2,2)
plt.plot(history_v3.history['loss'], label='Train Loss')
plt.plot(history_v3.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss (V3)')
plt.show()


# In[ ]:




