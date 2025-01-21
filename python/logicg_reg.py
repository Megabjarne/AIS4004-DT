from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns


#data = {data.csv}
df = pd.read_csv("data.csv")

# Split features (first 15 columns) and labels (16th column)
x = df.iloc[:, 1:-2].to_numpy()  # First 15 columns
y = df.iloc[:, -2].to_numpy()   # Last column
failure_threshold = 0.99
y_bool = list(map(lambda v: v < failure_threshold, df.iloc[:, -2]))  # Assuming second-to-last column is the target

#y = (y > 0.95).astype(int)



# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x)
x_test = scaler.transform(x)

x_train, x_test, y_train, y_test =\
    train_test_split(x, y_bool, test_size=0.2, random_state=0)

print(x_train.shape, x_test.shape)


model = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=0.05,
                           random_state=0))

model.fit(x_train, y_train)

x_test = scaler.transform(x_test)
y_pred = model.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

#----------------------------------------------------------------
feature_importance = model.estimators_[0].coef_[0]
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance)
plt.title("Feature Importance in Logistic Regression", fontsize=16)
plt.xlabel("Feature Index", fontsize=14)
plt.ylabel("Coefficient Value", fontsize=14)
plt.show()


# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

# Add labels, title, and axes
plt.title('Confusion Matrix for Logistic Regression Model', fontsize=16)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.xticks(ticks=[0, 1], labels=['False', 'True'])
plt.yticks(ticks=[0, 1], labels=['False', 'True'], rotation=0)
plt.show()

# Print classification report for additional details
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import roc_curve, auc

y_pred_proba = model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.title('ROC Curve', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.legend(loc='lower right')
plt.show()


