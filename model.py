import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib
# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder
import pickle # Import pickle to serialize the label encoder


# Load the dataset you uploaded
data = pd.read_csv('Book.csv')

# Select features and target
X = data[['Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic']]  # Input: Voltage and Current values
y = data['output ']  # Output: Fault type

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Assuming 'fault_labels' is a list or Series of your fault types
fault_labels = y.unique() # or replace with your actual fault_labels
le = LabelEncoder()
le.fit(fault_labels)

# Encode labels (optional, useful if retraining models)
# Assuming 'df' is your DataFrame, you might want to use 'data' instead if that's where your labels are
data['fault_type_encoded'] = le.transform(data['output ']) # Replace 'output ' with your fault type column name if different

# Save the LabelEncoder as .pkl
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("LabelEncoder saved as label_encoder.pkl")

# Initialize models
knn = KNeighborsClassifier(n_neighbors=3)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
dt = DecisionTreeClassifier(random_state=42)

# Train models
knn.fit(X_train, y_train)
rf.fit(X_train, y_train)
dt.fit(X_train, y_train)

# Print evaluations
print("KNN Report:\n", classification_report(y_test, knn.predict(X_test)))
print("Random Forest Report:\n", classification_report(y_test, rf.predict(X_test)))
print("Decision Tree Report:\n", classification_report(y_test, dt.predict(X_test)))

# Save models as .pkl
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(dt, 'dt_model.pkl')
print("✅ Models saved")