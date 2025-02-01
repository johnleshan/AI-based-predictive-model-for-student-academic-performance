import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from keras.models import Sequential
from keras.layers import Dense
from imblearn.over_sampling import SMOTE

# Generate synthetic data
np.random.seed(42)
n_students = 1000

data = {
    'attendance': np.random.randint(50, 100, n_students),
    'previous_grades': np.random.uniform(40, 100, n_students),
    'study_habits': np.random.randint(1, 5, n_students),  # 1-5 scale
    'socio_economic': np.random.randint(1, 4, n_students), # 1 (low) to 3 (high)
    'performance_score': np.random.randint(0, 2, n_students)  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)
df.to_csv('data/raw_data.csv', index=False)

# Load data
df = pd.read_csv('data/raw_data.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df['previous_grades'] = imputer.fit_transform(df[['previous_grades']])

# Normalize features
scaler = StandardScaler()
features = ['attendance', 'previous_grades', 'study_habits', 'socio_economic']
X = scaler.fit_transform(df[features])
y = df['performance_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(f"RF Accuracy: {accuracy_score(y_test, rf_pred)}")

# SVM
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred)}")

# Neural Network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
nn_pred = (model.predict(X_test) > 0.5).astype(int)
print(f"NN Accuracy: {accuracy_score(y_test, nn_pred)}")

# Handle class imbalance
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Model evaluation
from sklearn.metrics import classification_report

# Example for Random Forest
print(classification_report(y_test, rf_pred))

# Save the model (e.g., Random Forest)
import joblib
joblib.dump(rf, 'model/model.pkl')