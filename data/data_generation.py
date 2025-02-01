import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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