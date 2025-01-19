import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from flask import Flask, render_template, request
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load and preprocess the dataset
data = pd.read_csv('C:\\Users\\fathi\\Downloads\\Ml\\laptop_price.csv', encoding='ISO-8859-1')
data = data.drop('laptop_ID', axis=1)

# Clean the 'Ram' and 'Weight' columns
data['Ram'] = data['Ram'].astype(str).str.extract(r'(\d+)', expand=False).astype(float)
data['Weight'] = data['Weight'].astype(str).str.extract(r'(\d+\.?\d*)', expand=False).astype(float)

# Clean the 'Inches' column
data['Inches'] = data['Inches'].astype(str).apply(lambda x: float(x) if x.replace('.', '', 1).isdigit() else 15.6)

# Handle the 'ScreenResolution' column
if 'ScreenResolution' in data.columns:
    data[['Width', 'Height']] = data['ScreenResolution'].str.extract(r'(\d+)x(\d+)', expand=True)
    data['Width'] = pd.to_numeric(data['Width'], errors='coerce').fillna(1920)
    data['Height'] = pd.to_numeric(data['Height'], errors='coerce').fillna(1080)
    data['Total_Pixels'] = data['Width'] * data['Height']
    data = data.drop('ScreenResolution', axis=1)

# Encode categorical variables
label_encoders = {}
for column in ['Company', 'Product', 'TypeName', 'Cpu', 'Gpu', 'OpSys']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Separate features (X) and target (y)
X = data.drop('Price_euros', axis=1)
y = data['Price_euros']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features
scaler = StandardScaler()
numerical_features = ['Inches', 'Ram', 'Weight', 'Width', 'Height', 'Total_Pixels']
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Train multiple models
models = {}

# Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
models['Decision Tree'] = dt_model

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
models['Random Forest'] = rf_model

# Support Vector Machine
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)
models['SVM'] = svm_model

# K-Nearest Neighbors
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
models['KNN'] = knn_model

# Gradient Boosting Regressor
gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr_model.fit(X_train, y_train)
models['Gradient Boosting'] = gbr_model

# Serialize models and scaler
with open('models.pkl', 'wb') as f:
    pickle.dump(models, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Deserialize models and scaler
with open('models.pkl', 'rb') as f:
    models = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Home route to display the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        algorithm = request.form.get('algorithm')
        if not algorithm or algorithm not in models:
            raise ValueError("Invalid algorithm selected.")

        company = request.form['company']
        product = request.form['product']
        typename = request.form['typename']
        cpu = request.form['cpu']
        gpu = request.form['gpu']
        os = request.form['os']
        inches = float(request.form['inches'])
        ram = int(''.join(filter(str.isdigit, request.form['ram'])))
        weight = float(''.join(filter(lambda x: x.isdigit() or x == '.', request.form['weight'])))
        width = float(request.form['width'])
        height = float(request.form['height'])

        # Encode categorical input values
        company_encoded = label_encoders['Company'].transform([company])[0]
        product_encoded = label_encoders['Product'].transform([product])[0]
        typename_encoded = label_encoders['TypeName'].transform([typename])[0]
        cpu_encoded = label_encoders['Cpu'].transform([cpu])[0]
        gpu_encoded = label_encoders['Gpu'].transform([gpu])[0]
        os_encoded = label_encoders['OpSys'].transform([os])[0]

        # Create input feature array
        total_pixels = width * height
        features = [[company_encoded, product_encoded, typename_encoded, cpu_encoded, gpu_encoded, os_encoded, inches, ram, weight, width, height, total_pixels]]

        # Scale features
        features_df = pd.DataFrame(features, columns=X.columns)
        features_df[numerical_features] = scaler.transform(features_df[numerical_features])

        # Predict using the selected algorithm
        selected_model = models[algorithm]
        predicted_price = selected_model.predict(features_df)[0]

        # Ensure the predicted price is non-negative
        predicted_price = max(0, predicted_price)

        return render_template('index.html', predicted_price=round(predicted_price, 2), selected_algorithm=algorithm)

    except Exception as e:
        return render_template('index.html', error_message=str(e))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
