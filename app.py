
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load and train the model
def train_model():
    # Load the dataset
    sneakers_pd = pd.read_csv('snicker_dataset_with_dates.csv')

    # Define features
    numerical_features = ['price', 'sell_through_rate', 'damage_rate']
    categorical_features = ['edition', 'quarter', 'price_bucket']

    # One-hot encode categorical variables
    sneakers_pd_encoded = pd.get_dummies(sneakers_pd[categorical_features], drop_first=True)

    # Combine all features
    X = pd.concat([sneakers_pd[numerical_features], sneakers_pd_encoded], axis=1)
    y = sneakers_pd['unsold_inventory']

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Return the trained model and the columns used for training
    return model, X.columns

model, train_columns = train_model()

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    price = float(request.form['price'])
    sell_through_rate = float(request.form['sell_through_rate'])
    damage_rate = float(request.form['damage_rate'])
    edition = request.form['edition']
    quarter = request.form['quarter']
    price_bucket = request.form['price_bucket']

    # Create a new product as a single-row DataFrame
    new_product_data = pd.DataFrame([{
        'price': price,
        'sell_through_rate': sell_through_rate,
        'damage_rate': damage_rate,
        'edition': edition,
        'quarter': quarter,
        'price_bucket': price_bucket
    }])

    # One-hot encode categorical features
    new_dummies = pd.get_dummies(new_product_data[['edition', 'quarter', 'price_bucket']])
    
    # Align dummy columns with training set
    new_dummies_aligned = new_dummies.reindex(columns=train_columns, fill_value=0)

    # Combine numerical and dummy features
    X_new = pd.concat([new_product_data[['price', 'sell_through_rate', 'damage_rate']], new_dummies_aligned], axis=1)
    
    # Ensure the order of columns is the same as during training
    X_new = X_new[train_columns]

    # Predict unsold inventory
    predicted_unsold = model.predict(X_new)

    return render_template('index.html', prediction=predicted_unsold[0])

if __name__ == '__main__':
    app.run(debug=True)
