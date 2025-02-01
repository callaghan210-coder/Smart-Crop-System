from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for Flask-Login and Flask-WTF

# Database Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(BASE_DIR, "database.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Route to redirect unauthorized users

# Define User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Define CropRecommendation Model
class CropRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nitrogen = db.Column(db.Float, nullable=False)
    phosphorus = db.Column(db.Float, nullable=False)
    potassium = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    ph = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    recommended_crop = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Link to User

    def __repr__(self):
        return f'<Recommendation {self.id}: {self.recommended_crop}>'

# Load pre-trained model and scaler
model = joblib.load('model/crop_recommendation_model.pkl')

# Load and preprocess data (for visualization and analysis)
data = pd.read_csv('data/Crop_recommendation.csv')

# Create a mapping from numeric labels to crop names
crop_names = data['label'].unique()
crop_mapping = {index: name for index, name in enumerate(crop_names)}

# Flask-Login User Loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Registration Form
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=50)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

# Login Form
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, password=form.password.data)  # In production, hash the password!
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.password == form.password.data:  # In production, use password hashing!
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Get input data from the form
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        # Preprocess inputs
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

        
        # Predict the crop label
        prediction = model.predict(input_data)
        predicted_crop = crop_mapping.get(prediction[0], "Unknown Crop")
        
        # Save to database
        new_recommendation = CropRecommendation(
            nitrogen=nitrogen, phosphorus=phosphorus, potassium=potassium,
            temperature=temperature, humidity=humidity, ph=ph, rainfall=rainfall,
            recommended_crop=predicted_crop, user_id=current_user.id
        )
        db.session.add(new_recommendation)
        db.session.commit()
        
        # Return the predicted crop
        return jsonify({'recommended_crop': predicted_crop})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)})

@app.route('/history')
@login_required
def history():
    recommendations = CropRecommendation.query.filter_by(user_id=current_user.id).all()
    return render_template('history.html', recommendations=recommendations)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Ensure database tables are created before running
    app.run(debug=True)

# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import joblib
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from flask_sqlalchemy import SQLAlchemy
# import os

# # Initialize Flask app
# app = Flask(__name__)

# # Database Configuration
# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(BASE_DIR, "database.db")}'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)

# # Define Database Model
# class CropRecommendation(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     nitrogen = db.Column(db.Float, nullable=False)
#     phosphorus = db.Column(db.Float, nullable=False)
#     potassium = db.Column(db.Float, nullable=False)
#     temperature = db.Column(db.Float, nullable=False)
#     humidity = db.Column(db.Float, nullable=False)
#     ph = db.Column(db.Float, nullable=False)
#     rainfall = db.Column(db.Float, nullable=False)
#     recommended_crop = db.Column(db.String(50), nullable=False)

#     def __repr__(self):
#         return f'<Recommendation {self.id}: {self.recommended_crop}>'

# # Load pre-trained model
# model = joblib.load('model/crop_recommendation_model.pkl')

# # Load and preprocess data (for visualization and analysis)
# data = pd.read_csv('data/Crop_recommendation.csv')

# # Create a mapping from numeric labels to crop names
# crop_names = data['label'].unique()  # Assuming 'label' is the column with numeric crop labels
# crop_mapping = {index: name for index, name in enumerate(crop_names)}

# scaler = StandardScaler()
# data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']] = scaler.fit_transform(
#     data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
# )

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get input data from the form
#         nitrogen = float(request.form['nitrogen'])
#         phosphorus = float(request.form['phosphorus'])
#         potassium = float(request.form['potassium'])
#         temperature = float(request.form['temperature'])
#         humidity = float(request.form['humidity'])
#         ph = float(request.form['ph'])
#         rainfall = float(request.form['rainfall'])
        
#         # Preprocess inputs
#         input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
#         input_data = scaler.transform(input_data)
        
#         # Predict the crop label
#         prediction = model.predict(input_data)
        
#         # Map the numeric prediction to the crop name
#         predicted_crop = crop_mapping.get(prediction[0], "Unknown Crop")
#         # Optionally, if you want a range of suitable crops, you could get probabilities or 
#         # nearest neighbors (based on model support).
        
#         # For simplicity, let's assume we return the top 3 closest crops (this depends on your model's capability)
#         prediction_proba = model.predict_proba(input_data)  # Get probability distribution
#         top_n_indices = prediction_proba[0].argsort()[-3:][::-1]  # Get top 3 predictions
#         top_crops = [crop_mapping.get(i, "Unknown Crop") for i in top_n_indices]
        
#         # Return the top N predicted crops
#         return jsonify({'recommended_crops': top_crops})
#     except Exception as e:

#         # Save to database
#         new_recommendation = CropRecommendation(
#             nitrogen=nitrogen, phosphorus=phosphorus, potassium=potassium,
#             temperature=temperature, humidity=humidity, ph=ph, rainfall=rainfall,
#             recommended_crop=top_crops
#         )
#         db.session.add(new_recommendation)
#         db.session.commit()

#         return jsonify({'recommended_crop': predicted_crop})
    
#     except Exception as e:
#         print(f"Error: {e}")
#         return jsonify({'error': str(e)})

# @app.route('/history')
# def history():
#     recommendations = CropRecommendation.query.all()
#     return render_template('history.html', recommendations=recommendations)

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()  # Ensure database tables are created before running
#     app.run(debug=True)

# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import joblib
# import numpy as np
# from sklearn.preprocessing import StandardScaler

# # Initialize Flask app
# app = Flask(__name__)

# # Load pre-trained model
# model = joblib.load('model/crop_recommendation_model.pkl')

# # Load and preprocess data (for visualization and analysis)
# data = pd.read_csv('data/Crop_recommendation.csv')

# # Create a mapping from numeric labels to crop names
# crop_names = data['label'].unique()  # Assuming 'label' is the column with numeric crop labels
# crop_mapping = {index: name for index, name in enumerate(crop_names)}

# scaler = StandardScaler()
# data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']] = scaler.fit_transform(
#     data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
# )

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get input data from the form
#         nitrogen = float(request.form['nitrogen'])
#         phosphorus = float(request.form['phosphorus'])
#         potassium = float(request.form['potassium'])
#         temperature = float(request.form['temperature'])
#         humidity = float(request.form['humidity'])
#         ph = float(request.form['ph'])
#         rainfall = float(request.form['rainfall'])
        
#         # Preprocess inputs
#         input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
#         input_data = scaler.transform(input_data)
        
#         # Predict the crop label
#         prediction = model.predict(input_data)
        
#         # Map the numeric prediction to the crop name
#         predicted_crop = crop_mapping.get(prediction[0], "Unknown Crop")

#         # Optionally, if you want a range of suitable crops, you could get probabilities or 
#         # nearest neighbors (based on model support).
        
#         # For simplicity, let's assume we return the top 3 closest crops (this depends on your model's capability)
#         prediction_proba = model.predict_proba(input_data)  # Get probability distribution
#         top_n_indices = prediction_proba[0].argsort()[-3:][::-1]  # Get top 3 predictions
#         top_crops = [crop_mapping.get(i, "Unknown Crop") for i in top_n_indices]
        
#         # Return the top N predicted crops
#         return jsonify({'recommended_crops': top_crops})
#     except Exception as e:
#         # Debugging information (optional, remove in production)
#         print(f"Error: {e}")
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)


