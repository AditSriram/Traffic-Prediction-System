import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Reshape, GRU, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import joblib
import os

class TrafficPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        
    def prepare_data(self, data):
        # Prepare X and y
        X = data.values
        y = data.values
        
        def classify_traffic(value):
            # Adjusted thresholds based on your data
            if value < 50:  # Changed from 200
                return 0    # Less Traffic
            elif value < 90:  # Changed from 400
                return 1    # Medium Traffic
            else:
                return 2    # Heavy Traffic
        
        # Apply classification for deep learning models
        y_classified = np.apply_along_axis(
            lambda x: [classify_traffic(val) for val in x], 1, y
        )
        
        # Print distribution of classes
        total_samples = y_classified.size
        unique, counts = np.unique(y_classified, return_counts=True)
        print("\nClass distribution:")
        for label, count in zip(unique, counts):
            percentage = (count / total_samples) * 100
            if label == 0:
                category = "Less Traffic"
            elif label == 1:
                category = "Medium Traffic"
            else:
                category = "Heavy Traffic"
            print(f"{category}: {count} samples ({percentage:.2f}%)")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split the data
        return train_test_split(
            X_scaled, y_classified, test_size=0.2, random_state=42
        )
    
    def build_cnn(self, input_shape):
        model = Sequential([
            Input(shape=(input_shape[1], 1)),
            Conv1D(filters=64, kernel_size=2, activation='relu'),
            Conv1D(filters=32, kernel_size=2, activation='relu'),  # Added layer
            Flatten(),
            Dense(64, activation='relu'),  # Increased neurons
            Dense(32, activation='relu'),
            Dense(6 * 3, activation='softmax'),
            Reshape((6, 3))
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),  # Added learning rate
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        return model
    
    def build_gru(self, input_shape):
        model = Sequential([
            Input(shape=(input_shape[1], 1)),
            GRU(128, return_sequences=True),  # Increased units, added return_sequences
            GRU(64),  # Added second GRU layer
            Dense(64, activation='relu'),  # Increased neurons
            Dense(32, activation='relu'),
            Dense(6 * 3, activation='softmax'),
            Reshape((6, 3))
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),  # Added learning rate
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        return model
    
    def train_models(self, data_path):
        # Load data
        data = pd.read_csv(data_path)
        
        # Print data statistics
        print("\nData Statistics:")
        print(data.describe())
        
        X_train, X_test, y_train, y_test = self.prepare_data(data)
        
        # Prepare data for deep learning models
        X_train_dl = np.expand_dims(X_train, axis=2)
        X_test_dl = np.expand_dims(X_test, axis=2)
        
        y_train_cat = to_categorical(y_train, num_classes=3)
        y_test_cat = to_categorical(y_test, num_classes=3)
        
        y_train_cat_reshaped = y_train_cat.reshape(y_train.shape[0], 6, 3)
        y_test_cat_reshaped = y_test_cat.reshape(y_test.shape[0], 6, 3)
        
        print("\nTraining CNN model...")
        cnn = self.build_cnn((X_train_dl.shape))
        cnn.fit(X_train_dl, y_train_cat_reshaped, 
                epochs=100,  # Increased epochs
                batch_size=32,
                validation_split=0.2,  # Added validation split
                verbose=1)
        self.models['CNN'] = cnn
        
        print("\nTraining GRU model...")
        gru = self.build_gru((X_train_dl.shape))
        gru.fit(X_train_dl, y_train_cat_reshaped,
                epochs=100,  # Increased epochs
                batch_size=32,
                validation_split=0.2,  # Added validation split
                verbose=1)
        self.models['GRU'] = gru
        
        print("\nTraining Gradient Boosting model...")
        gb = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=200,  # Increased estimators
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        )
        gb.fit(X_train, y_train)
        self.models['GradientBoosting'] = gb
        
        print("\nTraining Random Forest model...")
        rf = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=200,  # Increased estimators
                max_depth=10,
                random_state=42
            )
        )
        rf.fit(X_train, y_train)
        self.models['RandomForest'] = rf
        
        self.save_models()
        
    def save_models(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.models['CNN'].save(os.path.join(current_dir, 'cnn_model.h5'))
        self.models['GRU'].save(os.path.join(current_dir, 'gru_model.h5'))
        joblib.dump(self.models['GradientBoosting'], 
                   os.path.join(current_dir, 'gb_model.pkl'))
        joblib.dump(self.models['RandomForest'], 
                   os.path.join(current_dir, 'rf_model.pkl'))
        joblib.dump(self.scaler, os.path.join(current_dir, 'scaler.pkl'))

def train_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'traffic-prediction-dataset.csv')
        
        print(f"Training models using dataset at: {file_path}")
        predictor = TrafficPredictor()
        predictor.train_models(file_path)
        print("All models trained and saved successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()