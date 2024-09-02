import logging
import joblib
from pydantic import BaseModel, Field
from typing import Literal, Dict
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Configure logging
logging.basicConfig(level=logging.INFO)


# Pydantic model for input features
class IrisInference(BaseModel):
    sepal_length: float = Field(description="Length of sepal in cm", ge=0)
    sepal_width: float = Field(description="Width of sepal in cm", ge=0)
    petal_length: float = Field(description="Length of petal in cm", ge=0)
    petal_width: float = Field(description="Width of petal in cm", ge=0)


# Pydantic model for prediction output
class IrisPrediction(BaseModel):
    prediction: Literal["setosa", "versicolor", "virginica"]
    probability: Dict[str, float]


# Function to create and return the pipeline
def create_model_pipeline():
    """Creates and returns the model pipeline with preprocessing and classifier."""
    pipeline = make_pipeline(
        SimpleImputer(strategy="mean"), StandardScaler(), LogisticRegression()
    )
    return pipeline


# Function to evaluate the model
def evaluate_model(pipe, X_test, y_test, target_names):
    """Evaluates the model on the test set and prints classification report and cross-validation score."""
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=target_names)
    logging.info("Classification Report:\n%s", report)


# Function to make a prediction
def make_prediction(pipe, input_data: IrisInference) -> IrisPrediction:
    """Make a prediction using the trained pipeline."""
    data = [
        [
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width,
        ]
    ]

    try:
        # Make prediction
        prediction = pipe.predict(data)[0]
        label = str(iris.target_names[prediction])  # Convert to Python string

        # Get prediction probabilities
        probabilities = pipe.predict_proba(data)[0]
        return IrisPrediction(
            prediction=label,
            probability={
                iris.target_names[i]: prob for i, prob in enumerate(probabilities)
            },
        )
    except Exception as e:
        logging.error("An error occurred during prediction: %s", e)
        raise


def save_model(pipe, filename: str) -> None:
    joblib.dump(pipe, filename)


def load_model(filename: str):
    model = joblib.load(filename)
    return model


if __name__ == "__main__":
    # Load dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.33, random_state=42
    )

    # Create and train the pipeline
    pipeline = create_model_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    evaluate_model(pipeline, X_test, y_test, iris.target_names)

    # Collect input data
    sep_len = float(input("Sepal Length: "))
    sep_wid = float(input("Sepal Width: "))
    pet_len = float(input("Petal Length: "))
    pet_wid = float(input("Petal Width: "))

    # Make a prediction
    input_data = IrisInference(
        sepal_length=sep_len,
        sepal_width=sep_wid,
        petal_length=pet_len,
        petal_width=pet_wid,
    )

    prediction = make_prediction(pipeline, input_data)

    # Display the prediction results
    print(f"Predicted Class: {prediction.prediction}")
    print(f"Probabilities: {prediction.probability}")

    # Save model
    name = "iris_pipeline.joblib"
    save_model(pipeline, name)
    print(f"Model Successfully Saved filename : {name} ")
    # Load model
    model = load_model(name)
    # Show model score
    test_acc = model.score(X_test, y_test)
    print(f"Model accuracy in test : {test_acc:.4f}")
