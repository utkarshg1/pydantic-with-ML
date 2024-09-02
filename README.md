# Iris Classification Pipeline

This script demonstrates creating a machine learning pipeline for classifying Iris flower species using logistic regression. It includes data loading, preprocessing, model training, evaluation, and prediction.

## Dependencies

- `logging`
- `joblib`
- `pydantic`
- `typing`
- `sklearn`

## Pydantic Models

### `IrisInference`

A model for input features used for making predictions.

- **Attributes**:
  - `sepal_length` (float): Length of the sepal in cm.
  - `sepal_width` (float): Width of the sepal in cm.
  - `petal_length` (float): Length of the petal in cm.
  - `petal_width` (float): Width of the petal in cm.

### `IrisPrediction`

A model for the output of predictions.

- **Attributes**:
  - `prediction` (Literal["setosa", "versicolor", "virginica"]): The predicted Iris species.
  - `probability` (Dict[str, float]): The probabilities for each Iris species.

## Functions

### `create_model_pipeline()`

Creates and returns a machine learning pipeline that includes:

- **Preprocessing**:
  - `SimpleImputer(strategy="mean")`: Fills missing values with the mean.
  - `StandardScaler()`: Standardizes features by removing the mean and scaling to unit variance.

- **Classifier**:
  - `LogisticRegression()`: Logistic regression classifier.

**Returns**: A `Pipeline` object.

### `evaluate_model(pipe, X_test, y_test, target_names)`

Evaluates the model on the test set and logs the classification report.

- **Parameters**:
  - `pipe` (Pipeline): The trained model pipeline.
  - `X_test` (array-like): Test features.
  - `y_test` (array-like): True labels for test features.
  - `target_names` (array-like): The names of the target classes.

### `make_prediction(pipe, input_data: IrisInference) -> IrisPrediction`

Makes a prediction using the trained pipeline and returns the predicted class and probabilities.

- **Parameters**:
  - `pipe` (Pipeline): The trained model pipeline.
  - `input_data` (IrisInference): Input data for prediction.

- **Returns**: An `IrisPrediction` object.

### `save_model(pipe, filename: str) -> None`

Saves the trained model pipeline to a file.

- **Parameters**:
  - `pipe` (Pipeline): The trained model pipeline.
  - `filename` (str): The name of the file where the model will be saved.

### `load_model(filename: str)`

Loads a model pipeline from a file.

- **Parameters**:
  - `filename` (str): The name of the file from which the model will be loaded.

- **Returns**: A `Pipeline` object.

## Execution Flow

1. **Load Dataset**: The Iris dataset is loaded and split into training and test sets.
2. **Create and Train Pipeline**: A pipeline is created and trained with the training data.
3. **Evaluate Model**: The model is evaluated using the test data, and the classification report is logged.
4. **Collect Input Data**: The user is prompted to input Iris flower features.
5. **Make Prediction**: The trained model makes a prediction based on the input data.
6. **Display Results**: The predicted class and probabilities are displayed.
7. **Save Model**: The trained model is saved to a file.
8. **Load Model**: The model is loaded from the file, and its accuracy on the test set is displayed.

## Usage

1. Run the script to train the model and evaluate it.
2. Input the Iris flower features when prompted.
3. The script will display the prediction and save the model to a file.

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

---