import os
import re
import cv2,tempfile
import numpy as np
import streamlit as st
from zipfile import ZipFile
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from dateutil.parser import parse
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import json
from sklearn.preprocessing import LabelEncoder



global epochs, batchs, drops, returseqs, bidis
epochs = 0
batchs = 0
drops = 0
returseqs = 0
bidis = 0

# Global variables from the second code
global epoch, batch, drop, returseq, bidi
epoch = 0
batch = 0
drop = 0
returnseq = 0
bidi = 0
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def extract_zip(zip_file_path, extract_dir):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

def count_images_in_folders(zip_file_path):
    image_counts = {}
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.is_dir():
                folder_name = os.path.dirname(file_info.filename)
                if folder_name not in image_counts:
                    image_counts[folder_name] = 0
            else:
                folder_name = os.path.dirname(file_info.filename)
                if folder_name not in image_counts:
                    image_counts[folder_name] = 1
                else:
                    image_counts[folder_name] += 1
    return image_counts

def train_model(zip_file_path):
    extract_dir = 'extracted_images'
    extract_zip(zip_file_path, extract_dir)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        extract_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        extract_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    st.write("Training progress:")
    progress_bar = st.progress(0)

    for epoch in range(10):
        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=1,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size
        )
        progress_bar.progress((epoch + 1) / 10)

    os.system(f'rm -rf {extract_dir}')

    return model, train_generator

def detect_objects(model, img):
    # Perform object detection
    results = model.predict(img)

    # Initialize annotator
    annotator = Annotator(img)

    # List to store cropped images
    cropped_images = []

    # Loop through the detections and annotate the image
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            result = results[0]
            class_id = result.names[box.cls[0].item()]
            annotator.box_label(b, str(class_id))

            # Crop the object from the image
            x1, y1, x2, y2 = map(int, b)
            cropped_img = img[y1:y2, x1:x2]
            cropped_images.append((cropped_img, class_id))

    # Get annotated image
    annotated_img = annotator.result()

    return annotated_img, cropped_images

def main():
    st.sidebar.title("Contents")

    # Create a dropdown for selecting the section
    selection = st.sidebar.selectbox("Select Section", ["Introduction", "Image Segmentation", "Dynamic Image Classification", "LSTM Datasets", "CSV Datasets", "Results", "About"])

    if selection == "Introduction":
        st.title("Project Overview")
        st.header("Problem Statement:")
        st.write("""
            Our project addresses the challenge of efficiently analyzing large volumes of unstructured data, including images and text. Traditional methods struggle with this task, leading to time-consuming and error-prone manual processing.
            We aim to develop an intelligent data analysis platform using machine learning and deep learning techniques along with training models. Our goal is to enable users to extract valuable insights from complex data sets, facilitating informed decision-making. Our platform will offer functionalities such as image segmentation, dynamic classification, and natural language processing, empowering users to unlock the full potential of their data.
                """)
        st.header("Target Audience:")
        st.write("""
            - **Data Scientist:** Explore machine learning techniques for data analysis and predictive modeling.
            - **Python Developer:** Enhance Python skills and learn about its applications in various domains.
            - **Machine Learning Practitioner:** Master machine learning algorithms and applications through practical examples.
            - **Computer Vision Engineer:** Delve into the field of computer vision and image processing.
            """)


    elif selection == "Image Segmentation":
        st.title("Image Segmentation App")

        uploaded_file = st.file_uploader("Upload a zip file", type="zip")

        if uploaded_file is not None:
            st.write("Counting images in each class...")
            image_counts = count_images_in_folders(uploaded_file)
            st.write("Number of images in each class:")
            for folder, count in image_counts.items():
                st.write(f"- {folder}: {count} images")

            st.write("Sit tight !!!")
            model, train_generator = train_model(uploaded_file)
            st.write("Training done!")

            uploaded_image = st.file_uploader("Upload an image to test the model", type=["jpg", "jpeg", "png"])

            if uploaded_image is not None:
                st.write("Testing Image:")
                st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
                img = load_img(uploaded_image, target_size=(224, 224))
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0  

                prediction = model.predict(img_array)

                st.write("Class probabilities:")
                class_indices = train_generator.class_indices
                for class_name, prob in zip(class_indices.keys(), prediction[0]):
                    st.write(f"- {class_name}: {prob*100:.2f}%")

                predicted_class = list(class_indices.keys())[np.argmax(prediction)]
                st.write(f"The image is predicted to belong to class: {predicted_class}")


    elif selection == "Dynamic Image Classification":
        st.title("Dynamic Image Classification")

        model = YOLO('yolov8n.pt')  

        # File uploader for zip file
        uploaded_zip_file = st.file_uploader("Upload a zip file containing images...", type="zip")

        if uploaded_zip_file is not None:
            # Create a temporary directory to extract images
            temp_dir = 'temp'
            os.makedirs(temp_dir, exist_ok=True)

            try:
                # Extract the uploaded zip file
                with ZipFile(uploaded_zip_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Perform object detection on each image in the zip file
                annotated_images = []
                classwise_cropped_images = {}
                for root, _, files in os.walk(temp_dir):
                    for image_file in files:
                        image_path = os.path.join(root, image_file)
                        img = cv2.imread(image_path)
                        if img is not None:
                            annotated_img, cropped_images = detect_objects(model, img)
                            annotated_images.append((annotated_img, image_file))
                
                            # Save cropped images classwise
                            for cropped_img, class_id in cropped_images:
                                class_dir = os.path.join(temp_dir, class_id)
                                os.makedirs(class_dir, exist_ok=True)
                                # Saving the cropped images with the appropriate extension
                                cv2.imwrite(os.path.join(class_dir, f"{image_file}_{class_id}.jpg"), cropped_img)
                                if class_id in classwise_cropped_images:
                                    classwise_cropped_images[class_id].append(cropped_img)
                                else:
                                    classwise_cropped_images[class_id] = [cropped_img]
                        else:
                            st.warning(f"Failed to read image file: {image_file}")


                if annotated_images:
                    # Create a zip file with annotated images
                    annotated_zip_file = 'annotated_images.zip'
                    with ZipFile(annotated_zip_file, 'w') as zip_output:
                        for annotated_img, image_file in annotated_images:
                            zip_output.writestr(image_file, cv2.imencode('.jpg', annotated_img)[1].tobytes())

                    # Provide download link for the zip file
                    st.download_button(label="Download Annotated Images", data=open(annotated_zip_file, 'rb').read(), file_name=annotated_zip_file)

                    # Create zip files for classwise cropped images
                    for class_id, images in classwise_cropped_images.items():
                        class_zip_file = f'{class_id}_cropped_images.zip'
                        with ZipFile(class_zip_file, 'w') as zip_output:
                            for i, image in enumerate(images):
                                zip_output.writestr(f"{class_id}_{i}.jpg", cv2.imencode('.jpg', image)[1].tobytes())

                        # Provide download link for classwise cropped images
                        st.download_button(label=f"Download {class_id} Cropped Images", data=open(class_zip_file, 'rb').read(), file_name=class_zip_file)

                    # Create a zip file containing all the zip files
                    all_zip_file = 'all_files.zip'
                    with ZipFile(all_zip_file, 'w') as zip_output:
                        zip_output.write(annotated_zip_file)
                        for class_id in classwise_cropped_images.keys():
                            class_zip_file = f'{class_id}_cropped_images.zip'
                            zip_output.write(class_zip_file)

                    # Provide download link for all zip files
                    st.header("Download All Zip Files")
                    st.download_button(label="Download All Files", data=open(all_zip_file, 'rb').read(), file_name=all_zip_file)

            except Exception as e:
                st.error(f"Error: {str(e)}")

            finally:
                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    for root, dirs, files in os.walk(temp_dir, topdown=False):
                        for name in files:
                            os.remove(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                    os.rmdir(temp_dir)

    elif selection == "LSTM Datasets":
        
        url = "https://raw.githubusercontent.com/sidd2305/DynamicLSTM/5e054d621262f5971ba1a5b54d8e7ec6b9573baf/hu.csv"
        dataset = pd.read_csv(url)

        class KNNUnsupervised:
            def __init__(self, k):
                self.k = k

            def fit(self, X, y):
                self.X_train = tf.constant(X, dtype=tf.float32)
                self.y_train = tf.constant(y, dtype=tf.float32)

            def predict(self, X):
                X_test = tf.constant(X, dtype=tf.float32)
                distances = tf.reduce_sum(tf.square(tf.expand_dims(X_test, axis=1) - self.X_train), axis=2)
                top_k_indices = tf.argsort(distances, axis=1)[:, :self.k]
                nearest_neighbor_labels = tf.gather(self.y_train, top_k_indices, axis=0)

                # Calculate average values of specified columns for nearest neighbors
                avg_values = tf.reduce_mean(nearest_neighbor_labels, axis=1)

                return avg_values.numpy()

        class KNNUnsupervisedLSTM:
            def __init__(self, k):
                self.k = k

            def fit(self, X, y):
                # Convert string representation of LSTM units to numeric arrays
                max_layers = 0
                y_processed = []
                for units in y[:, 5]:  # Assuming LSTM units are in the 5th column
                    units_array = eval(units) if isinstance(units, str) else [units]
                    max_layers = max(max_layers, len(units_array))
                    y_processed.append(units_array)
                
                # Pad arrays with zeros to ensure uniform length
                for i in range(len(y_processed)):
                    y_processed[i] += [0] * (max_layers - len(y_processed[i]))

                # Convert input and output arrays to TensorFlow constant tensors
                self.X_train = tf.constant(X, dtype=tf.float32)
                self.y_train = tf.constant(y_processed, dtype=tf.float32)

            def predict(self, X):
                X_test = tf.constant(X, dtype=tf.float32)
                distances = tf.reduce_sum(tf.square(tf.expand_dims(X_test, axis=1) - self.X_train), axis=2)
                top_k_indices = tf.argsort(distances, axis=1)[:, :self.k]
                nearest_neighbor_labels = tf.gather(self.y_train, top_k_indices, axis=0)
                neighbor_indices = top_k_indices.numpy()

                # Calculate average values of specified columns for nearest neighbors
                avg_values = tf.reduce_mean(nearest_neighbor_labels, axis=1)
                
                return avg_values.numpy(), neighbor_indices


        def split_data(sequence, n_steps):
            X, Y = [], []
            for i in range(len(sequence) - n_steps):
                x_seq = sequence[i:i + n_steps]
                y_seq = sequence.iloc[i + n_steps]
                X.append(x_seq)
                Y.append(y_seq)
            return np.array(X), np.array(Y)

        def handle_date_columns(dat, col):
            # Convert the column to datetime
            dat[col] = pd.to_datetime(dat[col], errors='coerce')
            # Extract date components
            dat[f'{col}_year'] = dat[col].dt.year
            dat[f'{col}_month'] = dat[col].dt.month
            dat[f'{col}_day'] = dat[col].dt.day
            # Extract time components
            dat[f'{col}_hour'] = dat[col].dt.hour
            dat[f'{col}_minute'] = dat[col].dt.minute
            dat[f'{col}_second'] = dat[col].dt.second
        def is_date(string):
            try:
                # Check if the string can be parsed as a date
                parse(string)
                return True
            except ValueError:
                # If parsing fails, also check if the string matches a specific date format
                return bool(re.match(r'^\d{2}-\d{2}-\d{2}$', string))

        def analyze_csv(df):
            # Get the number of records
            num_records = len(df)

            # Get the number of columns
            num_columns = len(df.columns)

            # Initialize counters for textual, numeric, and date columns
            num_textual_columns = 0
            num_numeric_columns = 0
            num_date_columns = 0

            # Identify textual, numeric, and date columns
            for col in df.columns:
                if pd.api.types.is_string_dtype(df[col]):
                    if all(df[col].apply(is_date)):
                        handle_date_columns(df, col)
                        num_date_columns += 1
                    else:
                        num_textual_columns += 1
                elif pd.api.types.is_numeric_dtype(df[col]):
                    num_numeric_columns += 1

            # Find highly dependent columns (you may need to define what "highly dependent" means)
            # For example, you can use correlation coefficients or other statistical measures

            # In this example, let's assume highly dependent columns are those with correlation coefficient above 0.8
            highly_dependent_columns = set()
            correlation_matrix = df.corr()
            for i in range(len(correlation_matrix.columns)):
                for j in range(i):
                    if abs(correlation_matrix.iloc[i, j]) > 0.8:
                        col1 = correlation_matrix.columns[i]
                        col2 = correlation_matrix.columns[j]
                        highly_dependent_columns.add(col1)
                        highly_dependent_columns.add(col2)

            num_highly_dependent_columns = len(highly_dependent_columns)

            # Output the results
            st.write("Number Of Records:", num_records)
            st.write("Number Of Columns:", num_columns)
            st.write("Number of Date Columns:", num_date_columns)

            st.write("Number of Textual Columns:", num_textual_columns)
            st.write("Number of Numeric Columns:", num_numeric_columns)

            st.write("Total Number of highly dependent columns:", num_highly_dependent_columns)
            X = dataset[['Number Of Records', 'Number Of Columns',
                        'Number of Textual Columns', 'Number of Numeric Columns', 'Total Number of highly dependent columns']].values
            y = dataset[['Bidirectional', 'Return Sequence=True', 'Dropout', 'Epochs', 'Batch Size']].values

            knn = KNNUnsupervised(k=3)
            knn.fit(X, y)

            # Input data for which we want to predict the average values
            q1 = np.array([[num_records,num_columns,num_textual_columns,num_numeric_columns,num_highly_dependent_columns]])  # Example input data, 1 row, 6 columns
            avg_neighbors = knn.predict(q1)

            # Apply sigmoid to the first two elements
            for i in range(len(avg_neighbors)):
                # avg_neighbors[i][0] = 1 / (1 + np.exp(-avg_neighbors[i][0]))
                # avg_neighbors[i][1] = 1 / (1 + np.exp(-avg_neighbors[i][1]))
                avg_neighbors[i][0] = 1 if avg_neighbors[i][0] >= 0.5 else 0
                avg_neighbors[i][1] = 1 if avg_neighbors[i][1] >= 0.5 else 0

            # st.write("Output using KNN of info 1-Bidirectional,Return Sequence,Dropout,Epochs,BatchSize:")
            # st.write(avg_neighbors)
            # st.write(avg_neighbors.shape)
            global epoch,batch,drop,returseq,bidi
            #poch,batch,drop,returseq,bidi
            epoch=avg_neighbors[0][3]
            batch=avg_neighbors[0][4]
            drop=avg_neighbors[0][2]
            bidi=avg_neighbors[0][0]
            returnseq=avg_neighbors[0][1]
            # st.write("epoch is",epoch)



            #LSTM Layer
            X = dataset[['Number Of Records', 'Number Of Columns', 
                        'Number of Textual Columns', 'Number of Numeric Columns', 'Total Number of highly dependent columns']].values
            y = dataset[['Bidirectional', 'Return Sequence=True', 'Dropout', 'Epochs', 'Batch Size', 'LSTM Layers']].values
            knn1 = KNNUnsupervisedLSTM(k=3)
            knn1.fit(X, y)
            
        
            avg_neighbors, neighbor_indices = knn1.predict(q1)

            # Extract LSTM units of k-nearest neighbors
            lstm_units = y[neighbor_indices[:, 0], 5]  # Extract LSTM units corresponding to the indices of k-nearest neighbors
            lstm_units_array = []
            for units in lstm_units:
                units_list = [int(x) for x in units.strip('[]').split(',')]
                lstm_units_array.append(units_list)

            # Get the maximum length of nested lists
            max_length = max(len(units) for units in lstm_units_array)

            # Pad shorter lists with zeros to match the length of the longest list
            padded_lstm_units_array = [units + [0] * (max_length - len(units)) for units in lstm_units_array]

            # Convert the padded list of lists to a numpy array
            lstm_units_array_transpose = np.array(padded_lstm_units_array).T

            # Calculate the average of each element in the nested lists
            avg_lstm_units = np.mean(lstm_units_array_transpose, axis=1)

            global output_array_l
            output_array_l = np.array(list(avg_lstm_units))
            # st.write("LSTM Layer Output")
            # st.write(output_array_l)

            #Dense Layer thing
            X = dataset[['Number Of Records', 'Number Of Columns', 
                        'Number of Textual Columns', 'Number of Numeric Columns', 'Total Number of highly dependent columns']].values
            y = dataset[['Bidirectional', 'Return Sequence=True', 'Dropout', 'Epochs', 'Batch Size', 'LSTM Layers', 'Dense Layers']].values
            knn = KNNUnsupervisedLSTM(k=3)
            knn.fit(X, y)
            
            
            avg_neighbors, neighbor_indices = knn.predict(q1)

            # Extract Dense layers of k-nearest neighbors
            dense_layers = y[neighbor_indices[:, 0], 6]  # Extract Dense layers corresponding to the indices of k-nearest neighbors
            dense_layers_array = []
            for layers in dense_layers:
                layers_list = [int(x) for x in layers.strip('[]').split(',')]
                dense_layers_array.append(layers_list)

            # Get the maximum length of nested lists
            max_length = max(len(layers) for layers in dense_layers_array)

            # Pad shorter lists with zeros to match the length of the longest list
            padded_dense_layers_array = [layers + [0] * (max_length - len(layers)) for layers in dense_layers_array]

            # Convert the padded list of lists to a numpy array
            dense_layers_array_transpose = np.array(padded_dense_layers_array).T

            # Calculate the average of each element in the nested lists
            avg_dense_layers = np.mean(dense_layers_array_transpose, axis=1)

            global output_array_d
            # Print the output in the form of an array
            output_array_d = np.array(list(avg_dense_layers))
            # st.write("Dense layer output:")
            # st.write(output_array_d)


        def load_data(file):
            df = pd.read_csv(file)
            st.subheader("1. Show first 10 records of the dataset")
            st.dataframe(df.head(10))
            analyze_csv(df)
            # Call analyze_csv function here

            return df

        def show_correlation(df):
            st.subheader("2. Show the correlation matrix and heatmap")
            numeric_columns = df.select_dtypes(include=['number']).columns
            correlation_matrix = df[numeric_columns].corr()
            st.dataframe(correlation_matrix)

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
            st.pyplot(fig)

        def show_missing_values(df):
            st.subheader("3. Show the number of missing values in each column")
            missing_values = df.isnull().sum()
            st.dataframe(missing_values)
            st.write(output_array_d)

        def handle_missing_values(df):
            st.subheader("4. Handle missing values")
            numeric_columns = df.select_dtypes(include=['number']).columns

            fill_option = st.radio("Choose a method to handle missing values:", ('Mean', 'Median', 'Mode', 'Drop'))

            if fill_option == 'Drop':
                df = df.dropna(subset=numeric_columns)
            else:
                fill_value = (
                    df[numeric_columns].mean() if fill_option == 'Mean'
                    else (df[numeric_columns].median() if fill_option == 'Median'
                        else df[numeric_columns].mode().iloc[0])
                )
                df[numeric_columns] = df[numeric_columns].fillna(fill_value)

            st.dataframe(df)

            return df

        def drop_column(df):
            st.subheader("5. Drop a column")
            columns_to_drop = st.multiselect("Select columns to drop:", df.columns)
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
                st.dataframe(df)

            return df

        def build_model(layer_sizes, dense_layers, return_sequence, bidirectional, dropout):
            model = tf.keras.Sequential()
            
            for i, size in enumerate(layer_sizes):
                size = int(size) 
                if i == 0:
                    # For the first layer, we need to specify input_shape
                    # model.add(LSTM(size, return_sequences=bool(return_sequence))) then did  model.add(LSTM(size,input_shape=(c,d), return_sequences=True))
                    model.add(LSTM(size,input_shape=(X_train.shape[1], 1), return_sequences=True))
                    
                else:
                    if bool(bidirectional):  # Bidirectional layer
                        model.add(Bidirectional(LSTM(size, return_sequences=True)))
                    else:
                        model.add(LSTM(size,return_sequences=True))

                if dropout > 0:  # Dropout
                    model.add(Dropout(dropout))

            for nodes in dense_layers:
                model.add(Dense(nodes, activation='relu'))

            model.add(Dense(1))  # Example output layer, adjust as needed

            model.compile(optimizer='adam', loss='mse')  # Compile the model
            model.build()  # Explicitly build the model

            return model

        def train_regression_model(df):
            st.subheader("6. Train a Custom Time Series model")

            if df.empty:
                st.warning("Please upload a valid dataset.")
                return

            st.write("Select columns for X (features):")
            st.write("Please DO NOT select your date column.We have automatically pre processed it into date,month,year(hour,min,sec if applicable)")
            st.write("Please do select our preproccesed date columns")
            x_columns = st.multiselect("Select columns for X:", df.columns)

            if not x_columns:
                st.warning("Please select at least one column for X.")
                return

            st.write("Select the target column for Y:")
            y_column = st.selectbox("Select column for Y:", df.columns)

            if not y_column:
                st.warning("Please select a column for Y.")
                return

            df = df.dropna(subset=[y_column])

            X = df[x_columns]
            y = df[y_column]

            # Handle textual data
            textual_columns = X.select_dtypes(include=['object']).columns
            if not textual_columns.empty:
                for col in textual_columns:
                    X[col] = X[col].fillna("")  # Fill missing values with empty strings
                    vectorizer = TfidfVectorizer()  # You can use any other vectorization method here
                    X[col] = vectorizer.fit_transform(X[col])

            numeric_columns = X.select_dtypes(include=['number']).columns
            scaler_option = st.selectbox("Choose a scaler for numerical data:", ('None', 'StandardScaler', 'MinMaxScaler'))

            if scaler_option == 'StandardScaler':
                scaler = StandardScaler()
                X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
            elif scaler_option == 'MinMaxScaler':
                scaler = MinMaxScaler()
                X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
            global X_train,y_train,a,b,c,d
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            a = X_train.shape
            b = y_train.shape

            c=a[0]
            d=b[0]
            
            st.subheader("6.1-Information About Training")
            
            st.write("We have dynamically determined the Architecture of your model using an KNN model trained on our CSV properties vs architecture dataset ")
            lstm = [int(x) for x in output_array_l]
            dense = [int(x) for x in output_array_d]
            model = build_model(lstm,dense,returnseq,bidi,drop)
            model.summary()
            print(model.summary())
        
            st.write("We are going to be training your dataset from our dynamically determined hyperparameters!")
            st.write("The Parameters for your CSV are:")
            st.write("Batch Size",int(batch)) 
            st.write("Epochs",int(epoch))
            st.write("Dropout Value",drop)
            st.write("Bidirectional is",bool(bidi))
            st.write("LSTM Layers",output_array_l)
            st.write("Dense Layers",output_array_d)

            st.write("While we train,here's a video that should keep you entertained while our algorithm works behind the scenes🎞️!")
            st.write("I mean,who doesn`t like a friends episode?🤔👬🏻👭🏻🫂")
            video_url = "https://www.youtube.com/watch?v=nvzkHGNdtfk&pp=ygUcZnJpZW5kcyBlcGlzb2RlIGZ1bm55IHNjZW5lcw%3D%3D"  # Example YouTube video URL
            st.video(video_url)


            # Train the model
        
            n_steps = 7

        # Call the split_data function with X_train and Y_train
            X_train_split, Y_train_split = split_data(X_train, n_steps), split_data(y_train, n_steps)
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=int(epoch), batch_size=int(batch))
            global train_loss
            global val_loss
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']

            st.subheader("Training Information➕➖")
            st.write("Final Training loss is-",train_loss[-1])
            st.write("Final Validation loss is-",val_loss[-1])
            st.write("Training losses",train_loss)
            st.write("Validation losses",val_loss)
            # st.write(f"LSTM Model: {model_option}")

            # # Evaluate the model
            # loss, accuracy = model.evaluate(X_test, y_test)
            # st.write(f"Loss: {loss}")
        

        # Assuming history is available with the 'loss' and 'val_loss' keys
        
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']
            
            ploty()

            model_filename = "model.h5"
            model.save(model_filename)
            st.success(f"Model saved as {model_filename}")

            st.subheader("8.Download the trained model")
            st.download_button(label="Download Model", data=open(model_filename, "rb").read(), file_name=model_filename)

        def ploty():
            st.subheader("7.Plotting the loss vs epoch graph")
            epochsi = range(1, len(train_loss) + 1)

            plt.plot(epochsi, train_loss, 'bo', label='Training loss') # 'bo' = blue dots
            plt.plot(epochsi, val_loss, 'r', label='Validation loss') # 'r' = red line   
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            st.write("Yayyyy yipeee!! Now we`re done with processing and training the model!🥳🎉")

            # Optionally, you can save the plot or display it
            # plt.savefig('loss_plot.png')  # Save the plot as a PNG file
            # plt.show()  # Display the plot
            #newest
            st.pyplot(plt)


        def download_updated_dataset(df):
            st.subheader("9. Download the updated dataset")
            csv_file = df.to_csv(index=False)
            st.download_button("Download CSV", csv_file, "Updated_Dataset.csv", key="csv_download")

        
        st.title("LSTM Time Series Dataset Analysis and Model Training App")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            st.info("File uploaded successfully!")
            df = load_data(uploaded_file)

            if not df.select_dtypes(include=['number']).empty:
                show_correlation(df)
                show_missing_values(df)
                df = handle_missing_values(df)

            df = drop_column(df)
            train_regression_model(df)
    
            download_updated_dataset(df)
        

    elif selection == "CSV Datasets":
        url = "sisi.csv"
        dataset = pd.read_csv(url)
        

        class LazyPredict:
            def __init__(self, df, x_columns, y_column):
                self.data = df
                self.target_column = y_column
                self.X = self.data[x_columns]
                self.y = self.data[y_column]
                self.is_regression = self.is_regression()
                self.models = {}  # Dictionary to store trained models

            def is_regression(self):
                # Calculate the number of unique values in the target column
                num_unique_values = self.y.nunique()
                
                # If the number of unique values is below a threshold, consider it as classification
                classification_threshold = 10  # You can adjust this threshold as needed
                if num_unique_values <= classification_threshold:
                    return False  # It's a classification problem
                else:
                    return True 

            def fit_predict(self):
                if self.is_regression:
                    models = {
                        "Linear Regression": LinearRegression(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "Random Forest": RandomForestRegressor(),
                        "XGBoost": XGBRegressor(),
                        "AdaBoost": AdaBoostRegressor(),
                        "SGDRegressor": SGDRegressor()
                    }
                    results = {}
                    for name, model in models.items():
                        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        variance = np.var(y_test)
                        accuracy = (1 - (mse / variance))*100
                        results[name] = accuracy

                        if accuracy > 80:  # Save the model if accuracy is greater than 80%
                            self.models[name] = model
                else:
                    models = {
                        "Logistic Regression": LogisticRegression(),
                        "Decision Tree": DecisionTreeClassifier(),
                        "Random Forest": RandomForestClassifier(),
                        "XGBoost": XGBClassifier(),
                        "AdaBoost": AdaBoostClassifier(),
                        "SGDClassifier": SGDClassifier()
                    }
                    results = {}
                    for name, model in models.items():
                        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)*100
                        results[name] = accuracy

                        if accuracy > 80:  # Save the model if accuracy is greater than 80%
                            self.models[name] = model

                return results

            


            def predict_new_data(self, new_data):
                if self.is_regression:
                    model = LinearRegression()
                else:
                    model = LogisticRegression()

                model.fit(self.X, self.y)
                predictions = model.predict(new_data)

                return predictions


        class KNNUnsupervised:
            def __init__(self, k):
                self.k = k

            def fit(self, X, y):
                self.X_train = tf.constant(X, dtype=tf.float32)
                self.y_train = tf.constant(y, dtype=tf.float32)

            def predict(self, X):
                X_test = tf.constant(X, dtype=tf.float32)
                distances = tf.reduce_sum(tf.square(tf.expand_dims(X_test, axis=1) - self.X_train), axis=2)
                top_k_indices = tf.argsort(distances, axis=1)[:, :self.k]
                nearest_neighbor_labels = tf.gather(self.y_train, top_k_indices, axis=0)

                # Calculate average values of specified columns for nearest neighbors
                avg_values = tf.reduce_mean(nearest_neighbor_labels, axis=1)

                return avg_values.numpy()

        class KNNUnsupervisedLSTM:
            def __init__(self, k):
                self.k = k

            def fit(self, X, y):
                # Convert string representation of LSTM units to numeric arrays
                max_layers = 0
                y_processed = []
                for units in y[:, 0]:  # Assuming LSTM units are in the 5th column
                    units_array = eval(units) if isinstance(units, str) else [units]
                    max_layers = max(max_layers, len(units_array))
                    y_processed.append(units_array)
                
                # Pad arrays with zeros to ensure uniform length
                for i in range(len(y_processed)):
                    y_processed[i] += [0] * (max_layers - len(y_processed[i]))

                # Convert input and output arrays to TensorFlow constant tensors
                self.X_train = tf.constant(X, dtype=tf.float32)
                self.y_train = tf.constant(y_processed, dtype=tf.float32)

            def predict(self, X):
                X_test = tf.constant(X, dtype=tf.float32)
                distances = tf.reduce_sum(tf.square(tf.expand_dims(X_test, axis=1) - self.X_train), axis=2)
                top_k_indices = tf.argsort(distances, axis=1)[:, :self.k]
                nearest_neighbor_labels = tf.gather(self.y_train, top_k_indices, axis=0)
                neighbor_indices = top_k_indices.numpy()

                # Calculate average values of specified columns for nearest neighbors
                avg_values = tf.reduce_mean(nearest_neighbor_labels, axis=1)
                
                return avg_values.numpy(), neighbor_indices

        def handle_date_columns(dat, col):
            # Convert the column to datetime
            dat[col] = pd.to_datetime(dat[col], errors='coerce')
            # Extract date components
            dat[f'{col}_year'] = dat[col].dt.year
            dat[f'{col}_month'] = dat[col].dt.month
            dat[f'{col}_day'] = dat[col].dt.day
            # Extract time components
            dat[f'{col}_hour'] = dat[col].dt.hour
            dat[f'{col}_minute'] = dat[col].dt.minute
            dat[f'{col}_second'] = dat[col].dt.second
        def is_date(string):
            try:
                # Check if the string can be parsed as a date
                parse(string)
                return True
            except ValueError:
                # If parsing fails, also check if the string matches a specific date format
                return bool(re.match(r'^\d{2}-\d{2}-\d{2}$', string))

        def analyze_csv(df):
            # Get the number of records
            num_records = len(df)

            # Get the number of columns
            num_columns = len(df.columns)
            
            # Initialize counters for textual, numeric, and date columns
            num_textual_columns = 0
            num_numeric_columns = 0
            num_date_columns = 0

            # Identify textual, numeric, and date columns
            for col in df.columns:
                if pd.api.types.is_string_dtype(df[col]):
                    if all(df[col].apply(is_date)):
                        handle_date_columns(df, col)
                        num_date_columns += 1
                    else:
                        num_textual_columns += 1
                elif pd.api.types.is_numeric_dtype(df[col]):
                    num_numeric_columns += 1
            
            textual_columns = df.select_dtypes(include=['object']).columns
            label_encoders = {}
            for col in textual_columns:
                if col not in df.columns:
                    continue
                le = LabelEncoder()
                df[col] = df[col].fillna("")  # Fill missing values with empty strings
                df[col] = le.fit_transform(df[col])
                # Store the label encoder for inverse transformation
                label_encoders[col] = le

                # Add another column for reverse inverse label encoding
                #df[f'{col}_inverse'] = le.inverse_transform(df[col])

        

            
            highly_dependent_columns = set()
            correlation_matrix = df.corr()
            for i in range(len(correlation_matrix.columns)):
                for j in range(i):
                    if abs(correlation_matrix.iloc[i, j]) > 0.8:
                        col1 = correlation_matrix.columns[i]
                        col2 = correlation_matrix.columns[j]
                        highly_dependent_columns.add(col1)
                        highly_dependent_columns.add(col2)

            num_highly_dependent_columns = len(highly_dependent_columns)

                ##Output the results
            st.write("Number Of Records:", num_records)
            st.write("Number Of Columns:", num_columns)
            st.write("Number of Date Columns:", num_date_columns)

            st.write("Number of Textual Columns:", num_textual_columns)
            st.write("Number of Numeric Columns:", num_numeric_columns)

            st.write("Total Number of highly dependent columns:", num_highly_dependent_columns)
            X = dataset[['Number Of Records', 'Number Of Columns',
                        'Number of Textual Columns', 'Number of Numeric Columns', 'Total Number of highly dependent columns']].values
            y = dataset[['Optimizer','Dropout', 'Epochs', 'Batch Size']].values

            knn = KNNUnsupervised(k=3)
            knn.fit(X, y)

            # Input data for which we want to predict the average values
            q1 = np.array([[num_records,num_columns,num_textual_columns,num_numeric_columns,num_highly_dependent_columns]])  # Example input data, 1 row, 6 columns
            avg_neighbors = knn.predict(q1)

            # Apply sigmoid to the first two elements
            for i in range(len(avg_neighbors)):
                # avg_neighbors[i][0] = 1 / (1 + np.exp(-avg_neighbors[i][0]))
                # avg_neighbors[i][1] = 1 / (1 + np.exp(-avg_neighbors[i][1]))
                avg_neighbors[i][0] = 1 if avg_neighbors[i][0] >= 0.5 else 0
                # avg_neighbors[i][1] = 1 if avg_neighbors[i][1] >= 0.5 else 0

            # st.write("Output using KNN of info 1-Bidirectional,Return Sequence,Dropout,epochss,BatchSize:")
            # st.write(avg_neighbors)
            # st.write(avg_neighbors.shape)
            global epochs,batchs,drops,returseqs,bidis,opi
            #poch,batch,drops,returseq,bidi
            epochs=avg_neighbors[0][2]
            batchs=avg_neighbors[0][3]
            drops=avg_neighbors[0][1]
            opi=avg_neighbors[0][0]

        

            # #Dense Layer thing
            X = dataset[['Number Of Records', 'Number Of Columns', 
                        'Number of Textual Columns', 'Number of Numeric Columns', 'Total Number of highly dependent columns']].values
            y = dataset[['Hidden units']].values
            knn = KNNUnsupervisedLSTM(k=3)
            knn.fit(X, y)
            
            
            avg_neighbors, neighbor_indices = knn.predict(q1)

            # Extract Dense layers of k-nearest neighbors
            dense_layers = y[neighbor_indices[:, 0], 0]  # Extract Dense layers corresponding to the indices of k-nearest neighbors
            dense_layers_array = []
            for layers in dense_layers:
                layers_list = [int(x) for x in layers.strip('[]').split(',')]
                dense_layers_array.append(layers_list)

            # Get the maximum length of nested lists
            max_length = max(len(layers) for layers in dense_layers_array)

            # Pad shorter lists with zeros to match the length of the longest list
            padded_dense_layers_array = [layers + [0] * (max_length - len(layers)) for layers in dense_layers_array]

            # Convert the padded list of lists to a numpy array
            dense_layers_array_transpose = np.array(padded_dense_layers_array).T

            # Calculate the average of each element in the nested lists
            avg_dense_layers = np.mean(dense_layers_array_transpose, axis=1)

            global output_array_d
            # Print the output in the form of an array
            output_array_d = np.array(list(avg_dense_layers))
            # st.write("Dense layer output:")
            # st.write(output_array_d)





        def load_data(file):
            df = pd.read_csv(file)
            st.subheader("1. Show first 10 records of the dataset")
            st.dataframe(df.head(10))
            analyze_csv(df)
            df.dropna(inplace=True)
            
            # Handle textual columns using label encoding
        
            # Call analyze_csv function here

            return df

        def show_correlation(df):
            st.subheader("3. Show the correlation matrix and heatmap")
            numeric_columns = df.select_dtypes(include=['number']).columns
            correlation_matrix = df[numeric_columns].corr()
            st.dataframe(correlation_matrix)

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
            st.pyplot(fig)

        def show_missing_values(df):
            st.subheader("2. Show the number of missing values in each column")
            missing_values = df.isnull().sum()
            st.dataframe(missing_values)
        # st.write(output_array_d)

        def handle_missing_values(df):
            st.subheader("4. Handle missing values")
            numeric_columns = df.select_dtypes(include=['number']).columns
        
            textual_columns = df.select_dtypes(include=['object']).columns

            fill_option = st.radio("Choose a method to handle missing values:", ('Mean', 'Median', 'Mode', 'Drop'))

            if fill_option == 'Drop':
                df = df.dropna(subset=numeric_columns)
            else:
                fill_value = (
                    df[numeric_columns].mean() if fill_option == 'Mean'
                    else (df[numeric_columns].median() if fill_option == 'Median'
                        else df[numeric_columns].mode().iloc[0])
                )
                df[numeric_columns] = df[numeric_columns].fillna(fill_value)

        
            return df

        def drop_column(df):
            st.subheader("5. Drop a column")
            columns_to_drop = st.multiselect("Select columns to drop:", df.columns)
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
                st.dataframe(df)

            return df

        def build_model(dense_layers,dropout):
            model = tf.keras.Sequential()
            
            for i, size in enumerate(dense_layers):
                size = int(size) 
                if i == 0:
                    # For the first layer, we need to specify input_shape
                    # model.add(LSTM(size, return_sequences=bool(return_sequence))) then did  model.add(LSTM(size,input_shape=(c,d), return_sequences=True))
                    model.add(Dense(size,input_shape=(X_train.shape[1], 1)))
                else:
                    model.add(Dense(size))
                
                
                    
                
        
                if dropout > 0:  # Dropout
                    model.add(Dropout(dropout))

            for nodes in dense_layers:
                model.add(Dense(nodes, activation='relu'))

            model.add(Dense(1))  # Example output layer, adjust as needed
            if(opi==0):
                model.compile(optimizer='adam', loss='mse')  # Compile the model
            else:
                model.compile(optimizer='sgd', loss='mse') 

            model.build()  # Explicitly build the model

            return model

        def train_regression_model(df):
            st.subheader("6. Train a model")

            if df.empty:
                st.warning("Please upload a valid dataset.")
                return

            st.write("Select columns for X (features):")
            x_columns = st.multiselect("Select columns for X:", df.columns)

            if not x_columns:
                st.warning("Please select at least one column for X.")
                return

            st.write("Select the target column for Y:")
            y_column = st.selectbox("Select column for Y:", df.columns)

            if not y_column:
                st.warning("Please select a column for Y.")
                return
            lp = LazyPredict(df, x_columns, y_column)
            results = lp.fit_predict()

            # Check if any model's accuracy is less than 80 percent
            proceed_with_ann = any(accuracy >= 80.0 for accuracy in results.values())

            df = df.dropna(subset=[y_column])

            X = df[x_columns]
            y = df[y_column]

            # Handle textual data
            textual_columns = X.select_dtypes(include=['object']).columns
            if not textual_columns.empty:
                for col in textual_columns:
                    X[col] = X[col].fillna("")  # Fill missing values with empty strings
                    vectorizer = TfidfVectorizer()  # You can use any other vectorization method here
                    X[col] = vectorizer.fit_transform(X[col])

            numeric_columns = X.select_dtypes(include=['number']).columns
            scaler_option = st.selectbox("Choose a scaler for numerical data:", ('None', 'StandardScaler', 'MinMaxScaler'))

            if scaler_option == 'StandardScaler':
                scaler = StandardScaler()
                X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
            elif scaler_option == 'MinMaxScaler':
                scaler = MinMaxScaler()
                X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
            global X_train,y_train,a,b,c,d
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            a = X_train.shape
            b = y_train.shape

            c=a[0]
            d=b[0]
            
            st.subheader("6.1-Information About Training")
            
            st.write("We have dynamically determined the Architecture of your model using an KNN model trained on our CSV properties vs architecture dataset ")
            # lstm = [int(x) for x in output_array_l]
            dense = [int(x) for x in output_array_d]
            
            # Use LazyPredict to get model accuracies
        
            if proceed_with_ann:
                st.write("One or more models from LazyPredict have accuracy more than 80%. Skipping ANN training.")
                sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
            
                for model, accuracy in sorted_results.items():
                    st.write(f"- {model}: {accuracy:.2f}%")
                max_accuracy_model = max(results, key=results.get)
                best_lp_model = lp.models[max_accuracy_model]

                # Save the best LazyPredict model
                lp_model_filename = f"best_lp_model.pkl"
                joblib.dump(best_lp_model, lp_model_filename)
                st.write("Yayyyy yipeee!! Now we`re done with processing and training the model!🥳🎉")
                # Provide a download button for the best LazyPredict model
                st.subheader("7.Download Best LazyPredict Model")
                st.write("Click the button below to download the best LazyPredict model:")
                st.download_button(label="Download LazyPredict Model", data=open(lp_model_filename, "rb").read(), file_name=lp_model_filename)
                
                
            else:
                model = build_model(dense, drops)
                model.summary()
                st.write("We are going to be training your dataset from our dynamically determined hyperparameters!")
                st.write("The Parameters for your CSV are:")
                st.write("Batch Size", int(batchs)) 
                st.write("Epochs", int(epochs))
                st.write("Dropout Value", drops)
                if opi == 0:
                    st.write("Adam Optimizer Chosen")
                else:
                    st.write("SGD Optimizer Chosen")
                st.write("Dense Layers", output_array_d)
                st.write("While we train, here`s a video that should keep you entertained while our algorithm works behind the scenes🎞️!")
                st.write("I mean, who doesn`t like a friends episode?🤔👬🏻👭🏻🫂")
                video_url = "https://www.youtube.com/watch?v=nvzkHGNdtfk&pp=ygUcZnJpZW5kcyBlcGlzb2RlIGZ1bm55IHNjZW5lcw%3D%3D"  # Example YouTube video URL
                st.video(video_url)

                history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=int(epochs), batch_size=int(batchs))

                global train_loss
                global val_loss
                train_loss = history.history['loss']
                val_loss = history.history['val_loss']

                st.subheader("Training Information➕➖")
                st.write("Final Training loss is-", train_loss[-1])
                st.write("Final Validation loss is-", val_loss[-1])
                st.write("Training losses", train_loss)
                st.write("Validation losses", val_loss)
            

                ploty()

                model_filename = "model.h5"
                model.save(model_filename)
                st.success(f"Model saved as {model_filename}")

                st.subheader("7.Download the trained model")
                st.download_button(label="Download Model", data=open(model_filename, "rb").read(), file_name=model_filename)
                # Save LazyPredict models


        def ploty():
            st.subheader("Plotting the loss vs epoch graph")
            epochsi = range(1, len(train_loss) + 1)

            plt.plot(epochsi, train_loss, 'bo', label='Training loss') # 'bo' = blue dots
            plt.plot(epochsi, val_loss, 'r', label='Validation loss') # 'r' = red line   
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            st.write("Yayyyy yipeee!! Now we`re done with processing and training the model!🥳🎉")


            st.pyplot(plt)


        def download_updated_dataset(df):
            st.subheader("8. Download the updated dataset")
            csv_file = df.to_csv(index=False)
            st.download_button("Download CSV", csv_file, "Updated_Dataset.csv", key="csv_download")

        
        st.title("CSV Dataset Analysis and Model Training App")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            st.info("File uploaded successfully!")
            df = load_data(uploaded_file)
            
            if not df.select_dtypes(include=['number']).empty or df.select_dtypes(include=['object']).empty :
                show_missing_values(df)#hi
                show_correlation(df)
                df = handle_missing_values(df)
                
            
            
            df = drop_column(df)
            train_regression_model(df)
    
            download_updated_dataset(df)
        

    elif selection == "Results":
        
        
        st.write("### LSTM Analysis")
        df_lstm = pd.DataFrame({
            'S.No': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Testing Loss Tabulation': ["Tesla Dataset","Traffic Dataset","Air Passengers","Panama Electricity","Google Train","Apple","Netflix","London Bike","Electricity_dah","LSTM-Multivariate_pollution"],
            'X Columns': ["Date_year,Date_month,Open,High,Low","Date_year,Date_month","Date_year,Date_month","datetime_year,datetime_month,T2M_toc","Date_year,Open,High,Volume","Date_year,Date_month,Open,High,Volume","Date_year,Date_month,Open,High,Volume","timestamp_year,timestamp_month,wind_speed,weather_code,is_holiday,is_weekend","date_year,date_month,date_day","date_day,date_hour,wnd_spd,pressdew,pollution"],
            'Y Columns': ["Close","Vehicles","Passengers","T2M_san","Close","Close","Close","hum","France","temp"],
            'Initial Loss': [1745838.125,952.472,96211.2266,204.77,3683.0974,908.415,187812.625,5417.0169,90821.7969,300.1845],
            'Final Loss': [475.4735,639.267,51822.6016,6.1987,3336.0879,10.7328,23196.4355,3242.8433,15822.5752,173.9059],
            'Decrease in Loss': [1745362.652,313.205,44388.625,198.5713,347.0095,897.6822,164616.1895,2174.1736,74999.2217,126.2786],
            'Percent Decrease %': [99.97276532,32.88338135,46.13663766,96.97284759,9.421675897,98.81851356,87.6491607,40.13599441,82.5784385,42.06699546]
        })
        st.write(df_lstm)
        st.write("Average Reduction In Loss % : 63.66364104 ")

        st.write("### CSV Dataset Analysis")
        df_csv = pd.DataFrame({
            'Dataset Name': ["Iris Species","Titanic","California Housing Prices","Mobile price prediction","heart_failure_clinical_records_dataset","breast cancer wisconsin","Churn Modelling","Rain in Australia","Bank Customer Churn Prediction","Activity Recognition in Senior Citizens","Medical Cost Personal Datasets","Digit Recognizer","Mushroom Classification","Stoke prediction","Credit Card Fraud Detection","Campus Recruitment","early_stage_diabetes_risk_prediction","Price of Used Toyota Corolla Cars","Supermarket store branches sales analysis","Biomechanical features of orthopedic patients","Pima Indians Diabetes Database","Amazon Musical Instruments Reviews","Natural Language Processing with Disaster Tweets","Passenger Satisfaction","Obesity or CVD risk","Santander Value Prediction","Hotels Booking Data"],
            'Number Of Records': [150, 891, 20640, 2000, 299, 569, 10000, 145460, 10000, 103860, 1338, 30402, 8124, 5110, 257457, 215, 520, 1436, 896, 310, 768, 10261, 7613, 129880, 2111, 4459, 119390],
            'Number Of Columns': [6, 12, 10, 21, 13, 33, 14, 23, 14, 8, 7, 785, 23, 12, 31, 15, 17, 39, 5, 7, 9, 9, 5, 24, 17, 4993, 32],
            'Number Of Date/Time Colums': [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2],
            'Number of Textual Columns': [1, 5, 1, 0, 0, 1, 3, 6, 3, 0, 3, 0, 23, 5, 0, 8, 16, 3, 0, 1, 0, 6, 3, 5, 9, 1, 10],
            'Number of Numeric Columns': [5, 7, 9, 21, 13, 32, 11, 16, 11, 7, 4, 785, 0, 7, 31, 7, 1, 36, 5, 6, 9, 2, 2, 19, 8, 4992, 20],
            'Number of highly dependent columns': [4, 0, 6, 2, 0, 23, 0, 6, 0, 2, 0, 250, 0, 0, 0, 0, 0, 8, 2, 2, 0, 0, 0, 2, 0, 937, 0],
            'Dropout': [0, 0, 0, 0, 0.25, 0, 0.1, 0.25, 0, 0.2, 0, 0.3, 0.2, 0, 0, 0.5, 0.1, 0.1, 0, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0],
            'Optimizer': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'Hidden units': [[48,16,3], [18,60,1], [50,50,20], [8,6,4], [16,8,4,1], [16,16,1], [6,6,1], [32,32,16,8,1], [6,6,1], [64,9], [12,8,4,1], [350,165,64,10], [20,20,1], [256,256,2], [15,6,1], [36,36,2], [9,9,9,1], [64,64,64,1], [100,100,50], [96,48,1], [128,128,1], [75,50,25,10,1], [128,32,1], [256,64,16,4,1], [128,128,7], [64,32,16,8], [27,27,1]],
            'Epochs': [5, 30, 1, 105, 500, 400, 100, 150, 100, 10, 300, 50, 50, 150, 70, 150, 100, 100, 100, 100, 20, 10, 12, 100, 5, 30, 10],
            'Batch Size': [1, 60, 15, 64, 32, 400, 10, 32, 10, 64, 50, 50, 32, 30, 40, 100, 12, 12, 896, 310, 32, 32, 32, 32, 32, 16, 32]
        })
        st.write(df_csv)
        st.write("### Image Segmentation Analysis")
        df_performance = pd.DataFrame({
        'Dataset Name': ["Dogs vs Cats", "Medical Images", "Autonomous Driving", "Satellite Imagery", "Histopathology", "Semantic Segmentation Benchmark", "Lung Nodule Detection", "Plant Disease Identification"],
        'Accuracy': ["90%", "95%", "85%", "91%", "88%", "90%", "91%", "92%"],
        'Precision': ["86%", "90%", "84%", "88%", "87%", "89%", "88%", "82%"],
        'Recall': ["88%", "91%", "87%", "90%", "89%", "92%", "91%", "87%"],
        'Intersection over Union': ["82%", "85%", "79%", "84%", "81%", "86%", "85%", "78%"],
        'Dice Coefficient': ["84%", "87%", "82%", "86%", "83%", "88%", "87%", "80%"]
        })
        st.write(df_performance)
        st.write("Average Accuracy % : 90")

    elif selection == "About":
        st.title("The Team")

        # Define information for each person
        people_info = [
            {
                "name": "Siddhanth Sridhar",
                "intro": "Meet Siddhanth Sridhar, a fervent Computer Science Engineering (CSE) undergraduate at PES University, deeply immersed in the realm of Machine Learning. Fueled by curiosity and boundless enthusiasm, he continuously delves into the intricacies of artificial intelligence. He staunchly believes in technology's potential to reshape industries and enhance livelihoods, propelling him to the forefront of this exhilarating revolution.",
                "linkedin": "https://www.linkedin.com/in/siddhanth-sridhar-4aa9aa222/",
                "github": "https://github.com/sidd2305",
                "image": "Sid_profile.jpeg"
            },
            {
                "name": "Swaraj Khan",
                "intro": "Meet Swaraj Khan, a driven B.Tech student at Dayananda Sagar University, immersing himself in the realm of Computer Science with a special focus on machine learning. With an unwavering commitment to tackling real-world challenges, Swaraj harnesses the power of technology to unravel complexities and pave the way for innovative solutions.",
                "linkedin": "https://www.linkedin.com/in/swaraj-khan/",
                "github": "https://github.com/swaraj-khan",
                "image": "Swaraj_profile.jpeg" 
            },
            {
                "name": "Shreya Chaurasia",
                "intro": "Introducing Shreya Chaurasia, a B.Tech Computer Science scholar driven by an insatiable curiosity for Machine Learning. Ambitious, passionate, and self-motivated, she finds the potential of ML to revolutionize industries utterly captivating. Delving into data to reveal patterns and derive insights, she thrives on crafting innovative solutions. Challenges are her stepping stones to growth, and she relentlessly pursues excellence in all her pursuits.",
                "linkedin": "https://www.linkedin.com/in/shreya-chaurasia-4b5407266/",
                "github": "https://github.com/shreyyasiaa",
                "image": "Shreya_profile.jpeg"
            }
        ]

        # Display information for each person
        for person in people_info:
            st.write(f"### {person['name']}")

            # Display profile image
            st.image(person['image'], caption=f"Profile Image - {person['name']}", width=150)

            # Display introduction text
            st.write(person['intro'])

            # Display LinkedIn and GitHub links
            st.markdown(f"**LinkedIn:** [{person['name']}'s LinkedIn Profile]({person['linkedin']})")
            st.markdown(f"**GitHub:** [{person['name']}'s GitHub Profile]({person['github']})")


if __name__ == "__main__":
    main()

