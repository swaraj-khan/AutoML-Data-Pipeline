# Dynamic AutoML: Comprehensive Solution for Diverse Data Tasks

Dynamic AutoML is a versatile platform designed to streamline various data tasks, including CSV analysis, LSTM modeling, and image classification and detection. Our platform offers advanced features and capabilities to empower developers in handling diverse datasets efficiently.


## Dynamic AutoML: A Closer Look

#### CSV Dataset Analysis and LazyPredict Model

**Dynamic Dataset Architecture**
- The architecture of CSV datasets is dynamically determined by analyzing their structure, including the number of records, columns, and their types (textual, numeric, date). This dynamic approach ensures that our platform can handle a wide range of dataset formats without requiring manual configuration.

**LazyPredict Model Implementation**
- Our LazyPredict model is implemented from scratch to provide developers with a comprehensive tool for model selection and comparison. By automating the process of evaluating multiple models with various configurations, developers can quickly identify the most suitable model for their specific task, saving time and effort.

#### Image Classification and Detection

**Automatic Model Training**
- Our platform simplifies the process of image classification by automating model training. Developers can upload a zip folder containing images organized into folders as classes. The platform then trains models using this data, enabling accurate classification of new images based on their content.

**Dynamic Image Segmentation**
- Using techniques like YOLO, our platform dynamically determines classes from datasets such as COCO128, enabling precise image segmentation. This capability allows developers to identify and isolate specific objects within images, opening up possibilities for applications such as medical image analysis, autonomous vehicles, and more.

#### LSTM Model Training

**Dynamic Architecture Determination**
- Similar to CSV datasets, the architecture of LSTM models is dynamically determined based on the dataset's characteristics. This approach ensures that the model architecture is optimized for the specific task and dataset, leading to improved performance and adaptability.

**Streamlined Model Training**
- Our platform streamlines the process of LSTM model training by automatically tuning hyperparameters based on dataset properties. This automation reduces the manual effort required for hyperparameter optimization, allowing developers to focus on model experimentation and refinement.

## Usage

1. **Upload Data:** Upload CSV files or image datasets to start your analysis and model training.
2. **Dataset Exploration:** Explore dataset properties, perform preprocessing tasks, and visualize data distributions.
3. **Model Training:** Choose appropriate models (LSTM, image classifiers, etc.) and train them using automated processes.
4. **Model Evaluation:** Evaluate model performance using metrics and visualizations provided by the platform.
5. **Deployment:** Download trained models for deployment or integrate them directly into your applications.

**Access:** Access the platform directly [here](https://automl-data-pipeline.streamlit.app/).


## How It Helps Developers

Dynamic AutoML offers a range of features designed to streamline the development process and empower developers in handling diverse data tasks efficiently:

- **Automated Data Analysis:** Quickly analyze time series datasets without the need for manual coding.
- **Efficient Data Preprocessing:** Built-in functionality for handling missing values and dropping columns.
- **Hyperparameter Tuning Automation:** Dynamically determines hyperparameters for LSTM regression models.
- **Visual Model Training Monitoring:** Visualize the training and validation loss versus epoch graph.
- **Easy Model Deployment:** Download trained LSTM regression models for deployment.
- **Streamlined Development Workflow:** User-friendly interface for seamless development experience.
- **Automated Image Analysis and Segmentation:** Simplifies image analysis and segmentation.
- **Efficient Image Classification:** Train image classification models effortlessly.

##Run the application
Execute these two lines of code,and you`re all set!
```python
pip install requirements.txt
streamlit run app.py
```

## The Team

Meet the passionate individuals behind Dynamic AutoML:

### Siddhanth Sridhar

<img src="Sid_profile.jpeg" width="150">

Siddhanth Sridhar is a fervent Computer Science Engineering (CSE) undergraduate at PES University, deeply immersed in the realm of Machine Learning. Fueled by curiosity and boundless enthusiasm, he continuously delves into the intricacies of artificial intelligence. He staunchly believes in technology's potential to reshape industries and enhance livelihoods, propelling him to the forefront of this exhilarating revolution.

- LinkedIn: [Siddhanth Sridhar's LinkedIn Profile](https://www.linkedin.com/in/siddhanth-sridhar/)
- GitHub: [Siddhanth Sridhar's GitHub Profile](https://github.com/siddhanth-sridhar)

### Swaraj Khan

<img src="Swaraj_profile.jpeg" width="150">

Swaraj Khan is a driven B.Tech student at Dayananda Sagar University, immersing himself in the realm of Computer Science with a special focus on machine learning. With an unwavering commitment to tackling real-world challenges, Swaraj harnesses the power of technology to unravel complexities and pave the way for innovative solutions.

- LinkedIn: [Swaraj Khan's LinkedIn Profile](https://www.linkedin.com/in/swaraj-khan/)
- GitHub: [Swaraj Khan's GitHub Profile](https://github.com/swaraj-khan)

### Shreya Chaurasia

<img src="Shreya_profile.jpeg" width="150">


Shreya Chaurasia is a B.Tech Computer Science scholar driven by an insatiable curiosity for Machine Learning. Ambitious, passionate, and self-motivated, she finds the potential of ML to revolutionize industries utterly captivating. Delving into data to reveal patterns and derive insights, she thrives on crafting innovative solutions. Challenges are her stepping stones to growth, and she relentlessly pursues excellence in all her pursuits.

- LinkedIn: [Shreya Chaurasia's LinkedIn Profile](https://www.linkedin.com/in/shreya-chaurasia/)
- GitHub: [Shreya Chaurasia's GitHub Profile](https://github.com/shreya-chaurasia)

## Contributions

Contributions are welcome! Please feel free to open issues or pull requests for any enhancements or bug fixes. Together, we can continue to improve and expand the capabilities of Dynamic AutoML to better serve the needs of developers.
