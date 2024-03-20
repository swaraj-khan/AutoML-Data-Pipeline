import streamlit as st
import pandas as pd
import zipfile
import os

def process_csv_dataset(csv_file):
    return "Preprocessed CSV dataset"

def process_time_series_dataset(csv_file):
    return "Preprocessed Time Series dataset"

def process_image_dataset(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.type == "application/zip":
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall("temp_images")
            
            
            os.system("rm -rf temp_images")
            
            return "Preprocessed Image dataset"
        else:
            return "Uploaded file is not a zip file. Please upload a zip file."

def main():
    active_section = "Landing Page"  

    st.sidebar.title("Sections")
    if st.sidebar.button("Landing Page"):
        active_section = "Landing Page"

    if st.sidebar.button("CSV Dataset"):
        active_section = "CSV Dataset"

    if st.sidebar.button("Time Series Dataset"):
        active_section = "Time Series Dataset"

    if st.sidebar.button("Image Dataset"):
        active_section = "Image Dataset"

    if st.sidebar.button("Background Study"):
        active_section = "Background Study"

    if active_section == "Landing Page":
        st.title("Welcome to Data Preprocessing App")
        st.write("This app lets the user upload a dataset and it will do the preprocessing and provide a model for download in return.")

    elif active_section == "CSV Dataset":
        st.title("CSV Dataset")
        csv_file = st.file_uploader("Upload CSV file", type=["csv"])
        if csv_file is not None:
            processed_data = process_csv_dataset(csv_file)
            st.write("CSV dataset processed successfully.")

    elif active_section == "Time Series Dataset":
        st.title("Time Series Dataset")
        csv_file = st.file_uploader("Upload CSV file", type=["csv"])
        if csv_file is not None:
            processed_data = process_time_series_dataset(csv_file)
            st.write("Time series dataset processed successfully.")

    elif active_section == "Image Dataset":
        st.title("Image Dataset")
        uploaded_file = st.file_uploader("Upload zip file or any other type of file")
        if uploaded_file is not None:
            processed_data = process_image_dataset(uploaded_file)
            st.write(processed_data)

    elif active_section == "Background Study":
        st.title("Our Homework")

if __name__ == "__main__":
    main()
