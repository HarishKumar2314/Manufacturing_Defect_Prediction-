import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model and the scaler
try:
    model = joblib.load("logistic_regression_model.pkl")
    scaler = joblib.load("standard_scaler.pkl")
    st.success("Model and scaler loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model files not found. Please run the training script first.")
    st.stop()

# Define the features the model was trained on
features = ["ProductionVolume", "SupplierQuality", "MaintenanceHours", "EnergyEfficiency"]

# Set the title and header
st.title("Manufacturing Defects Prediction App")
st.write("Predict if a manufacturing product is defective or not.")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Manual Input", "CSV File Upload"])

with tab1:
    st.header("Predict with Manual Input")
    st.write("Enter values for the four features to get a prediction.")

    # Create input fields with min/max values from the dataset
    with st.form(key='manual_form'):
        st.write("Production Volume (Min: 100, Max: 999)")
        production_volume = st.number_input("Production Volume", min_value=100, max_value=999, format="%d")
        
        st.write("Supplier Quality (Min: 80, Max: 100)")
        supplier_quality = st.number_input("Supplier Quality", min_value=80.0, max_value=100.0, format="%.2f")
        
        st.write("Maintenance Hours (Min: 0, Max: 23)")
        maintenance_hours = st.number_input("Maintenance Hours", min_value=0, max_value=23, format="%d")
        
        st.write("Energy Efficiency (Min: 0.1, Max: 0.5)")
        energy_efficiency = st.number_input("Energy Efficiency", min_value=0.1, max_value=0.5, format="%.4f")
        
        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Create a DataFrame from the manual input
        manual_data = pd.DataFrame([[production_volume, supplier_quality, maintenance_hours, energy_efficiency]],
                                   columns=features)
        
        # Scale the data using the pre-trained scaler
        scaled_data = scaler.transform(manual_data)
        
        # Make the prediction
        prediction = model.predict(scaled_data)[0]
        
        # Display the result
        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error("The product is predicted to be **Defective**.")
        else:
            st.success("The product is predicted to be **Non-defective**.")

with tab2:
    st.header("Predict from a CSV File")
    st.write("Upload a CSV file to get predictions for all records.")
    
    # File uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(df.head())

            # Check if the required features are in the uploaded CSV
            if not all(feature in df.columns for feature in features):
                missing_features = [feature for feature in features if feature not in df.columns]
                st.error(f"The uploaded CSV is missing the following required columns: {', '.join(missing_features)}")
            else:
                # Select the features for prediction
                data_to_predict = df[features]

                # Scale the data using the pre-trained scaler
                scaled_data = scaler.transform(data_to_predict)

                # Make predictions
                predictions = model.predict(scaled_data)

                # Map the predictions to "Defective" and "Non-defective"
                df['PredictedStatus'] = predictions
                df['PredictedStatus'] = df['PredictedStatus'].map({1: "Defective", 0: "Non-defective"})

                st.subheader("Prediction Results")
                # Display the full DataFrame with the new prediction column
                st.dataframe(df)

                # Count and display defective vs. non-defective items
                defective_count = (df['PredictedStatus'] == "Defective").sum()
                non_defective_count = (df['PredictedStatus'] == "Non-defective").sum()

                st.write(f"Total **Defective** items predicted: **{defective_count}**")
                st.write(f"Total **Non-defective** items predicted: **{non_defective_count}**")

                # Display a pie chart for visualization
                counts = [defective_count, non_defective_count]
                labels = ["Defective", "Non-defective"]
                
                fig, ax = plt.subplots()
                ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
                ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")