import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from io import BytesIO

def calculate_relative_weights(df, dependent_var, independent_vars):
    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(df[independent_vars])
    y = df[dependent_var].values

    # Ridge regression for multicollinearity adjustment
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X, y)
    ridge_coefs = ridge_model.coef_

    # Variance decomposition using PCA
    pca = PCA()
    pca.fit(X)
    explained_variance = pca.explained_variance_ratio_

    # Calculate relative weights
    relative_weights = np.abs(ridge_coefs * explained_variance)
    relative_weights = (relative_weights / relative_weights.sum()) * 100  # Convert to percentage

    # Prepare results DataFrame
    relative_weights_df = pd.DataFrame({
        'Variable': independent_vars,
        'Relative Weight (%)': relative_weights
    }).sort_values(by='Relative Weight (%)', ascending=False)
    
    return relative_weights_df

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Relative Weights')
    return output.getvalue()

# Streamlit app
st.title("Relative Weights Calculator")

# File upload (accepts CSV and Excel files)
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Determine file type and load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            excel_file = pd.ExcelFile(uploaded_file)
            if len(excel_file.sheet_names) == 0:
                st.error("The uploaded Excel file contains no worksheets. Please upload a valid file.")
            else:
                # Load the first sheet by default
                df = excel_file.parse(sheet_name=0)
        
        st.write("Data Preview:", df.head())

        # Variable selection
        dependent_var = st.selectbox("Select the dependent variable (Y)", options=df.columns)
        independent_vars = st.multiselect("Select the independent variables (X)", options=[col for col in df.columns if col != dependent_var])

        if dependent_var and independent_vars:
            if st.button("Calculate Relative Weights"):
                try:
                    # Calculate relative weights
                    result_df = calculate_relative_weights(df, dependent_var, independent_vars)
                    st.write("Relative Weights of Independent Variables:")
                    st.dataframe(result_df)

                    # Convert to Excel and provide download link
                    excel_data = to_excel(result_df)
                    st.download_button(
                        label="Download as Excel",
                        data=excel_data,
                        file_name="relative_weights.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                except Exception as e:
                    st.error(f"Error during calculation: {e}")
    except ValueError:
        st.error("Error reading the file. Please ensure the file is not corrupted and has a valid format.")
