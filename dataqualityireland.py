import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def filter_data(df, column, condition, value):
    return df[df[column].apply(lambda x: eval(f"x {condition} {value}"))]

def main():
    st.title('Data Exploration with Streamlit')
    
    uploaded_file = st.file_uploader('Upload a CSV, Excel, or JSON file', type=['csv', 'xls', 'xlsx', 'json'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            df = pd.read_excel(uploaded_file)
        
        st.subheader('Data Sample')
        st.write(df.head())

        # Faceting: Generate summary statistics for numerical columns
        numerical_columns = df.select_dtypes(include=[np.number])
        st.subheader('Summary Statistics')
        st.write(numerical_columns.describe())

        # Filtering
        st.subheader('Filtering Data')
        filter_column = st.selectbox('Select a column to filter', df.columns)
        filter_condition = st.selectbox('Select a filter condition', ['>', '>=', '<', '<=', '==', '!='])
        filter_value = st.number_input('Enter a filter value', value=0)

        if st.button('Apply Filter'):
            filtered_df = filter_data(df, filter_column, filter_condition, filter_value)
            st.write(filtered_df)

        # Data Visualization: Histogram of numerical column
        st.subheader('Data Visualization')
        numerical_columns = df.select_dtypes(include=[np.number])
        column_to_visualize = st.selectbox('Select a column for histogram', numerical_columns.columns)

        plt.hist(df[column_to_visualize], bins=10, edgecolor='black')
        plt.xlabel(column_to_visualize)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {column_to_visualize}')
        st.pyplot()

if __name__ == '__main__':
    main()
