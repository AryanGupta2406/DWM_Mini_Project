# app.py
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to upload dataset


def upload_data():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        return df

# Function for data analysis


# Function for data analysis
def analyze_data(df):
    st.subheader("Data Analysis")

    # User-selectable analysis options
    analysis_options = st.multiselect(
        "Select analysis options",
        ["Basic Statistics", "Correlation Analysis", "Descriptive Statistics", "Custom Analysis"]
    )

    # Display selected analyses
    if "Basic Statistics" in analysis_options:
        st.write("### Basic Statistics")
        st.write(df.describe())

    if "Correlation Analysis" in analysis_options:
        st.write("### Correlation Analysis")

        try:
            # Check if there are numeric columns for correlation analysis
            numeric_columns = df.select_dtypes(include='number').columns

            if len(numeric_columns) > 0:
                corr_matrix = df[numeric_columns].corr()
                st.write(corr_matrix)
            else:
                st.warning("No numeric columns available for correlation analysis.")
        except Exception as e:
            st.error(f"An error occurred during correlation analysis: {e}")

    if "Descriptive Statistics" in analysis_options:
        st.write("### Descriptive Statistics")
        st.write(df.describe(include='all'))

    if "Custom Analysis" in analysis_options:
        st.write("### Custom Analysis")

        # Custom analysis options
        custom_analysis_options = st.multiselect(
            "Select custom analysis options",
            ["Missing Values", "Unique Values", "Value Counts", "Custom Numeric Analysis"]
        )

        for custom_option in custom_analysis_options:
            try:
                if custom_option == "Missing Values":
                    missing_percentage = df.isnull().mean() * 100
                    missing_data = pd.DataFrame({
                        'Column': missing_percentage.index,
                        'Missing Percentage': missing_percentage.values
                    })
                    st.write("#### Missing Values Analysis")
                    st.write(missing_data)

                elif custom_option == "Unique Values":
                    unique_data = pd.DataFrame({
                        'Column': df.columns,
                        'Unique Values': [df[col].nunique() for col in df.columns]
                    })
                    st.write("#### Unique Values Analysis")
                    st.write(unique_data)

                elif custom_option == "Value Counts":
                    selected_column = st.selectbox("Select a column for value counts", df.columns)
                    value_counts_data = df[selected_column].value_counts().reset_index()
                    value_counts_data.columns = [selected_column, 'Count']
                    st.write(f"#### Value Counts for {selected_column}")
                    st.write(value_counts_data)

                elif custom_option == "Custom Numeric Analysis":
                    st.write("#### Custom Numeric Analysis")

                    selected_numeric_column = st.selectbox("Select a numeric column for analysis",
                                                           df.select_dtypes(include='number').columns)

                    st.write(f"#### Numeric Analysis for {selected_numeric_column}")

                    # User-input options for row range
                    start_row = st.text_input("Enter starting row (0-based index)", "0")
                    end_row = st.text_input("Enter ending row (0-based index)", str(len(df)-1))

                    try:
                        start_row = int(start_row)
                        end_row = int(end_row)

                        # Perform numeric analysis on the selected row range
                        selected_data = df.loc[start_row:end_row, selected_numeric_column]

                        # Additional custom numeric analysis options
                        numeric_operation = st.selectbox(
                            "Select a numeric operation",
                            ["Mean", "Median", "Minimum", "Maximum", "Standard Deviation"]
                        )

                        if numeric_operation == "Mean":
                            st.write(f"Mean of {selected_numeric_column} in rows {start_row}-{end_row}: {selected_data.mean()}")

                        elif numeric_operation == "Median":
                            st.write(f"Median of {selected_numeric_column} in rows {start_row}-{end_row}: {selected_data.median()}")

                        elif numeric_operation == "Minimum":
                            st.write(f"Minimum of {selected_numeric_column} in rows {start_row}-{end_row}: {selected_data.min()}")

                        elif numeric_operation == "Maximum":
                            st.write(f"Maximum of {selected_numeric_column} in rows {start_row}-{end_row}: {selected_data.max()}")

                        elif numeric_operation == "Standard Deviation":
                            st.write(f"Standard Deviation of {selected_numeric_column} in rows {start_row}-{end_row}: {selected_data.std()}")

                    except ValueError:
                        st.warning("Please enter valid numeric values for rows.")
                    except Exception as e:
                        st.error(f"An error occurred during custom numeric analysis: {e}")

            except Exception as e:
                st.error(f"An error occurred during custom analysis: {e}")

    # Additional analysis options can be added based on user selection


# Function for data visualization


import seaborn as sns
import matplotlib.pyplot as plt

# Function for data visualization


def visualize_data(df):
    st.write("### Visualization")

    # User-input options for visualization
    plot_type = st.selectbox("Select plot type", ["Scatter Plot", "Line Plot", "Bar Plot", "Box Plot", "Histogram"])
    x_axis = st.selectbox("Select X-axis column", df.columns)
    y_axis = st.selectbox("Select Y-axis column", df.columns)

    # User-input options for row range
    start_row_viz = st.text_input("Enter starting row for visualization (0-based index)", "0")
    end_row_viz = st.text_input("Enter ending row for visualization (0-based index)", str(len(df)-1))

    try:
        start_row_viz = int(start_row_viz)
        end_row_viz = int(end_row_viz)

        # Perform visualization on the selected row range
        selected_data_viz = df.loc[start_row_viz:end_row_viz, [x_axis, y_axis]]

        if plot_type == "Scatter Plot":
            st.write("#### Scatter Plot")
            chart = alt.Chart(selected_data_viz).mark_circle().encode(
                x=x_axis,
                y=y_axis,
                tooltip=[x_axis, y_axis]
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        elif plot_type == "Line Plot":
            st.write("#### Line Plot")
            chart = alt.Chart(selected_data_viz).mark_line().encode(
                x=x_axis,
                y=y_axis,
                tooltip=[x_axis, y_axis]
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        elif plot_type == "Bar Plot":
            st.write("#### Bar Plot")
            chart = alt.Chart(selected_data_viz).mark_bar().encode(
                x=x_axis,
                y='count()',
                tooltip=[x_axis, 'count()']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        elif plot_type == "Box Plot":
            st.write("#### Box Plot")
            chart = alt.Chart(selected_data_viz).mark_boxplot().encode(
                x=x_axis,
                y=y_axis,
                tooltip=[x_axis, y_axis]
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        elif plot_type == "Histogram":
            st.write("#### Histogram")
            chart = alt.Chart(selected_data_viz).mark_bar().encode(
                alt.X(y_axis, bin=True),
                y='count()',
                tooltip=['count()']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

    except ValueError:
        st.warning("Please enter valid numeric values for rows.")
    except Exception as e:
        st.error(f"An error occurred during visualization: {e}")



def perform_olap_operations(df, dimensions, measures):
    st.subheader("OLAP Operations")

    # Display available dimensions and measures
    st.write("**Available Dimensions:**", ", ".join(df.columns))
    st.write("**Available Measures:**", ", ".join(df.select_dtypes(include='number').columns))

    # User input for OLAP operations
    operation = st.selectbox("Select OLAP Operation", ["Pivot Table"])

    if operation == "Pivot Table":
        try:
            # User selects dimensions and measures for pivot table
            pivot_dimensions = st.multiselect("Select Dimensions for Pivot Table", dimensions)
            pivot_measures = st.multiselect("Select Measures for Pivot Table", measures)

            # Create a pivot table
            if pivot_dimensions and pivot_measures:
                pivot_table = pd.pivot_table(df, values=pivot_measures, index=pivot_dimensions, aggfunc='sum')
                st.write("### Pivot Table")
                st.write(pivot_table)
            else:
                st.warning("Please select at least one dimension and one measure for the Pivot Table.")

        except Exception as e:
            st.error(f"An error occurred during OLAP operation: {e}")

# Function for predictions
# def make_predictions(df):
#     st.subheader("Make Predictions")
#     # Perform your predictions here using st functions and machine learning models

# Main function
def main():
    st.title("Streamlit Dataset Analysis")

    # Upload dataset
    df = upload_data()

    if df is not None:
        # Sidebar with options
        option = st.sidebar.selectbox("Select an option", ["Analysis", "Visualization", "Predictions", "OLAP"])

        if option == "Analysis":
            analyze_data(df)
        elif option == "Visualization":
            visualize_data(df)
        elif option == "OLAP":
            # User selects dimensions and measures for OLAP operations
            dimensions = st.multiselect("Select Dimensions", df.columns)
            measures = st.multiselect("Select Measures", df.select_dtypes(include='number').columns)

            if dimensions or measures:
                perform_olap_operations(df, dimensions, measures)
            else:
                st.warning("Please select at least one dimension or one measure.")
        # elif option == "Predictions":
        #     make_predictions(df)

if __name__ == "__main__":
    main()
