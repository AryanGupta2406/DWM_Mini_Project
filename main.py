import streamlit as st
import pandas as pd


def load_data():

    uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("The dataset you uploaded has {} rows and {} columns.".format(df.shape[0], df.shape[1]))
        except Exception as e:
            st.error(f"Error: {e}")
            return None
        return df


def show_plots(df):
    st.subheader("Data Visualization")

    selected_columns = st.multiselect("Select columns for visualization", df.columns)
    if selected_columns:
        st.line_chart(df[selected_columns])
        st.bar_chart(df[selected_columns])
        st.scatter_chart(df[selected_columns])


def show_data_analysis(df):
    st.subheader("Data Analysis")

    selected_columns = st.multiselect("Select columns for analysis", df.columns)
    if selected_columns:
        st.write("Descriptive Statistics")
        st.write(df[selected_columns].describe())


def data_opns(df):
    st.subheader("Data Operations")

    drop_column = st.checkbox("Drop the column from dataset")

    if drop_column:
        columns = list(df.columns)
        column_to_drop = st.selectbox("Choose the column to drop", columns)
        df.drop(column_to_drop, axis=1, inplace=True)
        st.write("The column {} has been dropped from the dataset.".format(column_to_drop))
        st.write(df)


def main():
    st.title("DWM MINIPROJECT")

    df = load_data()

    if df is not None:
        page = st.sidebar.radio("Select a page", ["Home", "Data Visualization", "Data Analysis","Data Operations"])

        if page == "Home":
            st.write("Upload a dataset and navigate to other pages for analysis.")
        elif page == "Data Visualization":
            show_plots(df)
        elif page == "Data Analysis":
            show_data_analysis(df)
        elif page == "Data Operations":
            data_opns(df)

if __name__ == "__main__":
    main()