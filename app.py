
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


# Preprocess the data for clustering
def preprocess_data(df, columns):
    data = df[columns].fillna(df[columns].mean())  # Fill missing values with the mean of each column
    scaler = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Train the KMeans model
def train_kmeans_model(data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans.inertia_

# Streamlit App
def main():
    st.title("Crime Data Clustering and Visualization")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load the uploaded data
            df = pd.read_csv(uploaded_file)

            # Display the uploaded data
            st.write("Uploaded Data:")
            st.write(df.head())

            # Sidebar for column selection
            columns = st.sidebar.multiselect("Select Numeric Columns for Clustering", df.select_dtypes(include=[np.number]).columns.tolist())

            if columns:  # Ensure columns are selected
                # Preprocess the selected columns
                scaled_data = preprocess_data(df, columns)

                # Dynamic K-Means Clustering Visualization
                st.title("K-Means Clustering Visualization")
                k_values = st.sidebar.slider("Select Number of Clusters", min_value=1, max_value=10, value=4)
                fig, axs = plt.subplots(2, 2, figsize=(10, 8))
                
                inertias = []
                for idx, k in enumerate(range(1, 5)):
                    clusters, inertia = train_kmeans_model(scaled_data, n_clusters=k)
                    inertias.append(inertia)
                    
                    # Plotting each cluster
                    ax = axs[idx // 2, idx % 2]
                    ax.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clusters, s=50, cmap='viridis')

                    centers = KMeans(n_clusters=k).fit(scaled_data).cluster_centers_
                    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.75, marker='*')
                    ax.set_title(f'k = {k}')
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.grid(True)
                    plt.tight_layout()  # Adjust layout to minimize margins

                st.pyplot(fig)

                # Display Elbow Method Plot
                st.title("Elbow Method to Determine Optimal k")
                ks = list(range(1, 11))
                inertias = []
                for k in ks:
                    _, inertia = train_kmeans_model(scaled_data, n_clusters=k)
                    inertias.append(inertia)

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(ks, inertias, '-o')
                ax.set_xlabel('Number of clusters, k')
                ax.set_ylabel('Inertia')
                ax.set_xticks(ks)
                plt.tight_layout()  # Adjust layout to minimize margins
                st.pyplot(fig)

                # Train KMeans model with selected k
                clusters, _ = train_kmeans_model(scaled_data, n_clusters=k_values)
                df['Cluster'] = clusters

                # Sidebar for state selection
                state_column = st.sidebar.selectbox("Select State/Region Column", df.columns)
                selected_states = st.sidebar.multiselect("Select States/Regions", df[state_column].unique())

                if selected_states:  # Ensure states are selected
                    # Filter data by selected states
                    state_data = df[df[state_column].isin(selected_states)]

                    # Sidebar for crime type selection
                    crime_columns = [col for col in df.columns if 'crime' in col.lower()]
                    crime_column = st.sidebar.selectbox("Select Crime Column for Analysis", crime_columns)

                    # Display filtered data
                    st.write(f"Data for selected states:")
                    st.write(state_data)

                    # Analysis
                    total_crime = state_data.groupby(state_column)[crime_column].sum()
                    max_crime_state = total_crime.idxmax()
                    min_crime_state = total_crime.idxmin()

                    st.write(f"State/Region with Maximum Crime ({crime_column}): {max_crime_state} ({total_crime[max_crime_state]})")
                    st.write(f"State/Region with Minimum Crime ({crime_column}): {min_crime_state} ({total_crime[min_crime_state]})")

                    st.write(f"Total Crime by State/Region:")

                    # Plot the bar chart directly using Streamlit
                    st.bar_chart(total_crime)

                    # Show pie chart option
                    chart_type = st.sidebar.selectbox("Select Chart Type", ["Bar Chart", "Pie Chart"])

                    if chart_type == "Pie Chart":
                        st.write("Pie Chart for Total Crime:")
                        fig, ax = plt.subplots()
                        total_crime.plot.pie(autopct='%1.1f%%', ax=ax)
                        plt.tight_layout()  # Adjust layout to minimize margins
                        st.pyplot(fig)
                    elif chart_type == "Bar Chart":
                        st.write("Bar Chart for Total Crime:")
                        st.bar_chart(total_crime)

                    # Optionally display clustering information
                    cluster_option = st.sidebar.checkbox("Show Cluster Information")
                    if cluster_option:
                        st.write("Cluster Data:")
                        st.write(df[['Cluster'] + columns].head())

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
