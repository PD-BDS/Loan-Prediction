# Import required packages
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import zscore
import streamlit as st
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

df_new = pd.read_csv('kiva_loans_cleaned.csv', encoding='utf-8', on_bad_lines='skip')


# Page title
st.title("Exploratory Data Analysis, K-means Clustering and Recommendation Engine on Kiva Loans")
st.sidebar.header("Filters")

# Filter for Country
country = df_new['country'].unique()
selected_country = st.sidebar.selectbox("Select Country", country.tolist())
if selected_country:
    filtered_df = df_new[df_new['country'] == selected_country]
else:
    st.warning("Please select a country from the sidebar")
    st.stop()

# Filter for Gender 
borrower_genders = df_new['borrower_genders'].unique()
selected_genders = st.sidebar.multiselect("Select Gender", borrower_genders.tolist(), default=borrower_genders.tolist())
filtered_df = filtered_df[filtered_df['borrower_genders'].isin(selected_genders)]

# Filter for Loan Amount
min_loan, max_loan = float(df_new['loan_amount'].min()), float(df_new['loan_amount'].max())
selected_loan_amount = st.sidebar.slider("Select Loan Amount", min_value=min_loan, max_value=max_loan, value=(min_loan, max_loan))
filtered_df = filtered_df[(filtered_df['loan_amount'] >= selected_loan_amount[0]) & (filtered_df['loan_amount'] <= selected_loan_amount[1])]

# Filter for Years 
filtered_df['year'] = pd.to_datetime(filtered_df['date']).dt.year
years = sorted(filtered_df['year'].unique())
selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years)
filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]

# selected filters
st.caption(f"Data for Country: {selected_country} | Gender: {', '.join(selected_genders)} | Loan Amount: {selected_loan_amount} | Years: {', '.join(map(str, selected_years))}")


# Distribution of Loan Sector
st.subheader('Distribution of Loan Sector')
sector_chart = alt.Chart(filtered_df).mark_bar().encode(
    x=alt.X('count(sector):Q', title='Count'),
    y=alt.Y('sector:N', sort='-x', title='Sector'),
    color=alt.Color('sector:N', legend=None)
).properties(
    width=600,
    height=400   
)
st.altair_chart(sector_chart)

# Distribution of Loan Term 
st.subheader('Distribution of Loan Term (in Months)')
term_hist = alt.Chart(filtered_df).mark_bar().encode(
    x=alt.X('term_in_months:Q', bin=alt.Bin(maxbins=30), title='Term in Months'),
    y=alt.Y('count():Q', title='Frequency'),
    color=alt.Color('term_in_months:Q', legend=None)
).properties(
    width=600,
    height=400
)
st.altair_chart(term_hist)

# Monthly Loan Amounts Over Time
st.subheader('Monthly Loan Amounts Over Time')
filtered_df['month'] = pd.to_datetime(filtered_df['date']).dt.month
filtered_df['month_name'] = pd.to_datetime(filtered_df['date']).dt.strftime('%b')

monthly_loan_amount = filtered_df.groupby(['year', 'month_name', 'month']).sum().reset_index()

loan_time_series = alt.Chart(monthly_loan_amount).mark_line(point=True).encode(
    x=alt.X('month_name:N', sort=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], title='Month'),
    y=alt.Y('loan_amount:Q', title='Total Loan Amount'),
    color=alt.Color('year:N', title='Year'),
    tooltip=['year', 'month_name', 'loan_amount']
).properties(
    width=700,
    height=400
)
st.altair_chart(loan_time_series)

# Top 10 Countries with Highest Average Loan Amount
st.subheader('Top 10 Countries with Highest Average Loan Amount')
#df_clean = df_new[df_new['country'].notna() & (df_new['country'].str.strip() != '')]
#df_clean['country'] = df_clean['country'].str.strip()
top_10_countries_avg_loan = df_new.groupby('country')['loan_amount'].mean().nlargest(10).reset_index()

top_10_chart = alt.Chart(top_10_countries_avg_loan).mark_bar().encode(
    x=alt.X('loan_amount:Q', title='Average Loan Amount'),
    y=alt.Y('country:N', sort='-x', title='Country'),
    color=alt.Color('country:N', legend=None)
).properties(
    width=600,
    height=400
)
st.altair_chart(top_10_chart)

# Distribution of Genders 
st.subheader('Distribution of Borrower Genders')
gender_counts = filtered_df['borrower_genders'].value_counts().reset_index()
gender_counts.columns = ['borrower_genders', 'count']

gender_doughnut_chart = alt.Chart(gender_counts).mark_arc(innerRadius=80, outerRadius=120).encode(
    theta=alt.Theta(field="count", type="quantitative"),
    color=alt.Color(field="borrower_genders", type="nominal", title="Borrower Genders"),
    tooltip=[alt.Tooltip('borrower_genders:N', title="Gender"), alt.Tooltip('count:Q', title="Count")]
).properties(
    width=400,
    height=400
)

# text labels to the doughnut chart 
gender_doughnut_text = gender_doughnut_chart.mark_text(radius=150, size=15).encode(
    text=alt.Text('count:Q', format='.0f')
)

final_chart = alt.layer(gender_doughnut_chart, gender_doughnut_text).configure_legend(
    labelFontSize=12,
    titleFontSize=14
)
st.altair_chart(final_chart)


# Dataset Summary
st.header('Dataset Summary')
st.caption('Mean Loan Amount: ' + str(round(filtered_df['loan_amount'].mean(), 2)))
st.caption('Median Loan Amount: ' + str(round(filtered_df['loan_amount'].median(), 2)))
st.caption('Mode Loan Amount: ' + str(filtered_df['loan_amount'].mode()[0]))
st.write(filtered_df.describe())

# Filtered dataframe
st.header("Filtered Data")
st.dataframe(filtered_df)


st.header('K-Means Clustering')
 
# Columns for K-means clustering
filtered_df_reduced = filtered_df[['loan_amount', 'term_in_months']]
 
# Scaler
fig, ax = plt.subplots(figsize=(10, 5))
filtered_df_reduced.hist(bins=100, ax=ax)
st.pyplot(fig)

scaler = MinMaxScaler()
 
data_to_cluster_scaled = scaler.fit_transform(filtered_df_reduced)
 
Sum_of_squared_distances = []
 
K = range(1, 10)
 
for k in K:
    km = KMeans(n_clusters=k, n_init = "auto")
    km.fit(data_to_cluster_scaled)
    Sum_of_squared_distances.append(km.inertia_)
 
fig, ax = plt.subplots()
ax.plot(K, Sum_of_squared_distances, 'bx-')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Sum of Squared Distances')
ax.set_title('Elbow Method For Optimal k')
ax.grid(True)
 
st.pyplot(fig)
 
 
 
def k_means_simple(data, k, max_iters=100):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
 
    for _ in range(max_iters):
        distances = np.linalg.norm(data - centroids[:, np.newaxis], axis=2)
        labels = np.argmin(distances, axis=0)
 
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
 
        if np.all(centroids == new_centroids):
            break
 
        centroids = new_centroids
 
    return labels, centroids
 
labels, final_centroids = k_means_simple(data_to_cluster_scaled, 5)
 
 
distances = np.linalg.norm(data_to_cluster_scaled[:, np.newaxis] - final_centroids, axis=2)
nearest_centroid_indices = np.argmin(distances, axis=1)
 
data_df = pd.DataFrame({
    'x': data_to_cluster_scaled[:, 0],
    'y': data_to_cluster_scaled[:, 1],
    'centroid': nearest_centroid_indices
})
 
 
centroids_df = pd.DataFrame({
    'x': final_centroids[:, 0],
    'y': final_centroids[:, 1],
    'centroid': range(final_centroids.shape[0])
})
 
 
data_df['type'] = 'data'
centroids_df['type'] = 'centroid'
 
data_df['loan_amount'] = filtered_df['loan_amount'].values
data_df['term_in_months'] = filtered_df['term_in_months'].values
data_df['activity'] = filtered_df['activity'].values
data_df['sector'] = filtered_df['sector'].values
data_df['region'] = filtered_df['region'].values
 
combined_df = pd.concat([data_df, centroids_df])
 
scatter_plot = alt.Chart(combined_df).mark_circle(size=60).encode(
    x='x',
    y='y',
    color=alt.Color('centroid:N', scale=alt.Scale(scheme='category10')),
    opacity=alt.condition(
        alt.datum.type == 'data',  
        alt.value(0.6),
        alt.value(1)
    ),
    tooltip=[
        alt.Tooltip('loan_amount:Q', title='Loan Amount'),
        alt.Tooltip('term_in_months:Q', title='Term (Months)'),
        alt.Tooltip('activity:N', title='Activity'),
        alt.Tooltip('sector:N', title='Sector'),
        alt.Tooltip('region:N', title='Region')
    ]
).properties(
    title='Reduced Data and Initial Centroids'
)
 
st.altair_chart(scatter_plot, use_container_width=True)

# Fix session states
if 'country_selected' not in st.session_state:
    st.session_state['country_selected'] = None
if 'gender_selected' not in st.session_state:
    st.session_state['gender_selected'] = None
if 'sector_selected' not in st.session_state:
    st.session_state['sector_selected'] = None

# Recommendation Engine based on Country, Gender, and Sector
st.subheader("Loan Recommendation")

# Input for country
country_input = st.selectbox("Select Country", ["None"] + sorted(list(df_new['country'].unique())))
if country_input != "None":
    # Filter gender options based on selected country
    filtered_genders = df_new[df_new['country'] == country_input]['borrower_genders'].unique()
    gender_input = st.selectbox("Select Gender", ["None"] + list(filtered_genders))
else:
    gender_input = st.selectbox("Select Gender", ["None"] + list(df_new['borrower_genders'].unique()))

# Input for sector based on the selected country and gender
if country_input != "None" and gender_input != "None":
    filtered_sectors = df_new[(df_new['country'] == country_input) & (df_new['borrower_genders'] == gender_input)]['sector'].unique()
    sector_input = st.selectbox("Select Sector", ["None"] + list(filtered_sectors))
else:
    sector_input = st.selectbox("Select Sector", ["None"] + list(df_new['sector'].unique()))

# Recommendations based on country, gender, and sector
if country_input != "None" and gender_input != "None" and sector_input != "None":
    user_filtered_df = df_new[
        (df_new['country'] == country_input) &
        (df_new['borrower_genders'] == gender_input) &
        (df_new['sector'] == sector_input)
    ].reset_index(drop=True)
    
    if not user_filtered_df.empty:
        data_to_cluster_scaled_filtered = data_to_cluster_scaled[:len(user_filtered_df)]

        # Similarity matrix
        similarity_matrix = cosine_similarity(data_to_cluster_scaled_filtered)
        
        # Top 3 similar loans
        similar_loans_indices = np.argsort(similarity_matrix[0])[::-1][1:4]
        
        # Display recommended loans
        recommendations = user_filtered_df.iloc[similar_loans_indices][['country', 'borrower_genders', 'sector', 'loan_amount', 'term_in_months']]
        st.write("Recommended Loans:")
        st.dataframe(recommendations)
    else:
        st.write("No matching loans found for the selected country, gender, and sector.")
else:
    st.write("Please select a country, gender, and sector.")
