# Import required packages
import pandas as pd
import altair as alt
import streamlit as st

df_new = pd.read_csv('kiva_loans.csv')
df_new.drop(columns=['use', 'tags'], inplace=True)
df_new.dropna(subset=['region', 'partner_id', 'disbursed_time', 'funded_time', 'borrower_genders' ], inplace=True)

date_cols = ['posted_time', 'disbursed_time', 'funded_time', 'date']
df_new[date_cols] = df_new[date_cols].apply(pd.to_datetime, errors='coerce')
df_new['funding_days'] = (df_new['funded_time'] - df_new['posted_time']).dt.days

def gender_group(row):
    if row['borrower_genders'] == 'male':
        return 'male'
    elif row['borrower_genders'] == 'female':
        return 'female'
    else:
        return 'group'

df_new['borrower_genders'] = df_new.apply(gender_group, axis=1)

# Page title
st.title("Exploratory Data Analysis on Kiva Loans")
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
filtered_df['year'] = pd.to_datetime(filtered_df['date']).dt.year

monthly_loan_amount = filtered_df.groupby(['year', 'month_name', 'month'])['loan_amount'].sum().reset_index()

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
df_clean = df_new[df_new['country'].notna() & (df_new['country'].str.strip() != '')]
df_clean['country'] = df_clean['country'].str.strip()
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

#doughnut
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

