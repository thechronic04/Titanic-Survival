import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_excel('train.xlsx')

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nColumn names and data types:")
print(df.dtypes)
print("\nFirst few rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

# Check for missing data
print("Missing Data Analysis:")
print("=" * 40)
missing_data = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_data.index,
    'Missing Count': missing_data.values,
    'Missing Percentage': missing_percentage.values
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
print(missing_df)

# Overall survival rate
overall_survival_rate = df['Survived'].mean() * 100
print(f"\nOverall survival rate: {overall_survival_rate:.2f}%")

# Basic survival analysis by key factors
print("\n" + "=" * 50)
print("BASIC SURVIVAL ANALYSIS")
print("=" * 50)

# 1. Survival by Gender
print("\n1. Survival by Gender:")
gender_survival = df.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean']).round(4)
gender_survival['survival_rate'] = (gender_survival['mean'] * 100).round(2)
print(gender_survival)

# 2. Survival by Passenger Class
print("\n2. Survival by Passenger Class:")
class_survival = df.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean']).round(4)
class_survival['survival_rate'] = (class_survival['mean'] * 100).round(2)
print(class_survival)

# 3. Survival by Embarkation Point
print("\n3. Survival by Embarkation Point:")
embark_survival = df.groupby('Embarked')['Survived'].agg(['count', 'sum', 'mean']).round(4)
embark_survival['survival_rate'] = (embark_survival['mean'] * 100).round(2)
print(embark_survival)

# Extract titles from names
def extract_title(name):
    title = name.split(',')[1].split('.')[0].strip()
    return title

df['Title'] = df['Name'].apply(extract_title)

# Group rare titles
title_counts = df['Title'].value_counts()
print("Title distribution:")
print(title_counts)

# Simplify titles
def simplify_title(title):
    if title in ['Mr']:
        return 'Mr'
    elif title in ['Miss', 'Mlle', 'Ms']:
        return 'Miss'
    elif title in ['Mrs', 'Mme']:
        return 'Mrs'
    elif title in ['Master']:
        return 'Master'
    else:
        return 'Other'

df['Title_Simple'] = df['Title'].apply(simplify_title)

print("\nSimplified title distribution:")
print(df['Title_Simple'].value_counts())

# Survival by title
print("\n4. Survival by Title:")
title_survival = df.groupby('Title_Simple')['Survived'].agg(['count', 'sum', 'mean']).round(4)
title_survival['survival_rate'] = (title_survival['mean'] * 100).round(2)
print(title_survival)

# Create family size variable
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

print("\n5. Survival by Family Status:")
alone_survival = df.groupby('IsAlone')['Survived'].agg(['count', 'sum', 'mean']).round(4)
alone_survival['survival_rate'] = (alone_survival['mean'] * 100).round(2)
alone_survival.index = ['With Family', 'Alone']
print(alone_survival)

print("\n6. Survival by Family Size:")
family_survival = df.groupby('FamilySize')['Survived'].agg(['count', 'sum', 'mean']).round(4)
family_survival['survival_rate'] = (family_survival['mean'] * 100).round(2)
print(family_survival)

# Age analysis
# Create age groups
def categorize_age(age):
    if pd.isna(age):
        return 'Unknown'
    elif age < 13:
        return 'Child (0-12)'
    elif age < 18:
        return 'Teenager (13-17)'
    elif age < 30:
        return 'Young Adult (18-29)'
    elif age < 50:
        return 'Adult (30-49)'
    elif age < 65:
        return 'Middle Age (50-64)'
    else:
        return 'Senior (65+)'

df['AgeGroup'] = df['Age'].apply(categorize_age)

print("7. Survival by Age Group:")
age_survival = df.groupby('AgeGroup')['Survived'].agg(['count', 'sum', 'mean']).round(4)
age_survival['survival_rate'] = (age_survival['mean'] * 100).round(2)
print(age_survival)

# Fare analysis by class
print("\n8. Fare Analysis by Class:")
fare_by_class = df.groupby('Pclass')['Fare'].agg(['count', 'mean', 'median', 'min', 'max']).round(2)
print(fare_by_class)

# Deck analysis (extract from cabin)
df['Deck'] = df['Cabin'].str[0]
print("\n9. Survival by Deck:")
deck_survival = df.groupby('Deck')['Survived'].agg(['count', 'sum', 'mean']).round(4)
deck_survival['survival_rate'] = (deck_survival['mean'] * 100).round(2)
print(deck_survival)

# Combined analysis: Class and Gender
print("\n10. Survival by Class and Gender:")
class_gender_survival = df.groupby(['Pclass', 'Sex'])['Survived'].agg(['count', 'sum', 'mean']).round(4)
class_gender_survival['survival_rate'] = (class_gender_survival['mean'] * 100).round(2)
print(class_gender_survival)

# Create comprehensive summary dataset for visualizations
summary_data = []

# 1. Demographics and Survival Data
demographics_data = [
    {'Category': 'Gender', 'Group': 'Female', 'Count': 314, 'Survivors': 233, 'Survival_Rate': 74.20},
    {'Category': 'Gender', 'Group': 'Male', 'Count': 577, 'Survivors': 109, 'Survival_Rate': 18.89},
    {'Category': 'Class', 'Group': '1st Class', 'Count': 216, 'Survivors': 136, 'Survival_Rate': 62.96},
    {'Category': 'Class', 'Group': '2nd Class', 'Count': 184, 'Survivors': 87, 'Survival_Rate': 47.28},
    {'Category': 'Class', 'Group': '3rd Class', 'Count': 491, 'Survivors': 119, 'Survival_Rate': 24.24},
    {'Category': 'Embarkation', 'Group': 'Cherbourg', 'Count': 168, 'Survivors': 93, 'Survival_Rate': 55.36},
    {'Category': 'Embarkation', 'Group': 'Queenstown', 'Count': 77, 'Survivors': 30, 'Survival_Rate': 38.96},
    {'Category': 'Embarkation', 'Group': 'Southampton', 'Count': 644, 'Survivors': 217, 'Survival_Rate': 33.70}
]

demographics_df = pd.DataFrame(demographics_data)

# 2. Age Group Analysis Data
age_data = [
    {'AgeGroup': 'Child (0-12)', 'Count': 69, 'Survivors': 40, 'Survival_Rate': 57.97},
    {'AgeGroup': 'Teenager (13-17)', 'Count': 44, 'Survivors': 21, 'Survival_Rate': 47.73},
    {'AgeGroup': 'Young Adult (18-29)', 'Count': 271, 'Survivors': 95, 'Survival_Rate': 35.06},
    {'AgeGroup': 'Adult (30-49)', 'Count': 256, 'Survivors': 107, 'Survival_Rate': 41.80},
    {'AgeGroup': 'Middle Age (50-64)', 'Count': 63, 'Survivors': 26, 'Survival_Rate': 41.27},
    {'AgeGroup': 'Senior (65+)', 'Count': 11, 'Survivors': 1, 'Survival_Rate': 9.09},
    {'AgeGroup': 'Unknown', 'Count': 177, 'Survivors': 52, 'Survival_Rate': 29.38}
]

age_df = pd.DataFrame(age_data)

# 3. Title Analysis Data
title_data = [
    {'Title': 'Mr', 'Count': 517, 'Survivors': 81, 'Survival_Rate': 15.67},
    {'Title': 'Mrs', 'Count': 126, 'Survivors': 100, 'Survival_Rate': 79.37},
    {'Title': 'Miss', 'Count': 185, 'Survivors': 130, 'Survival_Rate': 70.27},
    {'Title': 'Master', 'Count': 40, 'Survivors': 23, 'Survival_Rate': 57.50},
    {'Title': 'Other', 'Count': 23, 'Survivors': 8, 'Survival_Rate': 34.78}
]

title_df = pd.DataFrame(title_data)

# 4. Family Size Analysis Data
family_data = [
    {'FamilySize': 1, 'Status': 'Alone', 'Count': 537, 'Survivors': 163, 'Survival_Rate': 30.35},
    {'FamilySize': 2, 'Status': 'Small Family', 'Count': 161, 'Survivors': 89, 'Survival_Rate': 55.28},
    {'FamilySize': 3, 'Status': 'Small Family', 'Count': 102, 'Survivors': 59, 'Survival_Rate': 57.84},
    {'FamilySize': 4, 'Status': 'Medium Family', 'Count': 29, 'Survivors': 21, 'Survival_Rate': 72.41},
    {'FamilySize': 5, 'Status': 'Large Family', 'Count': 15, 'Survivors': 3, 'Survival_Rate': 20.00},
    {'FamilySize': 6, 'Status': 'Large Family', 'Count': 22, 'Survivors': 3, 'Survival_Rate': 13.64},
    {'FamilySize': 7, 'Status': 'Large Family', 'Count': 12, 'Survivors': 4, 'Survival_Rate': 33.33},
    {'FamilySize': 8, 'Status': 'Large Family', 'Count': 6, 'Survivors': 0, 'Survival_Rate': 0.00},
    {'FamilySize': 11, 'Status': 'Large Family', 'Count': 7, 'Survivors': 0, 'Survival_Rate': 0.00}
]

family_df = pd.DataFrame(family_data)

# 5. Class and Gender Combined Data
class_gender_data = [
    {'Class': '1st Class', 'Gender': 'Female', 'Count': 94, 'Survivors': 91, 'Survival_Rate': 96.81},
    {'Class': '1st Class', 'Gender': 'Male', 'Count': 122, 'Survivors': 45, 'Survival_Rate': 36.89},
    {'Class': '2nd Class', 'Gender': 'Female', 'Count': 76, 'Survivors': 70, 'Survival_Rate': 92.11},
    {'Class': '2nd Class', 'Gender': 'Male', 'Count': 108, 'Survivors': 17, 'Survival_Rate': 15.74},
    {'Class': '3rd Class', 'Gender': 'Female', 'Count': 144, 'Survivors': 72, 'Survival_Rate': 50.00},
    {'Class': '3rd Class', 'Gender': 'Male', 'Count': 347, 'Survivors': 47, 'Survival_Rate': 13.54}
]

class_gender_df = pd.DataFrame(class_gender_data)

# Save all datasets as CSV files
demographics_df.to_csv('demographics_survival.csv', index=False)
age_df.to_csv('age_survival.csv', index=False)
title_df.to_csv('title_survival.csv', index=False)
family_df.to_csv('family_survival.csv', index=False)
class_gender_df.to_csv('class_gender_survival.csv', index=False)

print("Data analysis complete. CSV files created:")
print("1. demographics_survival.csv")
print("2. age_survival.csv")  
print("3. title_survival.csv")
print("4. family_survival.csv")
print("5. class_gender_survival.csv")

# Display sample of the main datasets
print("\nSample Demographics Data:")
print(demographics_df.head())

print("\nSample Class-Gender Data:")
print(class_gender_df)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the data
df = pd.read_excel('train.xlsx')

# Display the first few rows to understand the data structure
print("Data columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Calculate survival rates by different demographics
survival_data = []

# Gender survival rates
if 'Sex' in df.columns and 'Survived' in df.columns:
    gender_survival = df.groupby('Sex')['Survived'].agg(['mean', 'count']).reset_index()
    for _, row in gender_survival.iterrows():
        survival_data.append({
            'Category': 'Gender',
            'Group': row['Sex'].capitalize(),
            'Survival_Rate': row['mean'] * 100,
            'Count': row['count']
        })

# Class survival rates
if 'Pclass' in df.columns:
    class_survival = df.groupby('Pclass')['Survived'].agg(['mean', 'count']).reset_index()
    for _, row in class_survival.iterrows():
        survival_data.append({
            'Category': 'Class',
            'Group': f'Class {row["Pclass"]}',
            'Survival_Rate': row['mean'] * 100,
            'Count': row['count']
        })

# Embarkation survival rates
if 'Embarked' in df.columns:
    embarked_survival = df.groupby('Embarked')['Survived'].agg(['mean', 'count']).reset_index()
    embarked_survival = embarked_survival.dropna()  # Remove NaN values
    embarked_map = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
    for _, row in embarked_survival.iterrows():
        port_name = embarked_map.get(row['Embarked'], row['Embarked'])
        survival_data.append({
            'Category': 'Embarkation',
            'Group': port_name,
            'Survival_Rate': row['mean'] * 100,
            'Count': row['count']
        })

# Create DataFrame from survival data
survival_df = pd.DataFrame(survival_data)
print("\nSurvival rates by demographics:")
print(survival_df)

# Create the horizontal bar chart
fig = go.Figure()

# Define colors for each category
color_map = {
    'Gender': '#1FB8CD',      # Strong cyan
    'Class': '#DB4545',       # Bright red  
    'Embarkation': '#2E8B57'  # Sea green
}

# Add bars for each category
categories = survival_df['Category'].unique()
for i, category in enumerate(categories):
    category_data = survival_df[survival_df['Category'] == category]
    
    # Create abbreviated group names (15 char limit)
    abbreviated_groups = []
    for group in category_data['Group']:
        if len(group) > 15:
            if 'Class' in group:
                abbreviated_groups.append(group[:15])
            elif group == 'Southampton':
                abbreviated_groups.append('Southampton')
            elif group == 'Cherbourg':
                abbreviated_groups.append('Cherbourg')
            elif group == 'Queenstown':
                abbreviated_groups.append('Queenstown')
            else:
                abbreviated_groups.append(group[:15])
        else:
            abbreviated_groups.append(group)
    
    fig.add_trace(go.Bar(
        y=[f"{cat}: {grp}" for cat, grp in zip([category]*len(category_data), abbreviated_groups)],
        x=category_data['Survival_Rate'],
        name=category,
        orientation='h',
        marker_color=color_map[category],
        hovertemplate='<b>%{y}</b><br>Survival Rate: %{x:.1f}%<extra></extra>'
    ))

# Update layout
fig.update_layout(
    title="Titanic Survival Rates by Demographics",
    xaxis_title="Survival Rate (%)",
    yaxis_title="Groups",
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update traces
fig.update_traces(cliponaxis=False)

# Save as both PNG and SVG
fig.write_image("survival_demographics.png")
fig.write_image("survival_demographics.svg", format="svg")

fig.show()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the data
df = pd.read_excel('train.xlsx')

# Calculate survival counts and rates by class and gender
survival_data = df.groupby(['Pclass', 'Sex'])['Survived'].agg(['sum', 'count']).reset_index()
survival_data['rate'] = (survival_data['sum'] / survival_data['count']) * 100

print("Survival data by class and gender:")
print(survival_data)

# Pivot data for easier access
survival_counts = survival_data.pivot(index='Pclass', columns='Sex', values='sum').fillna(0)
survival_rates = survival_data.pivot(index='Pclass', columns='Sex', values='rate').fillna(0)

print("\nSurvival counts:")
print(survival_counts)
print("\nSurvival rates:")
print(survival_rates)

# Create the stacked bar chart using survivor counts but showing rates as labels
fig = go.Figure()

# Add female survivors
fig.add_trace(go.Bar(
    name='Female',
    x=['1st Class', '2nd Class', '3rd Class'],
    y=survival_counts['female'].values,
    text=[f'{rate:.1f}%' for rate in survival_rates['female'].values],
    textposition='inside',
    marker_color='#1FB8CD'
))

# Add male survivors
fig.add_trace(go.Bar(
    name='Male',
    x=['1st Class', '2nd Class', '3rd Class'],
    y=survival_counts['male'].values,
    text=[f'{rate:.1f}%' for rate in survival_rates['male'].values],
    textposition='inside',
    marker_color='#DB4545'
))

# Update layout
fig.update_layout(
    title='Survival Rates by Class and Gender',
    xaxis_title='Class',
    yaxis_title='Survivors Count',
    barmode='stack',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image('chart.png')
fig.write_image('chart.svg', format='svg')

print("Chart saved successfully!")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the data
df = pd.read_csv("age_survival.csv")

# Exclude the "Unknown" age group
df_filtered = df[df['AgeGroup'] != 'Unknown'].copy()

# Create bar chart
fig = px.bar(df_filtered, 
             x='AgeGroup', 
             y='Survival_Rate',
             title='Survival Rates by Age Group')

# Update layout and styling
fig.update_xaxes(title='Age Groups')
fig.update_yaxes(title='Survival Rate %')

# Add data labels on top of each bar
fig.update_traces(
    texttemplate='%{y:.1f}%',
    textposition='outside',
    cliponaxis=False
)

# Save the chart as both PNG and SVG
fig.write_image("chart.png")
fig.write_image("chart.svg", format="svg")

print("Chart created successfully!")
print(f"Filtered data shape: {df_filtered.shape}")
print("Age groups included:", df_filtered['AgeGroup'].tolist())

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

# Load the Excel file
df = pd.read_excel('train.xlsx')

# Extract titles from Name column
def extract_title(name):
    title = re.search(' ([A-Za-z]+)\.', name)
    if title:
        return title.group(1)
    else:
        return 'Unknown'

df['Title'] = df['Name'].apply(extract_title)

# Group rare titles into 'Other'
title_counts = df['Title'].value_counts()
rare_titles = title_counts[title_counts < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Other')

# Calculate survival rates by title
survival_rates = df.groupby('Title')['Survived'].agg(['count', 'sum']).reset_index()
survival_rates['survival_rate'] = (survival_rates['sum'] / survival_rates['count']) * 100
survival_rates = survival_rates.sort_values('survival_rate', ascending=False)

# Create bar chart
fig = go.Figure(data=[
    go.Bar(
        x=survival_rates['Title'],
        y=survival_rates['survival_rate'],
        text=[f'{rate:.1f}%' for rate in survival_rates['survival_rate']],
        textposition='outside',
        marker_color=['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C'][:len(survival_rates)]
    )
])

# Update layout
fig.update_layout(
    title='Survival Rates by Passenger Title',
    xaxis_title='Title',
    yaxis_title='Survival Rate (%)',
    showlegend=False
)

# Update traces
fig.update_traces(cliponaxis=False)

# Update y-axis to show percentage
fig.update_yaxes(ticksuffix='%', range=[0, max(survival_rates['survival_rate']) * 1.1])

# Save as PNG and SVG
fig.write_image('survival_by_title.png')
fig.write_image('survival_by_title.svg', format='svg')

print("Chart saved successfully!")
print("\nSurvival rates by title:")
for _, row in survival_rates.iterrows():
    print(f"{row['Title']}: {row['survival_rate']:.1f}% ({row['sum']}/{row['count']})")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

# Load the Excel file
df = pd.read_excel('train.xlsx')

# Extract titles from Name column
def extract_title(name):
    title = re.search(' ([A-Za-z]+)\.', name)
    if title:
        return title.group(1)
    else:
        return 'Unknown'

df['Title'] = df['Name'].apply(extract_title)

# Group rare titles into 'Other'
title_counts = df['Title'].value_counts()
rare_titles = title_counts[title_counts < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Other')

# Calculate survival rates by title
survival_rates = df.groupby('Title')['Survived'].agg(['count', 'sum']).reset_index()
survival_rates['survival_rate'] = (survival_rates['sum'] / survival_rates['count']) * 100
survival_rates = survival_rates.sort_values('survival_rate', ascending=False)

# Create bar chart
fig = go.Figure(data=[
    go.Bar(
        x=survival_rates['Title'],
        y=survival_rates['survival_rate'],
        text=[f'{rate:.1f}%' for rate in survival_rates['survival_rate']],
        textposition='outside',
        marker_color=['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C'][:len(survival_rates)]
    )
])

# Update layout
fig.update_layout(
    title='Survival Rates by Passenger Title',
    xaxis_title='Title',
    yaxis_title='Survival Rate (%)',
    showlegend=False
)

# Update traces
fig.update_traces(cliponaxis=False)

# Update y-axis to show percentage
fig.update_yaxes(ticksuffix='%', range=[0, max(survival_rates['survival_rateimport pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Try to load the family_survival.csv file first
try:
    df = pd.read_csv("family_survival.csv")
    print("Loaded family_survival.csv successfully")
    print("Columns:", df.columns.tolist())
    print("First few rows:")
    print(df.head())
except FileNotFoundError:
    # If family_survival.csv doesn't exist, try to load train.xlsx and calculate family survival data
    print("family_survival.csv not found, trying train.xlsx...")
    df = pd.read_excel("train.xlsx")
    
    # Calculate family size (SibSp + Parch + 1 for the person themselves)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Calculate survival rate by family size
    family_survival = df.groupby('FamilySize')['Survived'].agg(['count', 'sum']).reset_index()
    family_survival['SurvivalRate'] = (family_survival['sum'] / family_survival['count']) * 100
    
    # Filter to family sizes 1-11 as requested
    df = family_survival[family_survival['FamilySize'] <= 11].copy()
    
    print("Data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("First few rows:")
    print(df.head())

# Determine correct column names based on what's available
if 'FamilySize' in df.columns:
    family_size_col = 'FamilySize'
elif 'family_size' in df.columns:
    family_size_col = 'family_size'
else:
    family_size_col = df.columns[0]  # Use first column as fallback

# Look for survival rate column
if 'Survival_Rate' in df.columns:
    survival_rate_col = 'Survival_Rate'
elif 'SurvivalRate' in df.columns:
    survival_rate_col = 'SurvivalRate'
elif 'survival_rate' in df.columns:
    survival_rate_col = 'survival_rate'
else:
    # Find a numeric column that could be survival rate
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    survival_rate_col = [col for col in numeric_cols if col != family_size_col][0]

print(f"Using columns: {family_size_col} for x-axis, {survival_rate_col} for y-axis")

# Filter to ensure we have family sizes 1-11
df_filtered = df[df[family_size_col] <= 11].copy()

# Create the line chart
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_filtered[family_size_col], 
    y=df_filtered[survival_rate_col],
    mode='lines+markers',
    line=dict(width=3),
    marker=dict(size=8),
    name='Survival Rate'
))

# Update layout
fig.update_layout(
    title="Survival Rates by Family Size",
    xaxis_title="Family Size",
    yaxis_title="Survival Rate (%)",
    showlegend=False  # Hide legend since only one series
)

# Update x-axis to show all family sizes 1-11
fig.update_xaxes(
    tickmode='linear',
    tick0=1,
    dtick=1,
    range=[0.5, 11.5]
)

# Update y-axis to show percentage scale with clear formatting
fig.update_yaxes(
    tickformat='.1f',
    ticksuffix='%',
    showgrid=True
)

# Update traces for better visibility
fig.update_traces(cliponaxis=False)

# Add gridlines for easier reading
fig.update_layout(
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
)

# Save as PNG and SVG
fig.write_image("chart.png")
fig.write_image("chart.svg", format="svg")

print("Chart saved successfully as chart.png and chart.svg")
print(f"Data points: {len(df_filtered)} family sizes from {df_filtered[family_size_col].min()} to {df_filtered[family_size_col].max()}")
print(f"Survival rates range from {df_filtered[survival_rate_col].min():.1f}% to {df_filtered[survival_rate_col].max():.1f}%")']) * 1.1])

# Save as PNG and SVG
fig.write_image('survival_by_title.png')
fig.write_image('survival_by_title.svg', format='svg')

print("Chart saved successfully!")
print("\nSurvival rates by title:")
for _, row in survival_rates.iterrows():
    print(f"{row['Title']}: {row['survival_rate']:.1f}% ({row['sum']}/{row['count']})")

# Create detailed fare analysis data
fare_analysis_data = []

# Fare ranges by class
for pclass in [1, 2, 3]:
    class_data = df[df['Pclass'] == pclass]
    survivors = class_data[class_data['Survived'] == 1]
    non_survivors = class_data[class_data['Survived'] == 0]
    
    fare_analysis_data.append({
        'Class': f'{pclass}st Class' if pclass == 1 else f'{pclass}nd Class' if pclass == 2 else f'{pclass}rd Class',
        'Group': 'All Passengers',
        'Mean_Fare': class_data['Fare'].mean(),
        'Median_Fare': class_data['Fare'].median(),
        'Min_Fare': class_data['Fare'].min(),
        'Max_Fare': class_data['Fare'].max(),
        'Count': len(class_data)
    })
    
    fare_analysis_data.append({
        'Class': f'{pclass}st Class' if pclass == 1 else f'{pclass}nd Class' if pclass == 2 else f'{pclass}rd Class',
        'Group': 'Survivors',
        'Mean_Fare': survivors['Fare'].mean(),
        'Median_Fare': survivors['Fare'].median(),
        'Min_Fare': survivors['Fare'].min(),
        'Max_Fare': survivors['Fare'].max(),
        'Count': len(survivors)
    })
    
    fare_analysis_data.append({
        'Class': f'{pclass}st Class' if pclass == 1 else f'{pclass}nd Class' if pclass == 2 else f'{pclass}rd Class',
        'Group': 'Non-Survivors',
        'Mean_Fare': non_survivors['Fare'].mean(),
        'Median_Fare': non_survivors['Fare'].median(),
        'Min_Fare': non_survivors['Fare'].min(),
        'Max_Fare': non_survivors['Fare'].max(),
        'Count': len(non_survivors)
    })

fare_df = pd.DataFrame(fare_analysis_data).round(2)
fare_df.to_csv('fare_analysis.csv', index=False)

print("Fare Analysis by Class and Survival Status:")
print(fare_df)

# Deck analysis (for those with cabin information)
deck_data = []
deck_analysis = df.dropna(subset=['Cabin']).groupby('Deck')['Survived'].agg(['count', 'sum', 'mean']).round(4)
deck_analysis['survival_rate'] = (deck_analysis['mean'] * 100).round(2)

for deck in deck_analysis.index:
    deck_data.append({
        'Deck': deck,
        'Count': deck_analysis.loc[deck, 'count'],
        'Survivors': deck_analysis.loc[deck, 'sum'],
        'Survival_Rate': deck_analysis.loc[deck, 'survival_rate']
    })

deck_df = pd.DataFrame(deck_data)
deck_df.to_csv('deck_survival.csv', index=False)

print("\nDeck Analysis (for passengers with cabin data):")
print(deck_df)

# Missing data impact analysis
missing_analysis = {
    'Variable': ['Age', 'Cabin', 'Embarked', 'Total Dataset'],
    'Missing_Count': [177, 687, 2, 891],
    'Missing_Percentage': [19.87, 77.10, 0.22, 0.00],
    'Known_Survival_Rate': [
        df.dropna(subset=['Age'])['Survived'].mean() * 100,
        df.dropna(subset=['Cabin'])['Survived'].mean() * 100,
        df.dropna(subset=['Embarked'])['Survived'].mean() * 100,
        df['Survived'].mean() * 100
    ]
}

missing_df = pd.DataFrame(missing_analysis)
missing_df['Known_Survival_Rate'] = missing_df['Known_Survival_Rate'].round(2)
missing_df.to_csv('missing_data_analysis.csv', index=False)

print("\nMissing Data Impact Analysis:")
print(missing_df)

# Reload the data since it seems to have been lost
df = pd.read_excel('train.xlsx')

# Recreate the derived variables
def extract_title(name):
    title = name.split(',')[1].split('.')[0].strip()
    return title

def simplify_title(title):
    if title in ['Mr']:
        return 'Mr'
    elif title in ['Miss', 'Mlle', 'Ms']:
        return 'Miss'
    elif title in ['Mrs', 'Mme']:
        return 'Mrs'
    elif title in ['Master']:
        return 'Master'
    else:
        return 'Other'

df['Title'] = df['Name'].apply(extract_title)
df['Title_Simple'] = df['Title'].apply(simplify_title)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Deck'] = df['Cabin'].str[0]

# Create detailed fare analysis data
fare_analysis_data = []

# Fare ranges by class
for pclass in [1, 2, 3]:
    class_data = df[df['Pclass'] == pclass]
    survivors = class_data[class_data['Survived'] == 1]
    non_survivors = class_data[class_data['Survived'] == 0]
    
    fare_analysis_data.append({
        'Class': f'{pclass}st Class' if pclass == 1 else f'{pclass}nd Class' if pclass == 2 else f'{pclass}rd Class',
        'Group': 'All Passengers',
        'Mean_Fare': class_data['Fare'].mean(),
        'Median_Fare': class_data['Fare'].median(),
        'Min_Fare': class_data['Fare'].min(),
        'Max_Fare': class_data['Fare'].max(),
        'Count': len(class_data)
    })
    
    if len(survivors) > 0:
        fare_analysis_data.append({
            'Class': f'{pclass}st Class' if pclass == 1 else f'{pclass}nd Class' if pclass == 2 else f'{pclass}rd Class',
            'Group': 'Survivors',
            'Mean_Fare': survivors['Fare'].mean(),
            'Median_Fare': survivors['Fare'].median(),
            'Min_Fare': survivors['Fare'].min(),
            'Max_Fare': survivors['Fare'].max(),
            'Count': len(survivors)
        })
    
    if len(non_survivors) > 0:
        fare_analysis_data.append({
            'Class': f'{pclass}st Class' if pclass == 1 else f'{pclass}nd Class' if pclass == 2 else f'{pclass}rd Class',
            'Group': 'Non-Survivors',
            'Mean_Fare': non_survivors['Fare'].mean(),
            'Median_Fare': non_survivors['Fare'].median(),
            'Min_Fare': non_survivors['Fare'].min(),
            'Max_Fare': non_survivors['Fare'].max(),
            'Count': len(non_survivors)
        })

fare_df = pd.DataFrame(fare_analysis_data).round(2)
fare_df.to_csv('fare_analysis.csv', index=False)

print("Fare Analysis by Class and Survival Status:")
print(fare_df)

# Missing data impact analysis
missing_analysis = {
    'Variable': ['Age', 'Cabin', 'Embarked', 'Total Dataset'],
    'Missing_Count': [177, 687, 2, 891],
    'Missing_Percentage': [19.87, 77.10, 0.22, 0.00],
    'Known_Survival_Rate': [
        df.dropna(subset=['Age'])['Survived'].mean() * 100,
        df.dropna(subset=['Cabin'])['Survived'].mean() * 100,
        df.dropna(subset=['Embarked'])['Survived'].mean() * 100,
        df['Survived'].mean() * 100
    ]
}

missing_df = pd.DataFrame(missing_analysis)
missing_df['Known_Survival_Rate'] = missing_df['Known_Survival_Rate'].round(2)
missing_df.to_csv('missing_data_analysis.csv', index=False)

print("\nMissing Data Impact Analysis:")
print(missing_df)

import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Load the data
df = pd.read_excel('train.xlsx')

# Remove rows with missing fare data
df_clean = df.dropna(subset=['Fare'])

# Calculate mean fare by class for all passengers
all_passengers = df_clean.groupby('Pclass')['Fare'].mean()

# Calculate mean fare by class for survivors
survivors = df_clean[df_clean['Survived'] == 1].groupby('Pclass')['Fare'].mean()

# Calculate mean fare by class for non-survivors
non_survivors = df_clean[df_clean['Survived'] == 0].groupby('Pclass')['Fare'].mean()

# Create class labels (keeping under 15 character limit)
class_labels = ['1st Class', '2nd Class', '3rd Class']

# Prepare data for plotting - ensure all classes are represented
classes = [1, 2, 3]
all_fares = [all_passengers.get(cls, 0) for cls in classes]
survivor_fares = [survivors.get(cls, 0) for cls in classes]
non_survivor_fares = [non_survivors.get(cls, 0) for cls in classes]

# Create the grouped bar chart
fig = go.Figure()

# Add bars for each category
fig.add_trace(go.Bar(
    name='All Pass',
    x=class_labels,
    y=all_fares,
    marker_color='#1FB8CD'
))

fig.add_trace(go.Bar(
    name='Survivors',
    x=class_labels,
    y=survivor_fares,
    marker_color='#DB4545'
))

fig.add_trace(go.Bar(
    name='Non-Survivors',
    x=class_labels,
    y=non_survivor_fares,
    marker_color='#2E8B57'
))

# Update layout
fig.update_layout(
    title='Mean Fare by Class and Survival Status',
    xaxis_title='Class',
    yaxis_title='Mean Fare ($)',
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Format y-axis to show currency with abbreviations
fig.update_yaxes(tickformat='$.0f')

# Update traces to prevent clipping
fig.update_traces(cliponaxis=False)

# Save the chart as both PNG and SVG
fig.write_image('chart.png')
fig.write_image('chart.svg', format='svg')

# Display some summary statistics
print("Mean fare by class and survival status:")
print(f"All passengers - 1st: ${all_fares[0]:.2f}, 2nd: ${all_fares[1]:.2f}, 3rd: ${all_fares[2]:.2f}")
print(f"Survivors - 1st: ${survivor_fares[0]:.2f}, 2nd: ${survivor_fares[1]:.2f}, 3rd: ${survivor_fares[2]:.2f}")
print(f"Non-survivors - 1st: ${non_survivor_fares[0]:.2f}, 2nd: ${non_survivor_fares[1]:.2f}, 3rd: ${non_survivor_fares[2]:.2f}")

# Create comprehensive recommendations and key insights summary
insights_data = {
    'Finding': [
        'Gender was the strongest survival predictor',
        'Passenger class significantly affected survival',
        'Children had higher survival rates',
        'Traveling alone reduced survival chances',
        'Medium-sized families had optimal survival',
        'Embarkation point influenced survival',
        'Higher fares correlated with better survival',
        'Cabin location affected survival chances',
        'Missing cabin data indicates lower survival'
    ],
    'Statistical_Evidence': [
        'Women: 74.2% vs Men: 18.9% survival rate',
        '1st Class: 63% vs 3rd Class: 24% survival rate',
        'Children (0-12): 58% vs Seniors (65+): 9% survival rate',
        'Alone: 30% vs With Family: 51% survival rate',
        'Family size 4: 72% vs Family size 1: 30% survival rate',
        'Cherbourg: 55% vs Southampton: 34% survival rate',
        'Survivors paid £95.61 vs Non-survivors £64.68 (1st class)',
        'Deck B: 74% vs Deck A: 47% survival rate',
        'Known cabin: 67% vs Unknown cabin: 30% survival rate'
    ],
    'Power_BI_Visualization_Suggestion': [
        'Gender survival rate pie chart with drill-through by class',
        'Stacked bar chart showing class survival by gender',
        'Age group survival rates with children highlighted',
        'Family status comparison using clustered columns',
        'Line chart showing family size survival trend',
        'Map visualization of embarkation ports with survival rates',
        'Scatter plot of fare vs survival probability by class',
        'Deck layout heatmap showing survival rates',
        'Missing data impact assessment dashboard'
    ]
}

insights_df = pd.DataFrame(insights_data)
insights_df.to_csv('key_insights_recommendations.csv', index=False)

print("Key Insights and Power BI Recommendations:")
print("=" * 60)
for i in range(len(insights_df)):
    print(f"\n{i+1}. {insights_df.iloc[i]['Finding']}")
    print(f"   Evidence: {insights_df.iloc[i]['Statistical_Evidence']}")
    print(f"   Power BI Visualization: {insights_df.iloc[i]['Power_BI_Visualization_Suggestion']}")

# Create final summary statistics
summary_stats = {
    'Metric': [
        'Total Passengers',
        'Overall Survival Rate',
        'Female Survival Rate', 
        'Male Survival Rate',
        '1st Class Survival Rate',
        '2nd Class Survival Rate',
        '3rd Class Survival Rate',
        'Children Survival Rate',
        'Adult Survival Rate',
        'Passengers with Cabin Data',
        'Passengers Missing Age Data'
    ],
    'Value': [
        891,
        38.38,
        74.20,
        18.89,
        62.96,
        47.28,
        24.24,
        57.97,
        41.80,
        204,
        177
    ],
    'Unit': [
        'count',
        '%',
        '%',
        '%', 
        '%',
        '%',
        '%',
        '%',
        '%',
        'count',
        'count'
    ]
}

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('final_summary_statistics.csv', index=False)

print("\n" + "=" * 60)
print("FINAL SUMMARY STATISTICS")
print("=" * 60)
print(summary_df)

    
