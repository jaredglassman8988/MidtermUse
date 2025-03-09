import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

df = pd.read_csv("Life Expectancy Data.csv")
st.title("Welcome to: Life Expectancy Predictions with Jared and Sonia!")
st.write("Want to know how long you'll live? Look no further 😎")
st.image("IMAGE.jpg")

from streamlit_option_menu import option_menu #pip install streamlit-option-menu
selected = option_menu(menu_title=None, options=["01 Data", "02 Visualization", "03 Predictions"], default_index=0, orientation="horizontal")
if selected == "01 Data":
    st.markdown("### :violet[Let's take a look at the data!]")
    st.image("globe.jpg")
    num = st.number_input("No. of rows", 5,10)
    st.dataframe(df.head(num))
    st.image("politics.jpg")
    st.dataframe(df.describe())
    st.image("hi.png")

if selected== "02 Visualization":
    import plotly.express as px
    import streamlit as st
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    st.markdown("### :red[Data Visualization]")
    selected_vars = st.multiselect("Select variables for the correlation matrix",numeric_columns,default=numeric_columns[:3])
    st.title("Correlation Matrix of All Numerical Variables")
    st.image("correlation.jpg")
    st.write("Insights: The following correlation matrix displays the strength of the relationship between any two numerical variables in the dataset.")
    if len (selected_vars)>1:
        tab_corr,= st.tabs(["Correlation"])
    correlation = df[selected_vars].corr()
    fig = px.imshow(correlation.values, x=correlation.index, y=correlation.columns,labels=dict(color="Correlation"), text_auto=True)
    tab_corr.plotly_chart(fig, theme="streamlit",use_container_width=True)
   
    st.title('Line Graph of Life Expectancy by Adult Mortality Rates')
    st.write("Insights: The graph below clearly displays the strong, negative correlation between life expectancy and adult mortality rate. Although there is a unique bunch of outliers towards the left side of the graph, one can see clearly that overall, increases in adult mortality rates are associated with decreases in life expectancy.")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(x='Adult Mortality',y='Life expectancy ', data=df, marker='o', ax=ax)
    ax.set_title('Line Graph')
    ax.set_xlabel('Adult Mortality Rate (probability of dying between 15 and 60 years per 1000 population)')
    ax.set_ylabel('Life Expectancy in Age')
    st.pyplot(fig)

    st.title('Violin Plot of Life Expectancy by County Status')
    st.write("Insights: The following violin plot shows that life expectancies for individuals in developing countries are far more dispersed than those in developed countries. It appears that people in developed countries tend to live longer, and also have a mode life expectancy of around 80 years old, with the overwhelming majority of countries lying at about this age. People in developing countries seem to live shorter with a modal age of about 73, but do not have a strong pattern and there exists a lot more dispersion.")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(x='Status', y='Life expectancy ', data=df, ax=ax)
    ax.set_title('Violin Plot')
    ax.set_xlabel('Status')
    ax.set_ylabel('Life Expectancy')
    st.pyplot(fig)

    st.title("Count Plot of Country Status")
    st.write("Insights: The following count plot shows that most of the countries in this dataset are developing.")
    ax.set_title('Count Plot')
    fig, ax= plt.subplots(figsize=(8,5))
    sns.countplot(x=df["Status"], data=df, ax=ax)
    st.pyplot(fig)

if selected== "03 Predictions":
    import streamlit as st
    import pandas as pd
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.impute import SimpleImputer
    st.title("Life Expectancy Prediction Using Linear Regression")
    st.write("Insights: The below scatterplot with a line of best fit demonstrates that using the numerical variables in the dataset, this linear regression model we generated is strong at predicting actual life expectancies in countries across time. There are few outliers and most of the datapoints lie close to the line. Additionally, the mean absolute error reflects that on average, the model is not too inaccurate in predicting life expectancies.")
    
    df = df.dropna()
    X = df.drop(columns=['Life expectancy ', 'Country','Year','Status']) 
    y = df['Life expectancy ']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and axis
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax)  # Scatter plot
    ax.set_xlabel("Actual Life Expectancies")  # X-axis label
    ax.set_ylabel("Predicted Life Expectancies")  # Y-axis label
    ax.set_title("Actual vs Predicted Life Expectancies")  # Plot title
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, ls='-')
    st.pyplot(fig)
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.image("wonka.png")
