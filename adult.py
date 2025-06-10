import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from datasist.structdata import detect_outliers
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
import category_encoders as ce

st.set_page_config(page_title="Adult Income Data EDA", layout="wide")
st.title("Adult Income Data EDA, Cleaning, Feature Engineering, and Preprocessing")

# Sidebar Navigation
section = st.sidebar.radio(
    "Go to section:",
    [
        "1. Data Overview",
        "2. Data Cleaning",
        "3. Feature Engineering",
        "4. EDA",
        "5. Data Preprocessing"
    ]
)

# File upload (shared across sections)
uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# 1. Data Overview
if section == "1. Data Overview":
    st.header("1. Data Overview")
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
                 use_column_width=True, caption="Income Analysis Visualization")
    st.subheader("Dataset Summary")
    st.markdown("""
    - **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)
    - **Records:** 32,561
    - **Features:** 15 demographic and employment-related attributes
    - **Target:** Income level (<=50K or >50K)
    """)
    st.subheader("Key Analysis Questions")
    st.markdown("""
    1. **What demographic factors most strongly correlate with higher income?**
    2. **How does education level affect earning potential?**
    3. **What occupations have the highest proportion of high earners?**
    4. **Can we accurately predict income level based on these features?**
    """)
    st.subheader("Attribute Information")
    st.markdown("""
    - **age:** The individual's age  
    - **workclass:** Type of employment  
    - **fnlwgt:** Census sampling weight  
    - **education:** Highest level of education achieved  
    - **education-num:** Numeric version of education  
    - **marital-status:** Marital status  
    - **occupation:** Job or profession  
    - **relationship:** Role in the household  
    - **race:** Self-reported race  
    - **sex:** Gender  
    - **capital-gain:** Income from investments other than wages  
    - **capital-loss:** Investment losses  
    - **hours-per-week:** Weekly work hours  
    - **native-country:** Country of origin  
    - **income:** Target (<=50K or >50K annual income)  
    """)
    st.subheader("Data Preview")
    st.dataframe(df.head())
    st.subheader("Column Data Types")
    st.write(df.dtypes)
    st.subheader("Summary Statistics (Numerical)")
    st.dataframe(df.describe().round(2))
    st.subheader("Summary Statistics (Categorical)")
    st.dataframe(df.describe(include="O"))
    st.subheader("Data Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

# 2. Data Cleaning
elif section == "2. Data Cleaning":
    st.header("2. Data Cleaning")
    st.markdown("""
    Steps:
    - Replace '?' with NaN
    - Remove duplicates
    - Handle missing values
    - Outlier detection & log transformation
    - Encode the target column
    """)
    df.replace('?', np.nan, inplace=True)
    st.write("Replaced '?' with NaN.")
    st.write("Number of duplicates:", df.duplicated().sum())
    if st.button("Drop Duplicates"):
        df.drop_duplicates(inplace=True, ignore_index=True)
        st.success("Duplicates dropped.")
        st.write("Shape after dropping duplicates:", df.shape)
    else:
        st.info("Duplicates not dropped yet.")
    st.write("Missing values per column (%):")
    st.dataframe((df.isna().mean() * 100).round(2))
    st.subheader("Impute Missing Values")
    impute_cols = st.multiselect(
        "Select columns to impute (most frequent for categorical, median for numeric):",
        options=df.columns[df.isna().any()].tolist()
    )
    if st.button("Impute Selected Columns"):
        for col in impute_cols:
            if df[col].dtype == "O":
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        st.success(f"Imputed columns: {impute_cols}")
    st.subheader("Outlier Detection & Log Transformation for Capital Gain/Loss")
    if 'capital.gain' in df.columns:
        outlier_indices_gain = detect_outliers(data=df, n=0, features=['capital.gain'])
        st.write(f"Outliers in capital.gain: {len(outlier_indices_gain)} ({len(outlier_indices_gain)/df.shape[0]*100:.2f}%)")
    else:
        outlier_indices_gain = []
    if 'capital.loss' in df.columns:
        outlier_indices_loss = detect_outliers(data=df, n=0, features=['capital.loss'])
        st.write(f"Outliers in capital.loss: {len(outlier_indices_loss)} ({len(outlier_indices_loss)/df.shape[0]*100:.2f}%)")
    else:
        outlier_indices_loss = []
    if (outlier_indices_gain or outlier_indices_loss) and st.button("Remove detected outliers (gain+loss)"):
        total_outliers = set(outlier_indices_gain) | set(outlier_indices_loss)
        df = df.drop(index=total_outliers).reset_index(drop=True)
        st.success(f"Removed {len(total_outliers)} rows with outliers in capital.gain or capital.loss.")
    if 'capital.gain' in df.columns and 'capital.loss' in df.columns:
        df['capital-gain-log'] = df['capital.gain'].apply(lambda x: np.log1p(x))
        df['capital-loss-log'] = df['capital.loss'].apply(lambda x: np.log1p(x))
        df.drop(columns=['capital.gain', 'capital.loss'], inplace=True)
        st.success("Log1p transformation applied to capital.gain and capital.loss (original columns dropped).")
    st.subheader("Encode Target Column")
    target_col = st.selectbox(
        "Select target column (should be income):",
        options=[col for col in df.columns if "income" in col.lower()]
    )
    if target_col:
        if set(df[target_col].dropna().unique()) == set(['<=50K', '>50K']):
            df[target_col] = df[target_col].map({'<=50K': 0, '>50K': 1})
            st.success(f"Encoded {target_col} as 0/1.")
        elif set(df[target_col].dropna().unique()) == set([0, 1]):
            st.info("Target is already encoded.")
        else:
            st.warning("Target values not recognized for automatic encoding.")
        st.write(df[target_col].value_counts())
    st.subheader("Preview after cleaning")
    st.dataframe(df.head())

# 3. Feature Engineering
elif section == "3. Feature Engineering":
    st.header("3. Feature Engineering")
    st.write("Apply feature engineering steps as described:")
    if 'capital.gain' in df.columns and 'capital.loss' in df.columns:
        df['net_capital'] = df['capital.gain'] - df['capital.loss']
        st.success("Added 'net_capital' column (capital.gain - capital.loss)")
    if 'marital.status' in df.columns:
        def simplify_marital(status):
            if 'Married' in status:
                return 'Married'
            elif status in ['Divorced', 'Separated', 'Widowed']:
                return 'Not-Married'
            else:
                return 'Single'
        df['marital_status_simple'] = df['marital.status'].apply(simplify_marital)
        st.success("Added 'marital_status_simple' column.")
    if 'age' in df.columns:
        bins = [0, 30, 60, 100]
        labels = ['Young', 'Middle-aged', 'Senior']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
        st.success("Added 'age_group' column (Young, Middle-aged, Senior)")
    if 'capital.gain' in df.columns and 'hours.per.week' in df.columns:
        df['income_per_hour'] = df['capital.gain'] / df['hours.per.week'].replace({0: np.nan})
        st.success("Added 'income_per_hour' column (capital.gain / hours.per.week)")
    if 'hours.per.week' in df.columns:
        def categorize_hours(hours):
            if hours < 30:
                return 'Part-time'
            elif hours < 45:
                return 'Full-time'
            else:
                return 'Overtime'
        df['work_hours_cat'] = df['hours.per.week'].apply(categorize_hours)
        st.success("Added 'work_hours_cat' column (Part-time, Full-time, Overtime)")
    st.subheader("Preview with Engineered Features")
    st.dataframe(df.head(10))

# 4. EDA (with dropdown for Univariate, Bivariate, Multivariate)
elif section == "4. EDA":
    st.header("4. Exploratory Data Analysis (EDA)")
    st.markdown("""
    - Visualize distributions and relationships
    - Answer analysis questions
    """)
    eda_type = st.selectbox("Select EDA Type", [
        "Univariate Analysis",
        "Bivariate Analysis",
        "Multivariate Analysis"
    ])
    # --- Univariate Analysis ---
    if eda_type == "Univariate Analysis":
        st.subheader("Univariate Analysis")
        st.markdown("Explore distributions of individual features.")

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            num_col = st.selectbox("Numerical column:", num_cols, key="uni_num")
            if num_col:
                fig = px.histogram(df, x=num_col, nbins=30, title=f"Distribution of {num_col}")
                st.plotly_chart(fig, use_container_width=True)

        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            cat_col = st.selectbox("Categorical column:", cat_cols, key="uni_cat")
            if cat_col:
                chart_type = st.radio("Choose chart type for categorical column", ["Bar Chart", "Pie Chart"], key="uni_cat_chart")
                if chart_type == "Bar Chart":
                    fig = px.bar(df[cat_col].value_counts().reset_index(),
                                x='index', y=cat_col,
                                labels={'index': cat_col, cat_col: 'Count'},
                                title=f"Value Counts of {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.pie(df, names=cat_col, title=f"Pie Chart of {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)


    # --- Bivariate Analysis ---
    elif eda_type == "Bivariate Analysis":
        st.subheader("Bivariate Analysis")
        st.markdown("Explore relationships between two variables.")

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        target_col = [col for col in df.columns if "income" in col.lower()]

        # Scatter Plot (Numeric vs Numeric)
        st.markdown("**Scatter Plot (Numeric vs Numeric):**")
        if len(num_cols) >= 2:
            x_num = st.selectbox("X-axis (numeric):", num_cols, key="bi_xnum")
            y_num = st.selectbox("Y-axis (numeric):", [col for col in num_cols if col != x_num], key="bi_ynum")
            if x_num and y_num:
                fig = px.scatter(df, x=x_num, y=y_num, title=f"{x_num} vs {y_num}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric columns for scatter plot.")

        # Box Plot (Numeric vs Categorical)
        st.markdown("**Box Plot (Numeric vs Categorical):**")
        if num_cols and cat_cols:
            num_for_box = st.selectbox("Numeric column:", num_cols, key="bi_box_num")
            cat_for_box = st.selectbox("Categorical column:", cat_cols, key="bi_box_cat")
            if num_for_box and cat_for_box:
                fig = px.box(df, x=cat_for_box, y=num_for_box, title=f"{num_for_box} by {cat_for_box}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric or categorical columns for box plot.")

        # Grouped Bar Plot (Categorical vs Target)
        st.markdown("**Grouped Bar Plot (Categorical vs Target):**")
        if target_col and cat_cols:
            cat_for_bar = st.selectbox("Categorical column (grouped by target):", [c for c in cat_cols if c != target_col[0]], key="bi_bar_cat")
            if cat_for_bar:
                grouped = df.groupby([cat_for_bar, target_col[0]]).size().reset_index(name='count')
                fig = px.bar(grouped, x=cat_for_bar, y='count', color=target_col[0], barmode='group',
                             title=f"{cat_for_bar} vs {target_col[0]}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough categorical columns or target column for grouped bar plot.")

    # --- Multivariate Analysis ---
    elif eda_type == "Multivariate Analysis":
        st.subheader("Multivariate Analysis")
        st.markdown("Explore relationships among three or more variables.")

        st.subheader("Correlation Heatmap")
        corr = df.select_dtypes(include=np.number).corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        if len(num_cols) >= 2 and len(cat_cols) >= 1:
            st.subheader("Scatter Plot with Color (Numeric vs Numeric, color by Category)")
            x_multi = st.selectbox("X-axis (numeric):", num_cols, key="multi_x")
            y_multi = st.selectbox("Y-axis (numeric):", num_cols, key="multi_y")
            color_cat = st.selectbox("Color by (categorical):", cat_cols, key="multi_color")
            if x_multi and y_multi and color_cat and x_multi != y_multi:
                fig = px.scatter(df, x=x_multi, y=y_multi, color=color_cat,
                                title=f"{x_multi} vs {y_multi} colored by {color_cat}")
                st.plotly_chart(fig, use_container_width=True)

        # --- Bar Plot ---
        if cat_cols and num_cols:
            st.subheader("Bar Plot (Categorical X, Numeric Y, Optional Group)")
            bar_x = st.selectbox("Barplot X-axis (categorical):", cat_cols, key="multi_bar_x")
            bar_y = st.selectbox("Barplot Y-axis (numeric):", num_cols, key="multi_bar_y")
            bar_color = st.selectbox("Barplot color/group (categorical, optional):", ["None"] + cat_cols, key="multi_bar_color")
            agg_method = st.selectbox("Aggregation method:", ["mean", "sum", "count"], key="multi_bar_agg")
            if bar_x and bar_y:
                if bar_color != "None":
                    grouped = df.groupby([bar_x, bar_color])[bar_y]
                    grouped_df = getattr(grouped, agg_method)().reset_index()
                    fig = px.bar(
                        grouped_df, x=bar_x, y=bar_y, color=bar_color, barmode="group",
                        title=f"{agg_method.capitalize()} of {bar_y} by {bar_x}" + (f" and {bar_color}" if bar_color != "None" else "")
                    )
                else:
                    grouped = df.groupby(bar_x)[bar_y]
                    grouped_df = getattr(grouped, agg_method)().reset_index()
                    fig = px.bar(
                        grouped_df, x=bar_x, y=bar_y,
                        title=f"{agg_method.capitalize()} of {bar_y} by {bar_x}"
                    )
                st.plotly_chart(fig, use_container_width=True)

        # --- Histogram Plot ---
        if num_cols and cat_cols:
            st.subheader("Histogram (Numeric, color by Category)")
            hist_x = st.selectbox("Histogram X-axis (numeric):", num_cols, key="multi_hist_x")
            hist_color = st.selectbox("Histogram color by (categorical):", ["None"] + cat_cols, key="multi_hist_color")
            if hist_x:
                if hist_color != "None":
                    fig = px.histogram(df, x=hist_x, color=hist_color,
                                    barmode="overlay",
                                    title=f"Histogram of {hist_x} colored by {hist_color}")
                else:
                    fig = px.histogram(df, x=hist_x,
                                    title=f"Histogram of {hist_x}")
                st.plotly_chart(fig, use_container_width=True)
    st.header("Insights & Conclusion")

    # 1) Income vs Age Group
    st.markdown("### 1) What is the relation of income and age group?")
    st.markdown("**A:** Middle age is the group that earns the most money.")
    

    # 2) Income vs Workclass
    st.markdown("### 2) What is the relation of income and workclass?")
    st.markdown("**A:** Private workclass is the most common among high earners.")
    

    # 3) Which hours per week categories make the most income?
    st.markdown("### 3) Which hours per week categories make the most income?")
    st.markdown("**A:** Full time is the most common among high earners.")
    

    # 4) Is there a relationship between capital-gain-log and income?
    st.markdown("### 4) Is there a relationship between capital-gain-log and income?")
    st.markdown("**A:** No, there is no relationship.")
    
    # 5) Which native country is the most common for high earners?
    st.markdown("### 5) Which native country is the most common for high earners?")
    st.markdown("**A:** United States is the top income country.")
    

    # 6) What is the relationship between income, age group, and working hours?
    st.markdown("### 6) What is the relationship between income, age group, and working hours?")
    st.markdown("**A:** Middle-aged & full-time make the most income, then senior & full-time.")
    

    # 7) Does education level and workclass together affect income?
    st.markdown("### 7) Does education level and workclass together affect income?")
    st.markdown("**A:** Yes, higher education levels and certain work classes are associated with higher income levels.")
    

    # 8) What is the relationship between income, occupation, and working hours?
    st.markdown("### 8) What is the relationship between income, occupation, and working hours?")
    st.markdown("**A:** Adm-clerical & full-time make the most income, then craft-repair & full-time.")
    

    # 9) What is the relationship between native country, age group, and income?
    st.markdown("### 9) What is the relationship between native country, age group, and income?")
    st.markdown("**A:** United States & middle-aged make the most income, then United States & senior.")
    
    # 10) What is the relationship between net capital, age group, and income?
    st.markdown("### 10) What is the relationship between net capital, age group, and income?")
    st.markdown("**A:** No clear relationship.")
    
        
# 5. Data Preprocessing (with 4 steps)
elif section == "5. Data Preprocessing":
    st.header("5. Data Preprocessing")
    st.markdown("""
    1. Impute missing values (KNN Imputer)
    2. Encode categorical variables (One-Hot, Binary, Ordinal)
    3. Feature scaling (Robust Scaler)
    4. Prepare data for modeling
    """)

    # Step 1: KNN Imputer for missing values
    st.subheader("Step 1: Impute missing values (KNN Imputer)")
    impute_cols = df.columns[df.isna().any()].tolist()
    if impute_cols:
        st.write("Missing columns:", impute_cols)
        n_neighbors = st.number_input("Number of neighbors (KNN):", min_value=1, max_value=10, value=5)
        if st.button("Impute all missing now (KNN)", key="impute_knn"):
            # KNN Imputer works only on numeric data
            df_temp = df.copy()
            cat_cols = df_temp.select_dtypes(include="object").columns.tolist()
            # Temporarily encode categoricals as numbers for imputation
            for col in cat_cols:
                df_temp[col] = df_temp[col].astype('category').cat.codes
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_imputed = pd.DataFrame(imputer.fit_transform(df_temp), columns=df_temp.columns)
            # Restore categorical columns to int for further encoding
            for col in cat_cols:
                df_imputed[col] = df_imputed[col].round().astype(int)
            # Replace original df with imputed df
            for col in df.columns:
                df[col] = df_imputed[col]
            st.success("All missing values imputed using KNN Imputer.")
    else:
        st.write("No missing values to impute.")

    # Step 2: Encoding
    st.subheader("Step 2: Encode categorical variables")
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    encoding_method = st.selectbox("Encoding method", ["One-Hot", "Binary", "Ordinal"], key="encoding")
    if st.button("Encode Categorical Variables"):
        if encoding_method == "One-Hot":
            encoder = OneHotEncoder(drop='first', sparse=False)
            encoded = encoder.fit_transform(df[cat_cols])
            encoded_cols = encoder.get_feature_names_out(cat_cols)
            df = df.drop(columns=cat_cols)
            df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
            df = pd.concat([df, df_encoded], axis=1)
            st.success("Categorical variables one-hot encoded.")
        elif encoding_method == "Binary":
            encoder = ce.BinaryEncoder(cols=cat_cols)
            df = encoder.fit_transform(df)
            st.success("Categorical variables binary encoded.")
        elif encoding_method == "Ordinal":
            encoder = OrdinalEncoder()
            df[cat_cols] = encoder.fit_transform(df[cat_cols])
            st.success("Categorical variables ordinally encoded.")
    st.write("Current columns:", df.columns.tolist())

    # Step 3: Robust Scaler
    st.subheader("Step 3: Feature Scaling (Robust Scaler)")
    scale_cols = st.multiselect("Select columns to scale (Robust Scaler)", df.select_dtypes(include=np.number).columns.tolist())
    if st.button("Scale Selected Columns (Robust Scaler)"):
        scaler = RobustScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        st.success(f"Scaled columns: {scale_cols}")

    # Step 4: Download
    st.subheader("Step 4: Download Preprocessed Data")
    csv = df.to_csv(index=False).encode()
    st.download_button(
        label="Download Preprocessed CSV",
        data=csv,
        file_name="adult_preprocessed.csv",
        mime="text/csv"
    )
    st.success("Data preprocessing complete! You can now use the data for modeling.")
