import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Proiect Tehnici AI", page_icon="icon.png", layout="wide")

st.title("Proiect Tehnici si Metode Avansate de Inteligenta Artificiala")

# 1.
st.header("1. Incarca un fisier CSV/Excel")
uploaded_file = st.file_uploader("Incarca un fisier CSV sau Excel", type=['csv', 'xlsx'])
df = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("Fisier citit cu succes!")
        st.dataframe(df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Eroare la citirea fisierului: {str(e)}")

if df is not None:
    st.subheader("Filtrare Date")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Slidere pentru val numerice
    numeric_filters = {}
    for col in numeric_cols:
        # Eliminam NaN pentru slidere
        col_data = df[col].dropna()
        if col_data.empty:
            st.warning(f"Coloana numerica '{col}' contine doar valori lipsa si va fi ignorata pentru filtrare.")
            continue

        min_val, max_val = float(col_data.min()), float(col_data.max())
        val_range = st.slider(
            f"Filtrare {col}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
        )
        numeric_filters[col] = val_range

    # Multiselect pentru categoric
    cat_filters = {}
    for col in cat_cols:
        selected = st.multiselect(
            f"Filtrare {col}",
            options=df[col].dropna().unique(),
            default=list(df[col].dropna().unique())
        )
        cat_filters[col] = selected

    # Aplicare filtre
    df_filtered = df.copy()
    for col, (min_val, max_val) in numeric_filters.items():
        df_filtered = df_filtered[(df_filtered[col] >= min_val) & (df_filtered[col] <= max_val)]
    for col, selected in cat_filters.items():
        df_filtered = df_filtered[df_filtered[col].isin(selected)]

    st.markdown(f"**Numar randuri inainte de filtrare:** {df.shape[0]}")
    st.markdown(f"**Numar randuri dupa filtrare:** {df_filtered.shape[0]}")
    st.dataframe(df_filtered, use_container_width=True)


#2.
if df is not None:
    st.header("2. Statistici Descriptive")

    st.markdown(f"**Numar randuri si coloane:** {df.shape}")
    st.markdown("**Tipuri de date per coloana:**")
    st.dataframe(df.dtypes, use_container_width=True)

    st.subheader("Valori Lipsa")
    na_counts = df.isnull().sum()
    na_percent = (na_counts / len(df)) * 100
    na_df = pd.DataFrame({"Missing": na_counts, "Percent": na_percent})
    st.dataframe(na_df[na_df['Missing'] > 0], use_container_width=True)

    # Grafic valori lipsa
    fig_na = px.bar(na_df, x=na_df.index, y="Missing", text="Percent", title="Valori lipsa per coloana")
    fig_na.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig_na, use_container_width=True)

    st.subheader("Statistici pentru coloane numerice")
    st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

# 3.
if df is not None and numeric_cols:
    st.header("3. Histograma si Boxplot")

    col_hist = st.selectbox("Selecteaza coloana numerica", numeric_cols)
    n_bins = st.slider("Numar bins histograma", 10, 100, 20)

    # Histogram
    fig_hist = px.histogram(df_filtered, x=col_hist, nbins=n_bins, title=f"Histograma: {col_hist}")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Boxplot
    fig_box = px.box(df_filtered, y=col_hist, title=f"Boxplot: {col_hist}", points="all")
    st.plotly_chart(fig_box, use_container_width=True)

    # Statistici
    mean_val = df_filtered[col_hist].mean()
    median_val = df_filtered[col_hist].median()
    std_val = df_filtered[col_hist].std()
    st.markdown(f"**Media:** {mean_val:.2f}, **Mediana:** {median_val:.2f}, **Deviatie standard:** {std_val:.2f}")


# 4.
if df is not None and cat_cols:
    st.header("4. Analiza Coloane Categorice")

    col_cat = st.selectbox("Selecteaza coloana categorica", cat_cols)
    counts = df_filtered[col_cat].value_counts()
    percents = df_filtered[col_cat].value_counts(normalize=True) * 100
    cat_df = pd.DataFrame({"Frecvența": counts, "Procent": percents.round(2)})
    st.dataframe(cat_df, use_container_width=True)

    fig_bar = px.bar(cat_df, x=cat_df.index, y="Frecvența", text="Procent", title=f"Count Plot: {col_cat}")
    fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)


# 5.
if df is not None and numeric_cols:
    st.header("5. Corelatie si Outlieri")

    st.subheader("Matrice Corelatie")
    corr_matrix = df_filtered[numeric_cols].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", title="Heatmap Corelatie")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Scatter Plot intre doua variabile")
    col_x = st.selectbox("Variabila X", numeric_cols, index=0)
    col_y = st.selectbox("Variabila Y", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

    fig_scatter = px.scatter(df_filtered, x=col_x, y=col_y, trendline="ols",
                             title=f"Scatter Plot: {col_x} vs {col_y}")
    st.plotly_chart(fig_scatter, use_container_width=True)

    pearson_corr = df_filtered[col_x].corr(df_filtered[col_y])
    st.markdown(f"**Coeficient corelatie Pearson ({col_x}, {col_y}): {pearson_corr:.2f}**")

    st.subheader("Detectie Outlieri cu IQR")
    outlier_summary = []
    for col in numeric_cols:
        Q1 = df_filtered[col].quantile(0.25)
        Q3 = df_filtered[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df_filtered[(df_filtered[col] < lower) | (df_filtered[col] > upper)]
        pct_out = len(outliers) / len(df_filtered) * 100
        outlier_summary.append({
            "Coloana": col,
            "Numar outlieri": len(outliers),
            "Procent outlieri": round(pct_out, 2)
        })

        # Vizualizare outlieri
        fig_out = px.box(df_filtered, y=col, title=f"Outlieri: {col}", points="all")
        st.plotly_chart(fig_out, use_container_width=True)

    st.dataframe(pd.DataFrame(outlier_summary), use_container_width=True)
