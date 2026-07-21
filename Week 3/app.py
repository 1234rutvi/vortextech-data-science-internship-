import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Dashboard", layout="wide")
st.title("Interactive Sales Dashboard")

uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    c1,c2=st.columns(2)
    with c1:
        st.metric("Rows", len(df))
    with c2:
        st.metric("Columns", len(df.columns))

    if cat_cols and num_cols:
        cat=st.selectbox("Category",cat_cols)
        num=st.selectbox("Value",num_cols)
        grp=df.groupby(cat)[num].sum()
        fig,ax=plt.subplots()
        grp.plot(kind="bar",ax=ax)
        st.pyplot(fig)

    if len(num_cols)>=2:
        x=st.selectbox("Scatter X",num_cols,key="x")
        y=st.selectbox("Scatter Y",num_cols,key="y")
        fig,ax=plt.subplots()
        ax.scatter(df[x],df[y])
        ax.set_xlabel(x); ax.set_ylabel(y)
        st.pyplot(fig)

    st.subheader("Summary")
    st.write(df.describe())
else:
    st.info("Upload a CSV file to begin.")
