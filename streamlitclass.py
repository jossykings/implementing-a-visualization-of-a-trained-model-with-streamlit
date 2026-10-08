import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("iris flower prediction")
st.write(" A simple model")


iris = load_iris()
X= iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42, test_size=0.2)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train,y_train)

accuracy = model.score(X_test,y_test)

st.write(f"model accuracy on test data:{accuracy:.0%}")
st.header("Confirm your flower")

sepal_width = st.slider("sepal Width cm", 2.0,8.0,5.0)
          

