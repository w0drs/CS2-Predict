import streamlit as st
import requests

st.title("Прогноз на матч")
st.write("Чтобы сделать прогноз, заполните все поля ввода")

with st.form("Поле для ввода значений"):
    st.write("Первая команда (ваша прогнозируемая)")
    Team_A_avg_win_percentage = st.number_input("Средний процент побед")
    Team_A_avg_KR = st.number_input("Средний показатель: убийства/раунды, в команде")
    Team_A_avg_elo = st.number_input("Средний elo")
    st.write("________________________________________________________________")
    st.write("Вторая команда")
    Team_B_avg_win_percentage = st.number_input("Средний процент побед ")
    Team_B_avg_KR = st.number_input("Средний показатель: убийства/раунды, в команде ")
    Team_B_avg_elo = st.number_input("Средний elo ")

    submit = st.form_submit_button("Сделать прогноз")

if submit:
    features = {
        "Team_A_avg_win_percentage": Team_A_avg_win_percentage,
        "Team_A_avg_KR": Team_A_avg_KR,
        "Team_A_avg_elo": Team_A_avg_elo,
        "Team_B_avg_win_percentage": Team_B_avg_win_percentage,
        "Team_B_avg_KR": Team_B_avg_KR,
        "Team_B_avg_elo": Team_B_avg_elo
    }
    response = requests.post("http://127.0.0.1:8000/predict", json=features)
    if response.json()["score"]:
        st.success("Возможно, первая команда выиграет!")
    else:
        st.success("Возможно, вторая команда выиграет!")