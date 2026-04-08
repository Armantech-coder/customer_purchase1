import streamlit as st
import pandas as pd
import joblib

# Загрузка обученной модели
@st.cache_resource
def load_model():
    return joblib.load('customer_purchase_model.joblib')

# ─── Боковая панель ───
st.sidebar.header("⚙️ Настройки")
theme = st.sidebar.radio("Тема", ["Светлая", "Тёмная"])
st.sidebar.info(f"Выбрана тема: {theme}")
model = load_model()

st.title('Прогнозирование статуса покупки клиента')
st.write('Используйте этот интерфейс для прогнозирования, совершит ли клиент покупку.')

# Создание полей ввода для признаков
age = st.slider('Возраст', 18, 70, 30)

gender_options = [(0, 'Мужской'), (1, 'Женский')]
selected_gender_tuple = st.radio('Пол', options=gender_options, format_func=lambda x: x[1])
gender = selected_gender_tuple[0]

annual_income = st.number_input('Годовой доход ($)', 10000.0, 200000.0, 50000.0, step=1000.0)
number_of_purchases = st.slider('Количество покупок', 0, 20, 5)

product_category_options = [(0, 'Электроника'), (1, 'Одежда'), (2, 'Товары для дома'), (3, 'Косметика'), (4, 'Спорт')]
selected_product_category_tuple = st.selectbox('Категория товара', options=product_category_options, format_func=lambda x: x[1])
product_category = selected_product_category_tuple[0]

time_spent_on_website = st.number_input('Время, проведенное на сайте (минуты)', 0.0, 120.0, 20.0, step=1.0)

loyalty_program_options = [(0, 'Нет'), (1, 'Да')]
selected_loyalty_program_tuple = st.radio('Программа лояльности', options=loyalty_program_options, format_func=lambda x: x[1])
loyalty_program = selected_loyalty_program_tuple[0]

discounts_availed = st.slider('Полученные скидки', 0, 5, 1)

# Кнопка для прогнозирования
if st.button('Сделать прогноз'):
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'AnnualIncome': annual_income,
        'NumberOfPurchases': number_of_purchases,
        'ProductCategory': product_category,
        'TimeSpentOnWebsite': time_spent_on_website,
        'LoyaltyProgram': loyalty_program,
        'DiscountsAvailed': discounts_availed
    }])

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0, 1]

    st.subheader('Результат прогноза:')
    if prediction == 1:
        st.success(f'Клиент, скорее всего, совершит покупку! (Вероятность: {prediction_proba:.2f})')
    else:
        st.warning(f'Клиент, скорее всего, не совершит покупку. (Вероятность: {prediction_proba:.2f})')

st.sidebar.header('О приложении')
st.sidebar.info(
    'Это демонстрационное приложение Streamlit для прогнозирования статуса покупки клиента на основе предоставленных данных.'
)
