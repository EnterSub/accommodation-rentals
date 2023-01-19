import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from math import cos, asin, sqrt, pi
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import warnings

warnings.simplefilter("ignore")
plt.rcParams['font.size'] = 14.0
sns.set(rc={'figure.figsize':(10, 7)})
st.set_page_config(page_title="Practical Exam - Accommodation Rentals",
                   page_icon='⚙',
                   layout="centered",
                   initial_sidebar_state="collapsed",
                   menu_items=None)

st.title("Practical Exam - Accommodation Rentals")
with st.expander("Write report and FAQ"):
    st.text("""
        Practical Exam - Accommodation Rentals.
        FAQ:
        1. Write parameters about your rental accommodation you providing / searching
        2. The program will write expected price and some metrics with visualizations
        Developer:
        Dmitry Moskalev
        https://github.com/entersub
    """)

df = pd.read_csv('./data/file.csv')
latitude = st.number_input("Enter latitude", value=round(df.latitude.mean(), 5), format='%f')
longitude = st.number_input("Enter longitude", value=round(df.longitude.mean(), 5), format='%f')
property_type = st.selectbox("Select property type", ('Apartment', 'Condominium', 'Guest suite', 'House'))
room_type = st.selectbox("Select room type", ('Entire home/apt', 'Private room', 'Shared room'))
bathrooms = st.number_input("Enter bathrooms", min_value=0, max_value=6, step=1)
bedrooms = st.number_input("Enter bedrooms", min_value=0, max_value=5, step=1)
minimum_nights = st.number_input("Enter minimum nights for rental accommodation", min_value=1, max_value=30, step=1)
price = st.number_input("Enter price", min_value=1, step=1, value=200)
container = st.container()
container_visualizations = st.container()

@st.cache(ttl=3600, suppress_st_warning=True, allow_output_mutation=True)
def calculation(df):
    st.write('Detailed calculations:')
    rf_reg = joblib.load("./data/rf_reg.pkl")
    st.caption(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    df = df.drop(columns=['id'])
    df = df.dropna()
    df.price = pd.Series(df.price.str.replace('\..*$|[$,]', '', regex=True))
    df.price = df.price.astype(int)
    df.bathrooms = df.bathrooms.astype(int)
    df.bedrooms = df.bedrooms.astype(int)
    df.minimum_nights = df.minimum_nights.astype(int)
    df = df.loc[df.minimum_nights <= 30]
    df = df.reset_index(drop=True)
    rows_noise_property_type = []
    for i in list(set(df.property_type)):
        if len(df.loc[df.property_type == i]) < df.shape[0] // 50:
            rows_noise_property_type.append(i)
    df = df.drop(index=df.loc[df.property_type.isin(rows_noise_property_type)].index)
    st.caption(f"Less than 2% property_type count: {len(rows_noise_property_type)}")
    rows_noise_room_type = []
    for i in list(set(df.room_type)):
        if len(df.loc[df.room_type == i]) < df.shape[0] // 50:
            # st.write(i, len(df.loc[df.room_type == i]))
            rows_noise_room_type.append(i)
    st.caption(f"Less than 2% room_type count: {len(rows_noise_room_type)}")
    df = df.drop(index=df.loc[df.property_type.isin(rows_noise_room_type)].index)
    df = df.reset_index(drop=True)
    st.caption(f"Rows: {df.shape[0]}, Columns: {df.shape[1]} after pre-processing")
    df.loc[df.shape[0]] = [latitude, longitude, property_type, room_type, bathrooms, bedrooms, minimum_nights, price]
    df_accommodations = df[['latitude', 'longitude']]
    col1, col2 = container.columns(2)
    with col1:
        st.caption("Your coordinates of entered accommodation")
        st.map(df_accommodations[-1:], zoom=10)
    with col2:
        st.caption("All suitable accommodations")
        st.map(df_accommodations[:-1], zoom=10)
    st.write(f'Your input data for rental accommodation:')
    for count, i in enumerate(df.columns):
        st.caption(f'{str(i)}: {str(df.loc[df.shape[0]-1][count])}')
    latitude_center = 37.773972
    longitude_center = -122.431297
    placeholder.empty()
    def distance(lat1, lon1, lat2, lon2):
        p = pi / 180
        distance = 12742 * asin(
            sqrt(0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2))
        return round(distance, 2)
    df.insert(2, 'distance_from_center', '')
    df = df.reset_index(drop=True)
    for count, i in enumerate(df.distance_from_center):
        df.loc[count, 'distance_from_center'] = distance(latitude_center, longitude_center, df.latitude[count],
                                                         df.longitude[count])
    df.distance_from_center = df.distance_from_center.astype(float)
    df = df.drop(columns=['latitude', 'longitude'])
    def model_stat(y_test, y_pred):
        st.write('Output metrics:')
        st.caption(f'MAE: {mean_absolute_error(y_test, y_pred)}')
        st.caption(f'r2: {r2_score(y_test, y_pred)}')
    X = df.drop(columns='price').copy()
    y = df.price.copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)
    try:
        X_test.loc[df.shape[0] - 1] = X_train.loc[df.shape[0] - 1]
        y_test.loc[df.shape[0] - 1] = y_train.loc[df.shape[0] - 1]
    except KeyError:
        pass
    try:
        X_train = X_train.drop(df.shape[0] - 1)
        y_train = y_train.drop(df.shape[0] - 1)
    except KeyError:
        pass
    nume_col = list(X.select_dtypes('number').columns)
    cate_col = list(X.select_dtypes('object').columns)
    pipe_cate = Pipeline([('ohe', OneHotEncoder(sparse=False, drop=None))])
    pipe_nume = Pipeline([('scaler', StandardScaler())])
    preprocessor = ColumnTransformer([('nume_feat', pipe_nume, nume_col),
                                      ('cate_feat', pipe_cate, cate_col)])
    X_train_pr = pd.DataFrame(preprocessor.fit_transform(X_train),
                              columns=nume_col + list(preprocessor.named_transformers_['cate_feat'].
                                           named_steps['ohe'].get_feature_names_out(cate_col)))
    X_test_pr = pd.DataFrame(preprocessor.transform(X_test),
                             columns=nume_col + list(preprocessor.named_transformers_['cate_feat'].
                                          named_steps['ohe'].get_feature_names_out(cate_col)))
    y_pred_rf = rf_reg.predict(X_test_pr)
    model_stat(y_test, y_pred_rf)
    features_value = pd.DataFrame({'features': rf_reg.feature_importances_, 'value': X_train_pr.columns})
    features_plot = features_value.plot(kind='bar', x='value', y='features', figsize=(7, 7), color=list('rgbymc'), legend=False)
    plt.title('Feature importances')
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    st.caption(f'Feature importances: {rf_reg.feature_importances_}')
    st.caption(f'Sum: {rf_reg.feature_importances_.sum()}')
    # X_prediction = X_train_pr.head(1)
    # prediction = rf_reg.predict(X_prediction)[0]
    # print(prediction)
    df_visualization = df[
['distance_from_center', 'property_type', 'room_type', 'bathrooms', 'bedrooms', 'minimum_nights', 'price']]
    with container_visualizations.expander("Visualizations"):
        f, ax = plt.subplots(figsize=(10, 10))
        st.pyplot(features_plot.figure)
        st.pyplot(sns.heatmap(df.corr(method='pearson'), vmin=-1, vmax=1, annot=True, cmap='RdBu', square=True).
                  figure)

    calculated_price = y_pred_rf[y_pred_rf.shape[0] - 1]
    container.caption(f'Calculated price: {calculated_price}')
    if calculated_price - price > 25:
        container.info(f'Estimated rental price per night is more expensive for more than 25 dollars of actual price ~ {round(calculated_price-price, 1)}$ or ~ {round(((abs(calculated_price-price))/price)*100,1)}%')
    elif price - calculated_price > 25:
        container.info(f'Estimated rental price per night is more cheaper for more than 25 dollars of actual price ~ {round(price-calculated_price, 1)}$ or ~ {round(((abs(calculated_price-price))/calculated_price)*100,1)}%')
    else:
        container.info(f'Estimated rental price per night is in a range of 25 dollars from an actual price and the difference is ~ {round(abs(calculated_price-price), 1)}')
if st.button("Calculate"):
    placeholder = st.empty()
    with placeholder.container():
        st.success("Starting")
    calculation(df)