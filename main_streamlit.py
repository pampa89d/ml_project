# 1. Импорт всех необходимых библиотек для работы в проекте

import os
import numpy as np
import pandas as pd

from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

plt.rcParams["figure.dpi"] = 100
plt.style.use("ggplot")
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "{:.2f}".format(x))
pd.reset_option("display.float_format")
# from category_encoders import TargetEncoder

# Важная настройка для корректной настройки pipeline!
import sklearn

sklearn.set_config(transform_output="pandas")

# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    OrdinalEncoder,
    FunctionTransformer,
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import f_regression, chi2

# for model learning
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score,
    StratifiedKFold,
)

# models
from sklearn.neighbors import (
    KNeighborsClassifier,
    RadiusNeighborsClassifier,
    KNeighborsRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    BaggingClassifier,
    RandomForestRegressor,
    VotingRegressor,
)

# LightGBM
import lightgbm as lgb

# CatBoost
from catboost import CatBoostRegressor

# Metrics
from sklearn.metrics import accuracy_score, roc_auc_score

import seaborn as sns

# tunning hyperparamters model
import optuna

import streamlit as st
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

import joblib

st.write(
    """
# ML-сервис для автоматического предсказания будущих объектов недвижимости"""
)

st.subheader(" ", divider="gray")

st.write('## **Участники команды "ДЕДА"**')
st.write(
    """ 
### - **Дмитрий**
### - **Евгений**
### - **Динара**
### - **Андрей**
"""
)

st.subheader(" ", divider="gray")

# Путь относительно текущего файла
current_dir = os.path.dirname(__file__)  # Директория, где лежит скрипт

# Загружаем изображения
img4 = Image.open(os.path.join(current_dir, "Photos", "Eugene.jpg"))
img2 = Image.open(os.path.join(current_dir, "Photos", "Deema.jpg"))
img3 = Image.open(os.path.join(current_dir, "Photos", "Dinara.jpg"))
img1 = Image.open(os.path.join(current_dir, "Photos", "Andrey.jpg"))

# Создаём 4 колонки
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image(img2, caption="Дима", width=200)

with col2:
    st.image(img4, caption="Евгений", width=200)

with col3:
    st.image(img3, caption="Динара", width=200)

with col4:
    st.image(img1, caption="Андрей", width=200)



# Загрузка датасета
# Читаем датасет
main_data = pd.read_csv(os.path.join(current_dir, 'datasets', 'train.csv'))
main_data_copy = main_data.copy().set_index("Id")
main_data_copy.drop("3SsnPorch", axis=1, inplace=True)
main_data_copy["SalePrice"] = np.log(main_data_copy["SalePrice"])

main_data_copy.drop(
    ['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
    + [
        "HalfBath",
        "Electrical",
        "Neighborhood",
        "BedroomAbvGr",
        "RoofStyle",
        "MSZoning",
        "GarageQual",
        "Exterior2nd",
        "LotConfig",
        "Condition2",
    ],
    inplace=True,
    axis=1,
    )



cutter_quants_list = [
    "LotFrontage",
    "EnclosedPorch",
    "BsmtFinSF2",
    "OpenPorchSF",
    "WoodDeckSF",
    "MasVnrArea",
    "2ndFlrSF",
    "GarageArea",
    "BsmtFinSF1",
    "TotalBsmtSF",
    "1stFlrSF",
    "BsmtUnfSF",
    "GrLivArea",
    "LotArea",
]

def cutter_quants(X):
    X_copy = X.copy()
    for col in cutter_quants_list:
        q_low = X_copy[col].quantile(0.025)
        q_high = X_copy[col].quantile(0.975)
        X_copy[col] = np.clip(X_copy[col], q_low, q_high)
    return X_copy

my_inputer = ColumnTransformer(
    [
        (
            "all_num",
            SimpleImputer(strategy="median"),
            [
                "LotFrontage",
                "MasVnrArea",
                "BsmtHalfBath",
                "BsmtFullBath",
                "GarageCars",
                "GarageYrBlt",
                "BsmtFinSF2",
                "GarageArea",
                "BsmtFinSF1",
                "TotalBsmtSF",
                "BsmtUnfSF",
            ],
        ),
        (
            "all_ctgrs",
            SimpleImputer(strategy="most_frequent"),
            [
                "GarageFinish",
                "BsmtCond",
                "BsmtQual",
                "BsmtExposure",
                "GarageCond",
                "BsmtFinType1",
                "BsmtFinType2",
                "GarageType",
                "Utilities",
                "KitchenQual",
                "Functional",
                "SaleType",
                "Exterior1st",
            ],
        ),
    ],
    verbose_feature_names_out=False,
    remainder="passthrough",
)

my_cutter = ColumnTransformer(
    [("cutter", FunctionTransformer(cutter_quants), cutter_quants_list)],
    verbose_feature_names_out=False,
    remainder="passthrough",
)


ordinal = [
    "BsmtHalfBath",
    "BsmtFullBath",
    "FullBath",
    "Fireplaces",
    "KitchenAbvGr",
    "GarageCars",
    "OverallCond",
    "OverallQual",
    "MoSold",
    "TotRmsAbvGrd",
    "MSSubClass",
    "GarageYrBlt",
    "YearRemodAdd",
    "YrSold",
    "Street",
    "Utilities",
    "CentralAir",
    "LandSlope",
    "GarageFinish",
    "PavedDrive",
    "LandContour",
    "ExterQual",
    "KitchenQual",
    "BsmtQual",
    "BsmtExposure",
    "BsmtCond",
    "LotShape",
    "HeatingQC",
    "ExterCond",
    "GarageCond",
    "BldgType",
    "Heating",
    "BsmtFinType1",
    "BsmtFinType2",
    "Foundation",
    "GarageType",
    "SaleCondition",
    "Functional",
    "HouseStyle",
    "RoofMatl",
    "SaleType",
    "Condition1",
    "Exterior1st",
]

standart_sc = ["LotFrontage", "GrLivArea", "1stFlrSF", "LotArea", "YearBuilt"]

minmax = [
    "PoolArea",
    "MiscVal",
    "LowQualFinSF",
    "ScreenPorch",
    "EnclosedPorch",
    "BsmtFinSF2",
    "OpenPorchSF",
    "WoodDeckSF",
    "MasVnrArea",
    "2ndFlrSF",
    "GarageArea",
    "BsmtFinSF1",
    "TotalBsmtSF",
    "BsmtUnfSF",
]


my_scaler_and_encoder = ColumnTransformer(
    [
        (
            "ordinal",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ordinal,
        ),
        ("standart_sc", StandardScaler(), standart_sc),
        ("minmax", MinMaxScaler(), minmax),
    ],
    verbose_feature_names_out=False,
    remainder="passthrough",
)


preprocessor = Pipeline(
    [
        ("inputering", my_inputer),
        ("cuttering", my_cutter),
        ("scaling", my_scaler_and_encoder),
    ]
)

X, y = main_data_copy.drop("SalePrice", axis=True), main_data_copy["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)


# ФИНАЛ ДЛЯ KAGGLE

lr = LinearRegression()
dt = DecisionTreeRegressor(
    max_depth=8, min_samples_leaf=6, criterion="friedman_mse", random_state=42
)
knn = KNeighborsRegressor(n_neighbors=6, p=1, weights="distance", metric="minkowski")
rfr = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    criterion="poisson",
    random_state=42,
)
cb = CatBoostRegressor(
    iterations=1000,
    early_stopping_rounds=50,
    depth=5,
    learning_rate=0.034791305544360815,
    l2_leaf_reg=3.9996075411217293,
    bagging_temperature=0.692003923329511,
    border_count=53,
    random_strength=1.1760426195534421,
    grow_policy="SymmetricTree",
    subsample=0.9530745138068817,
    verbose=0,
    loss_function="RMSE",
    random_state=42,
)

voting = VotingRegressor(
    estimators=[("lr", lr), ("dt", dt), ("knn", knn), ("rfr", rfr), ("cb", cb)],
    weights=[
        0.744007544490073,
        0.036821482733168646,
        0.04938543280075221,
        0.08451613445303022,
        0.9988328004130205,
    ],
)

pipe = Pipeline([("preprocessor", preprocessor), ("model", voting)])

# КЭШИРУЕМ ЗАГРУЗКУ CSV
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)


uploaded_file = st.sidebar.file_uploader("**Загрузи сюда датасет** 👇", type="csv")

if uploaded_file is not None:
    real_test = load_csv(uploaded_file)
    real_test = real_test.copy().set_index("Id")
    real_test.drop("3SsnPorch", axis=1, inplace=True)

    # Удаляем ненужные колонки из тестовой выборки
    real_test.drop(
    ['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
    + [
        "HalfBath",
        "Electrical",
        "Neighborhood",
        "BedroomAbvGr",
        "RoofStyle",
        "MSZoning",
        "GarageQual",
        "Exterior2nd",
        "LotConfig",
        "Condition2",
    ],
    inplace=True,
    axis=1,
    )


    final_model = pipe.fit(X_train, y_train)


    y_final = final_model.predict(real_test)
    

    final_csv = pd.DataFrame(np.exp(y_final), real_test.index).reset_index()
    final_csv.columns = ["Id", "SalePrice"]
    st.write(
'''
Качество модели на тренировочных данных RMSE(%): 9.92
'''
    )
    st.write('''
### Прогнозные цены недвижимости на основе предоставленных вами данных:

''')

    st.dataframe(final_csv)

    @st.cache_data
    def convert_for_download(df):
        return df.to_csv().encode("utf-8")

    csv = convert_for_download(final_csv)
    
    st.download_button(
        label="Скачать цены",
        data=csv,
        file_name="Prices.csv",
        mime="text/csv",
        icon=":material/download:",
    )

else:
    st.write('''
Данные не валидные, загрузите '.csv'
''')
    st.stop()
