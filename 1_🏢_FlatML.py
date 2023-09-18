import streamlit as st
import pandas as pd

room = st.slider('room', min_value=1, max_value=5, value=3)

storeylevel = st.number_input('storey level', min_value=1, max_value=40, value=2, step=1)

town = st.selectbox(
    'town',
    ('PUNGGOL', 'ANG MO KIO', 'SEMBAWANG', 'JURONG WEST', 'CLEMENTI',
       'HOUGANG', 'KALLANG/WHAMPOA', 'QUEENSTOWN', 'TAMPINES',
       'WOODLANDS', 'SENGKANG', 'BISHAN', 'YISHUN', 'PASIR RIS',
       'BUKIT BATOK', 'BEDOK', 'CHOA CHU KANG', 'SERANGOON',
       'CENTRAL AREA', 'TOA PAYOH', 'BUKIT MERAH', 'BUKIT PANJANG',
       'JURONG EAST', 'GEYLANG', 'MARINE PARADE', 'BUKIT TIMAH'))
flat_model = st.selectbox(
    'flat_model',
    ('Improved', 'New Generation', 'Model A', 'DBSS', 'Model A2',
       'Premium Apartment', 'Adjoined flat', 'Simplified', 'Standard',
       'Model A-Maisonette', 'Type S1', 'Premium Apartment Loft',
       'Terrace', 'Type S2', 'Improved-Maisonette', '2-room'))
floor_area_sqm = st.number_input('floor area sqm', min_value=1, max_value=400,step=1)
remaining_lease_years =  st.slider('remaining lease years', min_value=20, max_value=99, value=60)

if st.button('execute'):

    lister = [[floor_area_sqm,remaining_lease_years,storeylevel,room,f'{town}',f'{flat_model}']]

    offer = pd.DataFrame(lister,columns=['floor_area_sqm', 'remaining_lease_years', 'storey_min', 'flat_type.1',
        'town', 'flat_model'])


    from ipynb.fs.full.flatpricez import predictit

    maballs2 = predictit(offer)
    st.write(maballs2)

st.header('Machine Learning process')
st.divider()

st.subheader('Load data')

processcode = '''
data = pd.read_csv('Flat_prices_noexecutives.csv')
data = pd.DataFrame(data)

data = data.dropna(axis=0)
data = data.drop_duplicates(keep='first')
data = data.drop(['remaining_lease','lease_commence_date','storey_range','block','month','flat_type'],axis = 1)

'''
st.code(processcode, language='python')
multi1 = '''
EDA shows data set has duplicates (drop)  
EDA shows data set has nulls (drop)

remaining_lease_years, remaining_lease, lease_commence are the same concept (drop unnecessary atrributes)  
block_value unrelated/bad correlation (drop block)  
street_name,town_name (drop street_name)  


'''
st.markdown(multi1)





st.subheader('Preprocessing')

pipelineimpute = '''datax = data.copy()
X_train, X_valid, y_train, y_valid = train_test_split(datax,data.resale_price)


categorical_column = [col for col in datax.columns if datax[col].dtypes == 'object']
numerical_column = [n for n in datax.columns if datax[n].dtypes == 'int64' or datax[n].dtypes == 'float64']

cols = numerical_column + categorical_column
X_train = X_train[cols]
X_valid = X_valid[cols]
'''
st.code(pipelineimpute, language='python')

multi2 = '''
split the dataset into training and validation sets (80/20)  
indicate & separate the columns in your dataset into categorical and numerical columns based on their data types.  
'''
st.markdown(multi2)



st.subheader('Feature engineering')

combineandtree = '''
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numpreprocessor = Pipeline(steps=[
    ('imp',SimpleImputer(strategy='constant'))
])

catpreprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numpreprocessor, numerical_column),
        ('cat', catpreprocessor, categorical_column)    
    ]
)

model = RandomForestRegressor(n_estimators=100, random_state=0)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

'''
st.code(combineandtree, language='python')
multi3 = '''two separate pipelines, each designed to preprocess different types of data 

numpreprocessor is a pipeline for numerical features that fills missing values with a constant (likely zero by default).  

catpreprocessor is a pipeline for categorical features that fills missing values with the most frequent category   
and converts categorical variables into one-hot encoded vectors.  



my_pipeline is the main pipeline that combines the preprocessing and modeling steps:  

The first step, 'preprocessor', applies the preprocessor (ColumnTransformer) to transform  
and preprocess the input data. It handles both numerical and categorical features accordingly.  

The second step, 'model', fits the random forest regressor to the preprocessed data.  
'''
st.markdown(multi3)


