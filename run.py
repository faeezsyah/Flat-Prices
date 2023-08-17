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

