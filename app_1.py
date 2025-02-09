import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import time
import seaborn as sns
import matplotlib.pyplot as plt

# Title of the form
with st. sidebar:
    st.subheader('Amazon Delivery Time')
    page_selection = st.radio('Page', ['Prediction', 'Plots'])

if page_selection == 'Prediction':
    raw_df = pd.read_csv('./Delivery_Time.csv')
    st.title("Delivery Time Predictor")
    # Create the form
    with st.form(key='Delivery_Time'):
    # Input fields
        c1, c2, c4 = st.columns(3)
        with c1:
            order_day = st.selectbox("Order Day", ['Mon', 'Tue', 'Wed','Thu', 'Fri', 'Sat', 'Sun' ])
        with c2:
            order_time = st.time_input("Order_Time")
        with c4:
            distance = st.number_input('Delivery Distance', min_value= 1, max_value=21, step = 1 )

        d1, d2, d3, d4  = st.columns(4)
        with d1:
            category = st.selectbox('Item Category',raw_df.Category.unique())
        with d2:
            area = st.selectbox('Area',raw_df.Area.unique())
        with d3:
            traffic = st.selectbox('Traffic Conditions',raw_df.Traffic.unique())
        with d4:
            weather = st.selectbox('Weather',raw_df.Weather.unique())


        e1, e2, e3 = st.columns(3)
        with e1:
            age = st.number_input('Delivery Agent Age', min_value = 20, max_value = 39, step = 1)
        with e2:
            rating = st.number_input('Agent Rating', min_value = 1.0, max_value = 5.0, step = 0.5) 
        with e3:
            vehicle = st.selectbox('Vehicle', raw_df.Vehicle.unique())    
        # Form submit button
        submit_button = st.form_submit_button(label='Submit')
        # If the form is submitted
        order_day_num = {'Mon':0,
                        'Tue': 1,
                        'Wed': 2,
                        'Thu': 3,
                        'Fri':4,
                        'Sat': 5,
                        'Sun': 6}
        if submit_button:
            # Create a dictionary with the form data
            form_data = {
                'Agent_Age': age,
                'Agent_Rating': rating,
                'weekdays': order_day_num[order_day],
                'Order_Time': order_time,
                'Weather': weather,
                'Category': category,
                'Traffic': traffic,
                'Vehicle': vehicle,
                'Area': area,
                'distance': distance
            }
            test_data1 = pd.DataFrame([form_data])
                
                # Display a success message
            st.success("Entry Successful!")
            
            # Display the form data as a table
            st.write("Here are your details:")
            st.table(test_data1)
                
            # Convert the dictionary to a DataFrame
            
        
            #convert order date and order time to ordinal value
            import datetime as dt
                # test_data1.Order_Date = pd.to_datetime(test_data1.Order_Date, dayfirst=True)
                # test_data1['weekdays'] = test_data1['Order_Date'].dt.dayofweek
                # test_data1['Order_Date']=test_data1['Order_Date'].map(dt.datetime.toordinal)

            test_data1['Order_Time'] = pd.to_datetime(test_data1['Order_Time'], format='%H:%M:%S')
            test_data1['Order_Time']=test_data1['Order_Time'].dt.strftime('%H%M%S').astype(int)

                #test_data1 is converted to encoded_data by one hot encoding
            encoded_data = pd.get_dummies(data = test_data1, dtype = int)
                    
            prediction_df_columns = ['Agent_Age', 'Agent_Rating', 'Order_Time', 'weekdays', 'distance',
       'Weather_Cloudy', 'Weather_Fog', 'Weather_Sandstorms', 'Weather_Stormy',
       'Weather_Sunny', 'Weather_Windy', 'Traffic_High ', 'Traffic_Jam ',
       'Traffic_Low ', 'Traffic_Medium ', 'Vehicle_motorcycle ',
       'Vehicle_scooter ', 'Vehicle_van', 'Area_Metropolitian ', 'Area_Other',
       'Area_Semi-Urban ', 'Area_Urban ', 'Category_Apparel', 'Category_Books',
       'Category_Clothing', 'Category_Cosmetics', 'Category_Electronics',
       'Category_Grocery', 'Category_Home', 'Category_Jewelry',
       'Category_Kitchen', 'Category_Outdoors', 'Category_Pet Supplies',
       'Category_Shoes', 'Category_Skincare', 'Category_Snacks',
       'Category_Sports', 'Category_Toys']
                    
            df = pd.DataFrame(columns = prediction_df_columns)
                    
            encoded_data_columns = encoded_data.columns
            for column in encoded_data_columns:
                df[column] = encoded_data[column]

            df.fillna(0, inplace=True)   
                
                
            x_test_mean_dict = {'Agent_Age': 2.4040613264047434e-16,
                                'Agent_Rating': -1.0699091572554667e-15,
                                'Order_Time': 1.36909255198643e-16,
                                'weekdays': -2.3633145242622897e-17,
                                'distance': 1.4465114760570912e-17,
                                'Weather_Cloudy': -2.0373401071226637e-17,
                                'Weather_Fog': -7.660398802781215e-17,
                                'Weather_Sandstorms': -7.578905198496309e-17,
                                'Weather_Stormy': 1.7113656899830374e-17,
                                'Weather_Sunny': 2.994889957470316e-17,
                                'Weather_Windy': -1.1918439626667583e-17,
                                'Traffic_High ': 2.200327315692477e-17,
                                'Traffic_Jam ': -8.190107230633107e-17,
                                'Traffic_Low ': -3.707958994963248e-17,
                                'Traffic_Medium ': 8.556828449915187e-18,
                                'Vehicle_motorcycle ': -9.473631498120386e-17,
                                'Vehicle_scooter ': -8.96429647133972e-18,
                                'Vehicle_van': -6.926956364217056e-18,
                                'Area_Metropolitian ': -7.619652000638762e-17,
                                'Area_Other': 5.663805497801005e-17,
                                'Area_Semi-Urban ': 4.2784142249575935e-17,
                                'Area_Urban ': -1.9150997006953038e-17,
                                'Category_Apparel': -9.351391091693027e-17,
                                'Category_Books': -2.7707825456868225e-17,
                                'Category_Clothing': 2.2818209199773833e-17,
                                'Category_Cosmetics': -1.5483784814132245e-17,
                                'Category_Electronics': 3.4125446794304616e-18,
                                'Category_Grocery': 3.993186609960421e-17,
                                'Category_Home': 4.380281230313727e-17,
                                'Category_Jewelry': 1.629872085698131e-18,
                                'Category_Kitchen': -3.6672121928207945e-18,
                                'Category_Outdoors': 5.867539508513271e-17,
                                'Category_Pet Supplies': -4.033933412102874e-17,
                                'Category_Shoes': -9.636618706690199e-17,
                                'Category_Skincare': -3.015263358541542e-17,
                                'Category_Snacks': -3.076383561755222e-17,
                                'Category_Sports': -3.463478182108528e-18,
                                'Category_Toys': 0.0}


            x_test_std_dict = {'Agent_Age': 1.0000573509592419,
                                'Agent_Rating': 1.000057350959242,
                                'Order_Time': 1.0000573509592419,
                                'weekdays': 1.000057350959242,
                                'distance': 1.0000573509592419,
                                'Weather_Cloudy': 1.0000573509592419,
                                'Weather_Fog': 1.000057350959242,
                                'Weather_Sandstorms': 1.000057350959242,
                                'Weather_Stormy': 1.0000573509592419,
                                'Weather_Sunny': 1.000057350959242,
                                'Weather_Windy': 1.0000573509592419,
                                'Traffic_High ': 1.0000573509592419,
                                'Traffic_Jam ': 1.0000573509592416,
                                'Traffic_Low ': 1.0000573509592419,
                                'Traffic_Medium ': 1.0000573509592419,
                                'Vehicle_motorcycle ': 1.000057350959242,
                                'Vehicle_scooter ': 1.0000573509592416,
                                'Vehicle_van': 1.0000573509592416,
                                'Area_Metropolitian ': 1.0000573509592416,
                                'Area_Other': 1.0000573509592419,
                                'Area_Semi-Urban ': 1.0000573509592423,
                                'Area_Urban ': 1.000057350959242,
                                'Category_Apparel': 1.0000573509592419,
                                'Category_Books': 1.0000573509592419,
                                'Category_Clothing': 1.000057350959242,
                                'Category_Cosmetics': 1.0000573509592419,
                                'Category_Electronics': 1.000057350959242,
                                'Category_Grocery': 1.000057350959242,
                                'Category_Home': 1.0000573509592419,
                                'Category_Jewelry': 1.000057350959242,
                                'Category_Kitchen': 1.0000573509592419,
                                'Category_Outdoors': 1.0000573509592423,
                                'Category_Pet Supplies': 1.000057350959242,
                                'Category_Shoes': 1.0000573509592419,
                                'Category_Skincare': 1.0000573509592419,
                                'Category_Snacks': 1.000057350959242,
                                'Category_Sports': 1.000057350959242,
                                'Category_Toys': 1.0000573509592419}

                
            df = (df-x_test_mean_dict)/x_test_std_dict
                
                #df.to_csv('user_data.csv')
                
                #st.spinner()
            mlflow.set_tracking_uri("http://localhost:5000")

            model_name = 'GradientBoostingCV'
            model_version = None
                #run_id = 'dd1bc81c18cc4b129ab49334cb62f49f'

            prediction_model = mlflow.pyfunc.load_model(model_uri=f'models:/{model_name}/{model_version}')


            predicted_delivery_time = prediction_model.predict(df)
            st.subheader(f"Predicted Delivery Time {np.round(*predicted_delivery_time, 2)} Hours")

else:
    # import EDA
    # EDA
    df = pd.read_csv('./Delivery_Time.csv')
    df = df.drop('Unnamed: 0', axis = 1)
    # st.write(df)

    st.subheader('Category')
    select_hue = st.selectbox("Select Hue", ['Weather','Traffic', 'Vehicle', 'Area', 'Category','weekdays'])
    if st.button('Submit'):
        a1, a2 = st.columns(2)
        with a1:
            st.subheader('Delivery Time')
            df2 = df.groupby(select_hue)['Delivery_Time'].mean()
            st.bar_chart(df2)
        with a2:
            st.subheader('Agent Rating')
            df4 = pd.DataFrame(df.groupby(select_hue)['Agent_Rating'].mean())
            st.bar_chart(df4)
        a3, a4 = st.columns(2)
        with a3:
            st.subheader('Agent Age')
            df5 = df.groupby(select_hue)['Agent_Age'].mean()
            st.bar_chart(df5)
        with a4:
            st.subheader('Distance')
            df8 = df.groupby(select_hue)['distance'].mean()
            st.bar_chart(df8)
        
    st.subheader('Delivery Time')
    
    selection1 = st.selectbox("Select Parmaeter", ['Agent Age','Agent Rating', 'distance'])
    
    if st.button('Submit '):
        if selection1 == 'Agent Rating':
            df1 = df.groupby('Agent_Rating')['Delivery_Time'].mean()
            st.bar_chart(df1)   
        
        if selection1 == 'Agent Age':
            
            df1 = df.groupby('Agent_Age')['Delivery_Time'].mean()
            st.bar_chart(df1)
        
        if selection1 == 'distance':
            st.scatter_chart(data = df, x = 'distance', y = 'Delivery_Time')
            
    st.subheader('pair Plot')
    if st.button('Plot'):
        fig3 = sns.pairplot(df, hue = select_hue)
        fig3.figure.savefig('file3.png')
        st.image('./file3.png')