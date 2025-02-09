This project aims to predict delivery times for e-commerce orders based on a variety of factors such as product size, distance, traffic conditions, and shipping method. Using the provided dataset, learners will preprocess, analyze, and build regression models to accurately estimate delivery times. The final application will allow users to input relevant details and receive estimated delivery times via a user-friendly interface.

Business Use Cases  
Enhanced Delivery Logistics:  
●	Predict delivery times to improve customer satisfaction and optimize delivery schedules.  
Dynamic Traffic and Weather Adjustments:  
●	Adjust delivery estimates based on current traffic and weather conditions.  
Agent Performance Evaluation:  
●	Evaluate agent efficiency and identify areas for training or improvement.  
Operational Efficiency:  
●	Optimize resource allocation for deliveries by analyzing trends and performance metrics.  


Approach  
1.	Data Preparation:  
○	Load and preprocess the dataset.  
○	Handle missing or inconsistent data.  
○	Perform feature engineering (e.g., calculating distance between store and drop locations).  
2.	Data Cleaning:  
○	Remove duplicates and handle missing values.  
○	Standardize categorical variables (e.g., weather, traffic).  
3.	Exploratory Data Analysis (EDA):  
○	Analyze trends in delivery times, agent performance, and external factors.  
○	Visualize the impact of traffic, weather, and other variables on delivery times.  
4.	Feature Engineering:  
○	Calculate geospatial distances using store and drop coordinates.  
○	Extract time-based features (e.g., hour of day, day of the week).  
5.	Regression Model Development:  
○	Train multiple regression models, including:  
■	Linear Regression  
■	Random Forest Regressor  
■	Gradient Boosting Regressor  
○	Evaluate models using metrics like RMSE, MAE, and R-squared.  
○	Compare models and track performance metrics using MLflow.  
6.	Application Development:  
○	Build a user interface using Streamlit to:  
■	Input order details (e.g., distance, traffic, weather, etc.).  
■	Display predicted delivery times.  
7.	Model Comparison and Tracking:  
○	Use MLflow to log, compare, and manage different regression models.  
○	Document the hyperparameters, performance metrics, and model versions.  
8.	Deployment:  
○	Deploy the application in streamlit for accessibility and scalability.


Dataset Explanation  
The dataset contains detailed information about orders, agents, and delivery conditions:  
●	Order_ID: Unique identifier for each order.  
●	Agent_Age: Age of the delivery agent.  
●	Agent_Rating: Rating of the delivery agent.  
●	Store_Latitude/Longitude: Geographic location of the store.  
●	Drop_Latitude/Longitude: Geographic location of the delivery address.  
●	Order_Date/Order_Time: Date and time when the order was placed.  
●	Pickup_Time: Time when the delivery agent picked up the order.  
●	Weather: Weather conditions during delivery.  
●	Traffic: Traffic conditions during delivery.  
●	Vehicle: Mode of transportation used for delivery.  
●	Area: Type of delivery area (Urban/Metropolitan).  
●	Delivery_Time: Target variable representing the actual time taken for delivery (in hours).  
●	Category: Category of the product being delivered.  
 
Streamlit App:  
Prediction Using:  
Order Day  
Order Time  
Delivery Distance  
Item Category  
Area  
Traffic Conditions  
Weather  
Delivery Agent Age  
Delivery Agent Rating  
Vehicle  

Plots:  
Category-wise Plot for  
1. Delivery Time
2. Agent rating
3. Agent Age
4. Distance

Delivery Time variation Plots for  
1. Agent Age
2. Agent rating
3. Distance

Pait Plot with Categoris
