# Product Demand Forecasting

## Project Description
This project focuses on forecasting product demand for a manufacturing company to improve inventory planning and reduce stock-outs. In a global supply chain, long production and shipping lead times (often over a month) can lead to inventory shortages if demand is higher than expected . By predicting future demand with reasonable accuracy, the company can ensure products are shipped to warehouses in time to meet customer needs. The goal of this project is to build a model that forecasts monthly demand for each product **one month ahead** (i.e. forecasting the *month after next*). Such forecasts help the company maintain supply chain continuity and optimize its production and shipping schedule.

## Dataset Description
**Source**: The dataset used is the **“Forecasts for Product Demand”** dataset from Kaggle. It contains historical demand records for a global manufacturing firm.

**Timeframe and Size**: The data spans **7 years (2011–2017)**, with the latest available data from January 2017. It includes demand information for approximately **2,172 unique products** across **4 central warehouses** . There are **5 features** in the dataset, as described below.

**Features**: Each record in the dataset has the following fields:
- **Product_Code** – an identifier for the product.
- **Warehouse** – the warehouse/location fulfilling the order.
- **Product_Category** – category or type of the product.
- **Date** – the date of the order (format: day/month/year).
- **Order_Demand** – the quantity of the product ordered (demand).

**Data Preprocessing**: Several preprocessing steps were applied to prepare the data for modeling:
- *Datetime Conversion*: The `Date` field was converted to datetime format and sorted chronologically for each product. This allowed aggregation and time-series indexing.
- *Cleaning Demand Values*: The `Order_Demand` field, initially read as a string, was converted to numeric. Some entries were found in parentheses (e.g. "(100)") indicating canceled orders or negative demand. These were handled by interpreting them as negative values or treating them as cancellations (and thus removed or adjusted) so that demand values accurately reflect actual orders.
- *Aggregation*: Since the forecasting goal is monthly demand, daily records were aggregated to **monthly demand per product per warehouse**. For each product-warehouse combination, order quantities were summed for each month to create a time series of monthly demand.
- *Handling Missing Data*: If a product had no demand in certain months, those months were filled with zero demand to maintain a continuous time series. Products with very sparse data (e.g. introduced late or discontinued early) that could not support modeling were noted or excluded from modeling as needed.
- *Train/Test Split*: For model evaluation, the time series data was split into a training set (e.g. up to end of 2016) and a test set (the remaining period, such as the first few months of 2017). This allowed us to train the model on historical data and evaluate forecasts on unseen future data.

## Installation and Dependencies
To run this project, you need **Python 3.x** and the following libraries installed:
- **pandas** (data manipulation)
- **numpy** (numerical computations)
- **matplotlib** and **seaborn** (visualizations)
- **statsmodels** (time series modeling, e.g. ARIMA)
- **pmdarima** (for auto ARIMA parameter tuning, optional)
- **prophet** (Facebook Prophet for forecasting, optional)
- **scikit-learn** (for utility functions and metrics)

You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn statsmodels pmdarima prophet scikit-learn
```

Alternatively, use the provided `requirements.txt` (if available) to install all dependencies:

```bash
pip install -r requirements.txt
```

Ensure you have access to the dataset file (e.g., `Historical Product Demand.csv`). You can download the dataset from Kaggle and place it in a `data/` directory within the project folder.

## Usage Instructions
Follow these steps to run the project and reproduce the results:

1. **Clone the repository**: Download or clone this GitHub repository to your local machine.  
   ```bash
   git https://github.com/JeevaSaravanan/Product-Demand-Forecasting-.git
   cd forecasting-product-demand
   ``` 

2. **Install dependencies**: Install the required libraries as mentioned above. If using `requirements.txt`, run the install command shown in the Installation section.

3. **Download the dataset**: Obtain the *Historical Product Demand* dataset from Kaggle. You can download it directly from the Kaggle website or use the Kaggle CLI:  
   ```bash
   kaggle datasets download -d felixzhao/productdemandforecasting
   ```  
   Extract the downloaded file and place `Historical Product Demand.csv` into the `data/` folder of the repository.

4. **Run the analysis**: You can run the project using the Jupyter Notebook or the provided Python scripts:
   - **Jupyter Notebook**: Open the notebook (e.g., `Forecasting_Product_Demand.ipynb`) in Jupyter Notebook or JupyterLab. Run the cells sequentially to execute data loading, preprocessing, model training, and forecasting steps. This will reproduce the analysis and output charts/metrics as documented.
   - **Python Script**: If a script version is provided (e.g., `train_model.py`), run it from the command line:  
     ```bash
     python train_model.py
     ``` 
     This will load the data, train the forecasting model, and output the results (and possibly save forecasts to a file).

5. **Make predictions**: After training, the notebook/script will demonstrate how to forecast future demand. For example, you can adjust the code to forecast the next N months by specifying the forecast horizon. The model can generate predictions for each product and warehouse for the month after the last date in the training data (and further into the future as needed).

6. **Interpret results**: The output will include evaluation metrics on the test set and visualizations of the actual vs. predicted demand for sample products or aggregate demand. Review these results to assess the model performance (explained in the next section).

## Model Details

This project uses three main time-series forecasting models:

- **ARIMA/SARIMA**: Captures autocorrelation and seasonal patterns (using seasonal period S=12) by tuning parameters (p, d, q) and (P, D, Q, S) via ACF/PACF analysis and automated methods.
- **Holt-Winters**: Applies triple exponential smoothing to model level, trend, and seasonality, optimizing smoothing parameters (alpha, beta, gamma) for monthly data.
- **Prophet**: Facebook’s additive model that automatically detects seasonality, trends, and holiday effects, handling outliers robustly with minimal tuning.

Each model was tuned using statistical analysis or auto-optimization to leverage their strengths in forecasting product demand.

## Results and Evaluation

Models were evaluated on a held-out test set using:

- **MAPE**: Measures average percentage error.
- **RMSE**: Assesses absolute forecast error with higher weight on large errors.
- **MAE**: Provides a linear error measure.

Key observations include:

- Seasonal models (Holt-Winters and SARIMA) excelled for products with consistent seasonal trends.
- Prophet delivered robust forecasts with comparable accuracy.
- Simpler models struggled with intermittent or volatile demand, leading to higher errors.
- Forecast accuracy generally decreased as the forecast horizon extended.


Visualizations comparing actual versus forecasted demand confirmed that the models effectively captured overall trends and seasonality, with further improvements possible via additional features, advanced models, and ensemble methods.