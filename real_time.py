from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import time
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class EnhancedStockForecastingPipeline:
    def __init__(self):
        try:
            # Set environment variables
            os.environ['PYSPARK_PYTHON'] = 'python'
            os.environ['PYSPARK_DRIVER_PYTHON'] = 'python'

            # Initialize Spark Session with optimized configurations
            self.spark = SparkSession.builder \
                .appName("EnhancedStockForecasting") \
                .config("spark.streaming.stopGracefullyOnShutdown", "true") \
                .config("spark.python.worker.timeout", "600") \
                .config("spark.executor.memory", "2g") \
                .config("spark.driver.memory", "2g") \
                .config("spark.sql.shuffle.partitions", "10") \
                .getOrCreate()
            
            # Set log level
            self.spark.sparkContext.setLogLevel("ERROR")
            
            # Define schema for input data
            self.schema = StructType([
                StructField("Datetime", TimestampType(), True),
                StructField("Open", DoubleType(), True),
                StructField("High", DoubleType(), True),
                StructField("Low", DoubleType(), True),
                StructField("Close", DoubleType(), True),
                StructField("Volume", LongType(), True)
            ])
            
            # Load the pre-trained model
            try:
                model_data = joblib.load('linear_regression_model.pkl')
                self.model = model_data['model']
                self.features = model_data['features']
                logging.info(f"Model loaded successfully. Training date: {model_data['training_date']}")
                logging.info(f"Features used: {self.features}")
            except FileNotFoundError:
                logging.error("Model file not found. Please ensure model is saved in 'models' directory.")
                raise
            
            # Register EMA UDF
            self.register_ema_udf()
            
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}")
            raise
    
    def register_ema_udf(self):
        """Register UDF for EMA calculation"""
        def calculate_ema(data, span):
            if not data or len(data) == 0:
                return None
            
            alpha = 2 / (span + 1)
            ema_values = [float(data[0])]
            
            for i in range(1, len(data)):
                if data[i] is not None:
                    ema = float(data[i]) * alpha + ema_values[-1] * (1 - alpha)
                    ema_values.append(ema)
                else:
                    ema_values.append(ema_values[-1])
            
            return ema_values[-1]
        
        self.spark.udf.register("calculate_ema", calculate_ema, DoubleType())
    
    def fetch_real_time_data(self):
        """Fetch real-time Netflix stock data"""
        try:
            stock = yf.Ticker("NFLX")
            data = stock.history(period="1d", interval="1m")
            pdf = data.reset_index()
            
            # Ensure column names match schema
            pdf.columns = pdf.columns.str.title()
            if 'Datetime' not in pdf.columns and 'Date' in pdf.columns:
                pdf = pdf.rename(columns={'Date': 'Datetime'})
            
            # Convert to Spark DataFrame
            df = self.spark.createDataFrame(pdf)
            logging.info(f"Fetched {df.count()} records of real-time data")
            logging.info(f"Columns in fetched data: {df.columns}")
            return df
            
        except Exception as e:
            logging.error(f"Error fetching data: {str(e)}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate all technical indicators"""
        try:
            window = Window.orderBy("Datetime")
            
            # Calculate returns
            df = df \
                .withColumn("prev_close", lag("Close", 1).over(window)) \
                .withColumn("Returns", when(col("prev_close").isNotNull(),
                                         (col("Close") - col("prev_close")) / col("prev_close")).otherwise(0)) \
                .withColumn("Log_Returns", when(col("prev_close").isNotNull(),
                                             log(col("Close") / col("prev_close"))).otherwise(0))
            
            # Price Range calculations
            df = df \
                .withColumn("Price_Range", col("High") - col("Low")) \
                .withColumn("Price_Range_Pct", col("Price_Range") / col("Open"))
            
            # Moving Averages
            df = df \
                .withColumn("SMA_5", avg("Close").over(window.rowsBetween(-4, 0))) \
                .withColumn("SMA_20", avg("Close").over(window.rowsBetween(-19, 0)))
            
            # EMAs
            df = df \
                .withColumn("close_list_5", collect_list("Close").over(window.rowsBetween(-4, 0))) \
                .withColumn("close_list_20", collect_list("Close").over(window.rowsBetween(-19, 0))) \
                .withColumn("EMA_5", expr("calculate_ema(close_list_5, 5)")) \
                .withColumn("EMA_20", expr("calculate_ema(close_list_20, 20)"))
            
            # RSI
            df = df \
                .withColumn("price_diff", col("Close") - lag("Close", 1).over(window)) \
                .withColumn("gain", when(col("price_diff") > 0, col("price_diff")).otherwise(0)) \
                .withColumn("loss", when(col("price_diff") < 0, -col("price_diff")).otherwise(0)) \
                .withColumn("avg_gain", avg("gain").over(window.rowsBetween(-13, 0))) \
                .withColumn("avg_loss", avg("loss").over(window.rowsBetween(-13, 0))) \
                .withColumn("rs", when(col("avg_loss") != 0, col("avg_gain") / col("avg_loss")).otherwise(0)) \
                .withColumn("RSI", when(col("rs") != 0, 100 - (100 / (1 + col("rs")))).otherwise(0))
            
            # Clean up intermediate columns
            columns_to_drop = ["close_list_5", "close_list_20", "price_diff", "gain", 
                             "loss", "avg_gain", "avg_loss", "rs", "prev_close"]
            df = df.drop(*columns_to_drop)
            
            logging.info(f"Technical indicators calculated. Available columns: {df.columns}")
            return df
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {str(e)}")
            return None
    
    def make_predictions(self, df):
        """Make predictions using the loaded model"""
        try:
        # Convert Spark DataFrame to Pandas
            pdf = df.toPandas()
        
        # Extract features for prediction
            X = pdf[self.features]
        
        # Make predictions
            predictions = self.model.predict(X)
        
        # Create a new DataFrame with only the columns we need
            result_pdf = pd.DataFrame({
            'Datetime': pd.to_datetime(pdf['Datetime']),
            'Open': pdf['Open'].astype(float),
            'High': pdf['High'].astype(float),
            'Low': pdf['Low'].astype(float),
            'Close': pdf['Close'].astype(float),
            'Volume': pdf['Volume'].astype(np.int64),
            'Returns': pdf['Returns'].astype(float),
            'Log_Returns': pdf['Log_Returns'].astype(float),
            'Price_Range': pdf['Price_Range'].astype(float),
            'Price_Range_Pct': pdf['Price_Range_Pct'].astype(float),
            'SMA_5': pdf['SMA_5'].astype(float),
            'SMA_20': pdf['SMA_20'].astype(float),
            'EMA_5': pdf['EMA_5'].astype(float),
            'EMA_20': pdf['EMA_20'].astype(float),
            'RSI': pdf['RSI'].astype(float),
            'Predicted_Price': predictions.astype(float),
            'Prediction_Time': pd.to_datetime([datetime.now() for _ in range(len(pdf))]),
        })
        
        # Calculate prediction errors
            result_pdf['Prediction_Error'] = (result_pdf['Predicted_Price'] - result_pdf['Close']).abs().astype(float)
            result_pdf['Prediction_Error_Pct'] = (result_pdf['Prediction_Error'] / result_pdf['Close'] * 100).astype(float)
        
        # Define schema for result DataFrame
            result_schema = StructType([
            StructField("Datetime", TimestampType(), True),
            StructField("Open", DoubleType(), True),
            StructField("High", DoubleType(), True),
            StructField("Low", DoubleType(), True),
            StructField("Close", DoubleType(), True),
            StructField("Volume", LongType(), True),
            StructField("Returns", DoubleType(), True),
            StructField("Log_Returns", DoubleType(), True),
            StructField("Price_Range", DoubleType(), True),
            StructField("Price_Range_Pct", DoubleType(), True),
            StructField("SMA_5", DoubleType(), True),
            StructField("SMA_20", DoubleType(), True),
            StructField("EMA_5", DoubleType(), True),
            StructField("EMA_20", DoubleType(), True),
            StructField("RSI", DoubleType(), True),
            StructField("Predicted_Price", DoubleType(), True),
            StructField("Prediction_Time", TimestampType(), True),
            StructField("Prediction_Error", DoubleType(), True),
            StructField("Prediction_Error_Pct", DoubleType(), True)
        ])
        
        # Convert back to Spark DataFrame with explicit schema
            result_df = self.spark.createDataFrame(result_pdf, schema=result_schema)
        
        # Log latest prediction
            latest = result_pdf.iloc[-1]
            logging.info(f"Latest prediction - Actual: ${latest['Close']:.2f}, "
                    f"Predicted: ${latest['Predicted_Price']:.2f}, "
                    f"Error: {latest['Prediction_Error_Pct']:.2f}%")
        
            return result_df
        
        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}")
            logging.error(f"Available columns: {df.columns}")
        # Print detailed error information
            import traceback
            logging.error(f"Detailed error: {traceback.format_exc()}")
            return None
    
    def run_pipeline(self):
        """Run the enhanced pipeline"""
        try:
            logging.info("Starting real-time forecasting pipeline...")
            
            while True:
                # Fetch and process data
                df = self.fetch_real_time_data()
                if df is not None:
                    logging.info("Calculating technical indicators...")
                    df = self.calculate_technical_indicators(df)
                    if df is not None:
                        logging.info("Making predictions...")
                        predictions_df = self.make_predictions(df)
                        if predictions_df is not None:
                            # Display latest prediction
                            latest = predictions_df.toPandas().iloc[-1]
                            print("\nLatest Prediction:")
                            print(f"Time: {latest['Datetime']}")
                            print(f"Current Price: ${latest['Close']:.2f}")
                            print(f"Predicted Price: ${latest['Predicted_Price']:.2f}")
                            print(f"Prediction Error: {latest['Prediction_Error_Pct']:.2f}%")
                            print("-" * 50)
                
                # Wait before next update
                time.sleep(60)
                
        except KeyboardInterrupt:
            logging.info("Shutting down pipeline gracefully...")
        except Exception as e:
            logging.error(f"Pipeline error: {str(e)}")
        finally:
            self.spark.stop()

if __name__ == "__main__":
    try:
        pipeline = EnhancedStockForecastingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"Failed to start pipeline: {str(e)}")