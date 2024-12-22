from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark import SparkConf
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import time
import os
import logging
import gc
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OptimizedStockForecastingPipeline:
    def __init__(self):
        try:
            # Set environment variables
            os.environ['PYSPARK_PYTHON'] = 'python'
            os.environ['PYSPARK_DRIVER_PYTHON'] = 'python'

            # Create optimized Spark configuration
            conf = SparkConf() \
                .setAppName("OptimizedStockForecasting") \
                .set("spark.streaming.stopGracefullyOnShutdown", "true") \
                .set("spark.python.worker.timeout", "600") \
                .set("spark.executor.memory", "4g") \
                .set("spark.driver.memory", "4g") \
                .set("spark.sql.shuffle.partitions", "10") \
                .set("spark.default.parallelism", "8") \
                .set("spark.memory.offHeap.enabled", "true") \
                .set("spark.memory.offHeap.size", "2g") \
                .set("spark.sql.adaptive.enabled", "true") \
                .set("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .set("spark.sql.adaptive.skewJoin.enabled", "true") \
                .set("spark.dynamicAllocation.enabled", "true") \
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .set("spark.sql.execution.arrow.pyspark.enabled", "true")

            # Initialize Spark Session
            self.spark = SparkSession.builder \
                .config(conf=conf) \
                .getOrCreate()
            
            # Set log level
            self.spark.sparkContext.setLogLevel("ERROR")
            
            # Initialize ThreadPoolExecutor
            self.executor = ThreadPoolExecutor(max_workers=4)
            
            # Cache commonly used DataFrames
            self.cached_data = {}
            
            # Load and broadcast the model
            self._load_and_broadcast_model()
            
            # Register UDFs
            self._register_udfs()
            
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}")
            raise

    def _load_and_broadcast_model(self):
        """Load and broadcast the model to all workers"""
        try:
            model_data = joblib.load('linear_regression_model.pkl')
            self.model = model_data['model']
            self.features = model_data['features']
            
            # Broadcast model to workers
            self.broadcast_model = self.spark.sparkContext.broadcast(self.model)
            self.broadcast_features = self.spark.sparkContext.broadcast(self.features)
            
            logging.info(f"Model broadcast successful. Features: {self.features}")
        except Exception as e:
            logging.error(f"Error broadcasting model: {str(e)}")
            raise

    def _register_udfs(self):
        """Register UDFs for technical indicators"""
        def calculate_ema(data):
            if not data or len(data) == 0:
                return None
            
            data = [float(x) if x is not None else 0.0 for x in data]
            alpha = 2 / (len(data) + 1)
            
            # Vectorized calculation
            weights = np.array([(1 - alpha) ** i for i in range(len(data))])
            weights = weights / weights.sum()
            
            return float(np.dot(data, weights))
        
        self.spark.udf.register("calculate_ema", calculate_ema, DoubleType())

    def fetch_real_time_data(self):
        """Fetch real-time stock data"""
        try:
            stock = yf.Ticker("NFLX")
            data = stock.history(period="1d", interval="1m")
            pdf = data.reset_index()
            
            # Ensure column names match schema
            pdf.columns = pdf.columns.str.title()
            if 'Datetime' not in pdf.columns and 'Date' in pdf.columns:
                pdf = pdf.rename(columns={'Date': 'Datetime'})
            
            # Drop unnecessary columns
            columns_to_keep = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            pdf = pdf[columns_to_keep]
            
            # Convert to Spark DataFrame
            df = self.spark.createDataFrame(pdf)
            
            # Optimize DataFrame
            df = self.optimize_dataframe(df)
            
            logging.info(f"Fetched {df.count()} records of real-time data")
            return df
            
        except Exception as e:
            logging.error(f"Error fetching data: {str(e)}")
            return None

    def optimize_dataframe(self, df):
        """Optimize DataFrame performance"""
        try:
            # Cache frequently accessed data
            df.cache()
            
            # Repartition if needed
            num_partitions = self.spark.sparkContext.defaultParallelism
            if df.rdd.getNumPartitions() > num_partitions:
                df = df.repartition(num_partitions)
            
            return df
        except Exception as e:
            logging.error(f"Error optimizing DataFrame: {str(e)}")
            return df

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        try:
            # Create window specifications
            window = Window.orderBy("Datetime")
            window_5 = window.rowsBetween(-4, 0)
            window_20 = window.rowsBetween(-19, 0)
            window_rsi = window.rowsBetween(-13, 0)
            
            # Calculate returns
            df = df \
                .withColumn("prev_close", lag("Close", 1).over(window)) \
                .withColumn("Returns", 
                    when(col("prev_close").isNotNull(),
                        (col("Close") - col("prev_close")) / col("prev_close")
                    ).otherwise(0.0)
                ) \
                .withColumn("Log_Returns",
                    when(col("prev_close").isNotNull(),
                        log(col("Close") / col("prev_close"))
                    ).otherwise(0.0)
                )
            
            # Price range calculations
            df = df \
                .withColumn("Price_Range", col("High") - col("Low")) \
                .withColumn("Price_Range_Pct", col("Price_Range") / col("Open"))
            
            # Moving averages
            df = df \
                .withColumn("SMA_5", avg("Close").over(window_5)) \
                .withColumn("SMA_20", avg("Close").over(window_20))
            
            # EMAs
            df = df \
                .withColumn("close_list_5", collect_list("Close").over(window_5)) \
                .withColumn("close_list_20", collect_list("Close").over(window_20)) \
                .withColumn("EMA_5", expr("calculate_ema(close_list_5)")) \
                .withColumn("EMA_20", expr("calculate_ema(close_list_20)"))
            
            # RSI calculation
            df = df \
                .withColumn("price_diff", col("Close") - lag("Close", 1).over(window)) \
                .withColumn("gain", when(col("price_diff") > 0, col("price_diff")).otherwise(0.0)) \
                .withColumn("loss", when(col("price_diff") < 0, -col("price_diff")).otherwise(0.0)) \
                .withColumn("avg_gain", avg("gain").over(window_rsi)) \
                .withColumn("avg_loss", avg("loss").over(window_rsi)) \
                .withColumn("rs", when(col("avg_loss") != 0, col("avg_gain") / col("avg_loss")).otherwise(0.0)) \
                .withColumn("RSI", when(col("rs") != 0, 100 - (100 / (1 + col("rs")))).otherwise(0.0))
            
            # Clean up intermediate columns
            columns_to_drop = [
                "close_list_5", "close_list_20", "price_diff", 
                "gain", "loss", "avg_gain", "avg_loss", "rs", 
                "prev_close"
            ]
            df = df.drop(*columns_to_drop)
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {str(e)}")
            return None

    def make_predictions(self, df):
        """Make predictions using the model"""
        try:
        # Convert to pandas for prediction
            pdf = df.toPandas()
        
        # Extract features
            X = pdf[self.broadcast_features.value]
        
        # Make predictions
            predictions = self.broadcast_model.value.predict(X)
        
        # Calculate metrics using numpy for vectorized operations
            actual_prices = pdf['Close'].values
            prediction_errors = np.abs(predictions - actual_prices)
            prediction_errors_pct = (prediction_errors / actual_prices) * 100
        
        # Create results DataFrame using numpy arrays
            result_pdf = pd.DataFrame({
            'Datetime': pdf['Datetime'],
            'Predicted_Price': predictions,
            'Actual_Price': actual_prices,
            'Prediction_Error': prediction_errors,
            'Prediction_Error_Pct': prediction_errors_pct
        })
        
        # Define schema for Spark DataFrame
            result_schema = StructType([
            StructField("Datetime", TimestampType(), True),
            StructField("Predicted_Price", DoubleType(), True),
            StructField("Actual_Price", DoubleType(), True),
            StructField("Prediction_Error", DoubleType(), True),
            StructField("Prediction_Error_Pct", DoubleType(), True)
        ])
        
        # Ensure all numeric columns are float type
            numeric_columns = ['Predicted_Price', 'Actual_Price', 'Prediction_Error', 'Prediction_Error_Pct']
            for col in numeric_columns:
                result_pdf[col] = result_pdf[col].astype(float)
        
        # Convert datetime column
            result_pdf['Datetime'] = pd.to_datetime(result_pdf['Datetime'])
        
        # Convert to Spark DataFrame
            result_df = self.spark.createDataFrame(result_pdf, schema=result_schema)
        
        # Log success
            logging.info("Predictions made successfully")
            logging.info(f"Number of predictions: {result_df.count()}")
        
        # Cache the result DataFrame
            result_df.cache()
        
            return result_df
        
        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return None

    def cleanup(self):
        """Clean up resources"""
        try:
            # Unpersist cached DataFrames
            for df in self.cached_data.values():
                if df is not None:
                    try:
                        df.unpersist()
                    except Exception as e:
                        logging.warning(f"Error unpersisting DataFrame: {str(e)}")
            
            # Clear broadcast variables
            if hasattr(self, 'broadcast_model') and self.broadcast_model is not None:
                try:
                    self.broadcast_model.unpersist()
                except Exception as e:
                    logging.warning(f"Error unpersisting broadcast_model: {str(e)}")
            
            if hasattr(self, 'broadcast_features') and self.broadcast_features is not None:
                try:
                    self.broadcast_features.unpersist()
                except Exception as e:
                    logging.warning(f"Error unpersisting broadcast_features: {str(e)}")
            
            # Shutdown thread pool
            if hasattr(self, 'executor'):
                try:
                    self.executor.shutdown(wait=False)
                except Exception as e:
                    logging.warning(f"Error shutting down executor: {str(e)}")
            
            # Stop Spark session
            if hasattr(self, 'spark'):
                try:
                    self.spark.stop()
                except Exception as e:
                    logging.warning(f"Error stopping Spark session: {str(e)}")
            
            # Force garbage collection
            gc.collect()
            
            logging.info("Cleanup completed successfully")
            
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")

    def run_pipeline(self):
        """Run the optimized pipeline"""
        try:
            logging.info("Starting optimized forecasting pipeline...")
            
            while True:
                start_time = time.time()
                
                # Fetch and process data
                df = self.fetch_real_time_data()
                if df is not None:
                    # Calculate indicators
                    df = self.calculate_technical_indicators(df)
                    if df is not None:
                        # Make predictions
                        predictions_df = self.make_predictions(df)
                        if predictions_df is not None:
                            # Display results
                            latest = predictions_df.toPandas().iloc[-1]
                            print(f"\nPrediction Time: {time.time() - start_time:.2f} seconds")
                            print(f"Time: {latest['Datetime']}")
                            print(f"Current Price: ${latest['Actual_Price']:.2f}")
                            print(f"Predicted Price: ${latest['Predicted_Price']:.2f}")
                            print(f"Error: {latest['Prediction_Error_Pct']:.2f}%")
                            print("-" * 50)
                
                # Clean up memory
                gc.collect()
                
                # Wait for next iteration
                time.sleep(60)
                
        except KeyboardInterrupt:
            logging.info("Shutting down pipeline gracefully...")
        except Exception as e:
            logging.error(f"Pipeline error: {str(e)}")
        finally:
            self.cleanup()

if __name__ == "__main__":
    try:
        pipeline = OptimizedStockForecastingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"Failed to run pipeline: {str(e)}")