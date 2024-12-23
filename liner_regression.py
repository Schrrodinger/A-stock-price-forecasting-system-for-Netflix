from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

# Khởi tạo SparkSession
spark = SparkSession.builder.appName("LinearRegressionWithSpark").getOrCreate()

# Load dữ liệu
df = spark.read.csv('dataset/FE_NFLX.csv', header=True, inferSchema=True)

# Chuyển đổi cột "Date" thành kiểu ngày
df = df.withColumn("Date", col("Date").cast("date"))

# Xử lý thiếu dữ liệu
df = df.na.drop()

# Lựa chọn các đặc trưng
features = ['Open', 'High', 'Low', 'Volume', 'Returns', 'Log_Returns',
            'Price_Range', 'Price_Range_Pct', 'SMA_5', 'SMA_20',
            'EMA_5', 'EMA_20', 'RSI']

# Loại bỏ các đặc trưng có tổng bằng 0
non_zero_features = [f for f in features if df.select(f).rdd.map(lambda row: row[0]).sum() != 0]

# Chuyển đổi dữ liệu thành Vector
assembler = VectorAssembler(inputCols=non_zero_features, outputCol="features")
df = assembler.transform(df)

# Tách tập dữ liệu
(training_data, testing_data) = df.randomSplit([0.8, 0.2], seed=42)

# Huấn luyện mô hình Linear Regression
lr = LinearRegression(featuresCol="features", labelCol="Close")
lr_model = lr.fit(training_data)

# Dự đoán trên tập kiểm tra
predictions = lr_model.transform(testing_data)

# Đánh giá mô hình
evaluator = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
r2 = lr_model.summary.r2

print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# Visualize Actual vs Predicted
actual_vs_predicted = predictions.select("Close", "prediction").toPandas()
plt.scatter(actual_vs_predicted['Close'], actual_vs_predicted['prediction'])
plt.plot([actual_vs_predicted['Close'].min(), actual_vs_predicted['Close'].max()],
         [actual_vs_predicted['Close'].min(), actual_vs_predicted['Close'].max()],
         'r--', lw=2)
plt.xlabel("Actual Close Prices")
plt.ylabel("Predicted Close Prices")
plt.title("Actual vs Predicted Close Prices")
plt.show()

# Kết thúc SparkSession
spark.stop()
