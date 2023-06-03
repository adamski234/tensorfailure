import tensorflow
import pandas
import glob

# Load training data
source_data_frames = []
source_files = glob.glob("./training_data/*.xlsx")
for file in source_files:
    frame = pandas.read_excel(file)
    frame.dropna(inplace=True)  # Remove measurements with no data
    frame.drop(frame.columns[0], axis=1, inplace=True)
    frame.drop(columns=["Unnamed: 0", "version", "alive", "tagId", "success", "timestamp", "data__anchorData", "errorCode"], inplace=True, errors="ignore")
    source_data_frames.append(frame)
source_data = pandas.concat(source_data_frames)
source_data_reference = source_data[["reference__x", "reference__y"]]
source_data.drop(columns=["reference__x", "reference__y"], inplace=True)

# Create normalizer
normalizer = tensorflow.keras.layers.Normalization(axis=-1)
normalizer.adapt(source_data)


# Create model
model = tensorflow.keras.models.Sequential([
    normalizer,
    tensorflow.keras.layers.Dense(128, input_shape=(source_data.shape[1],), activation="sigmoid"),
    tensorflow.keras.layers.Dropout(0.1),
    tensorflow.keras.layers.Dense(64, activation="relu"),
    tensorflow.keras.layers.Dropout(0.1),
    tensorflow.keras.layers.Dense(32, activation="relu"),
    tensorflow.keras.layers.Dense(16, activation="relu"),
    tensorflow.keras.layers.Dense(8, activation="relu"),
    tensorflow.keras.layers.Dense(4, activation="relu"),
    tensorflow.keras.layers.Dense(2, activation="relu"),
    tensorflow.keras.layers.Dense(2)
])

# Set model to training
model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.1), loss="mape")

# Train model

print("Training begins")
pandas.set_option("display.max_columns", 1000)
print(source_data_reference.describe())
model.fit(source_data, source_data_reference, epochs=50)

model.summary()

model.save("./model")