import tensorflow
import pandas
import glob

model = tensorflow.keras.models.load_model("./model")
model.summary()

for file in glob.glob("input_data/*.xlsx"):
    frame = pandas.read_excel(file)
    frame.drop(frame.columns[0], axis=1, inplace=True)
    frame.drop(columns=["Unnamed: 0", "version", "alive", "tagId", "success", "timestamp", "data__anchorData", "errorCode"], inplace=True, errors="ignore")
    frame.drop(columns=["reference__x", "reference__y"], inplace=True)
    corrected = model.predict(frame)
    print(corrected)