import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

CSV_PATH = r"C:\Users\annac\Downloads\pokemon_all.csv"
IMAGE_DIR = r"C:\Users\annac\Downloads\pokemon_png"
PREDICT_IMAGE_PATH = r"C:\Users\annac\Downloads\pokemon_predict_image.png"

if not any(f.endswith(".png") for f in os.listdir(IMAGE_DIR)):
    nested = os.path.join(IMAGE_DIR, "pokemon_png")
    if os.path.isdir(nested):
        IMAGE_DIR = nested

print("[setup] Using image folder:", IMAGE_DIR)

#STEP 1.

#1.1 Load the CSV as a DataFrame and remove rows with missing Type_2
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["Type_2"]).reset_index(drop=True)
print("[1.1] Rows remaining after dropping missing Type_2:", df.shape[0])

#1.2 For each unique "Number", load the matching image, convert to
#     grayscale, and turn it into a 256x256 NumPy array. Stack all of
#     them into one 3D NumPy array. Numbers with no matching image file 
#     are skipped.
image_arrays = []
matched_row_indices = []

for idx, row in df.iterrows():
    number = row["Number"]
    img_filename = str(number) + ".png"
    img_path = os.path.join(IMAGE_DIR, img_filename)
    try:
        img = Image.open(img_path).convert("L")   #L=grayscale
        image_arrays.append(np.array(img))
        matched_row_indices.append(idx)
    except FileNotFoundError:
        #No image exists for this Number so skip it
        continue
    
#Keep only the CSV rows that actually had a matching image, in the same order
df = df.loc[matched_row_indices].reset_index(drop=True)

X = np.array(image_arrays)
print("[1.2] 3D NumPy array shape:", X.shape)   #(342, 256, 256)

#If nothing matched, IMAGE_DIR is almost certainly wrong. Stop here.
if X.shape[0] == 0:
    print("[1.2] No images were matched. Checked this folder:", IMAGE_DIR)
    print("[1.2] Make sure it directly contains files named 1.png, 2.png, etc.")
    raise FileNotFoundError("No matching Pokemon images found -- fix IMAGE_DIR above and rerun.")
    
#1.3 Normalize the pixel values to the range [0, 1.0]
X_norm = (X - X.min()) / (X.max() - X.min())

#QUESTION 1: value of the [0,70,35] cell in the 3D NumPy array
print("[QUESTION 1]", X_norm[0, 70, 35])

#1.4 Dummy code the target attribute ("Type_2") into a NumPy array "y"
y_dummies = pd.get_dummies(df["Type_2"])
class_names = y_dummies.columns.tolist()   #map predictions to labels
y = y_dummies.values.astype(int)

#QUESTION 2: shape of the y array
print("[QUESTION 2]", y.shape)


#STEP 2. MODEL BUILDING

# Conv2D layers require an explicit channel dimension, so reshape the
# images from (N, 256, 256) to (N, 256, 256, 1) before feeding the model.
X_cnn = X_norm.reshape(X_norm.shape[0], 256, 256, 1)

#2.1 Empty Sequential model with 3 Convolutional Blocks
#     (Conv2D + ReLU, followed by 2x2 Max Pooling)
model = models.Sequential()
model.add(layers.Input(shape=(256, 256, 1)))

#Block 1 - 16 filters, 3x3 (low-level features)
model.add(layers.Conv2D(16, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

#Block 2 - 32 filters, 3x3 (more complex features)
model.add(layers.Conv2D(32, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

#Block 3 - 64 filters, 3x3 (more abstract features)
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))


#2.2 Flatten layer: 2D feature maps -> 1D vector
model.add(layers.Flatten())

#2.3 Dense layer with 64 neurons
model.add(layers.Dense(64, activation="relu"))

#2.4 Dense output layer (one node per Type_2 class), softmax activation
model.add(layers.Dense(len(class_names), activation="softmax"))

# QUESTION 3: model architecture summary
print("[QUESTION 3]")
model.summary()

# 2.5 Compile the model and fit it to the 3D image data
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(X_cnn, y, epochs=15, batch_size=16)

#QUESTION 4: accuracy score printed in the console output
print("[QUESTION 4] final training accuracy:", history.history["accuracy"][-1])

#STEP 3. MODEL DEPLOYMENT

#3.1 / 3.2 Load pokemon_predict_image.png and convert it to a grayscale NumPy array (same procedure as 1.2)
predict_img = Image.open(PREDICT_IMAGE_PATH).convert("L")
predict_array = np.array(predict_img)

#3.3 Normalize to [0, 1.0] and reshape to add the batch dimension
predict_norm = (predict_array - predict_array.min()) / (predict_array.max() - predict_array.min())
predict_input = np.expand_dims(predict_norm, axis=0)      #(1, 256, 256)
predict_input = predict_input.reshape(1, 256, 256, 1)

#3.4 Use .predict() to get class probabilities for the image
probabilities = model.predict(predict_input)[0]
predicted_class = class_names[np.argmax(probabilities)]
predicted_prob = np.max(probabilities)

print("[QUESTION 5] predicted class:", predicted_class, "| probability:", predicted_prob)

#Full probability breakdown for every Type_2 class, highest first
prob_table = pd.Series(probabilities, index=class_names).sort_values(ascending=False)
print(prob_table)

# BONUS: visualize the prediction image next to a bar chart
# of class probabilities, similar to the examples shown.
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(predict_array, cmap="gray")
axes[0].set_title("Input image")
axes[0].axis("off")

top5 = prob_table.head(5)
axes[1].barh(top5.index[::-1], top5.values[::-1])
axes[1].set_xlabel("Probability")
axes[1].set_title("Top 5 predicted Type_2 classes")

plt.tight_layout()
plt.savefig("prediction_result.png", dpi=150)
plt.show()

