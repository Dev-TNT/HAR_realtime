import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, BatchNormalization, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1.
# read data
data_standing = pd.read_csv('standing.csv')
data_sitting = pd.read_csv('sitting.csv')
data_waving = pd.read_csv('waving.csv')
data_swat = pd.read_csv('swat.csv')
data_drinking = pd.read_csv('drinking.csv')
data_clapping = pd.read_csv('clapping.csv')
data_Tpose = pd.read_csv('Tpose.csv')

'''
data_crossing = pd.read_csv('crossing.csv')
data_pushup = pd.read_csv('pushup.csv')
'''

# data management
actions = {
    "standing": data_standing,
    "sitting": data_sitting,
    "waving": data_waving,
    "clapping": data_clapping,
#    "crossing": data_crossing,
    "drinking": data_drinking,
#    "pushup": data_pushup,
    "swat": data_swat,
    "Tpose": data_Tpose
}


# 2. LABEL MAP
label_map = {action: i for i, action in enumerate(actions.keys())}
label_names = list(actions.keys())

with open("label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)
print("✅ label_map.json saved:", label_map)

# 3. create sequence
X, Y = [], []
sequence_length = 30
stride = 5

for action, data in actions.items():
    dataset = data.values
    n_sample = len(dataset)
    for i in range(sequence_length, n_sample, stride):
        sequence = dataset[i - sequence_length:i]
        X.append(sequence)
        Y.append(label_map[action])

X = np.array(X)
Y = np.array(Y)

print(f"\n📊 Dataset shape  : X={X.shape}, Y={Y.shape}")
print(f"📊 Samples/class  :")
for action, idx in label_map.items():
    print(f"   [{idx}] {action}: {np.sum(Y == idx)} sequences")

# 4. TRAIN / TEST SPLIT  (stratify giữ tỉ lệ class đồng đều)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=Y
)

print(f"\n🔀 Train samples  : {len(X_train)}")
print(f"🔀 Test  samples  : {len(X_test)}")


# 5. XÂY DỰNG MODEL
model = Sequential([
    # Layer 1: Bidirectional GRU — học ngữ cảnh cả 2 chiều
    Bidirectional(GRU(64, return_sequences=True), input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    Dropout(0.3),

    # Layer 2: GRU thứ hai
    Bidirectional(GRU(64)),
    BatchNormalization(),
    Dropout(0.3),

    # Fully connected
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(actions), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 6. CALLBACKS
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,                # tăng từ 5 lên 10 — tránh dừng quá sớm
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# 7. TRAIN
history = model.fit(
    X_train, Y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, Y_test),
    callbacks=[early_stop, reduce_lr]
)

# 8. ĐÁNH GIÁ — CLASSIFICATION REPORT + CONFUSION MATRIX
print("\n" + "="*60)
print("📈 EVALUATION RESULTS")
print("="*60)

Y_pred = np.argmax(model.predict(X_test), axis=1)

# Classification report (precision, recall, F1 per class)
print("\n📋 Classification Report:")
print(classification_report(Y_test, Y_pred, target_names=label_names))

# Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_names,
    yticklabels=label_names
)
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("✅ confusion_matrix.png saved")

# Learning curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'],     label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(history.history['loss'],     label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)
plt.show()
print("✅ learning_curves.png saved")

# 9. LƯU MODEL
model.save("GRU_action_model.keras")
print("\n✅ MODEL SAVED → GRU_action_model.keras")
print("✅ LABEL MAP SAVED → label_map.json")
