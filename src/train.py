import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras

CLASS_NAMES = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_training(history, out_path):
    plt.figure()
    epochs = range(1, len(history.history['loss'])+1)
    plt.plot(epochs, history.history['loss'], label='train_loss')
    plt.plot(epochs, history.history.get('val_loss', []), label='val_loss')
    plt.plot(epochs, history.history['accuracy'], label='train_acc')
    if 'val_accuracy' in history.history:
        plt.plot(epochs, history.history['val_accuracy'], label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_sample_predictions(model, x, y, out_path, n=25):
    preds = np.argmax(model.predict(x[:n], verbose=0), axis=1)
    plt.figure(figsize=(8,8))
    for i in range(n):
        plt.subplot(5,5,i+1)
        plt.imshow(x[i].squeeze(), cmap='gray')
        title = f"{CLASS_NAMES[preds[i]]}\n(true: {CLASS_NAMES[y[i]]})"
        plt.title(title, fontsize=7)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main(args):
    assets = pathlib.Path('assets'); assets.mkdir(exist_ok=True, parents=True)
    models_dir = pathlib.Path('models'); models_dir.mkdir(exist_ok=True, parents=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = (x_train / 255.0).astype('float32')[..., None]
    x_test = (x_test / 255.0).astype('float32')[..., None]

    model = build_model()
    callbacks = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save artifacts
    plot_training(history, 'assets/training_curves.png')
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(include_values=False, xticks_rotation=45, cmap='Blues')
    plt.tight_layout()
    plt.savefig('assets/confusion_matrix.png', dpi=200)
    plt.close()

    save_sample_predictions(model, x_test, y_test, 'assets/sample_predictions.png')

    model.save(args.model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--model-path', type=str, default='models/fashion_mnist_cnn.keras')
    args = parser.parse_args()
    main(args)