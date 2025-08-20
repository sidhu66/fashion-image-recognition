import os, json
os.environ["KERAS_BACKEND"] = "tensorflow"
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from src.models.build_model import buid_mlp


LABELS = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

def plot_curves(history, out_direc = "reports"):
    if not os.path.exists(out_direc): os.mkdir(out_direc)
    # Accuracy
    plt.figure()
    plt.plot(history.history['sparse_categorical_accuracy'], label='train')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='val')
    plt.title('Accuracy'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_direc, "accuracy.png")); plt.close()
    
    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_direc, "loss.png")); plt.close()
    

def main(epoch=30):
    print()
    # Reproducibility
    tf.random.set_seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_valid, X_train = X_train[:5000]/255., X_train[5000:]/255.
    y_valid, y_train = y_train[:5000], y_train[5000:]
    X_test = X_test/255.
    
    model = buid_mlp()
    if not os.path.exists('models'): os.mkdir('models')
    ckpt = os.path.join('models', 'mlp_model.keras')
    
    history = model.fit(X_train, y_train, epochs=epoch, validation_data=(X_valid, y_valid))
    
    model.save(ckpt)
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")
    
    # Plotting Testing Evalutions
    plot_curves(history)

    # Save Labels for the Interface App
    with open(os.path.join('models', 'labels.json'), 'w') as f:
        json.dump({i: lbl for i, lbl in enumerate(LABELS)}, f, indent=2)    
        
    
    
if __name__ == "__main__":
    main()
        
