import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

CLASS_NAMES = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

def main(args):
    model = keras.models.load_model(args.model_path)
    (_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_test = (x_test / 255.0).astype('float32')[..., None]

    preds = np.argmax(model.predict(x_test[:25], verbose=0), axis=1)
    plt.figure(figsize=(8,8))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.imshow(x_test[i].squeeze(), cmap='gray')
        plt.title(f"{CLASS_NAMES[preds[i]]}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('assets/quick_infer_grid.png', dpi=200)

    print("Saved assets/quick_infer_grid.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='models/fashion_mnist_cnn.keras')
    args = parser.parse_args()
    main(args)