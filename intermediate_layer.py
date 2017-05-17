import argparse
from keras.models import Model
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )

    args = parser.parse_args()

    model = load_model(args.model)
    print(len(model.layers))
    
    f = './data_definitivo/IMG/center_2017_04_25_12_35_59_054.jpg'
    data = cv2.imread(f)
    data = np.expand_dims(data, axis=0)
    plot_model(model, to_file='model.png', show_shapes=True)
    for l in range(len(model.layers)):
        intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[l].output)
        intermediate_output = intermediate_layer_model.predict(data)
        intermediate_output = np.moveaxis(intermediate_output, -1, 1)
        print('layer {}: {}'.format(l, intermediate_output.shape))
        
        fig, axes = plt.subplots(10, 10, figsize=(20, 20))
        fig.subplots_adjust(hspace=0.02, wspace=0.02)
        last_label = -1
        r = 0
        c = 0
        for layer in intermediate_output[0]:
            try:
                if(c > 9): 
                   c = 0
                   r += 1
                axes[r,c].imshow(layer)
                c += 1
            except KeyError:
                pass
            
        plt.savefig(filename='layer{}.jpg'.format(l))
        

