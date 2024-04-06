# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from keras.utils import to_categorical, plot_model
#from keras.utils.vis_utils import model_to_dot
from keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from IPython.display import SVG
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


IMG_ROWS = 28
IMG_COLS = 28
NUM_CLASSES = 10
TEST_SIZE = 0.2
RANDOM_STATE = 2018

NO_EPOCHS = 20
BATCH_SIZE = 128

train_file = 'archive/fashion-mnist_train.csv'
test_file = 'archive/fashion-mnist_test.csv'

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot'
    }

def get_classes_distribution(data):
    label_counts = data['label'].value_counts()
    total_samples = len(data)
    
    for i in range(len(label_counts)):
        label = labels[label_counts.index[i]]
        count = label_counts.values[i]
        percent = (count / total_samples) * 100
        print("{:<20s}: {} or {}%".format(label, count, percent))
        
get_classes_distribution(train_data)

def plot_label_per_class(data):
    f, ax = plt.subplots(1,1, figsize=(12,4))
    g = sns.countplot(x = 'label', data=data, order = data["label"].value_counts().index)
    g.set_title("Number of labels for each class")

    for p, label in zip(g.patches, data["label"].value_counts().index):
        g.annotate(labels[label], (p.get_x(), p.get_height()+0.1))
    plt.show()  
    
plot_label_per_class(train_data)

def sample_images_data(data):
    sample_images = []
    sample_labels = []
    
    for k in labels.keys():
        
        samples = data[data['label'] == k].head(4)
        for j, s in enumerate(samples.values):
            img = np.array(samples.iloc[j, 1:]).reshape(IMG_ROWS, IMG_COLS)
            
            sample_images.append(img)
            sample_labels.append(samples.iloc[j, 0])
            
    print("Total number of images to plot: ", len(sample_images))
    return sample_images, sample_labels
    
train_sample_images, train_sample_labels = sample_images_data(train_data)

def plot_sample_images(data_sample_images, data_sample_labels, cmap="Blues"):
    f, ax = plt.subplots(5, 8, figsize=(16, 10))
    for i, img in enumerate(data_sample_images):
        ax[i//8, i%8].imshow(img, cmap = cmap)
        ax[i//8, i%8].axis('off')
        ax[i//8, i%8].set_title(labels[data_sample_labels[i]])
    plt.show()
    
plot_sample_images(train_sample_images, train_sample_labels, "Greens")

test_sample_images, test_sample_labels = sample_images_data(test_data)
plot_sample_images(test_sample_images, test_sample_labels)

def data_preprocessing(raw):
    out_y = to_categorical(raw.label, NUM_CLASSES)
    num_images = raw.shape[0]
    x_as_array = raw.values[:, 1:]
    x_shaped_array = x_as_array.reshape(num_images, IMG_ROWS, IMG_COLS, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y

X, y = data_preprocessing(train_data)
X_test, y_test = data_preprocessing(test_data)
    
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

print("Fashion MNIST train - rows: ", X_train.shape[0], " columns: ", X_train.shape[1:4])
print("Fashion MNIST valid - rows: ", X_val.shape[0], " columns: ", X_val.shape[1:4])
print("Fashion MNIST test - rows: ", X_test.shape[0], " columns: ", X_test.shape[1:4])

def plot_count_per_class(yd):
    ydf = pd.DataFrame(yd)
    f, ax  = plt.subplots(1, 1, figsize = (12, 4))
    g = sns.countplot(x = ydf[0], order = np.arange(0, 10))
    g.set_title("Number of items for each class")
    g.set_xlabel("Category")
    
    for p, label in zip(g.patches, np.arange(0, 10)):
        g.annotate(labels[label], (p.get_x(), p.get_height() + 0.2))
        
    plt.show()
    
def get_count_per_class(yd):
    ydf = pd.DataFrame(yd)
    label_counts = ydf[0].value_counts()
    total_samples = len(yd)
    
    for i in range(len(label_counts)):
        label = labels[label_counts.index[i]]
        count = label_counts.values[i]
        percent = (count / total_samples) * 100
        print("{:<20s}: {} or {}%" .format(label, count, percent))
    
plot_count_per_class(np.argmax(y_train, axis = 1))
get_count_per_class(np.argmax(y_train, axis = 1))

plot_count_per_class(np.argmax(y_val, axis = 1))
get_count_per_class(np.argmax(y_val, axis = 1))

model = Sequential()
model.add(Conv2D(32, 
                 kernel_size = (3, 3), 
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=(IMG_ROWS, IMG_COLS, 1)
                 ))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size = (3, 3), 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

plot_model(model, to_file='model.png')
#SVG(model_to_dot(model).create(prog = 'dot', format='svg'))

train_model = model.fit(X_train, y_train, 
                        batch_size=BATCH_SIZE,
                        epochs=NO_EPOCHS,
                        verbose=1,
                        validation_data=(X_val, y_val))

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

def create_trace(x, y, ylabel, color):
    trace = go.Scatter(
        x = x, y = y,
        name = ylabel,
        marker = dict(color = color),
        mode = "markers+lines",
        text = x)
    return trace

def plot_accuracy_and_loss(train_model):
    hist = train_model.history
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = list(range(1, len(acc) + 1))
    
    trace_ta = create_trace(epochs, acc, "Training accuracy", "Green")
    trace_va = create_trace(epochs, val_acc, "Validation accuarcy", "Red")
    trace_tl = create_trace(epochs, loss, "Training loss", "Blue")
    trace_vl = create_trace(epochs, val_loss, "Validation loss", "Magenta")
    
    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=("training and validation accuracy", "Training and validation loss"))
    
    fig.append_trace(trace_ta, 1, 1)
    fig.append_trace(trace_va, 1, 1)
    fig.append_trace(trace_tl, 1, 2)
    fig.append_trace(trace_vl, 1, 2)
    fig['layout']['xaxis'].update(title='Epoch')
    fig['layout']['xaxis2'].update(title='Epoch')
    fig['layout']['yaxis'].update(title = 'Accuracy', range=[0, 1])
    fig['layout']['yaxis2'].update(title = 'Loss', range=[0, 1])
    
    iplot(fig, filename='accuracy-loss')

plot_accuracy_and_loss(train_model)

predicted_classes = model.predict(X_test)
predicted_classes = np.argmax(predicted_classes, axis = 1)
y_true = test_data.iloc[:, 0]
p = predicted_classes[:10000]
y = y_true[:10000]
correct = np.nonzero(p == y)[0]
incorrect = np.nonzero(p != y)[0]

print("Number of correctly predicted classes: ", correct.shape[0])
print("Number of incorrectly predicted classes: ", incorrect.shape[0])

def plot_images(data_index, cmap="Blues"):
    f, ax = plt.subplots(4, 4, figsize=(15, 15))
    
    for i, indx in enumerate(data_index[:16]):
        ax[i//4, i%4].imshow(X_test[indx].reshape(IMG_ROWS, IMG_COLS), cmap = cmap)
        ax[i//4, i%4].axis('off')
        ax[i//4, i%4].set_title("True: {} Pred: {}".format(labels[y_true[indx]], labels[predicted_classes[indx]]))
        
    plt.show()
    
plot_images(correct, "Greens")
plot_images(incorrect, "Reds")














































