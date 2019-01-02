#%% download mnist datasets
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# prepare validation data
from sklearn.model_selection import train_test_split
X_train_, X_valid, y_train_, y_valid = train_test_split(X_train, y_train)

print(X_train.shape)
print(y_train.shape)

print(X_valid.shape)
print(y_valid.shape)

print(X_test.shape)
print(y_test.shape)


#%% create model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
layers = [
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),

    Dense(10, activation='softmax'),
]
for l in layers:
    model.add(l)

from keras.optimizers import RMSprop

model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['accuracy'])


#%% pre-processing
print(X_train.shape)
print(y_train.shape)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_valid = X_valid.reshape(X_valid.shape[0], 28, 28, 1)
X_test  = X_test.reshape( X_test.shape[0],  28, 28, 1)

# Normalization
X_train = X_train.astype('float32') / 255
X_valid = X_valid.astype('float32') / 255
X_test  = X_test.astype('float32') / 255

# convert one-hot vector
import keras
y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test, 10)
y_valid = keras.utils.to_categorical(y_valid, 10)

#%% Training
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=10,
    verbose=1,
    validation_data=(X_valid, y_valid))

model.save('./mnist_model.h5')

#%% evaluate
print(model.summary())

score = model.evaluate(X_test, y_test, verbose=0)

print("Loss: {}".format(score[0]))
print("Accuracy: {}".format(score[1]))



#%% Plot
def plot_result(history):
    '''
    plot result
    全ての学習が終了した後に、historyを参照して、accuracyとlossをそれぞれplotする
    '''

    # accuracy
    plt.figure()
    plt.plot(history.history['acc'], label='acc', marker='.')
    plt.plot(history.history['val_acc'], label='val_acc', marker='x')
    plt.grid()
    plt.legend(loc='best')
    plt.title('accuracy')
    plt.savefig('graph_accuracy.png')
    plt.show()

    # loss
    plt.figure()
    plt.plot(history.history['loss'], label='loss', marker='.')
    plt.plot(history.history['val_loss'], label='val_loss', marker='x')
    plt.grid()
    plt.legend(loc='best')
    plt.title('loss')
    plt.savefig('graph_loss.png')
    plt.show()

#%% Plot
plot_result(history)