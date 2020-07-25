from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy
#%matplotlib inline

#Red data from csv file for training and validation data
TrainingSet = numpy.genfromtxt("training.csv", delimiter=",", skip_header=True)
ValidationSet = numpy.genfromtxt("validation.csv", delimiter=",", skip_header=True)

# split into input (X) and output (Y) variables
X1 = TrainingSet[:,0:5]
Y1 = TrainingSet[:,5]

X2 = ValidationSet[:,0:5]
Y2 = ValidationSet[:,5]

# create model
for i in range(len(5))
    model = Sequential()
    model.add(Dense(20, activation="tanh", input_dim=5, kernel_initializer="uniform"))
    model.add(Dense(1, activation="linear", kernel_initializer="uniform"))

    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X1, Y1, epochs=300, batch_size=10,  verbose=2)

    # Calculate predictions
    PredTestSet = model.predict(X1)
    PredValSet = model.predict(X2)

    # Save predictions
    numpy.savetxt("trainresults.csv", PredTestSet, delimiter=",")
    numpy.savetxt("valresults.csv", PredValSet, delimiter=",")

    #Plot actual vs predition for training set
    TestResults = numpy.genfromtxt("trainresults.csv", delimiter=",")
    plt.plot(Y1,TestResults,'ro')
    plt.title('Training Set')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    #Compute R-Square value for training set
    TestR2Value = r2_score(Y1,TestResults)
    print("Training Set R-Square=", TestR2Value)

    with open("Output.txt", "w") as text_file:
        text_file.write(TestR2Value)