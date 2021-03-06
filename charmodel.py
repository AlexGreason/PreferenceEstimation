import itertools

import numpy as np
import pylab
import scipy.misc as misc
import scipy.ndimage as ndimage
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Reshape, Flatten, UpSampling2D, \
    AveragePooling2D
from keras.models import Model, Sequential, Graph
from matplotlib.widgets import Slider, Button

from BT_GLM import BTModel

batch_size = 64
sidelen = 96
original_shape = (batch_size, 1, sidelen, sidelen)
latent_dim = 32
intermediate_dim = 512
intermediate_dim_2 = 256
nb_epoch = 0

x = Input(batch_shape=original_shape)
a = Convolution2D(128, 5, 5, border_mode='same', activation='relu')(x)
b = MaxPooling2D(pool_size=(2, 2))(a)
c = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(b)
b2 = MaxPooling2D(pool_size=(2, 2))(c)
c2 = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(b2)
d = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(c2)
d_reshaped = Flatten()(d)
h = Dense(intermediate_dim, activation='relu')(d_reshaped)
i = Dense(intermediate_dim_2, activation='relu')(d_reshaped)
z_mean = Dense(latent_dim)(i)

# def sampling(args):
#   z_mean, z_log_var = args
#   epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
#   return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_h_2 = Dense(intermediate_dim_2, activation='relu')
i = Dense(8 * 24 * 24, activation='relu')
j = Reshape((8, 24, 24))
k = Convolution2D(128, 3, 3, border_mode='same', activation='relu')
l = UpSampling2D((2, 2))
m = Convolution2D(128, 3, 3, border_mode='same', activation='relu')
l2 = UpSampling2D((2, 2))
m2 = Convolution2D(128, 3, 3, border_mode='same', activation='relu')
n = Convolution2D(128, 3, 3, border_mode='same', activation='relu')
decoder_mean = Convolution2D(1, 3, 3, border_mode='same', activation='sigmoid')

h_decoded = decoder_h(z_mean)
h2_decoded = decoder_h_2(h_decoded)
i_decoded = i(h2_decoded)
j_decoded = j(i_decoded)
k_decoded = k(j_decoded)
l_decoded = l(k_decoded)
m_decoded = m(l_decoded)
l2_decoded = l(m_decoded)
m2_decoded = m(l2_decoded)
n_decoded = n(m2_decoded)
x_decoded_mean = decoder_mean(n_decoded)

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss="binary_crossentropy")
print(decoder_mean.output_shape)

vae.load_weights("autoencoderparams-2.sav")
# build a model to project inputs on the latent space
encoder = Model(x, z_mean)
print(encoder.output_shape)

decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_h2_decoded = decoder_h_2(_h_decoded)
_i_decoded = i(_h2_decoded)
_j_decoded = j(_i_decoded)
_k_decoded = k(_j_decoded)
_l_decoded = l(_k_decoded)
_m_decoded = m(_l_decoded)
_l2_decoded = l(_m_decoded)
_m2_decoded = m(_l2_decoded)
_n_decoded = n(_m2_decoded)
_x_decoded_mean = decoder_mean(_n_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# The graph API is outdated, use the model functional API instead

evaluatorbase = Sequential()
evaluatorbase.add(Convolution2D(64, 3, 3, activation="relu", border_mode="same", input_shape=(1, sidelen, sidelen)))
evaluatorbase.add(Convolution2D(64, 3, 3, activation="relu", border_mode="same"))
evaluatorbase.add(AveragePooling2D())
evaluatorbase.add(Convolution2D(64, 3, 3, activation="relu", border_mode="same"))
evaluatorbase.add(Convolution2D(256, 3, 3, activation="relu", border_mode="same"))
evaluatorbase.add(AveragePooling2D())
evaluatorbase.add(Convolution2D(128, 3, 3, activation="relu", border_mode="same"))
evaluatorbase.add(Convolution2D(128, 3, 3, activation="relu", border_mode="same"))
evaluatorbase.add(AveragePooling2D())
evaluatorbase.add(Convolution2D(128, 3, 3, activation="relu", border_mode="same"))
evaluatorbase.add(Convolution2D(128, 3, 3, activation="relu", border_mode="same"))
evaluatorbase.add(Convolution2D(64, 3, 3, activation="relu", border_mode="same"))
evaluatorbase.add(Flatten())
evaluatorbase.add(Dense(256, activation="relu"))

evaluator = Graph()
evaluator.add_input(name="image1", input_shape=(1, sidelen, sidelen))
evaluator.add_input(name="image2", input_shape=(1, sidelen, sidelen))
evaluator.add_shared_node(evaluatorbase, name="shared", inputs=["image1", "image2"], merge_mode="concat")
evaluator.add_node(Dense(256, activation="relu"), name="dense1", input="shared")
evaluator.add_node(Dense(128, activation="relu"), name="dense2", input="dense1")
evaluator.add_node(Dense(256, activation="softmax"), name="dense3", input="dense2")
evaluator.add_output(name="output", input="dense3")

evaluator.compile(optimizer="adam", loss={"output": "categorical_crossentropy"})
evaluator.summary()

# use weights so that earlier datapoints have exponentially decreasing weight until some threshold where they're eliminated?
# Since I'm training based non on the ground truth, but rather based on an update to the estimate it produces, even
# experience replay isn't an exact analogy, since at least there you get the actual immediate reward and can compute
# the expected future reward based on the net at the time of training

# try with only training on each datapoint once to start?
# Oh, wait, I was going to make it take pairs of images. Merge layer?

# try a thing that takes in the latent vectors rather than the images themselves
# it loses the portability and generator-agnostic nature of the one that takes the images, but it should just be tremendouly faster
# and technically I could adapt it to take the latent vectors from other generators (very inefficiently)
# by decoding them with the new generator and then encoding them with the old encoder
np.random.seed(0)


def genvec():
    return np.random.uniform(-20, 20, latent_dim)


def process(frame, x, y):
    base = (frame * 255).reshape((sidelen, sidelen))
    base = ndimage.gaussian_filter(base, sigma=1)
    resized = misc.imresize(base, (x, y), interp='bicubic')
    resized[resized < 128] = 0
    resized[resized >= 128] = 255
    # resized = resized[:, :, None].repeat(3, -1).astype("uint8")
    return resized


f = pylab.figure(num=1, figsize=(10, 10))
f.add_subplot(1, 2, 1)
# pylab.subplots_adjust(left=0.3, bottom=0.4)
image1 = pylab.imshow(np.array([[1] + [0] * 599] * 600, dtype='int64'), cmap="Greys", animated=True)
f.add_subplot(1, 2, 2)
image2 = pylab.imshow(np.array([[1] + [0] * 599] * 600, dtype='int64'), cmap="Greys", animated=True)
# pylab.axis([0, 1, -10, 10])
lower = -20
upper = 20
axcolor = 'lightgoldenrodyellow'
sliders = []
axis = pylab.axes([0.15, 0.25, 0.75, 0.01], axisbg=axcolor)
slider = (Slider(axis, "0", -1, 1, valinit=0))

# Maybe have two comparisons, each starting at .5 left -.5 right, and shift the second one as the slider goes from -1 to 0
# so that it goes to .5/-.5 -.5/.5 at 0, to simulate a tie, and then shift the first one as the slider goes from 0 to 1
# so that it ends at -.5/.5 on both?
numimages = 30
model = BTModel([[0] * (numimages - 1)], numvals=numimages - 1)
model.addDummyData()
userchoices = []


def getComparison(index1, index2, compresult):
    global userchoices
    comp1 = [0] * len(vectorset)
    comp2 = [0] * len(vectorset)
    if compresult <= 0:
        comp1[index1] = .5
        comp1[index2] = -.5
        comp2[index1] = -.5 - compresult
        comp2[index2] = .5 + compresult
    else:
        comp2[index1] = -.5
        comp2[index2] = .5
        comp1[index1] = .5 - compresult
        comp1[index2] = -.5 + compresult
    print(comp1, comp2)
    model.addData([comp1[1:], comp2[1:]])
    userchoices += [comp1[1:], comp2[1:]]


def generateData(numimages):
    return [genvec() for i in range(numimages)]


# reevaluate the results of the dummy data and evaluator data and add on userchoices after every epoch, so it can just
# keep training? Making sure that the userchoices is fairly extensive (at least one (two?) comparisons for every image
# might have to change the question generation, it should be much more likely to ask about things that haven't
# been asked about yet. As it is it takes ages to cover everything, especially with large sets

evaluator.load_weights("evaluatorparams-1.sav")


def trainEval(evaluator, worthmodel, vectorset):
    batchx = []
    batchy = []
    worths = [0] + list(worthmodel.worths)
    error = 10
    iteration = 0
    while iteration < 200:
        iteration += 1
        for a, b in itertools.permutations(list(range(len(vectorset))), 2):
            if len(batchy) < 32:
                image1 = generator.predict(vectorset[a].reshape((1, latent_dim))).reshape(1, 96, 96)
                image2 = generator.predict(vectorset[b].reshape((1, latent_dim))).reshape(1, 96, 96)
                diff = min(14.89, max(-15, worths[a] - worths[b]))
                diffvector = [0] * 256
                diffvector[int((diff + 15) / (30 / 256))] = 1
                batchx.append([image1, image2])
                batchy.append(diffvector)
            else:
                batchx, batchy = np.array(batchx), np.array(batchy)
                error = evaluator.train_on_batch({"image1": batchx[:, 0], "image2": batchx[:, 1], "output": batchy})
                print(error)
                batchx = []
                batchy = []
    evaluator.save_weights("evaluatorparams-1.sav", overwrite=True)


vectorset = generateData(numimages)
resetax = pylab.axes([0, 0.025, 0.1, 0.04])
resetb = Button(resetax, 'Next', color=axcolor, hovercolor='0.975')
i = 0
enoughQuestions = False


# maybe calculate the probability it assigns to A winning, and give A... no, because I need a measurement of /uncertainty/
# calculate the information content (sum minus log) of the distribution and... divide by a constant... and multiply that
# by the probabilities of a and b winning... and add both those datapoints (for both a-wins and b-wins scenarios)

# alright, just do the simplest thing to start
# I'll be totally outweighed if I do that, though.

# by the central tendency theorem standard deviation is still a valid measure for nonnormal data, just a less powerful one
# I need to write a visualizer. I can just make a bar graph of how much weight it's putting on each value.
def newData():
    global evaluator
    global model
    global vectorset
    global enoughQuestions
    enoughquestions = False
    trainEval(evaluator, model, vectorset)
    print("trained")
    # vectorset = generateData(numimages)
    print("generated")
    model = BTModel([[0] * (numimages - 1)], numvals=numimages - 1)
    model.addDummyData()
    for a, b in itertools.permutations(list(range(len(vectorset))), 2):
        image1 = generator.predict(vectorset[a].reshape((1, latent_dim))).reshape(1, 1, 96, 96)
        image2 = generator.predict(vectorset[b].reshape((1, latent_dim))).reshape(1, 1, 96, 96)
        distribution = evaluator.predict({"image1": image1, "image2": image2})["output"]
        meandiff = np.mean(distribution.reshape((256)) * np.arange(256)) * (30 / 256) - 15
        prob = np.exp(meandiff) / (np.exp(meandiff) + 1)
        data = [0] * len(vectorset)
        data[a] = prob
        data[b] = -prob
        model.addData([data[1:]])
    model.fit()
    print("Results after running evaluator:")
    print(model.worths, model.errors)


enoughax = pylab.axes([0, 0.125, 0.125, 0.04])
enoughb = Button(enoughax, 'End epoch', color=axcolor, hovercolor='0.975')

saveax = pylab.axes([.2, 0.125, 0.125, 0.04])
saveb = Button(saveax, 'Save', color=axcolor, hovercolor='0.975')

loadax = pylab.axes([.2, 0.025, 0.125, 0.04])
loadb = Button(loadax, 'Load', color=axcolor, hovercolor='0.975')


def enough(event):
    newData()
    index1, index2 = model.genQuestion()
    data = process(generator.predict((vectorset[index1]).reshape((1, latent_dim))).reshape(96, 96), 600, 600)
    image1.set_data(data)
    data = process(generator.predict((vectorset[index2]).reshape((1, latent_dim))).reshape(96, 96), 600, 600)
    image2.set_data(data)
    slider.set_val(0)


enoughb.on_clicked(enough)


def reset(event):
    global i
    global model
    global vectorset
    index1, index2 = model.genQuestion()
    data = process(generator.predict((vectorset[index1]).reshape((1, latent_dim))).reshape(96, 96), 600, 600)
    image1.set_data(data)
    data = process(generator.predict((vectorset[index2]).reshape((1, latent_dim))).reshape(96, 96), 600, 600)
    image2.set_data(data)
    compresult = slider.val
    slider.set_val(0)
    if i > 0:
        getComparison(index1, index2, compresult)
        model.fit()
        print(model.worths, model.errors)
    i = 1


resetb.on_clicked(reset)


def save(event):
    global i
    global model
    global vectorset
    global userchoices
    filename = input("Enter filename: ")
    file = open(filename, "w+")
    for vector in vectorset:
        file.write(str(vector) + "\n")
    file.write("\n")
    for comp in userchoices:
        file.write(str(comp) + "\n")
    file.write("\n")
    file.write(str(model.worths))
    file.write("\n")
    file.write(str(model.errors))
    file.close()


saveb.on_clicked(save)


def load(event):
    global i
    global model
    global vectorset
    global userchoices
    filename = input("Enter filename: ")
    file = open(filename, "r")
    vectors = ""
    line = file.readline()
    while (line != "\n"):
        vectors += line
        line = file.readline()
    vectorset = parsearray(vectors)
    line = file.readline()
    choices = ""
    while (line != "\n"):
        choices += line
        line = file.readline()
    userchoices = parsearray(choices)
    model.data = []
    model.addDummyData()
    for i in range(len(userchoices)):
        model.addData([list(userchoices[i])])
    model.questionOrder = np.random.permutation(np.arange(model.numvals + 1))


def parsearray(string):
    val = string.replace("[", "")
    val = val.replace("\n", "")
    val = val.replace(",", "")
    val = val.split("]")
    vallist = [x.split(" ") for x in val]
    floatlist = [[float(x) for x in sublist if x != ""] for sublist in vallist if sublist != [""]]
    result = [np.array(sublist) for sublist in floatlist]
    return result


loadb.on_clicked(load)

pylab.show()

# Y'know, the evaluator isn't really working, but I can try running a regular ol' evolutionary algorithm directly on the output of the bradley-terry model
