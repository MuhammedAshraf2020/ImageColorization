from keras.layers import Dense , Flatten , Input
from keras.layers import MaxPooling2D , Conv2D , UpSampling2D 
from keras.models import Model , Sequential
from keras.applications.vgg16 import VGG16


def VggModel():
	vgg = VGG16()
	feature_extract_model = Sequential()
	for i in vgg.layers[0:19]:
    	i.trainable = False
    	feature_extract_model.add(i)
	print(feature_extract_model.summary())
	return feature_extract_model



def model():
	#Build model take Vgg output and return model output
	Inp = Input(shape(7 , 7 , 512))
	# Decoder Model
	x = Conv2D(128, (3,3), activation='relu', padding='same' )(Inp)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(2, (3, 3), activation='tanh', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	model = Model(inputs = Inp , outputs = x)
	model.compile( optimizer='adam', loss='mse' , metrics=['accuracy'])
	return model
