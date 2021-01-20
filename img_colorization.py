from processing import *
from decodingModel import *

# Using Transfer learning
feature_extract_model = VggModel()

#Decoding model
colorize = model()

#prepare data in hard disk
PrepareData(datapath = "/content/data" , save_file = "/content/processed/" , 
			target_size = (224 , 224) , batch_size = 32 , feature_extract_model = feature_extract_model)


training_dir = "/content/processed"
num_train_samples = 1000
batch_size = 32
steps_per_epoch = np.floor(num_train_samples/batch_size)
epochs = 200

for i in range(epochs):
  generator = data_generator_baseline(training_dir, num_train_samples, batch_size)
  fit_history = colorize.fit_generator(generator, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)
  if i % 10 == 0:
    colorize.save('model_merge_' + str(i) + '.h5')
    X = test_images(path = "/content/oldes" , shape = (224 , 224) , batch_size = 2 ,
                 feature_extract_model = feature_extract_model , model = colorize )
    show_images(X , width = 20 , hight = 20 , columns = 2 , rows = 1)


