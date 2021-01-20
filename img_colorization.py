from processing import *
from decodingModel import *

# Using Transfer learning
feature_extract_model = VggModel()

PrepareData(datapath = "/content/data" , savepath = "/content/processed/"
			target_size = (224 , 224) , batch_size = 32 , apply = feature_extract_model)

colorize = model()

training_dir = "/content/processed"
num_train_samples = 2000
batch_size = 100
steps_per_epoch = np.floor(num_train_samples/batch_size)
epochs = 501

for i in range(epochs):
  generator = data_generator_baseline(training_dir, num_train_samples, batch_size)
  fit_history = model.fit_generator(generator, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)
  if epochs % 10 == 0:
    model.save('model_merge_' + str(i) + '.h5')
    X = test_images(path = "/content/oldes" , shape = (224 , 224) , batch_size = 4 ,
                 feature_extract_model = feature_extract_model , model = model )

