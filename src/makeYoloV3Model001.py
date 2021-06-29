from src.yolo3_one_file_to_detect_them_all import make_yolov3_model, WeightReader

# define the model
model = make_yolov3_model()

# load the model weights
weight_reader = WeightReader('yolov3_1.weights')

# set the model weights into the model
weight_reader.load_weights(model)

# save the model to file
model.save('yolov3-grey.h5')