"""Integrating the model in our API Server

Link to documentation: 
https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
The server we wrote is quite trivial and may not do everything you
need for your production application. So, here are some things you
can do to make it better:

-   The endpoint `/predict` assumes that always there will be a
    image file in the request. This may not hold true for all
    requests. Our user may send image with a different parameter or
    send no images at all.
-   The user may send non-image type files too. Since we are not
    handling errors, this will break our server. Adding an explicit
    error handing path that will throw an exception would allow us
    to better handle the bad inputs
-   Even though the model can recognize a large number of classes of
    images, it may not be able to recognize all images. Enhance the
    implementation to handle cases when the model does not recognize
    anything in the image.
-   We run the Flask server in the development mode, which is not
    suitable for deploying in production. You can check out [this
    tutorial](https://flask.palletsprojects.com/en/1.1.x/tutorial/deploy/)
    for deploying a Flask server in production.
-   You can also add a UI by creating a page with a form which takes
    the image and displays the prediction. Check out the
    [demo](https://pytorch-imagenet.herokuapp.com/) of a similar
    project and its [source
    code](https://github.com/avinassh/pytorch-flask-api-heroku).
-   In this tutorial, we only showed how to build a service that
    could return predictions for a single image at a time. We could
    modify our service to be able to return predictions for multiple
    images at once. In addition, the
    [service-streamer](https://github.com/ShannonAI/service-streamer)
    library automatically queues requests to your service and
    samples them into mini-batches that can be fed into your model.
    You can check out [this
    tutorial](https://github.com/ShannonAI/service-streamer/wiki/Vision-Recognition-Service-with-Flask-and-service-streamer).
-   Finally, we encourage you to check out our other tutorials on
    deploying PyTorch models linked-to at the top of the page."""


import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request


app = Flask(__name__)
imagenet_class_index = json.load(open('<PATH/TO/.json/FILE>/imagenet_class_index.json'))
model = models.densenet121(weights='IMAGENET1K_V1')
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()


