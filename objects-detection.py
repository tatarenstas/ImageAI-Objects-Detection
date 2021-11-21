from imageai.Detection import ObjectDetection

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath("yolo-tiny.h5")
detector.loadModel()

array_detection = detector.detectObjectsFromImage(input_image = "test.jpg", output_image_path = "detected_test.jpg")
print (array_detection)
