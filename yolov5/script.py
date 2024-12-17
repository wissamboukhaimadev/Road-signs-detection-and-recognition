import detect 

def labview_function():
    opt = detect.parse_opt()
    detect.check_requirements(exclude=('tensorboard', 'thop'))
    detect.run(**vars(opt))
    

labview_function()

print("===============")
print(detect.detected_label)
print(detect.detected_label_no_confidence)

print("===============")
