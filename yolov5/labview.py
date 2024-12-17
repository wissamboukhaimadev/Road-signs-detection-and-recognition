import subprocess

def execute_command(image_source):
    # Define the command as a list of arguments
    command = [
        "python3_12.exe", 
        "C:/Users/hp/Desktop/projects/python_yolo/road_sign_recognition/Traffic-Sign-Recognition-Using-YOLO/yolov5_git/yolov5/script.py", 
        "--weights", "C:/Users/hp/Desktop/projects/python_yolo/road_sign_recognition/Traffic-Sign-Recognition-Using-YOLO/yolov5_git/yolov5/weights/best_93.pt", 
        "--source", image_source
    ]
    print("running")
    try:
        # Run the command
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,  
            stderr=subprocess.PIPE,  
            text=True                
        )
        

        # Check the result
        if result.returncode == 0:
            print("Command executed successfully!")
            print("Output of the prediction:")
            output_lines = result.stdout.splitlines()

            print(output_lines[2])
            return output_lines[3]
        else:
            print("Command failed!")
            print("Error:")
            print(result.stderr)
            return "result count"

    except FileNotFoundError as e:
        print("Error: Python 3.12 or the script was not found.")
        print(e)
        return "file not found"
    except Exception as e:
        print("An unexpected error occurred.")
        print(e)
        return "exception"

