import numpy as np
import cv2
import tqdm
from torchvision import transforms
import torch
from torchsummary import summary
from model import * 
from PIL import Image

def test(model, device, frame, preprocess):

    model.eval()
    
    with torch.no_grad():

        frame = preprocess(frame).to(device)
        frame = frame.unsqueeze(0)
        output = model(frame)

        pred = torch.argmax(output)
        pred = pred.cpu().numpy()
        
    return pred
           
def load_model(device):
    
    model = TLClassification()
    # print(model)
    model = model.to(device)

    # summary(model, (3, 40, 40) )

    # model = nn.DataParallel(model)
    model_path = "./checkpoints/final_model.h5"
    model.load_state_dict(torch.load(model_path))

    return model

def main():

    preprocess = transforms.Compose([
        transforms.Resize(40),
        transforms.CenterCrop(40),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(device)

    cap = cv2.VideoCapture("./tl_violation.MP4")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    tl_id_0_0 = (2908, 759, 36, 67)
    tl_id_0_1 = (3141, 742, 37, 72)
    tl_id_1_0 = (1167, 783, 35, 73)
    tl_id_1_1 = (1314, 790, 32, 68)

    bboxes = [tl_id_0_0, tl_id_0_1, tl_id_1_0, tl_id_1_1]

    ret = True
    frame_id = 0
    count = 0
    while(ret):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_debug = frame.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        for bbox in bboxes:

            c_x = bbox[0] + ( bbox[2]/2 )
            c_y = bbox[1] + ( bbox[3]/2 )
            c_x = int(c_x)
            c_y = int(c_y)
            side = int(min(bbox[2], bbox[3]))

            data_frame = frame.crop((c_x-side, c_y-side, c_x+side, c_y+side)) 
            # file_name = "./new_dataset/img_" + str(count)
            # data_frame.save(file_name, "JPEG")
            count += 1
            pred = test(model, device, data_frame, preprocess)
            
            if pred == 2:
                color_class = (0, 255, 0)
            elif pred == 1:
                color_class = (0, 255, 255)
            elif pred == 0:
                color_class = (0, 0, 255)
            
            cv2.rectangle(frame_debug, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color_class, 5)
            
            # print(pred)

        cv2.imshow('frame', frame_debug)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

        # break 

    cap.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()