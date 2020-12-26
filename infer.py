import numpy as np
import cv2
import tqdm
from torchvision import transforms
import torch
from torchsummary import summary
from model import * 
from PIL import Image
import pandas as pd

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
    
    tl_state_map = {0:"red", 1:"yellow", 2:"green"}

    preprocess = transforms.Compose([
        transforms.Resize(40),
        transforms.CenterCrop(40),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(device)

    cap = cv2.VideoCapture("./inputs/tl_violation.MP4")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    width  = cap.get(3) # float
    height = cap.get(4) # float
    out = cv2.VideoWriter('outputs/tl_violation_processed.mp4', 
        cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (int(width),int(height)))


    # # 001 001 002
    # tl_id_l = (773, 391, 17, 40)
    # tl_id_r = (1767, 407, 15, 41)
    # offset = 0 # 0 or 2

    # # 001 003 002
    # tl_id_l = (774, 392, 15, 38)
    # tl_id_r = (1423, 396, 17, 44)
    # offset = 2 # 0 or 2

    # 002 001 001
    tl_id_l = (568, 383, 10, 26)
    tl_id_r = (1625, 391, 18, 47)
    offset = 0

    # # 002 003 001
    # tl_id_l = (569, 389, 9, 21)
    # tl_id_r = (1010, 385, 10, 26)
    # offset = 2 # 0 or 2

    bboxes = [tl_id_l, tl_id_r]
    
    tl_id = []
    frame_ids = []
    states = []

    count = 0
    for frame_id in range(length):
        
        ret, frame = cap.read()

        if not ret:
            break

        frame_debug = frame.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        for tid, bbox in enumerate(bboxes):

            c_x = bbox[0] + ( bbox[2]/2 )
            c_y = bbox[1] + ( bbox[3]/2 )
            c_x = int(c_x)
            c_y = int(c_y)
            side = int(min(bbox[2], bbox[3]))

            data_frame = frame.crop((c_x-side, c_y-side, c_x+side, c_y+side)) 
            # if tid==1:
            #     file_name = "./new_dataset_1/img_" + str(count)
            #     data_frame.save(file_name, "JPEG")
            count += 1
            pred = test(model, device, data_frame, preprocess)
            
            if pred == 2:
                color_class = (0, 255, 0)
            elif pred == 1:
                color_class = (0, 255, 255)
            elif pred == 0:
                color_class = (0, 0, 255)
            
            cv2.rectangle(frame_debug, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color_class, 5)
            
            tl_id.append(tid + offset)
            frame_ids.append(frame_id+1)
            states.append(tl_state_map[int(pred)])
        
    
        cv2.imshow('frame', frame_debug)
        out.write(frame_debug)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    d = {'frame_id': frame_ids, 'tl_id': tl_id, 'tl_state': states}
    df = pd.DataFrame(data=d)
    df.to_csv("outputs/tl_violation.csv")


if __name__ == '__main__':
    main()
