from data import  ImageDetectionsField,TextField
import torch
import json
import cv2
import matplotlib.pyplot as plt
import random
import pickle

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def draw_picture(image_path,num):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox,labels = get_bbox_from_feature(image_id,feature_path)
    cmap = get_cmap(len(bbox))
    ax = plt.gca()
    plt.xticks([])
    plt.yticks([])
    for i in range(num):
        b = bbox[i]
        color = cmap(random.randint(0, len(bbox)))
        plt.gca().add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                          fill=False, edgecolor=color, linewidth=1,
                                          alpha=0.5))
        plt.gca().text(b[0], b[3],
                       '%s' % (labels[i]),
                       bbox=dict(facecolor=color, alpha=0.5),
                       fontsize=8, color='white')

    plt.imshow(img)
    plt.show()
def get_bbox_from_feature(image_id,feature_path):
    image_val_field = ImageDetectionsField(detections_path=feature_path, max_detections=50, load_in_tmp=False)
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)
    feature, bbox, labels = image_val_field.preprocess(str(image_id))
    text_field.vocab = pickle.load(open('vocab_MyImageCaption.pkl', 'rb'))
    labels_name = text_field.decode(labels, join_words=False)
    return bbox,labels_name
    # bbox = torch.tensor(bbox)
    # print("----------bbox_info-------------")
    # print(bbox)
def get_json_file(file_path):
    file = open(file_path,"r")
    fileJson = json.load(file)
    images = fileJson["images"]
    annotations = fileJson["annotations"]
    return images,annotations
def print_gen_caption(res_json,image_id):
    file = open(res_json, "r")
    fileJson = json.load(file)
    print("----------gen_caption-----------------")
    for image in fileJson:
        if image["image_id"]==image_id:
            print(image["caption"])
    print("--------------------------------------")
def print_image_info(file_path,image_id):
    file_name =""
    images,annotations = get_json_file(file_path)
    print("----------image_info----------------------")
    for image in images:
        if image["id"]==image_id:
            file_name = image["file_name"]
            print(image["file_name"])
            print(image["coco_url"])
    print("----------ground_true_caption-------------")
    for caption in annotations:
        if caption["image_id"]==image_id:
            print(caption["caption"])
    print("------------------------------------------")
    return file_name

if __name__ == '__main__':
    image_id = input("请输入COCO图片编号:")
    num = input("请输入需要目标数量[0-50]:")
    #num = input("输入需要目标数量:")
    json_file = '/home/User/wangyi/annontions/captions_val2014.json'
    feature_path = '/home/datasets/coco_detections.hdf5'
    pic_path = '/home/User/ljd/fairseq-image-captioning-master/ms-coco/images/val2014/'
    res_json = '/home/User/wangyi/meshed-memory-transformer-master/res_val.json'
    file_name = print_image_info(json_file,int(image_id))
    print_gen_caption(res_json,int(image_id))
    draw_picture(pic_path+file_name,int(num))

