from info import args
from data import ImageDetectionsField, TextField, RawField
import json
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory,VisualPlusSemeticEncoder
import torch
import pickle
def resloveJson(path):
    file = open(path,"rb")
    fileJson = json.load(file)
    images = fileJson["images"]
    return images


def get_image_id(images):
    image_ids=[]
    for image in images:
        image_ids.append(image["id"])

    return image_ids

def generate_model(model,images,bbox,labels,text_field):
    images = torch.tensor(images).to(device)
    out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 3, out_size=1)
    caps_gen = text_field.decode(out, join_words=False)
    return " ".join(i for i in caps_gen[0])

def write_json(data,filename):

    data = json.dumps(data)
    file = open(filename, "w")
    file.write(data)
    file.close()


if __name__ == '__main__':
    device = torch.device('cuda:0')

    #加载测试集与预测集
    test_json = args.annotation_folder+"/captions_test2014.json"
    val_json = args.annotation_folder+"/captions_val2014.json"
    image_val_ids = get_image_id(resloveJson(val_json))
    image_test_ids = get_image_id(resloveJson(test_json))

    #读取特征集
    image_val_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)
    #feature,bbox,classes = image_val_field.preprocess()
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)
    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))
    #加载模型
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': 40})
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    data = torch.load('/home/User/wangyi/meshed-memory-transformer-master/saved_models/meshed_memory_transformer.pth')
    model.load_state_dict(data['state_dict'])
    res = []
    #生成描述，并保存
    for image_id in image_val_ids:
        feature,bbox,labels = image_val_field.preprocess(str(image_id))
        caption = generate_model(model,feature,bbox,labels,text_field)
        print(caption)
        res.append({"image_id":int(image_id),"caption":caption})

    write_json(res,"res_val.json")