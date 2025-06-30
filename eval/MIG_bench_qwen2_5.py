import torch
import warnings
import torch
import json
import datetime
import random
import copy
from tqdm import tqdm
import torch.multiprocessing as mp
from queue import Empty
from PIL import Image
import string
from decord import VideoReader, cpu 
import warnings
import os
from utils import *
### required by Migician ###
# from transformers import Qwen2VLForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, LlamaTokenizer

from transformers.image_utils import load_image

### required by LLaVA-OneVision ###
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
    from llava.conversation import conv_templates, SeparatorStyle
except:
    print('[WARNING] LLaVA hasn\'t been installed')

from modelscope import AutoTokenizer, AutoConfig, AutoModel

warnings.filterwarnings("ignore")

################# This implementation supports the evaluation of a series of mainstream multi-image models and diverse tasks #################
################# Note: Due to the conflicts of required transformers versions by different MLLMs, you may need to set the   #################
################# correct version for your target.                                                                           #################

PROMPT_TEMPLATE = {
    'format_univg': ' First output the thinking process in <think> </think> tags and then output the bounding box in <answer> </answer> tags.',
    'format_qwen': 'Format:<|box_start|>(x1,y1),(x2,y2)<|box_end|>. Don\'t generate addtional words.', # For more efficient evaluation
}

def compute_iou(ground_truth, prediction, acc, task):
    iou = calculate_iou(ground_truth, prediction)
    if not acc.get(task):
        acc[task]={'IOU@0.7':0,'IOU@0.5':0,'IOU@0.3':0,'AVE_IOU':0,'TOTAL':0}
    if iou >= 0.7: acc[task]['IOU@0.7'] += 1
    if iou >= 0.5: acc[task]['IOU@0.5'] += 1
    if iou >= 0.3: acc[task]['IOU@0.3'] += 1
    acc[task]['AVE_IOU'] += iou
    acc[task]['TOTAL'] += 1
    return iou, acc

def replace_image(match):
    replace_image.counter += 1
    return f"Image-{replace_image.counter}"

def post_processing(acc, output_path, output):
    for task, obj in acc.items():
        print(f"✅ Results for [{task}]:")
        print(obj)
        for item, value in obj.items():
            if item != 'TOTAL':
                print(f"|——> {item}:{100*value/acc[task]['TOTAL']}%  ✨")
    output.append(acc)
    with open(output_path,'w') as file2:
        json.dump(output, file2, indent=4, ensure_ascii=False)

################# Input process functions for specific MLLMs #################
def qwen2_process(obj):
    messages = [{"role": "user","content": []}]
    for path in obj['images']:
        messages[0]["content"].append({"type": "image","image": path})
    
    obj['question'] += PROMPT_TEMPLATE['format_qwen']
    messages[0]["content"].append({"type": "text", "text": obj['question']})
    return messages

def qwen2_5_process(obj):
    message = {"role": "user", "content": []}
    for img_path in obj["images"]:
        message["content"].append({"type": "image", "image": img_path})
    
    prompt = (
        obj["question"]
        + "\nYou must respond only with the bounding box in the format:"
        + " <|box_start|>(x1,y1),(x2,y2)<|box_end|>."
        + " Do not include any other words or formatting."
    )
    # print(f"Prompt: {prompt}")
    message["content"].append({"type": "text", "text": prompt})
    return [message]

def univgr1_process(obj):
    messages = [{"role": "user","content": []}]
    for path in obj['images']:
        messages[0]["content"].append({"type": "image","image": path})
    
    obj['question'] += PROMPT_TEMPLATE['format_univg']
    messages[0]["content"].append({"type": "text", "text": obj['question']})
    return messages

def qwen2_respond(model, processor, messages, device):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,return_tensors="pt")
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return response

def qwen2_5_respond(model, processor, messages, device):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return response

################# Model responding functions for specific MLLMs #################
def qwen2_vl_eval(model, test_data, processor, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"qwen2_vl_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        messages = qwen2_process(obj)
        response = qwen2_respond(model, processor, messages, device)
        prediction = extract_bbox(response[0]) # 正则匹配生成的bbo
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task':obj['task'], 'question':obj['question'], 'answer':response[0], 'filter_answer':prediction, 'iou':iou, 'groundtruth':obj['answer']})
            
    post_processing(acc, output_path, output)

def qwen2_5_vl_eval(model, test_data, processor, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"qwen2_5_vl_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        messages = qwen2_5_process(obj)  
        response = qwen2_5_respond(model, processor, messages, device)
        # print(f"Response: {response[0]}")
        prediction = extract_bbox(response[0])  # if it's a spatial task
        
        # print(f"[DEBUG] Raw response: {response[0]} → Prediction: {prediction}")
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task': obj['task'], 'question': obj['question'], 'answer': response[0],'filter_answer': prediction,'iou': iou,'groundtruth': obj['answer']})

    post_processing(acc, output_path, output)

def univgr1_eval(model, test_data, processor, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"univgr1_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        messages = univgr1_process(obj)
        response = qwen2_respond(model, processor, messages, device)
        prediction = extract_bbox(response[0]) # 正则匹配生成的bbox
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        
        # print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task':obj['task'], 'question':obj['question'], 'answer':response[0], 'filter_answer':prediction, 'iou':iou, 'groundtruth':obj['answer']})
            
    post_processing(acc, output_path, output)



def model_selection(model_type, model_path, test_data, device, output_path):
    if model_type=='univg-r1':
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
        processor = AutoProcessor.from_pretrained(model_path, min_pixels=3136, max_pixels=401408)
        univgr1_eval(model, test_data, processor, device, output_path)
    
    elif model_type=='qwen2_vl':
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
        processor = AutoProcessor.from_pretrained(model_path, min_pixels=3136, max_pixels=401408)
        qwen2_vl_eval(model, test_data, processor, device, output_path)
    
    elif model_type == 'qwen2_5_vl':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
        qwen2_5_vl_eval(model, test_data, processor, device, output_path)
        
    else:
        raise NotImplementedError

########################### Task-specific Calling ###########################
def MIG_all(model_type, model_path, test_data, device, output_path):
    model_selection(model_type, model_path, test_data, device, output_path)
    
def common_object(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'common_object']
    model_selection(model_type, model_path, test_data, device, output_path)

def diff(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'diff']
    model_selection(model_type, model_path, test_data, device, output_path)

def group_grounding(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'group_grounding']
    model_selection(model_type, model_path, test_data, device, output_path)

def multi_view(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'multi_view']
    model_selection(model_type, model_path, test_data, device, output_path)

def object_tracking(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'object_tracking']
    model_selection(model_type, model_path, test_data, device, output_path)

def view_diff(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'view_diff']
    model_selection(model_type, model_path, test_data, device, output_path)

def refer_grounding(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'refer_grounding']
    model_selection(model_type, model_path, test_data, device, output_path)

def region(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'region']
    model_selection(model_type, model_path, test_data, device, output_path)

def correspondence(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'correspondence']
    model_selection(model_type, model_path, test_data, device, output_path)

def reasoning(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'reasoning']
    model_selection(model_type, model_path, test_data, device, output_path)

def worker(gpu_id, task_queue, model_type, test_data, model_path, output_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f"cuda:{str(gpu_id)}")
    
    while True:
        try:
            task = task_queue.get(timeout=5)
        except Empty:
            break
        task(model_type, model_path, test_data, device, output_path)
        task_queue.task_done()
        

if __name__ == "__main__":

    mp.set_start_method("spawn")
    
    with open('/home/tajamul/UniVG-R1/eval/MIG_bench/revised_MIG_bench.json') as file:
        test_data = json.load(file)
        
    output_path = f'./results/qwen2_5_check' # log information output
    os.makedirs(output_path, exist_ok=True)
    
    ############################################## supported models ###############################################
    ### univg-r1, migician, qwen2_vl, internvl2_8b, llava_onevision, minicpm, mantis, cogvlm, mplug_owl3(transformers==4.37.2)
    ### we support multi-processing parallel for migician, qwen2_vl, minicpm on MIG-Bench
    num_gpus, model_type = 1, 'qwen2_5_vl'
    # model_path = "/share/data/drive_4/tj/UniVG-R1"
    # model_path ="/share/data/drive_4/tj/Qwen2-VL-7B-Instruct"
    model_path ="/share/data/drive_4/tj/Qwen2.5-VL-7B-Instruct"
    # model_path = "/share/data/drive_4/umair/Grounding-Work/UniVG-R1/Qwen2.5-VL-32B"
    print(f'😊 In case you forget~ The model you are evaluating is: {model_type} ❤️\n📂 From path: {model_path}')
    ###############################################################################################################
    
    ############################################### supported tasks ###############################################
    ### MIG-Bench ###
    ### MIG_all, common_object, view_diff, correspondence, diff, group_grounding
    ### multi_view, region, refer_grounding, object_tracking, reasoning
    # tasks = [common_object, view_diff, correspondence, diff, group_grounding, multi_view, region, refer_grounding, object_tracking, reasoning]
    tasks = [view_diff]
    ###############################################################################################################
    
    processes = []
    
    task_queue = mp.JoinableQueue()
    for task in tasks:
        task_queue.put(task)

    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker, args=(gpu_id, task_queue, model_type, test_data, model_path, output_path))
        processes.append(p)
        p.start()

    task_queue.join()
    for p in processes:
        p.join()
        
    print(f'😊 In case you forget~ The model you are evaluating is: {model_type} ❤️\n📂 From path: {model_path}')