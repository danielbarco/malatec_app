import numpy as np
import streamlit as st

from cellpose import models
from cellpose.utils import outlines_list, masks_to_outlines
import cv2
import tifffile
from PIL import Image
import time, os
import json
import requests
import io

import matplotlib.pyplot as plt

# import torch
import torchvision.transforms as transforms

from tensorflow.keras.models import load_model
from tensorflow.image import resize_with_pad
from tensorflow import expand_dims
from tensorflow.nn import softmax

# from streamlit group
from load_css import local_css
local_css("style.css")

from skimage.util import img_as_ubyte
import openflexure_microscope_client as ofm_client

def imread(image_up):
    ext = os.path.splitext(image_up.name)[-1]
    if ext== '.tif' or ext=='tiff':
        img = tifffile.imread(image_up)
        return img
    else:
        img = plt.imread(image_up)
        img = img_as_ubyte(img)
    return img

@st.cache(show_spinner=False)
def run_segmentation(model, image, diam, channels, flow_threshold, cellprob_threshold):
    masks, flows, styles, diams = model.eval(image, 
            # batch_size = 8,
            diameter = diam, # 100
            channels = channels,
            invert = True,
            # rescale = 0.5,
            net_avg=False,
            flow_threshold = flow_threshold, # 1
            cellprob_threshold = cellprob_threshold, # -4
                            )
    return masks, flows, styles, diams

#from cellpose
# @st.cache(show_spinner=False)
def show_cell_outlines(img, maski, color_mask):

    outlines = masks_to_outlines(maski)
    
    # plot the WordCloud image     
    fig, ax = plt.subplots(figsize = (8, 8))                   
    outX, outY = np.nonzero(outlines)
    imgout= img.copy()
    h = color_mask.lstrip('#')
    hex2rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    imgout[outX, outY] = hex2rgb
    # imgout[outX, outY] = np.array([255,75,75])
    ax.imshow(imgout)
    #for o in outpix:
    #    ax.plot(o[:,0], o[:,1], color=[1,0,0], lw=1)
    ax.set_title('Predicted outlines')
    ax.axis('off')
    
    return fig

@st.cache(show_spinner=False)
def get_cell_outlines(masks):
    outlines_ls = outlines_list(masks)
    return outlines_ls

@st.cache(show_spinner=False)
def transform_image(arr):
    my_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                        # transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
#     image = Image.open(io.BytesIO(image_bytes))
    im = Image.fromarray(arr)
    return my_transforms(im).unsqueeze(0)

class_names = ['parasitized', 'uninfected']

def get_prediction(arr):
    tensor = transform_image(arr)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return class_names[y_hat]

def get_prediction_tf(arr):
    '''function for tf model; resizes and pads an image array and then returns the prediction i.e. the most likely class'''
    arr = resize_with_pad(arr, 100, 100, method= 'bilinear', antialias=False)
    arr = expand_dims(arr, 0) # Create a batch
    predictions = pair_D_ensemble_model.predict(arr)
    score = softmax(predictions[0])
    #print(class_names[np.argmax(score)])
    return class_names[np.argmax(score)]

@st.cache(show_spinner=False)
def check_openflexure(client_name = ''):
    '''sanity checking openflexure microscope'''
    try:
        microscope = ofm_client.MicroscopeClient(client_name)
        pos = microscope.position
        starting_pos = pos.copy()
        pos['x'] += 100
        microscope.move(pos)
        assert microscope.position == pos
        pos['x'] -= 100
        microscope.move(pos)
        assert microscope.position == starting_pos
        # Check the microscope will autofocus
        ret = microscope.autofocus()
        return microscope
    except:
        pass

@st.cache(show_spinner=False)
def capture_full_resolution_image(base_uri, params: dict = None):
    """Capture an image and return it as a image """
    payload = {
        "use_video_port": False,
        "bayer": False,
    }
    if params:
        payload.update(params)
    r = requests.post(base_uri + "/actions/camera/capture", json=payload, headers={'Accept': 'image/jpeg'}, timeout =60)
    r.raise_for_status()
    action_href = json.loads(r.content.decode('utf-8'))['href']
    time.sleep(2)
    r = requests.get(action_href)
    if json.loads(r.content.decode('utf-8'))['output'] is not None:
        r = requests.get(json.loads(r.content.decode('utf-8'))['output']['links']['download']['href'])
        image = np.array(Image.open(io.BytesIO(r.content)))
        return image
    else:
        time.sleep(2)
        if json.loads(r.content.decode('utf-8'))['output'] is not None:
            r = requests.get(json.loads(r.content.decode('utf-8'))['output']['links']['download']['href'])
            image = np.array(Image.open(io.BytesIO(r.content)))
            return image
        
def capture_full_resolution_image_no_cache(base_uri, params: dict = None):
    """Capture an image and return it as a image streamlit no cache"""
    payload = {
        "use_video_port": False,
        "bayer": False,
    }
    if params:
        payload.update(params)
    r = requests.post(base_uri + "/actions/camera/capture", json=payload, headers={'Accept': 'image/jpeg'}, timeout =60)
    r.raise_for_status()
    action_href = json.loads(r.content.decode('utf-8'))['href']
    time.sleep(2)
    r = requests.get(action_href)
    if json.loads(r.content.decode('utf-8'))['output'] is not None:
        r = requests.get(json.loads(r.content.decode('utf-8'))['output']['links']['download']['href'])
        image = np.array(Image.open(io.BytesIO(r.content)))
        return image
    else:
        time.sleep(2)
        if json.loads(r.content.decode('utf-8'))['output'] is not None:
            r = requests.get(json.loads(r.content.decode('utf-8'))['output']['links']['download']['href'])
            image = np.array(Image.open(io.BytesIO(r.content)))
            return image
        
# @st.cache(show_spinner=False)
def grab_image(x,y,z):
    image = microscope.grab_image_array()
    return image

@st.cache(allow_output_mutation=True)
def return_inputs():
    return {"x": None, "y": None, "z": None, "autofocus": None,}


def openflexure_autofocus(x,y):
    microscope.autofocus()
    x, y, z = microscope.get_position_array()
    return x, y, z

st.title('P. falciparum Malaria Detection')
# st.text('Segmentataion -> Single cell ROI -> Classification')

page = st.sidebar.selectbox("Choose a stain", ('Giemsa', 'Stain type 2', 'Sample images'))

st.sidebar.title("About")
st.sidebar.info(" - Segmentation: [Cellpose] (https://github.com/MouseLand/cellpose)    \n \
- Classification: SqueezeNet + VGG19 [Article] (https://peerj.com/articles/6977.pdf) [GitHub] (https://github.com/sivaramakrishnan-rajaraman/Deep-Neural-Ensembles-toward-Malaria-Parasite-Detection-in-Thin-Blood-Smear-Images)     \n \
- Trained on Giemsa stained P. _falsiparum_  [NIH Data] (https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html)   \n \
- Powered by PyTorch, TensorFlow [Streamlit] (https://docs.streamlit.io/en/stable/api.html) ")

with st.spinner("Connecting to Openflexure Microscope"):   
    microscope = check_openflexure()
    microscope = check_openflexure('microscope')
    

file_up = None

# if we detect a microscope run this code:
if microscope:
    
    image = capture_full_resolution_image(microscope.base_uri)
    x, y, z = microscope.get_position_array()
    # fov: [4100, 3146]

    status_inputs = return_inputs()
    colx, coly = st.beta_columns(2)
    with colx:
        user_x = colx.number_input('x-axis', -20000, 20000, x, 800)
        if user_x:
            x = user_x
            status_inputs.update({"x": True})
    with coly:
        user_y = coly.number_input('y-axis', -20000, 20000, y, 640)
        if user_y:
            y = user_y
            status_inputs.update({"y": True})
    # with colz:
    #     user_z = colz.number_input('z-axis', -30000, 30000, z, 2)
    #     if user_z:
    #         z = user_z
    #         status_inputs.update({"z": True})
    # with autofocus:
    #     if autofocus.button('Autofocus'):
    #         status_inputs.update({"autofocus": True})
        
    # if st.button('High resolution image'):
    #     print(image.shape)
    
    if status_inputs["x"] or status_inputs["y"]:
        microscope.move([x,y,z])
        x,y,z = openflexure_autofocus(x, y)  
        image = capture_full_resolution_image_no_cache(microscope.base_uri)
        
    # if status_inputs['z']:
    #     microscope.move([x,y,z])
    #     image = capture_full_resolution_image_no_cache(microscope.base_uri)
        
         

    # if status_inputs["autofocus"]:
        
    #     image = capture_full_resolution_image_no_cache(microscope.base_uri)
   
    
            

    file_up = True

elif page == 'Sample images':
    img_list = ["./images/T0D2_1.tif", "./images/T14D2_2.tif", "./images/T38D2_2.tif", "./images/T48D2_2.tif" ]
    img_captions = ["After 2 hours", "After 14 hours", "After 38 hours", "After 48 hours", "<choose>"]
    st.image(img_list, caption = img_captions[:4], width = int(698/2))
    selected_image = st.selectbox("Choose a sample image to analyze", img_captions, 4)
    if selected_image != "<choose>":
        selected_image = img_captions.index(selected_image)
        file_up = img_list[selected_image]
        # st.text(file_up)
        image = tifffile.imread(file_up)
else:
    file_up = st.file_uploader("Upload an image", type=["tif", "tiff", "png", "jpg", "jpeg"])
    if file_up:
        image = imread(file_up)

if file_up:
    # @st.cache
    # image = Image.open(file_up)
    

    fig, ax = plt.subplots(figsize = (8,8))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title('Selected image')
    st.pyplot(fig)
    if microscope:
        diameter = 190
    else:
        st.subheader('Segmentation parameters')
        diameter = st.number_input('Diameter of the cells [pix]', 0, 500, 190, 10)
        st.write('The current diameter is ', diameter)


    flow_threshold = 1
    # flow_threshold = st.slider('Flow threshold (increase -> more cells)', .0, 1.1, 1.0, 0.1)
    # st.write("", flow_threshold)

    cellprob_threshold = 0
    # cellprob_threshold = st.slider('Cell probability threshold (decrease -> more cells)', -6, 6, -4, 1)
    # st.write("", cellprob_threshold)

    color_mask = '#000000'
    # color_mask = st.color_picker('Pick a color for cell outlines', '#000000')
    # st.write('The current color is', color_mask)

    if st.button('Analyze'):
        #print(x, y, z)
        # DEFINE CELLPOSE MODEL
        # model_type='cyto' or model_type='nuclei'
        with st.spinner("Running segmentation"):
            model = models.Cellpose(gpu=False, model_type ='cyto')
            # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
            channels = [[0,0]] #* len(files) # IF YOU HAVE GRAYSCALE

            since = time.time()
            # img = io.imread(filename)
            masks, flows, styles, diams = run_segmentation(model, image, diameter, channels, 
                                                flow_threshold, cellprob_threshold)
            st.text('Initial cell count: {} '.format(masks.max()))
            
            time_elapsed = time.time() - since
            st.write('Time spent on segmentation {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        # if st.button('Show results'):
            # DISPLAY RESULTS
            fig = show_cell_outlines(image, masks, color_mask)
            st.pyplot(fig)
        with st.spinner("Getting single cells"):
            outlines_ls = get_cell_outlines(masks)

        
        with st.spinner("Loading Model"):
            PATH = 'ensemblemodel_pairD.h5'
            pair_D_ensemble_model=load_model(PATH)
            pair_D_ensemble_model.summary()
            # device = torch.device('cpu')
            # # Load cnn model
            # PATH = "model.pth"
            # model = torch.load(PATH, map_location = device)
            # model.eval()
        
        size_thres = diameter*0.5
        tmp_img = image.copy()
        d_results = {"parasitized": [],
                    "uninfected": [],
                    }
        
        with st.spinner("Running inference..."):
        # st.text("Running inference ...")
            since = time.time()
            for idx, cell in enumerate(outlines_ls[:]):
                
                x = cell.flatten()[::2]
                y = cell.flatten()[1::2]

                if (y.max() - y.min()) < size_thres or (x.max() - x.min()) < size_thres:
                    continue

                # mask outline
                mask = np.zeros(tmp_img.shape, dtype=np.uint8)
                channel_count = tmp_img.shape[2]  # i.e. 3 or 4 depending on your image
                ignore_mask_color = (255,)*channel_count
                # fill contour
                cv2.fillConvexPoly(mask, cell, ignore_mask_color)
                # extract roi
                masked_image = cv2.bitwise_and(tmp_img, mask)
                # crop the box around the cell
                (topy, topx) = (np.min(y), np.min(x))
                (bottomy, bottomx) = (np.max(y), np.max(x))
                out = masked_image[topy:bottomy+1, topx:bottomx+1,:]
                # predict the stage of the cell p
                stage = get_prediction_tf(out)
                d_results[stage].append(idx)
  
        time_elapsed = time.time() - since
        st.write('time spent on classification {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


        with st.spinner("Plotting results"):
            t = "<div> <span class='highlight darkgreen'> Uninfected </span> \
            <span class='highlight red'> Parasitized </span>"
            st.markdown(t, unsafe_allow_html=True)

            colors_stage = { "parasitized": "#FF0000", "uninfected": '#458B00'}
            fig, ax = plt.subplots(figsize = (8,8))
            # yellow: ring; magenta: troph; cyan: shiz
            ax.imshow(image)

            for k in class_names:
                # if k!= 'uninfected' and len(d_results[k]) > 0:
                if len(d_results[k]) > 0:
                    for cell in d_results[k]:
                        coord = outlines_ls[cell]
                        ax.plot(coord[:,0], coord[:,1], c = colors_stage[k], lw=1)
            ax.set_title('Predicted infected cells')
            ax.axis('off')
            st.pyplot(fig)

            total_count = sum(len(v)for v in d_results.values())
            st.write("Final cell count", total_count)
            out_stat = []
            for key in class_names:
                stage_count = len(d_results[key])
                # st.write(key, stage_count, round(stage_count/total_count, 3))
                paras = round(stage_count/total_count, 3)
                out_stat.append((stage_count, paras))
            par = (out_stat[0][1])*100
            st.write("Parasitemia (%)", round(par, 2) )
            st.markdown(f"""
                | Stage       |      Count         |       %             |
                | ------------| -------------      | ----------          |
                | Parasitized | {out_stat[0][0]}   |  {out_stat[0][1]}   | 
                | Uninfected  | {out_stat[1][0]}   |  {out_stat[1][1]}   |

            """)