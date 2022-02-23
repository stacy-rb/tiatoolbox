import requests
from tiatoolbox.models.engine.semantic_segmentor import IOSegmentorConfig, SemanticSegmentor
from tiatoolbox.utils.misc import imread
from tiatoolbox.wsicore.wsireader import WSIReader
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from shapely.geometry import Polygon
import shapely.wkt
import geojson

mpl.rcParams['figure.dpi'] = 150 # for high resolution figure in notebook

SAMPLE_WSI_PATH = r"C:\\Analysis_Studies\\22-108 LifeNet\\Images\\22-108 HE 2122623.mrxs"
ON_GPU = False
MODEL_FILE_NAME = r".\\tissue_mask_model.pth"

TISSUE_MASK_OUTPUT = r"tissue_mask_results_lifenet"

# [Loading the raw prediction]
print('prediction method output is: {}'.format(TISSUE_MASK_OUTPUT))
mask_prediction = np.load(TISSUE_MASK_OUTPUT+'\\0.raw.0.npy') # Loading the first prediction [0] based on the output address [1]
print('Raw prediction dimensions: {}'.format(mask_prediction.shape))

# [Post-processing]
# Simple processing of the raw prediction to generate semantic segmentation task
wsi_tissue_mask = np.argmax(mask_prediction, axis=-1) # select the class with highest probability
print('Processed prediction dimensions: {}'.format(wsi_tissue_mask.shape))

mask = wsi_tissue_mask.astype(np.uint8)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = list(map(np.squeeze, contours))  # removing redundant dimensions)


polygons = [Polygon(c) for c in contours if len(c) > 2]

def wkt_to_geojson(wkt_string):
    geom = shapely.wkt.loads(wkt_string)
    geom = geojson.Feature(geometry=geom, properties={})
    return geom.geometry

def resize_coordinates(geom, ds, px_size):
    geom["coordinates"] = np.array(geom["coordinates"]) * (ds/px_size)
    geom["coordinates"] = geom["coordinates"].tolist()
    return geom

geoms = [wkt_to_geojson(p.wkt) for p in polygons]

write_shapes = []
for i, geom in enumerate(geoms):
    resize_coordinates(geom, 8.0, 0.2423)
    write_shapes.append({ "type": "Feature",
                          "id": "PathAnnotationObject",
                          "geometry": geom,
                          "properties": {"isLocked": False,
                                         "measurements": [],
                                         "classification": {"name": "Other",
                                         "colorRGB": -377282}}
                                    })

with open('lifenet.json', 'w') as j:
     geojson.dump(write_shapes, j)