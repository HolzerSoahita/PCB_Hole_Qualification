import torch
import cv2
import numpy as np
import pandas as pd
from torch import device
from .YOLOv5Singleton import YOLOv5Singleton
from .SamSingleton import SamSingleton


class InferenceModel:
    def __init__(self, yolov5_model_path, sam_model_path):
        self.yolo_model = YOLOv5Singleton.get_instance(yolov5_model_path)
        self.mask_predictor = SamSingleton.get_instance(sam_model_path)

    def infer(self, img_path):
        results = self.yolo_model(img_path)

        arr_point = results.pandas(
        ).xyxy[0][['xmin', 'ymin', 'xmax', 'ymax']].values
        data_point = np.round(arr_point).astype(int)

        label_point = list(results.pandas().xyxy[0]['class'])

        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        self.mask_predictor.set_image(image_rgb)

        input_boxes = torch.tensor(
            data_point, device=self.mask_predictor.device)

        transformed_boxes = self.mask_predictor.transform.apply_boxes_torch(
            input_boxes, image_rgb.shape[:2])
        masks, _, _ = self.mask_predictor.predict_torch(point_coords=None, point_labels=label_point,
                                                        boxes=transformed_boxes, multimask_output=False)

        data_hole_component = results.pandas().xyxy[0]
        mask_areas = []
        for mask in masks:
            mask = mask.cpu().numpy().astype(np.uint8)
            area = cv2.countNonZero(mask)
            mask_areas.append(area)

        data_hole_component['area'] = mask_areas

        components_data = results.pandas(
        ).xyxy[0][results.pandas().xyxy[0]['class'] == 0].values

        num_component = 1
        for j, row_data in data_hole_component.iterrows():
            if row_data['class'] == 0:
                data_hole_component.at[j, 'component_number'] = num_component
                num_component += 1

        for i in range(len(components_data)):
            row_component = components_data[i]
            for j, row_data in data_hole_component.iterrows():
                if row_data['class'] == 1:
                    if (row_component[0] <= row_data['xmin']) and (row_component[1] <= row_data['ymin']) and \
                            (row_component[2] >= row_data['xmax']) and (row_component[3] >= row_data['ymax']):
                        data_hole_component.at[j, 'component_of_void'] = i + 1

        data = {
            'Component': [],
            'Area': [],
            'void %': [],
            'Max void %': []
        }

        df_results = pd.DataFrame(data)

        for i in range(len(components_data)):
            num = i + 1
            area = data_hole_component[data_hole_component['component_number'] == num]['area'].item(
            )
            total_void = 0
            max_void = 0
            for i, row in data_hole_component.iterrows():
                if row['component_of_void'] == num:
                    total_void += row['area']
                    if row['area'] > max_void:
                        max_void = row['area']

            void_percent = (total_void / area) * 100
            max_void_percent = (max_void / area) * 100

            df_results.loc[len(df_results)] = list({
                'Component': num,
                'Area': area,
                'void %': round(void_percent, 2),
                'Max void %': round(max_void_percent, 2)
            }.values())

        df_results['Component'] = df_results['Component'].astype(int)
        df_results['Area'] = df_results['Area'].astype(int)
        df_results['void %'] = df_results['void %'].astype(float)
        df_results['Max void %'] = df_results['Max void %'].astype(float)

        return df_results
