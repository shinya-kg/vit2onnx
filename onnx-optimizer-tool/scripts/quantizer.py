import os
import onnx
import glob
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, CalibrationDataReader
import numpy as np
from PIL import Image

class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self):
        self.image_path = glob.glob('../images/*.jp*g')
        self.data = [{"input": self._transform(Image.open(img))} for img in self.image_path]
        self.enum_data = iter(self.data)

        
    def _transform(self, image):
        image = image.resize((224,224))
        image = np.array(image).astype(np.float32) / 255.0
        image = np.transpose(image, (2,0,1))
        image = np.expand_dims(image, axis=0)
        return image
    
    def get_next(self):
        return next(self.enum_data, None)
    
    def rewind(self):
        if not self.data:
            raise ValueError("校正データが空です！ ./images フォルダに画像を追加してください。")
        self.enum_data = iter(self.data)
        


class Quantizer:
    def __init__(self, quant_type, input_model, output_model):
        self.quant_type = quant_type
        self.input_model = input_model
        self.output_model = output_model
        self.dir_path = './models'
        
        
    def dynamic_quantization(self):
        
        quantized_model = quantize_dynamic(
            model_input = os.path.join(self.dir_path,self.input_model),
            model_output = os.path.join(self.dir_path,self.output_model),
            weight_type = QuantType.QInt8
        )

        print(f'動的量子化されたモデルを保存しました：{self.output_model}')
        
    def static_quantization(self):
        
        quantized_model = quantize_static(
            model_input = os.path.join(self.dir_path,self.input_model),
            model_output = os.path.join(self.dir_path,self.output_model),
            calibration_data_reader = ImageCalibrationDataReader(),
            weight_type = QuantType.QInt8,
            activation_type = QuantType.QInt8
        )
        
        print(f'静的量子化されたモデルを保存しました：{self.output_model}')
        
        
if __name__ == "__main__":
    quant_type = input('量子化の種類を指定してください。（dynamic:動的量子化, static:静的量子化）')
    if quant_type not in ["dynamic", "static"]:
        print("正しい種類を選択してください")
    else:
        pass
    input_model = input('変換元のモデル名を入力してください：')
    if not input_model.endswith('.onnx'):
        raise ValueError('.onnx形式を選択してください')
    output_model = input('変換後のモデル名を入力してください：')
    if not output_model.endswith('.onnx'):
        raise ValueError('.onnx形式を選択してください')
    
    quantizer = Quantizer(quant_type, input_model, output_model)
    
    if quantizer.quant_type == "dynamic":
        quantizer.dynamic_quantization()
        
    elif quantizer.quant_type == "static":
        quantizer.static_quantization()