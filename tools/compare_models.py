import re
from _prediction import predict_pth, predict_onnx

# モデルのクラスの定義
class model_info:
    def __init__(self, name):
        self.name = name
        self.type = None
        self.accuracy = None
        self.ave_time = None


def compare(models:list[str]):
    """入力された2つのモデルの精度と推論速度を測定し比較する関数

    Args:
        models list[str]：入力されたモデルのファイル名のリスト
    """
    
    for model in models:
        m = model_info(model)
        if m.name.endswith('onnx'):
            m.type = "onnx"
            m.accuracy, m.ave_time = predict_onnx(m.name)
        else:
            m.type = "pth"
            m.accuracy, m.ave_time = predict_pth(m.name)
            
        print(f'選んだモデル名{m.name}')
        print(f'正解率：{m.accuracy} 推論時間：{m.ave_time}')

    

if __name__ == "__main__":
    models = []
    flag = True
    while flag:
        file_name = input('モデル名を入力してください：')
        if not file_name.endswith(("pth","onnx")):
            print('.pthまたは.onnxファイルを選択してください')
            break
        
        models.append(file_name)
        continue_input = input('他に入力するモデルはありますか（y or n）')
        
        # 例外処理
        if continue_input == "n":
            flag = False
        elif continue_input not in ["y", "n"]:
            print('yまたはnを入力してください')
            break
        
    compare(models)
