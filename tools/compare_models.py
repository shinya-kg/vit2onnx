import re
from _prediction import predict_pth, predict_onnx

# モデルのクラスの定義
class Model_Info:
    def __init__(self, name):
        self.name = name
        self.type = None
        self.accuracy = None
        self.ave_time = None
        
def process_model(model_name):
    model_info = Model_Info(model_name)
    if model_info.name.endswith('onnx'):
        model_info.type = "onnx"
        model_info.accuracy, model_info.ave_time = predict_onnx(model_info.name)
    else:
        model_info.type = "pth"
        model_info.accuracy, model_info.ave_time = predict_pth(model_info.name)
    
    return model_info
    
def _print_result(result):
    print("-" * 40)
    print("モデル比較")
    print("-" * 40)
    print(f"{'モデル':<15} {'精度':<10} {'推論時間[秒]':<15}")
    print("-" * 40)
    
    for model, results in result.items():
        print(f"{model:<15} {results['accuracy']:<10.3f} {results['ave_time']:<15.3f}")
    
    print("-" * 40)

def compare(models:list[str]):
    """入力された2つのモデルの精度と推論速度を測定し比較する関数

    Args:
        models list[str]：入力されたモデルのファイル名のリスト
    """
    print('モデルが入力されました')
    result = {}
    for model in models:
        model_info = process_model(model)

        print(f'{model_info.name}の計算が完了しました。')
        result[model_info.name] = {
            "type": model_info.type,
            "accuracy": model_info.accuracy,
            "ave_time": model_info.ave_time
        }
        print("-" * 40)
    return result
    

    

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
        print("-" * 40)
        
    result = compare(models)
    print(_print_result(result))
