import re
from _prediction import predict_pth, predict_onnx


def compare(first_model, second_model):
    """入力された2つのモデルの精度と推論速度を測定し比較する関数

    Args:
        first_model (str): モデル名
        second_model (str): モデル名
    """

    # 他の形式についてはエラーを返す
    if any(not file.endswith(("pth", "onnx")) for file in [first_model, second_model]):
        print(".pthまたは.onnx形式のファイルを選択してください")
        return
    
    print('モデルの入力を受け取りました')
    
    # 1つ目のモデルの精度と推論時間を測定
    res_first = {}
    if first_model.endswith('pth'):
        accuracy, ave_inference_time = predict_pth(first_model)
        res_first['accuracy'] = accuracy
        res_first['ave_inference_time'] = ave_inference_time
    elif first_model.endswith('onnx'):
        accuracy, ave_inference_time = predict_onnx(first_model)
        res_first['accuracy'] = accuracy
        res_first['ave_inference_time'] = ave_inference_time
    
    print('----------------------------')
    print('1つ目のモデルの計算が終了しました')
    
    # 2つ目のモデルの精度と推論時間を測定
    res_second = {}
    if second_model.endswith('pth'):
        accuracy, ave_inference_time = predict_pth(second_model)
        res_second['accuracy'] = accuracy
        res_second['ave_inference_time'] = ave_inference_time
    elif second_model.endswith('onnx'):
        accuracy, ave_inference_time = predict_onnx(second_model)
        res_second['accuracy'] = accuracy
        res_second['ave_inference_time'] = ave_inference_time
    
    print('----------------------------')
    print('2つ目のモデルの計算が終了しました')
    
    # 結果の出力
    print("モデル比較")
    print("------------------------------------")
    print("{:<10} {:<10} {:<10}".format("モデル", "精度", "推論時間[秒]"))
    print("------------------------------------")
    print("{:<10} {:<10.3f} {:<10.3f}".format(first_model, res_first['accuracy'], res_first['ave_inference_time']))
    print("{:<10} {:<10.3f} {:<10.3f}".format(second_model, res_second['accuracy'], res_second['ave_inference_time']))
    print("------------------------------------")
    

if __name__ == "__main__":
    first_model = input("1つ目のモデル名を入力してください：")
    second_model = input("2つ目のモデル名を入力してください：")
    compare(first_model, second_model)
