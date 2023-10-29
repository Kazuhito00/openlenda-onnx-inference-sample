# openlenda-onnx-inference-sample
[OpenLenda（日本の信号機検出＋認識）](https://github.com/turingmotors/openlenda)のPythonでのONNX推論サンプルです。<br>
<br>
![image](https://github.com/Kazuhito00/openlenda-onnx-inference-sample/assets/37477845/3965521a-5c9c-4f7b-8a92-3b17ebafb025)

# Requirement 
* OpenCV 4.8.1.78 or later
* onnxruntime 1.14.1 or later ※GPU推論する際は「onnxruntime-gpu」

# Demo
デモの実行方法は以下です。
```bash
python sample_onnx.py --image=sample.jpg
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --image<br>
画像ファイルの指定 ※指定時はカメラデバイス・動画ファイルより優先<br>
デフォルト：指定なし
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：model/openlenda_s.onnx
* --score_th<br>
スコア閾値<br>
デフォルト：0.5
* --nms_th<br>
NMS閾値<br>
デフォルト：0.01
* --grouping_th<br>
同じバウンディングボックスの結果をグルーピングするIOU閾値<br>
デフォルト：0.8
* --use_gpu<br>
GPU推論<br>
デフォルト：指定なし

# ToDo
* [ ] マルチクラスNMSの暫定実装の見直し

# Reference
* [turingmotors/openlenda](https://github.com/turingmotors/openlenda)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
openlenda-onnx-inference-sample is under [Apache2.0 License](LICENSE).

# License(Image)
サンプル画像は[ぱくたそ](https://www.pakutaso.com/)様の[信号と車のライトがカラフルに反射する雨の道路](https://www.pakutaso.com/20230304087post-46153.html)を使用しています。
